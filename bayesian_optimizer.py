"""
Model 4 定数のベイズ最適化スクリプト
--------------------------------------
使い方:
  1. LESを手動で実行してCSVを出力
  2. このスクリプトを実行 → 次に試すべき定数セットを表示
  3. Cコードの #define を書き換えて再度LES実行
  4. 2に戻る

対象定数:
  FLAME_ACCEL_STRENGTH  (強さ)
  FLAME_ACCEL_WIDTH     (幅: ガウス分布のσ)
  FLAME_ACCEL_CENTER    (中心G値)
  CS_UNBURNED           (未燃側Cs)
  CS_BURNT              (既燃側Cs)
  CS_FLAME_BOOST        (火炎帯ブースト)
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

warnings.filterwarnings('ignore')
plt.rcParams['agg.path.chunksize'] = 10000


# ==============================================================================
# 1. 実験値（論文Fig. 3.1, 3.2から近似）
# ==============================================================================

def experimental_reynolds_stress(G):
    """
    実験値のレイノルズ応力プロファイルの近似関数
    Excelの近似式から構築:
        f(G) = E1 * exp(-E2 * G) + E3 * exp(-E4 * G)

    係数:
        E1 =  0.020  (正の速い減衰成分)
        E2 = 15.0    (速い減衰の時定数)
        E3 =  0.005  (正の緩やかな減衰成分)
        E4 =  1.5    (緩やかな減衰の時定数)

    挙動:
        G = 0.0: 0.020 + 0.005 = 0.025 (最大値)
        G = 0.3: 速い成分がほぼ消え、緩やか成分のみ残る
        G = 1.0: 0.005 * exp(-1.5) ≈ 0.0011 に収束
    """
    G = np.atleast_1d(np.array(G, dtype=float))

    E1 = 0.020
    E2 = 15.0
    E3 = 0.005
    E4 = 1.5

    return E1 * np.exp(-E2 * G) + E3 * np.exp(-E4 * G)


# ==============================================================================
# 2. CSVからレイノルズ応力プロファイルを計算
# ==============================================================================

def load_and_compute_reynolds_stress(csv_dir, loc_index=0, num_bins=20):
    """
    probe_data_flame_region_loc{loc_index}.csv を読み込み、
    G値でビニングしてレイノルズ応力 <u'v'> を計算する。

    Args:
        csv_dir: CSVファイルがあるディレクトリ
        loc_index: 計測位置のインデックス（0〜NUM_PROBE_POSITIONS-1）
        num_bins: Gのビン数

    Returns:
        G_bins: 各ビンのG値中心
        stress_vals: 各ビンのレイノルズ応力値
    """
    csv_path = os.path.join(csv_dir, f"probe_data_flame_region_loc{loc_index}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)

    # 必要な列の確認
    required_cols = ['u', 'v', 'G']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' がCSVにありません。列名: {df.columns.tolist()}")

    # Gのビニング
    G_edges = np.linspace(0.0, 1.0, num_bins + 1)
    G_centers = 0.5 * (G_edges[:-1] + G_edges[1:])

    stress_vals = np.full(num_bins, np.nan)

    for b in range(num_bins):
        mask = (df['G'] >= G_edges[b]) & (df['G'] < G_edges[b + 1])
        subset = df[mask]

        if len(subset) < 10:
            continue

        # <u'v'> = <uv> - <u><v>
        u_mean = subset['u'].mean()
        v_mean = subset['v'].mean()
        uv_mean = (subset['u'] * subset['v']).mean()
        stress_vals[b] = uv_mean - u_mean * v_mean

    return G_centers, stress_vals


# ==============================================================================
# 3. グラフ描画
# ==============================================================================

def plot_reynolds_stress_profile(
    csv_dir, loc_index, num_bins,
    G_centers, stress_sim,
    current_params, score,
    trial_number,
    save_dir=None,
    smooth_window=5
):
    """
    レイノルズ応力プロファイルのグラフを描画・保存する。

    上段: 詳細プロファイル（dG=0.01）＋スムージング曲線＋実験近似
    下段: スコア計算に使った粗いビニング結果＋定数テキスト
    """
    if save_dir is None:
        save_dir = csv_dir
    os.makedirs(save_dir, exist_ok=True)

    # --- 詳細プロファイル（dG=0.01）の計算 ---
    csv_path = os.path.join(csv_dir, f"probe_data_flame_region_loc{loc_index}.csv")
    G_fine, stress_fine, sizes_fine = None, None, None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df = df.dropna(subset=['u', 'v', 'G'])
            bins_fine = np.arange(0.0, 1.01, 0.01)
            results_fine = []
            for i in range(len(bins_fine) - 1):
                g_min, g_max = bins_fine[i], bins_fine[i + 1]
                g_c = (g_min + g_max) / 2.0
                sub = df[(df['G'] >= g_min) & (df['G'] < g_max)]
                if len(sub) < 10:
                    continue
                u_m = sub['u'].mean()
                v_m = sub['v'].mean()
                uv_m = (sub['u'] * sub['v']).mean()
                results_fine.append((g_c, uv_m - u_m * v_m, len(sub)))
            if results_fine:
                res_df = pd.DataFrame(results_fine, columns=['G', 'Stress', 'N'])
                G_fine      = res_df['G'].values
                stress_fine = res_df['Stress'].values
                sizes_fine  = res_df['N'] / res_df['N'].max() * 80 + 10
        except Exception as e:
            print(f"  [警告] 詳細プロファイル計算エラー: {e}")

    # --- スムージング曲線 ---
    valid = ~np.isnan(stress_sim)
    G_smooth, sim_smooth = None, None
    if valid.sum() >= 5:
        G_smooth, sim_smooth = smooth_profile(
            G_centers[valid], stress_sim[valid], smooth_window=smooth_window
        )

    # --- 実験近似曲線 ---
    G_exp     = np.linspace(0.0, 1.0, 200)
    stress_exp = experimental_reynolds_stress(G_exp)

    # --- 描画 ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(
        f'Reynolds Stress Profile  |  Trial {trial_number}  |  Score: {score:.5f}',
        fontsize=14
    )

    # ---- 上段: 詳細プロファイル（dG=0.01）----
    ax = axes[0]
    ax.set_title("Detailed Profile (dG=0.01)", fontsize=12)
    if G_fine is not None:
        sc = ax.scatter(
            G_fine, stress_fine,
            s=sizes_fine, c=G_fine, cmap='jet',
            edgecolors='k', linewidths=0.3,
            alpha=0.8, zorder=3, label='LES (dG=0.01)'
        )
        ax.plot(G_fine, stress_fine, 'k-', alpha=0.2, linewidth=0.8, zorder=2)
        plt.colorbar(sc, ax=ax, label='G value')
    if G_smooth is not None:
        ax.plot(G_smooth, sim_smooth, 'b-', linewidth=2.5, zorder=4,
                label=f'Smoothed (window={smooth_window})')
    ax.plot(G_exp, stress_exp, 'r--', linewidth=2, zorder=5,
            label='Experiment (approx.)')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Reaction Progress Variable G')
    ax.set_ylabel("<u'v'> (m\u00b2/s\u00b2)")
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)

    # ---- 下段: 粗いビニング + 定数情報 ----
    ax2 = axes[1]
    ax2.set_title(f"Score Evaluation Profile (dG={1/num_bins:.2f})", fontsize=12)
    valid_mask = ~np.isnan(stress_sim)
    ax2.scatter(
        G_centers[valid_mask], stress_sim[valid_mask],
        s=60, color='steelblue', edgecolors='k', linewidths=0.5,
        zorder=3, label='LES (binned)'
    )
    ax2.plot(G_centers[valid_mask], stress_sim[valid_mask],
             'k-', alpha=0.3, linewidth=0.8, zorder=2)
    if G_smooth is not None:
        ax2.plot(G_smooth, sim_smooth, 'b-', linewidth=2, zorder=4,
                 label='Smoothed')
    ax2.plot(G_exp, stress_exp, 'r--', linewidth=2, zorder=5,
             label='Experiment (approx.)')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Reaction Progress Variable G')
    ax2.set_ylabel("<u'v'> (m\u00b2/s\u00b2)")
    ax2.grid(True, alpha=0.4)
    ax2.legend(loc='upper right', fontsize=9)

    # 定数テキストボックス
    param_text = '\n'.join([
        f"FLAME_ACCEL_STRENGTH = {current_params['FLAME_ACCEL_STRENGTH']:.4f}",
        f"FLAME_ACCEL_WIDTH    = {current_params['FLAME_ACCEL_WIDTH']:.4f}",
        f"FLAME_ACCEL_CENTER   = {current_params['FLAME_ACCEL_CENTER']:.4f}",
        f"CS_UNBURNED          = {current_params['CS_UNBURNED']:.4f}",
        f"CS_BURNT             = {current_params['CS_BURNT']:.4f}",
        f"CS_FLAME_BOOST       = {current_params['CS_FLAME_BOOST']:.4f}",
    ])
    ax2.text(
        0.02, 0.98, param_text,
        transform=ax2.transAxes, fontsize=8,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = os.path.join(save_dir, f"trial_{trial_number:03d}_stress_profile.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  グラフを保存しました: {filename}")


def plot_optimization_history(optimizer, save_dir):
    """スコアの推移と最良値の累積グラフを保存する"""
    if len(optimizer.y_obs) < 2:
        return
    os.makedirs(save_dir, exist_ok=True)
    scores      = np.array(optimizer.y_obs)
    trials      = np.arange(1, len(scores) + 1)
    best_so_far = np.minimum.accumulate(scores)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trials, scores, 'o-', color='steelblue', linewidth=1.5,
            markersize=6, alpha=0.7, label='Score per trial')
    ax.plot(trials, best_so_far, 'r-', linewidth=2.5, label='Best score so far')
    best_idx = int(np.argmin(scores))
    ax.scatter([trials[best_idx]], [scores[best_idx]],
               s=150, color='red', zorder=5, marker='*', label='Best trial')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('Score (lower is better)')
    ax.set_title('Bayesian Optimization: Score History')
    ax.grid(True, alpha=0.4)
    ax.legend()
def plot_velocity_distribution(
    csv_dir, loc_index,
    current_params, score,
    trial_number,
    save_dir=None,
    kde_sample_limit=100000
):
    """
    流速分布グラフを描画・保存する。
    参考コードの task_1_1_bimodal_distribution および
    task_1_2_conditional_pdfs に相当。

    構成（3行2列）:
        左列: u, v, w の全体ヒストグラム + KDE
        右列: u, v の未燃/既燃 条件付きKDE（bi-modal確認）
              w はG値3ゾーン別 KDE（火炎帯の影響確認）
    """
    if save_dir is None:
        save_dir = csv_dir
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, f"probe_data_flame_region_loc{loc_index}.csv")
    if not os.path.exists(csv_path):
        print(f"  [警告] 流速分布: CSVが見つかりません: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df = df.dropna(subset=['u', 'v', 'w', 'G'])
    except Exception as e:
        print(f"  [警告] 流速分布: CSV読み込みエラー: {e}")
        return

    if df.empty:
        print("  [警告] 流速分布: 有効データがありません。")
        return

    unburnt = df[df['G'] < 0.5]
    burnt   = df[df['G'] >= 0.5]

    from scipy.stats import gaussian_kde

    def _kde_line(series, n_points=500):
        s = series.dropna()
        if len(s) < 10:
            return None, None
        s_kde = s.sample(min(len(s), kde_sample_limit), random_state=42)
        try:
            kde = gaussian_kde(s_kde)
            x   = np.linspace(s.min(), s.max(), n_points)
            return x, kde(x)
        except Exception:
            return None, None

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(
        f'Velocity Distribution  |  Trial {trial_number}  |  Score: {score:.5f}',
        fontsize=13
    )

    components = ['u', 'v', 'w']
    colors     = ['steelblue', 'tomato', 'seagreen']

    # ---- 左列: 全体ヒストグラム + KDE ----
    for row, (comp, color) in enumerate(zip(components, colors)):
        ax = axes[row, 0]
        vel = df[comp].dropna()
        ax.hist(vel, bins=100, density=True, color=color, alpha=0.4, label='All data')
        x_kde, y_kde = _kde_line(vel)
        if x_kde is not None:
            ax.plot(x_kde, y_kde, color=color, lw=2, label='KDE')
        ax.set_xlabel(f'{comp}-velocity (m/s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{comp}-velocity: Overall Distribution')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

    # ---- 右列 上2段: u, v の条件付き分布（未燃 / 既燃） ----
    for row, comp in enumerate(['u', 'v']):
        ax = axes[row, 1]
        if len(unburnt) > 10:
            x_u, y_u = _kde_line(unburnt[comp])
            if x_u is not None:
                ax.fill_between(x_u, y_u, alpha=0.3, color='blue')
                ax.plot(x_u, y_u, color='blue', lw=2, label='Unburnt (G<0.5)')
        if len(burnt) > 10:
            x_b, y_b = _kde_line(burnt[comp])
            if x_b is not None:
                ax.fill_between(x_b, y_b, alpha=0.3, color='red')
                ax.plot(x_b, y_b, color='red', lw=2, label='Burnt (G\u22650.5)')
        ax.set_xlabel(f'{comp}-velocity (m/s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'{comp}-velocity: Conditional PDF (Unburnt vs Burnt)')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

    # ---- 右列 下段: w のGゾーン別分布 ----
    ax = axes[2, 1]
    bins_g = [
        (0.0, 0.3, 'blue',   'G < 0.3 (Unburnt)'),
        (0.3, 0.7, 'orange', '0.3 \u2264 G < 0.7 (Flame zone)'),
        (0.7, 1.0, 'red',    'G \u2265 0.7 (Burnt)'),
    ]
    for g_lo, g_hi, col, lbl in bins_g:
        sub = df[(df['G'] >= g_lo) & (df['G'] < g_hi)]['w'].dropna()
        if len(sub) < 10:
            continue
        x_k, y_k = _kde_line(sub)
        if x_k is not None:
            ax.fill_between(x_k, y_k, alpha=0.25, color=col)
            ax.plot(x_k, y_k, color=col, lw=2, label=lbl)
    ax.set_xlabel('w-velocity (m/s)')
    ax.set_ylabel('Probability Density')
    ax.set_title('w-velocity: Conditional PDF by G zone')
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    # 定数テキストボックス（図の下部）
    param_text = '  '.join([
        f"STRENGTH={current_params['FLAME_ACCEL_STRENGTH']:.3f}",
        f"WIDTH={current_params['FLAME_ACCEL_WIDTH']:.3f}",
        f"CENTER={current_params['FLAME_ACCEL_CENTER']:.3f}",
        f"CS_U={current_params['CS_UNBURNED']:.3f}",
        f"CS_B={current_params['CS_BURNT']:.3f}",
        f"CS_BOOST={current_params['CS_FLAME_BOOST']:.3f}",
    ])
    fig.text(
        0.5, 0.01, param_text,
        fontsize=8, fontfamily='monospace',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    filename = os.path.join(save_dir, f"trial_{trial_number:03d}_velocity_dist.png")
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  グラフを保存しました: {filename}")


# ==============================================================================
# 4. スコア計算（目的関数）
# ==============================================================================

def smooth_profile(G_v, sim_v, n_points=50, smooth_window=5):
    """
    固定点データを一様グリッドに補間してスムージングする。

    手順:
        1. 有効点をスプライン補間で一様グリッドへ変換
        2. 移動平均でスムージング（数値振動を除去）
        3. スムージング後の曲線を返す

    Args:
        G_v      : 有効なGビンの代表値（NaN除外済み）
        sim_v    : 対応するレイノルズ応力値
        n_points : 補間後のグリッド点数
        smooth_window : 移動平均のウィンドウ幅（奇数推奨）

    Returns:
        G_smooth  : スムージング後のG配列
        sim_smooth: スムージング後のレイノルズ応力配列
    """
    from scipy.interpolate import UnivariateSpline

    # 重複するG値を平均してからソート
    df_tmp = pd.DataFrame({'G': G_v, 'stress': sim_v})
    df_tmp = df_tmp.groupby('G', as_index=False).mean().sort_values('G')
    G_sorted = df_tmp['G'].values
    s_sorted = df_tmp['stress'].values

    if len(G_sorted) < 4:
        # 点数が少なすぎる場合はそのまま返す
        return G_sorted, s_sorted

    # スプライン補間（k=3: 3次、s=0: 全点通過）
    # データが荒い場合は s > 0 で平滑化スプラインも選択可
    try:
        spl = UnivariateSpline(G_sorted, s_sorted, k=3, s=0, ext=3)
    except Exception:
        # フォールバック: 線形補間
        spl = UnivariateSpline(G_sorted, s_sorted, k=1, s=0, ext=3)

    G_uniform = np.linspace(G_sorted[0], G_sorted[-1], n_points)
    sim_interp = spl(G_uniform)

    # 移動平均でスムージング
    w = smooth_window
    kernel = np.ones(w) / w
    # 境界は端の値でパディング
    sim_padded = np.pad(sim_interp, w // 2, mode='edge')
    sim_smooth = np.convolve(sim_padded, kernel, mode='valid')

    return G_uniform, sim_smooth


def compute_score(G_centers, stress_sim, smooth_window=5, verbose=False):
    """
    シミュレーション結果のスコアを計算。

    考え方:
        固定点計測 + 数値振動があっても、スムージング後の曲線で
        「全体的な減衰傾向」が実験と一致していれば高評価とする。

        生データの上下振動は無視し、スムージング曲線だけを評価する。

    スコアの構成:
        1. 傾向スコア  : スムージング曲線の単調減少度合い
        2. 形状スコア  : 実験近似曲線との相関（形が似ているか）
        3. レンジスコア: G=0〜1 の変化量が実験と同程度あるか

    Args:
        G_centers   : Gのビン代表値配列
        stress_sim  : シミュレーションのレイノルズ応力（生データ）
        smooth_window: スムージングのウィンドウ幅
        verbose     : Trueなら各スコアの内訳を表示

    Returns:
        score: スコア（小さいほど良い）
    """
    # NaNを除外
    valid = ~np.isnan(stress_sim)
    if valid.sum() < 5:
        return 1e6

    G_v   = G_centers[valid]
    sim_v = stress_sim[valid]

    # ------------------------------------------------------------------
    # ステップ1: スムージング
    #   数値振動を除去した「傾向曲線」を生成する
    # ------------------------------------------------------------------
    G_smooth, sim_smooth = smooth_profile(G_v, sim_v, smooth_window=smooth_window)

    # スムージング後の実験値（同じGグリッドで評価）
    exp_smooth = experimental_reynolds_stress(G_smooth)

    # ------------------------------------------------------------------
    # スコア1: 単調減少ペナルティ（スムージング曲線に対して評価）
    #   スムージング後に隣接点間で「増加」していた場合のみペナルティ。
    #   数値振動は吸収済みなので、真の傾向の逆転だけを捉える。
    # ------------------------------------------------------------------
    diffs = np.diff(sim_smooth)
    increase_penalty = float(np.sum(np.maximum(0.0, diffs)))

    # ------------------------------------------------------------------
    # スコア2: 形状相関（スムージング曲線 vs 実験近似曲線）
    #   ピアソン相関係数を使用。1に近いほど形が似ている。
    # ------------------------------------------------------------------
    sim_std = sim_smooth.std()
    exp_std = exp_smooth.std()
    if sim_std < 1e-10 or exp_std < 1e-10:
        shape_score = 1.0
    else:
        corr = np.corrcoef(sim_smooth, exp_smooth)[0, 1]
        corr = float(np.clip(corr, -1.0, 1.0))
        shape_score = 1.0 - corr  # 相関が高いほど0に近い

    # ------------------------------------------------------------------
    # スコア3: 変化レンジ
    #   実験の変化量に対してシミュの変化量が小さすぎる場合にペナルティ。
    #   （変化がほぼゼロ = 傾向が出ていない を検出）
    # ------------------------------------------------------------------
    sim_range = float(sim_smooth.max() - sim_smooth.min())
    exp_range = float(exp_smooth.max() - exp_smooth.min())
    range_ratio = sim_range / (exp_range + 1e-10)
    range_score = max(0.0, 1.0 - range_ratio)

    # ------------------------------------------------------------------
    # 重み付き合算
    # ------------------------------------------------------------------
    w_trend = 5.0
    w_shape = 2.0
    w_range = 1.0

    score = w_trend * increase_penalty + w_shape * shape_score + w_range * range_score

    if verbose:
        print(f"  [スコア内訳]")
        print(f"    単調減少ペナルティ : {increase_penalty:.6f}  (×{w_trend} = {w_trend*increase_penalty:.6f})")
        print(f"    形状相関スコア     : {shape_score:.6f}  (×{w_shape} = {w_shape*shape_score:.6f})")
        print(f"    レンジスコア       : {range_score:.6f}  (×{w_range} = {w_range*range_score:.6f})")
        print(f"    合計スコア         : {score:.6f}")

    return score


# ==============================================================================
# 4. ベイズ最適化クラス
# ==============================================================================

class BayesianOptimizer:
    """
    ガウス過程回帰を用いたベイズ最適化
    """

    # 最適化対象の定数と探索範囲
    PARAM_BOUNDS = {
        'FLAME_ACCEL_STRENGTH': (0.1, 5.0),
        'FLAME_ACCEL_WIDTH':    (0.05, 0.30),
        'FLAME_ACCEL_CENTER':   (0.40, 0.80),
        'CS_UNBURNED':          (0.08, 0.20),
        'CS_BURNT':             (0.30, 0.70),
        'CS_FLAME_BOOST':       (0.10, 0.40),
    }

    def __init__(self, history_file='optimization_history.json'):
        self.param_names = list(self.PARAM_BOUNDS.keys())
        self.bounds = np.array([self.PARAM_BOUNDS[k] for k in self.param_names])
        self.history_file = history_file

        # GPRモデルの初期化
        kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

        # 履歴の読み込みまたは初期化
        self.X_obs = []  # 試した定数セット
        self.y_obs = []  # 対応するスコア
        self._load_history()

    def _load_history(self):
        """過去の試行履歴を読み込む"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            self.X_obs = data.get('X', [])
            self.y_obs = data.get('y', [])
            print(f"履歴を読み込みました: {len(self.y_obs)} 件の試行")
        else:
            print("履歴なし。新規スタートです。")

    def _save_history(self):
        """履歴を保存する"""
        with open(self.history_file, 'w') as f:
            json.dump({'X': self.X_obs, 'y': self.y_obs}, f, indent=2)

    def _normalize(self, x):
        """定数を [0, 1] に正規化"""
        return (x - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _denormalize(self, x_norm):
        """[0, 1] から実際の定数値に戻す"""
        return x_norm * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

    def _acquisition(self, X_cand, xi=0.01):
        """
        獲得関数: Expected Improvement (EI)
        現在の最良スコアをどれだけ改善できるかを推定する。
        """
        if len(self.y_obs) < 2:
            return np.random.rand(len(X_cand))

        X_norm = np.array([self._normalize(x) for x in self.X_obs])
        self.gp.fit(X_norm, self.y_obs)

        X_cand_norm = np.array([self._normalize(x) for x in X_cand])
        mu, sigma = self.gp.predict(X_cand_norm, return_std=True)

        # スコアは小さいほど良いので最小化問題として扱う
        y_best = np.min(self.y_obs)
        z = (y_best - mu - xi) / (sigma + 1e-9)
        ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-10] = 0.0

        return ei

    def suggest_next(self, n_candidates=10000):
        """
        次に試すべき定数セットを提案する。

        Returns:
            dict: パラメータ名 → 推奨値
        """
        if len(self.y_obs) < 3:
            # 試行が少ない場合はランダムに提案
            x_rand = np.random.uniform(
                self.bounds[:, 0], self.bounds[:, 1]
            )
            suggestion = dict(zip(self.param_names, x_rand))
            print("（試行数が少ないためランダム提案）")
            return suggestion

        # ランダム候補からEIが最大のものを選ぶ
        X_cand = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(n_candidates, len(self.param_names))
        )
        ei_vals = self._acquisition(X_cand)
        best_idx = np.argmax(ei_vals)
        best_x = X_cand[best_idx]

        return dict(zip(self.param_names, best_x))

    def register_result(self, params_dict, score):
        """
        試行結果を登録する。

        Args:
            params_dict: パラメータ名 → 値 の辞書
            score: そのときのスコア（小さいほど良い）
        """
        x = [params_dict[k] for k in self.param_names]
        self.X_obs.append(x)
        self.y_obs.append(score)
        self._save_history()
        print(f"結果を登録しました（スコア: {score:.6f}、累計 {len(self.y_obs)} 件）")

    def best_result(self):
        """これまでの最良結果を返す"""
        if not self.y_obs:
            return None, None
        best_idx = np.argmin(self.y_obs)
        best_x = dict(zip(self.param_names, self.X_obs[best_idx]))
        best_score = self.y_obs[best_idx]
        return best_x, best_score


# ==============================================================================
# 5. メインフロー
# ==============================================================================

def print_c_defines(params_dict):
    """
    Cコードに貼り付ける #define の形式で出力する
    """
    print("\n" + "=" * 55)
    print("★ 次のLES実行に使用する定数（Cコードに貼り付け）")
    print("=" * 55)
    print(f"#define FLAME_ACCEL_STRENGTH {params_dict['FLAME_ACCEL_STRENGTH']:.4f}")
    print(f"#define FLAME_ACCEL_WIDTH    {params_dict['FLAME_ACCEL_WIDTH']:.4f}")
    print(f"#define FLAME_ACCEL_CENTER   {params_dict['FLAME_ACCEL_CENTER']:.4f}")
    print(f"#define CS_UNBURNED          {params_dict['CS_UNBURNED']:.4f}")
    print(f"#define CS_BURNT             {params_dict['CS_BURNT']:.4f}")
    print(f"#define CS_FLAME_BOOST       {params_dict['CS_FLAME_BOOST']:.4f}")
    print("=" * 55 + "\n")


def main():
    print("=" * 55)
    print("  Model 4 定数ベイズ最適化スクリプト")
    print("=" * 55)

    # 設定
    CSV_DIR = r"D:\program\KUWAHATA\final\Force"  # LESのCSV出力先
    LOC_INDEX = 0      # 使用する計測位置（0始まり）
    NUM_BINS = 20      # Gのビン数

    optimizer = BayesianOptimizer()

    # --- モード選択 ---
    print("\n操作を選択してください:")
    print("  1: 新しいLES結果を登録してスコア化 → 次の定数を提案")
    print("  2: 次の定数だけ提案（LES結果なし）")
    print("  3: これまでの最良結果を表示")
    print("  4: 実験値プロファイルを確認")

    try:
        mode = int(input("\n番号を入力: ").strip())
    except ValueError:
        mode = 2

    if mode == 1:
        # 現在の定数を入力
        print("\n現在のLESで使用した定数を入力してください（Enterで既定値）:")
        current_params = {}
        defaults = {
            'FLAME_ACCEL_STRENGTH': 1.0,
            'FLAME_ACCEL_WIDTH':    0.15,
            'FLAME_ACCEL_CENTER':   0.60,
            'CS_UNBURNED':          0.120,
            'CS_BURNT':             0.500,
            'CS_FLAME_BOOST':       0.200,
        }
        for name, default in defaults.items():
            val_str = input(f"  {name} [{default}]: ").strip()
            current_params[name] = float(val_str) if val_str else default

        # CSVを読み込んでスコア計算
        print(f"\nCSVを読み込んでいます: {CSV_DIR}")
        try:
            G_centers, stress_sim = load_and_compute_reynolds_stress(
                CSV_DIR, LOC_INDEX, NUM_BINS
            )
            score = compute_score(G_centers, stress_sim, verbose=True)
            print(f"スコア合計: {score:.6f}")

            # 結果を登録
            optimizer.register_result(current_params, score)

            # グラフを描画・保存
            trial_number = len(optimizer.y_obs)  # 登録後の件数 = 今回の試行番号
            graph_dir = os.path.join(CSV_DIR, "optimization_graphs")
            plot_reynolds_stress_profile(
                csv_dir=CSV_DIR,
                loc_index=LOC_INDEX,
                num_bins=NUM_BINS,
                G_centers=G_centers,
                stress_sim=stress_sim,
                current_params=current_params,
                score=score,
                trial_number=trial_number,
                save_dir=graph_dir,
            )
            plot_velocity_distribution(
                csv_dir=CSV_DIR,
                loc_index=LOC_INDEX,
                current_params=current_params,
                score=score,
                trial_number=trial_number,
                save_dir=graph_dir,
            )
            plot_optimization_history(optimizer, save_dir=graph_dir)

        except FileNotFoundError as e:
            print(f"エラー: {e}")
            print("CSVファイルが見つかりません。パスを確認してください。")
            return
        except Exception as e:
            print(f"エラー: {e}")
            return

        # 次の定数を提案
        next_params = optimizer.suggest_next()
        print_c_defines(next_params)

    elif mode == 2:
        # 次の定数だけ提案
        next_params = optimizer.suggest_next()
        print_c_defines(next_params)

    elif mode == 3:
        # 最良結果を表示
        best_params, best_score = optimizer.best_result()
        if best_params is None:
            print("まだ試行結果がありません。")
        else:
            print(f"\n最良スコア: {best_score:.6f}")
            print_c_defines(best_params)

            # スコアの推移を表示
            print("スコアの推移:")
            for i, (x, y) in enumerate(zip(optimizer.X_obs, optimizer.y_obs)):
                marker = " ← 最良" if y == best_score else ""
                print(f"  試行 {i+1}: スコア = {y:.6f}{marker}")

            # 最適化履歴グラフを再描画
            graph_dir = os.path.join(CSV_DIR, "optimization_graphs")
            plot_optimization_history(optimizer, save_dir=graph_dir)

    elif mode == 4:
        # 実験値プロファイルを確認
        G_test = np.linspace(0, 1, 50)
        stress_exp = experimental_reynolds_stress(G_test)
        print("\n実験値プロファイル（近似関数）:")
        print(f"{'G':>8} | {'Reynolds stress':>16}")
        print("-" * 28)
        for g, s in zip(G_test[::5], stress_exp[::5]):
            print(f"{g:8.3f} | {s:16.6f}")
        print("\n※ Cコード内の実験値はFig. 3.1, 3.2からの読み取り近似値です。")
        print("  実際のデータがあれば experimental_reynolds_stress() を修正してください。")

    else:
        print("無効な選択です。")


if __name__ == '__main__':
    main()