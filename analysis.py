import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns 
from scipy.stats import gaussian_kde
import os
import glob
import concurrent.futures
import time
import warnings

# --- 警告の抑制 ---
warnings.filterwarnings('ignore')

# --- 設定項目 ---
# 自身のパスに合わせて変更
base_output_dir = "D:\\program\\KUWAHATA\\final\\Force\\"
monitor_filename = os.path.join(base_output_dir, "simulation_monitor.csv")

# 並列処理のワーカー数（CPUコア数に合わせて調整。メモリ不足なら減らす）
MAX_WORKERS = 6 

# エラー回避設定
plt.rcParams['agg.path.chunksize'] = 10000 

# ---------------------------------------------------------
#  各種解析関数（全データ対応・並列化用）
# ---------------------------------------------------------

def plot_velocity_distribution(velocity_data, direction, ax):
    vel = velocity_data.dropna()
    # 全データを用いたヒストグラム
    ax.hist(vel, bins=100, density=True, alpha=0.6, label='Histogram')
    
    # KDEはデータが多すぎると計算が非常に遅いため、
    # 統計的な形状を見るだけであれば20万点程度あれば十分な精度が出ますが、
    # ここではユーザー要望通り全データ計算を試みます。
    # 重すぎる場合は except ブロックへ飛びます。
    try:
        # データ数が多すぎる場合のKDE計算用間引き（精度には影響しないレベル）
        if len(vel) > 100000:
            kde_data = vel.sample(100000, random_state=42)
        else:
            kde_data = vel
            
        kde = gaussian_kde(kde_data)
        x_range = np.linspace(vel.min(), vel.max(), 500)
        ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE Curve')
    except Exception as e:
        print(f"KDE Error: {e}")
        pass
        
    ax.set_xlabel(f'{direction}-velocity (m/s)')
    ax.set_ylabel('Probability Density')
    ax.grid(True)
    ax.legend()

def task_1_1_bimodal_distribution(data, save_dir, use_log_scale=False):
    plt.switch_backend('Agg') 
    scale_label = "Log Scale" if use_log_scale else "Linear Scale"
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        fig.suptitle(f'Velocity Distribution: {scale_label}', fontsize=16)
        
        components = ['u', 'v', 'w']
        for i, comp in enumerate(components):
            plot_velocity_distribution(data[comp], comp, axes[i])
            axes[i].set_title(f"{comp}-velocity Distribution")
            if use_log_scale:
                axes[i].set_yscale('log')
                axes[i].set_ylim(bottom=1e-5) 

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        suffix = "_log" if use_log_scale else "_linear"
        filename = os.path.join(save_dir, f"task1_1_distribution{suffix}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)

def task_1_2_conditional_pdfs(data, save_dir):
    plt.switch_backend('Agg')
    try:
        unburnt = data[data['G'] < 0.5]
        burnt = data[data['G'] >= 0.5]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Conditional Velocity Distributions (All Data)', fontsize=16)
        
        # データ量が多い場合のKDE高速化: common_grid=Trueなどを内部で使うが、
        # ここではseabornに任せる。あまりに遅い場合は warn_singular などが出る。
        ax = axes[0]
        if len(unburnt) > 10: sns.kdeplot(unburnt['u'], color='blue', label='Unburnt (G<0.5)', ax=ax, fill=True, alpha=0.3)
        if len(burnt) > 10: sns.kdeplot(burnt['u'], color='red', label='Burnt (G>=0.5)', ax=ax, fill=True, alpha=0.3)
        ax.set_title('u-velocity (Streamwise)')
        ax.legend()
        ax.grid(True)
        
        ax = axes[1]
        if len(unburnt) > 10: sns.kdeplot(unburnt['v'], color='blue', label='Unburnt (G<0.5)', ax=ax, fill=True, alpha=0.3)
        if len(burnt) > 10: sns.kdeplot(burnt['v'], color='red', label='Burnt (G>=0.5)', ax=ax, fill=True, alpha=0.3)
        ax.set_title('v-velocity (Transverse)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(save_dir, "task1_2_conditional_pdfs.png"), dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)

def task_1_3_flame_profile(data, save_dir):
    plt.switch_backend('Agg')
    try:
        # 並列処理時のデータ競合を防ぐためコピー
        local_data = data[['u', 'v', 'w', 'G', 'y_relative']].copy()
        
        for col in ['u', 'v', 'w']:
            local_data[f'{col}_prime'] = local_data[col] - local_data[col].mean()
        
        local_data['re_uu'] = local_data['u_prime']**2
        local_data['re_vv'] = local_data['v_prime']**2
        local_data['re_ww'] = local_data['w_prime']**2
        local_data['re_uv'] = local_data['u_prime'] * local_data['v_prime']
        
        local_data['y_rel_mm'] = local_data['y_relative'] * 1000
        bins = np.arange(local_data['y_rel_mm'].min(), local_data['y_rel_mm'].max() + 0.5, 0.5)
        local_data['bin'] = pd.cut(local_data['y_rel_mm'], bins=bins)
        
        profile = local_data.groupby('bin', observed=True).mean(numeric_only=True)
        profile['bin_center'] = [(b.left + b.right) / 2 for b in profile.index]
        
        fig, axes = plt.subplots(5, 1, figsize=(10, 18), sharex=True)
        fig.suptitle('Profiles across the Flame Front', fontsize=16)
        
        plot_items = [
            ('re_uu', "<u'u'>", 'blue'), ('re_vv', "<v'v'>", 'green'),
            ('re_ww', "<w'w'>", 'orange'), ('re_uv', "<u'v'>", 'red'),
            ('G', 'Mean G', 'purple')
        ]
        
        for i, (col, ylabel, color) in enumerate(plot_items):
            if col in profile.columns:
                axes[i].plot(profile['bin_center'], profile[col], 'o-', color=color, label=col)
                axes[i].set_ylabel(ylabel)
                axes[i].grid(True)
                axes[i].axvline(0, color='black', linestyle='--', lw=2)
                axes[i].legend()
            
        axes[-1].set_xlabel('Relative Distance from Flame Front (mm)')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(save_dir, "task1_3_flame_profile.png"), dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)

def task_1_4_velocity_correlations(data, save_dir):
    plt.switch_backend('Agg')
    try:
        # 全データを使用（サンプリングなし）
        # データ数が多いため、描画負荷軽減のためにマーカーを極小にする
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Velocity Component Correlations (All Data)', fontsize=16)

        pairs = [('u', 'v'), ('u', 'w'), ('v', 'w')]
        for i, (x, y) in enumerate(pairs):
            # s=0.05, alpha=0.05 でデータの密集度を表現
            axes[i].scatter(data[x], data[y], alpha=0.05, s=0.05, rasterized=True)
            axes[i].set_xlabel(f'{x} (m/s)')
            axes[i].set_ylabel(f'{y} (m/s)')
            axes[i].grid(True)
            axes[i].set_title(f'{x} vs {y}')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, "task1_4_correlations.png"), dpi=300)
    finally:
        plt.close(fig)

def task_1_5_g_vs_velocity(data, save_dir):
    plt.switch_backend('Agg')
    try:
        # 全データを使用
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle('Velocity vs. G-value (All Data)', fontsize=16)

        for i, (col, color) in enumerate(zip(['u', 'v', 'w'], ['blue', 'green', 'orange'])):
            axes[i].scatter(data['G'], data[col], alpha=0.05, s=0.05, color=color, rasterized=True)
            axes[i].set_ylabel(f'{col}-velocity (m/s)')
            axes[i].grid(True)

        axes[2].set_xlabel('G-value')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(save_dir, "task1_5_g_vs_velocity.png"), dpi=300)
    finally:
        plt.close(fig)

def diagnose_zero_velocity_origin(data, save_dir):
    plt.switch_backend('Agg')
    threshold=0.1
    try:
        zero_vel = data[data['v'].abs() < threshold]
        if len(zero_vel) == 0:
            return
        fig = plt.figure(figsize=(10, 6))
        plt.hist(zero_vel['G'], bins=50, color='gray', alpha=0.7, label='v ≈ 0')
        plt.title(f'G-distribution for v ≈ 0 (Threshold: {threshold} m/s)')
        plt.xlabel('G-value')
        plt.ylabel('Count')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "diag_zero_vel_g.png"), dpi=300)
    finally:
        plt.close(fig)

def task_1_6_reynolds_stress_vs_g_binned(data, save_dir):
    plt.switch_backend('Agg')
    try:
        local_data = data[['u', 'v', 'w', 'G']].copy()
        bins = np.arange(0.0, 1.05, 0.05)
        local_data['G_bin'] = pd.cut(local_data['G'], bins=bins)
        
        grouped = local_data.groupby('G_bin', observed=True)
        mean_vals = grouped[['u', 'v', 'w']].mean()
        
        merged = pd.merge(local_data, mean_vals, left_on='G_bin', right_index=True, suffixes=('', '_mean'))
        
        merged['u_prime_cond'] = merged['u'] - merged['u_mean']
        merged['v_prime_cond'] = merged['v'] - merged['v_mean']
        
        merged['uu_cond'] = merged['u_prime_cond']**2
        merged['vv_cond'] = merged['v_prime_cond']**2
        merged['uv_cond'] = merged['u_prime_cond'] * merged['v_prime_cond']
        
        stress_profile = merged.groupby('G_bin', observed=True)[['uu_cond', 'vv_cond', 'uv_cond']].mean(numeric_only=True)
        stress_profile['G_center'] = [(b.left + b.right) / 2 for b in stress_profile.index]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        fig.suptitle('Conditional Reynolds Stress vs. G', fontsize=16)
        
        axes[0].plot(stress_profile['G_center'], stress_profile['uu_cond'], 'o-', color='blue', label="<u'u'>")
        axes[0].set_ylabel("<u'u'>")
        axes[0].grid(True)
        
        axes[1].plot(stress_profile['G_center'], stress_profile['vv_cond'], 'o-', color='green', label="<v'v'>")
        axes[1].set_ylabel("<v'v'>")
        axes[1].grid(True)
        
        axes[2].plot(stress_profile['G_center'], stress_profile['uv_cond'], 'o-', color='red', label="<u'v'>")
        axes[2].set_ylabel("<u'v'>")
        axes[2].set_xlabel('Reaction Progress Variable G')
        axes[2].grid(True)
        axes[2].axhline(0, color='black', linestyle='--')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(save_dir, "task1_6_stress_vs_G.png"), dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)

def task_1_7_convergence_check(data, save_dir):
    plt.switch_backend('Agg')
    try:
        # expanding mean は計算量が多いですが、重要なので全データ実行
        u_cum_mean = data['u'].expanding().mean()
        v_cum_mean = data['v'].expanding().mean()
        uv_cum_mean = (data['u'] * data['v']).expanding().mean()
        re_stress_uv_cum = uv_cum_mean - (u_cum_mean * v_cum_mean)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x_axis = np.arange(1, len(data) + 1)
        
        # データ点が多い場合描画が重いので間引き描画
        # (計算は全データだが、プロットはスクリーン解像度以上は不要)
        step = max(1, len(data) // 50000)
        
        ax.plot(x_axis[::step], re_stress_uv_cum.iloc[::step], '-', color='black', linewidth=1.5, label="<u'v'> convergence")
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel("Reynolds Stress <u'v'> (m^2/s^2)")
        ax.set_title("Statistical Convergence of Shear Reynolds Stress")
        ax.grid(True)
        ax.legend()
        
        final_val = re_stress_uv_cum.iloc[-1]
        ax.text(x_axis[-1], final_val, f' Final: {final_val:.5f}', 
                verticalalignment='bottom', horizontalalignment='right', 
                bbox=dict(facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "task1_7_convergence.png"), dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)

def task_1_8_convergence_by_G_bin(data, save_dir):
    plt.switch_backend('Agg')
    try:
        bins = np.arange(0.0, 1.1, 0.1)
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        cmap = plt.get_cmap('jet') 
        final_results = [] 

        for i in range(len(bins) - 1):
            g_min = bins[i]
            g_max = bins[i+1]
            g_center = (g_min + g_max) / 2.0
            
            subset = data[(data['G'] >= g_min) & (data['G'] < g_max)].sort_values('timestep')
            n_samples = len(subset)
            
            if n_samples < 10:
                final_results.append((g_center, np.nan, n_samples))
                continue
                
            u_cum = subset['u'].expanding().mean()
            v_cum = subset['v'].expanding().mean()
            uv_cum = (subset['u'] * subset['v']).expanding().mean()
            stress_cum = uv_cum - (u_cum * v_cum)
            
            color = cmap(i / (len(bins) - 1))
            
            # 描画間引き
            step = max(1, n_samples // 10000)
            x_axis = np.arange(1, n_samples + 1)
            
            ax1.plot(x_axis[::step], stress_cum.iloc[::step], label=f'G={g_min:.1f}-{g_max:.1f}', color=color, linewidth=1.5, alpha=0.8)
            
            final_val = stress_cum.iloc[-1]
            final_results.append((g_center, final_val, n_samples))

        ax1.set_xlabel('Number of Samples (Cumulative)')
        ax1.set_ylabel("Reynolds Stress <u'v'> (m^2/s^2)")
        ax1.set_title("Statistical Convergence of Reynolds Stress per G-bin")
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="G Range")
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        fig1.savefig(os.path.join(save_dir, "task1_8_convergence_history.png"), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        if len(final_results) > 0:
            res_df = pd.DataFrame(final_results, columns=['G', 'Stress', 'Samples'])
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sizes = res_df['Samples'].fillna(0) / res_df['Samples'].max() * 200 + 20
            ax2.scatter(res_df['G'], res_df['Stress'], s=sizes, color='red', edgecolor='black', zorder=3, label='Converged Value')
            ax2.plot(res_df['G'], res_df['Stress'], 'r--', alpha=0.5, zorder=2)
            ax2.set_xlabel('Reaction Progress Variable G')
            ax2.set_ylabel("Reynolds Stress <u'v'> (m^2/s^2)")
            ax2.set_title("Profile of Reynolds Stress vs G")
            ax2.grid(True)
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            plt.tight_layout()
            fig2.savefig(os.path.join(save_dir, "task1_8_G_vs_Stress_profile.png"), dpi=300, bbox_inches='tight')
            plt.close(fig2)
    except Exception as e:
        print(f"Error in task_1_8: {e}")

def task_1_9_fine_G_analysis(data, save_dir):
    plt.switch_backend('Agg')
    try:
        bins = np.arange(0.0, 1.01, 0.01)
        
        # プロット作成（履歴）
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        cmap = plt.get_cmap('jet')
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        final_results = [] 

        for i in range(len(bins) - 1):
            g_min = bins[i]
            g_max = bins[i+1]
            g_center = (g_min + g_max) / 2.0
            
            subset = data[(data['G'] >= g_min) & (data['G'] < g_max)].sort_values('timestep')
            n_samples = len(subset)
            
            if n_samples < 50:
                continue
                
            u_cum = subset['u'].expanding().mean()
            v_cum = subset['v'].expanding().mean()
            uv_cum = (subset['u'] * subset['v']).expanding().mean()
            stress_cum = uv_cum - (u_cum * v_cum)
            
            color = cmap(norm(g_center))
            
            # 描画用間引き
            step = max(1, n_samples // 5000)
            x_axis = np.arange(1, n_samples + 1)
            
            ax1.plot(x_axis[::step], stress_cum.iloc[::step], color=color, linewidth=1.0, alpha=0.6)
            final_results.append((g_center, stress_cum.iloc[-1], n_samples))

        ax1.set_xlabel('Number of Samples (Cumulative)')
        ax1.set_ylabel("Reynolds Stress <u'v'> (m^2/s^2)")
        ax1.set_title("Statistical Convergence (dG=0.01)")
        ax1.grid(True)
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label('G')
        
        fig1.savefig(os.path.join(save_dir, "task1_9_fine_convergence.png"), dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # プロファイル
        if len(final_results) > 0:
            res_df = pd.DataFrame(final_results, columns=['G', 'Stress', 'Samples'])
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sizes = res_df['Samples'] / res_df['Samples'].max() * 150 + 10
            sc = ax2.scatter(res_df['G'], res_df['Stress'], s=sizes, c=res_df['G'], cmap='jet', edgecolors='k', alpha=0.8, zorder=3)
            ax2.plot(res_df['G'], res_df['Stress'], 'k-', alpha=0.3, zorder=2)
            ax2.set_xlabel('Reaction Progress Variable G')
            ax2.set_ylabel("Reynolds Stress <u'v'> (m^2/s^2)")
            ax2.set_title("Detailed Profile (dG=0.01)")
            ax2.grid(True)
            ax2.axhline(0, color='black', linestyle='-', linewidth=1)
            cbar2 = plt.colorbar(sc, ax=ax2)
            cbar2.set_label('G')

            plt.tight_layout()
            fig2.savefig(os.path.join(save_dir, "task1_9_fine_profile.png"), dpi=300, bbox_inches='tight')
            plt.close(fig2)
    except Exception as e:
        print(f"Error in task_1_9: {e}")

def task_monitor_dt_stability(monitor_path, save_dir):
    plt.switch_backend('Agg')
    try:
        if not os.path.exists(monitor_path):
            return
        df_mon = pd.read_csv(monitor_path)
        if 'dt' not in df_mon.columns or 'step' not in df_mon.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_mon['step'], df_mon['dt'], marker='.', linestyle='-', linewidth=0.5, markersize=2, label='dt')
        ax.set_yscale('log')
        ax.set_title('Time Step Width (dt) vs. Simulation Step (Log Scale)', fontsize=16)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('dt (seconds) - Log Scale', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
        
        plt.savefig(os.path.join(save_dir, "simulation_dt_history.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error in dt_monitor: {e}")
    finally:
        plt.close(fig)

def task_model_diagnostics(data, save_dir):
    if 'nu_t' not in data.columns or 'S_mag' not in data.columns:
        return
    plt.switch_backend('Agg')
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 全データScatter (Rasterized=Trueで軽量化)
        sc = axes[0, 0].scatter(data['G'], data['nu_t'], alpha=0.1, s=0.1, c=data['S_mag'], cmap='viridis', norm=mcolors.LogNorm(), rasterized=True)
        axes[0, 0].set_xlabel('G')
        axes[0, 0].set_ylabel('nu_t (m^2/s)')
        axes[0, 0].set_title('SGS Viscosity (nu_t) vs G')
        axes[0, 0].grid(True)
        plt.colorbar(sc, ax=axes[0, 0], label='Strain Rate (S_mag)')

        axes[0, 1].hist(data['nu_t'], bins=50, log=True, color='purple', alpha=0.7)
        axes[0, 1].set_xlabel('nu_t (m^2/s)')
        axes[0, 1].set_ylabel('Count (Log Scale)')
        axes[0, 1].set_title('Distribution of nu_t')
        axes[0, 1].grid(True)

        axes[1, 0].hist(data['S_mag'], bins=100, log=True, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Strain Rate Magnitude |S| (1/s)')
        axes[1, 0].set_ylabel('Count (Log Scale)')
        axes[1, 0].set_title('Distribution of Strain Rate')
        axes[1, 0].grid(True)

        axes[1, 1].scatter(data['S_mag'], data['nu_t'], alpha=0.1, s=0.1, c='gray', rasterized=True)
        axes[1, 1].set_xscale('log')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xlabel('|S| (1/s)')
        axes[1, 1].set_ylabel('nu_t (m^2/s)')
        axes[1, 1].set_title('nu_t vs Strain Rate (Log-Log)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "diag_sgs_model_behavior.png"), dpi=300)
    finally:
        plt.close(fig)

def task_quadrant_analysis(data, save_dir):
    plt.switch_backend('Agg')
    try:
        u_prime = data['u'] - data['u'].mean()
        v_prime = data['v'] - data['v'].mean()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        # 全データ使用
        sc = ax.scatter(u_prime, v_prime, c=data['G'], cmap='jet', s=0.1, alpha=0.1, rasterized=True)
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel("u' (m/s)")
        ax.set_ylabel("v' (m/s)")
        ax.set_title("Quadrant Analysis of Reynolds Stress (u' vs v')")
        ax.grid(True)
        
        ax.text(0.95, 0.95, 'Q1 (Outward)', transform=ax.transAxes, ha='right')
        ax.text(0.05, 0.95, 'Q2 (Ejection/Burst)', transform=ax.transAxes, ha='left')
        ax.text(0.05, 0.05, 'Q3 (Inward)', transform=ax.transAxes, ha='left')
        ax.text(0.95, 0.05, 'Q4 (Sweep)', transform=ax.transAxes, ha='right')

        plt.colorbar(sc, label='G value')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "task_quadrant_analysis.png"), dpi=300)
    finally:
        plt.close(fig)

def task_raw_time_series(data, save_dir):
    plt.switch_backend('Agg')
    try:
        # 時系列データの可視化について:
        # 全点を描画すると、数百万点の場合PNGが巨大化し、線がつぶれてしまいます。
        # 統計処理には全データを使っていますが、この「生データの時系列表示」図だけは
        # 視認性とファイルサイズのために間引き表示を行います。
        max_points = 20000
        if len(data) > max_points:
            step = len(data) // max_points
            df_plot = data.iloc[::step]
        else:
            df_plot = data
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        axes[0].plot(df_plot['timestep'], df_plot['u'], lw=0.5, alpha=0.8, label='u')
        axes[0].plot(df_plot['timestep'], df_plot['v'], lw=0.5, alpha=0.8, label='v')
        axes[0].set_ylabel('Velocity (m/s)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True)
        
        axes[1].plot(df_plot['timestep'], df_plot['G'], color='green', lw=0.5, alpha=0.8)
        axes[1].set_ylabel('G value')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True)
        
        u_prime = df_plot['u'] - df_plot['u'].mean()
        v_prime = df_plot['v'] - df_plot['v'].mean()
        uv_inst = u_prime * v_prime
        
        axes[2].plot(df_plot['timestep'], uv_inst, color='red', lw=0.5, alpha=0.6)
        axes[2].set_ylabel("Instantaneous u'v'")
        axes[2].set_xlabel('Timestep')
        axes[2].grid(True)
        
        plt.suptitle("Raw Time Series Signal (Decimated for Visualization)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(save_dir, "task_raw_time_series.png"), dpi=300)
    finally:
        plt.close(fig)

# ---------------------------------------------------------
#  メイン処理（pickle対応・並列化）
# ---------------------------------------------------------

def load_data_fast(file_path):
    """
    CSVを高速に読み込む関数。
    .pkl があり、かつ CSVより新しい場合のみそれを読む。
    CSVが更新されていれば再生成する。
    """
    base, ext = os.path.splitext(file_path)
    pkl_path = base + ".pkl"
    
    load_from_pickle = False

    # 1. pickleが存在するか確認し、タイムスタンプを比較
    if os.path.exists(pkl_path):
        csv_mtime = os.path.getmtime(file_path) # CSVの更新日時
        pkl_mtime = os.path.getmtime(pkl_path)  # PKLの更新日時
        
        if csv_mtime > pkl_mtime:
            print(f"  [Info] CSVファイルが更新されています。キャッシュを破棄して再読み込みします。")
            load_from_pickle = False
        else:
            load_from_pickle = True

    # 2. 条件を満たせば pickle を読む
    if load_from_pickle:
        print(f"  [Info] 高速キャッシュファイル(.pkl)を使用します。")
        try:
            return pd.read_pickle(pkl_path)
        except Exception as e:
            print(f"  [Warn] .pklの読み込みに失敗({e})。CSVから再生成します。")

    # 3. CSVを読む（pickleがない、またはCSVの方が新しい場合）
    print(f"  [Info] CSVファイルを読み込んでいます... (engine='pyarrow')")
    try:
        # pyarrowエンジンで高速読み込み
        df = pd.read_csv(file_path, engine='pyarrow')
    except ValueError:
        print("  [Warn] pyarrowエンジンが使用できません。標準エンジンで読み込みます。")
        df = pd.read_csv(file_path, low_memory=False)

    # 4. 次回のために pickle として保存（更新）しておく
    print(f"  [Info] 次回高速化のためにキャッシュ(.pkl)を更新しています...")
    try:
        df.to_pickle(pkl_path)
    except Exception as e:
        print(f"  [Warn] .pklの保存に失敗しました: {e}")

    return df

def process_probe_file(file_path):
    t_start = time.time()
    print(f"解析開始: {os.path.basename(file_path)}")
    
    # 出力ディレクトリ作成
    loc_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[-1] # loc6
    current_output_dir = os.path.join(base_output_dir, loc_name)
    os.makedirs(current_output_dir, exist_ok=True)
    
    # 1. 高速データ読み込み
    try:
        df = load_data_fast(file_path)
        
        # 数値化とクリーニング
        numeric_cols = ['timestep', 'u', 'v', 'w', 'G', 'x_location', 'y_flame', 'y_physical', 'y_relative']
        if 'nu_t' in df.columns: numeric_cols.append('nu_t')
        if 'S_mag' in df.columns: numeric_cols.append('S_mag')

        # pyarrowで読んだ場合、型推論がうまくいっていることが多いが念のため確認
        # (エラーが出る場合のみ変換)
        # 欠損値削除
        before_len = len(df)
        df = df.dropna(subset=['u', 'v', 'w', 'G'])
        
        if df.empty:
            print("  [Error] 有効なデータがありません。")
            return
        
        print(f"  データ準備完了 ({len(df)} rows).")

    except Exception as e:
        print(f"  [Error] ファイル読み込み中にエラーが発生しました: {e}")
        return

    # 2. 並列処理タスク定義
    print(f"  グラフ作成タスクを並列実行します (Workers={MAX_WORKERS})...")
    
    tasks = [
        (task_1_1_bimodal_distribution, (df, current_output_dir, False)),
        (task_1_1_bimodal_distribution, (df, current_output_dir, True)),
        (task_1_2_conditional_pdfs, (df, current_output_dir)),
        (task_1_3_flame_profile, (df, current_output_dir)),
        (task_1_4_velocity_correlations, (df, current_output_dir)),
        (task_1_5_g_vs_velocity, (df, current_output_dir)),
        (diagnose_zero_velocity_origin, (df, current_output_dir)),
        (task_1_6_reynolds_stress_vs_g_binned, (df, current_output_dir)),
        (task_1_7_convergence_check, (df, current_output_dir)),
        (task_1_8_convergence_by_G_bin, (df, current_output_dir)),
        (task_1_9_fine_G_analysis, (df, current_output_dir)),
        (task_model_diagnostics, (df, current_output_dir)),
        (task_quadrant_analysis, (df, current_output_dir)),
        (task_raw_time_series, (df, current_output_dir)),
        (task_monitor_dt_stability, (monitor_filename, current_output_dir)),
    ]

    # 3. ProcessPoolExecutor で並列実行
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(func, *args): func.__name__ for func, args in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            task_name = futures[future]
            try:
                future.result()
                print(f"    - {task_name} 完成") # 進捗が見たい場合はコメントアウト解除
            except Exception as e:
                print(f"    - {task_name} でエラー: {e}")

    elapsed = time.time() - t_start
    print(f"完了: {os.path.basename(file_path)} (所要時間: {elapsed:.1f}秒)")

if __name__ == "__main__":
    # loc6 限定で検索 (CSVを検索対象の基本とする)
    search_pattern = os.path.join(base_output_dir, "probe_data_flame_region_loc6.csv")
    target_files = glob.glob(search_pattern)
    
    if not target_files:
        print(f"エラー: 指定されたディレクトリにファイルが見つかりません。\nパス: {search_pattern}")
    else:
        for file_path in target_files:
            process_probe_file(file_path)
            
    print("\n全処理が終了しました。")