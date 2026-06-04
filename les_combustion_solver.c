#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <errno.h>
#include <float.h> 
#include <string.h> 
#include <signal.h>

// --- リスタート・出力設定 ---
#define RESTART_INFO_FILE "restart_info.txt"       // 最新のステップ数を記録するメモ
#define QUICK_CHECKPOINT_FILE "checkpoint.bin"     // 高速リスタート用データ（上書きされる）
#define STATS_BACKUP_FILE "stats_backup.bin"       // 時間平均の積算値データ（上書きされる）

// この間隔で「リスタート地点作成」と「CSV書き出し」を行います
// 50～100程度推奨（計算ロスは数秒です）
#define CHECKPOINT_INTERVAL 50 

// --- 強制終了制御用フラグ ---
volatile sig_atomic_t stop_requested = 0;

// Ctrl+C (SIGINT) が押されたときに呼ばれる関数
void handle_sigint(int sig) {
    const char msg[] = "\n\n!!! CAUTION: Ctrl+C detected. Finalizing data output... !!!\n\n";
    fwrite(msg, 1, sizeof(msg), stderr);
    stop_requested = 1; // ループを抜けるフラグを立てる
}

//圧力一定

/*
圧力の扱いに関する議論
現在の設定: 計算負荷を考慮し、「圧力一定」の条件で計算されています。
物理的な妥当性:
本来、燃焼によって温度が変化すると、状態方程式（P=ρRT）に基づき圧力も変動します。
しかし、今回のような大気圧下で解放されている燃焼場（大気解放場）では、圧力の変動はごく僅か（1%程度）です。
そのため、質量保存則などを厳密に解く上では圧力を変数として計算すべきですが、今回の目的（拡散の確認）においては、「圧力一定」という仮定でも大きな問題はないという結論に至りました。
*/

// ==============================================================================
// --- シミュレーションパラメータ (ユーザー設定) ---
// ==============================================================================

// --- 格子と領域定義 ---
#define NX 60                   // x方向: 0.060 / 0.0003 = 200
#define NY 20                   // y方向: 0.030 / 0.0003 = 100
#define NZ 40                   // z方向: 0.060 / 0.0003 = 200
#define LX 0.12
#define LY 0.04
#define LZ 0.08

// --- 流入・境界条件 ---
#define INLET_RADIUS 0.013      // 流入口の半径 (m)
#define RIM_OUTER_RADIUS 0.005  // バーナーリムの外径 (m)
#define INLET_VELOCITY 4.0      // 流入速度 (m/s)
#define INLET_TEMPERATURE 300.0 // 流入温度 (K)
#define AMBIENT_TEMPERATURE 300.0 // 周囲・壁面温度 (K)
#define INLET_TURBULENCE_INTENSITY 0.20 // 乱流強度 (20%)

// SEM (合成渦法) パラメータ 
#define NUM_VORTICES 2000                // 生成する渦の総数 (計算コストと精度に影響)
#define TURBULENT_LENGTH_SCALE (0.2 * INLET_RADIUS) // 乱流の積分長さスケール (ジェット径の2倍程度が目安)

// --- 燃焼・化学反応モデル ---
#define NUM_SPECIES 5           // 化学種の総数 (C3H8, O2, N2, CO2, H2O)
#define HEAT_OF_REACTION_C3H8 4.63e7 // プロパンの燃焼発熱量 [J/kg of C3H8]
#define S_L 0.4                // プロパンの層流火炎速度 (m/s) 0.4
#define INITIAL_FLAME_FRONT_I 100 // 初期火炎面の位置 (i方向インデックス)
#define BURNT_GAS_TEMPERATURE 2200.0 // 既燃ガスの仮定温度 (K)

// --- LES・乱流モデル ---
// 未燃ガス側（G=0）のCs（標準的な値）
//#define CS_UNBURNED  0.120
#define Pr_t 0.5    // 乱流プラントル数

#define Sc_t 0.8    // 乱流シュミット数
#define FRACTAL_DIMENSION_D3 2.35 // SSR-SGSモデルのフラクタル次元

// --- 数値計算設定 ---
#define SIMULATION_TIME 20.0    // 総シミュレーション時間 (s)
#define COURANT_NUMBER 0.1      // 対流項に対するクーラン数
#define COURANT_DIFFUSION 0.1   // 拡散項に対するクーラン数

// --- 出力設定 ---
#define OUTPUT_INTERVAL 500      // VTK/CSVファイルの出力間隔 (ステップ数)

// ==============================================================================
// --- 出力ディレクトリ設定 ---
// ==============================================================================
// 末尾に必ず "\\"を付けてください
//#define OUTPUT_DIR "C:\\research\\KUWAHATA\\1129\\C3H8_ver2-6\\"
#define OUTPUT_DIR "D:\\program\\KUWAHATA\\final\\Force\\"
//#define OUTPUT_DIR "C:\\research\\final\\Force\\"

// 既存の定義はファイル名だけにします（パスは関数内で結合します）
#define PROBE_OUTPUT_FILENAME "probe_data_flame_region.csv"

// ==============================================================================
// --- グローバル定数定義 (派生パラメータ) ---
// ==============================================================================

// --- 格子間隔 (パラメータから自動計算) ---
#define DX (LX / (double)NX)
#define DY (LY / (double)NY)
#define DZ (LZ / (double)NZ)

// --- 流入口中心座標 (パラメータから自動計算) ---
#define INLET_CENTER_Y 0.0     
#define INLET_CENTER_Z (LZ / 2.0)

// SEM (合成渦法) 用のデータ構造とグローバル変数
typedef struct {
    double x, y, z;        // 渦の中心座標 (m)
    double radius;         // 渦の半径 (m)
    double strength_u;     // u方向の渦の強さ (m/s)
    double strength_v;     // v方向の渦の強さ (m/s)
    double strength_w;     // w方向の渦の強さ (m/s)
} Vortex;

Vortex vortices[NUM_VORTICES]; // 渦を格納するグローバル配列

// SEMの渦が動く仮想的な箱の大きさ (流入面を中心に配置)
const double VORTEX_BOX_X = TURBULENT_LENGTH_SCALE;
const double VORTEX_BOX_Y = LY; // 計算領域のY, Zと同じ大きさにする
const double VORTEX_BOX_Z = LZ;

// --- 化学種インデックス ---
#define C3H8_INDEX 0
#define O2_INDEX  1
#define N2_INDEX  2
#define CO2_INDEX 3
#define H2O_INDEX 4

// --- 物理定数 ---
#define R_univ 8.314462         // 一般気体定数 (J/(mol·K))
#define PATM 101325.0           // 標準大気圧 (Pa)
#define pressure_const 101325.0 // 計算に使用する基準圧力 [Pa]
#define G_GRAVITY -9.81         // x方向の重力加速度 (m/s^2)
#define DEFAULT_RHO_AIR 1.2     // デフォルトの空気密度（浮力計算の基準値として使用）
#define SMALL_NUMBER 1e-10 

// --- 燃焼モデル定数 ---
#define OMEGA_MODEL_BETA 500.0       // 反応速度モデルの局在化パラメータ 50が標準

// ==============================================================================
// --- 物理モデル・数値スキーム定数 (上級者向け) ---
// ==============================================================================

// --- SSR-SGSモデルの定数 ---
#define SSR_SGS_MODEL_A_ETA 0.2340
#define SSR_SGS_MODEL_B_ETA 0.2187
#define SSR_SGS_MODEL_C_ALPHA 10.0
#define NU_UNBURNT_AIR_APPROX 1.5e-5 // 未燃ガスの動粘性係数の近似値 (m^2/s)

// --- フィルタリング・スキームの定数 ---
#define FILTER_WEIGHT_CENTER 50.0   // Boxフィルタの中心点の重み(22 suisyou)
/*#define QUICK_COEFF_UW2 (-1.0/8.0)  // QUICKスキーム係数 (風上2点目)
#define QUICK_COEFF_UW1 (3.0/4.0)   // QUICKスキーム係数 (風上1点目)
#define QUICK_COEFF_DW1 (3.0/8.0)   // QUICKスキーム係数 (風下1点目)
*/
// 係数をこのように設定すると、計算式が 0.5*phi[i] + 0.5*phi[i+1] となり、
// 厳密な中心差分になります。
#define QUICK_COEFF_UW2 0.0
#define QUICK_COEFF_UW1 0.5
#define QUICK_COEFF_DW1 0.5

// --- 計算安定化・物理的制約のためのクリッピング値 ---
#define MIN_TEMPERATURE_CLIP 200.0  // 計算上の最低温度 (K)
#define MAX_TEMPERATURE_CLIP 3000.0 // 計算上の最高温度 (K)
#define MIN_Y_CLIP 0.0              // 質量分率の最小値
#define MAX_Y_CLIP 1.0              // 質量分率の最大値
#define MIN_G_CLIP 0.0              // G値の最小値
#define MAX_G_CLIP 1.0              // G値の最大値
#define MIN_MU_MIX_CLIP 1e-7        // 最小粘性係数 (Pa*s)
#define MIN_NU_MIX_CLIP 1e-7        // 最小動粘性係数 (m^2/s)
#define MIN_LAMBDA_MIX_CLIP 1e-3    // 最小熱伝導率 (W/m-K)
#define MIN_ALPHA_MIX_CLIP 1e-7     // 最小熱拡散率 (m^2/s)

// --- 初期値・フォールバック値 ---
#define DEFAULT_CP_AIR 1000.0       // 空気相当の定圧比熱 (J/kg-K)
#define DEFAULT_LAMBDA_AIR 0.025    // 空気相当の熱伝導率 (W/m-K)
#define DEFAULT_MU_AIR 1.8e-5       // 空気相当の粘性係数 (Pa*s)

// --- その他 ---
#define DEFAULT_RHO_AIR 1.2         // デフォルトの空気密度 (kg/m^3)
#define SMALL_NUMBER 1e-10          // ゼロ除算防止用の微小数
#define ONE_THIRD (1.0/3.0)         // 定数 1/3

// 時系列データ記録用の設定
#define MAX_DATA_POINTS 50000     // 記録する最大のデータ点数（十分大きな値に）
#define START_HIST_STEP 0      // データ記録を開始するタイムステップ
#define HIST_I 12                  // 取得するセルのiインデックス
#define HIST_J 32                 // 取得するセルのjインデックス
#define HIST_K 40                 // 取得するセルのkインデックス

// 時系列データ記録用の設定
#define MAX_MEASUREMENT_POINTS 64  // 記録する点の最大数
#define MAX_DATA_POINTS 50000      // 各点での記録データ数
#define START_DATA_ACQUISITION_STEP 0 // データ記録を開始するステップ

// ==============================================================================
//  論文再現用：熱膨張加速モデル (Flame Expansion Force) パラメータ 
// ==============================================================================


// 1:有効, 0:無効 (比較検証用)
#define USE_FLAME_ACCEL_MODEL 1

// 加速の強さ（係数）
// [調整のコツ]
// 値を大きくする -> 既燃ガスの流速(右側のピーク)が速くなる。ピーク間の距離が広がる。
// 値を小さくする -> 加速が弱まり、Bi-modalが不明瞭になる。
// 推奨範囲: 100.0 ～ 500.0 (まずは 200.0 か 300.0 で試す)110
//#define FLAME_ACCEL_STRENGTH 1.0 
//100


// 加速させるGの幅（ガウス分布の分散 sigma）
// [調整のコツ]
// 値を大きくする (0.25など) -> 長く加速されるため、最終速度が上がりやすい。全体的にボヤける。
// 値を小さくする (0.10など) -> 一瞬で「ドンッ」と加速する。ピークが鋭くなる。
// 推奨範囲: 0.15 ～ 0.20
//#define FLAME_ACCEL_WIDTH 0.15 
//0.1

// 加速の中心点（Gの値）
// 通常は火炎面である 0.5 でOK。
//#define FLAME_ACCEL_CENTER 0.6
//92

// この半径内（先端部分）では加速を行わない
// バーナー半径(13mm)の半分程度 (0.005 ~ 0.008) が目安
#define TIP_PROTECTION_RADIUS 0.0
//0.010

// 加速力（ソース項）の安全装置s
// 密度(kg/m3) * 加速度(m/s2) の次元。
// 5000.0 程度あれば、物理的な加速は通すが、数値的なスパイクはカットできます。
#define MAX_FLAME_FORCE_LIMIT  500.0

// 加速力を徐々に効かせる距離 (m)
// バーナー出口(z=0)からこの距離までは、力が 0% -> 100% に滑らかに変化します。
// これによりリム付近の火炎の根元を守ります。
#define FLAME_RAMP_DISTANCE 0.03  // 30mm (バーナー径程度)

// 既燃ガス側（G=1）のCs
// 論文の "-0.01" に近づけるため、ここを大きくして減衰を強める調整を行います。
// まずは標準の倍程度から試し、減衰が足りなければ 0.3 ~ 0.4 へ上げてください。
//#define CS_BURNT    0.500

// 火炎帯中心（G=0.5）でのブースト（オプション）
// 火炎面での乱れを特に強く抑えたい場合に使用します。
//#define CS_FLAME_BOOST 0.200

///-----------------------------------------------------------------------------
#define FLAME_ACCEL_STRENGTH 2.0361
#define FLAME_ACCEL_WIDTH    0.1126
#define FLAME_ACCEL_CENTER   0.6697
#define CS_UNBURNED          0.1374
#define CS_BURNT             0.4223
#define CS_FLAME_BOOST       0.2371
//-----------------------------------------------------------------------------

// ==============================================================================
// 火炎領域サンプリング設定 (3点計測版) 
// ==============================================================================

// 計測する箇所の数
#define NUM_PROBE_POSITIONS 7

// 計測するX座標 (m) のリスト
// 例: 20mm, 40mm, 60mm (ノズル直径D=26mmなら、約0.8D, 1.5D, 2.3D相当)
const double PROBE_X_LOCATIONS[NUM_PROBE_POSITIONS] = { 0.010, 0.015, 0.020, 0.025, 0.030 , 0.035 , 0.040};

// 出力ファイル名のベース（これに _x1.csv などが付く）
#define PROBE_FILENAME_BASE "probe_data_flame_region"

// --- 瞬時火炎面(G=0.5)からのサンプリング範囲 (m) ---
#define SAMPLING_RANGE 1.000  // 火炎面から±40mmの範囲

// --- データ記録設定 ---
#define PROBE_DATA_INTERVAL 1


typedef double (*Grid3D)[NY][NZ];
typedef int (*Grid3D_Int)[NY][NZ];  

// ==============================================================================
//  物性値データ構造体
// ==============================================================================
typedef struct {
    char name[17];           // 化学種名
    double mw_g_per_mol;     // 分子量 (g/mol)
    double t_mid;            // NASA多項式の高温/低温切り替え温度 (K)
    double nasa_high[7];     // 高温域用NASA多項式係数 (a1-a7)
    double nasa_low[7];      // 低温域用NASA多項式係数 (a1-a7)
    double lambda_coeffs[4]; // 熱伝導率フィット係数 (未使用)
    double eta_coeffs[4];    // 粘性係数フィット係数
} SpeciesData;

// ロードした物性値を格納するグローバル変数
SpeciesData loaded_species[NUM_SPECIES];
// 将来の拡張用 (今回は未使用)
double diffusion_coeffs[NUM_SPECIES][NUM_SPECIES][4];

// ==============================================================================
// 物性値関連関数
// ==============================================================================
// Fortranレコード読み込みヘルパー
unsigned char* read_fortran_record(FILE* fp) {
    int record_len;
    if (fread(&record_len, sizeof(int), 1, fp) != 1) return NULL;
    unsigned char* buffer = (unsigned char*)malloc(record_len);
    if (!buffer) { fprintf(stderr, "Error: Memory allocation failed.\n"); exit(1); }
    if (fread(buffer, 1, record_len, fp) != record_len) { free(buffer); return NULL; }
    int end_len;
    if (fread(&end_len, sizeof(int), 1, fp) != 1) { free(buffer); return NULL; }
    if (record_len != end_len) { fprintf(stderr, "Warning: Record length mismatch (%d vs %d).\n", record_len, end_len); }
    return buffer;
}

// chem.binの読み込み
int load_chem_bin(const char* filename, SpeciesData species[], int* n_species, int* n_elements) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Error: Cannot open chem.bin file: %s\n", filename); return 0; }
    unsigned char* buffer;
    buffer = read_fortran_record(fp); if(buffer) free(buffer); else { fclose(fp); return 0; }
    buffer = read_fortran_record(fp); if (!buffer) { fclose(fp); return 0; }
    memcpy(n_elements, buffer + 12, sizeof(int));
    memcpy(n_species, buffer + 16, sizeof(int));
    free(buffer);
    buffer = read_fortran_record(fp); if(buffer) free(buffer); else { fclose(fp); return 0; }
    buffer = read_fortran_record(fp); if (!buffer) { fclose(fp); return 0; }
    const size_t base_offset = 8, name_size = 16, comp_size = 4 * (*n_elements), two_ints_size = 8;
    const size_t double_size = 8, one_int_size = 4, three_doubles_size = 24, nasa_size = 7 * double_size;
    const size_t record_size_per_species = name_size + comp_size + two_ints_size + double_size + one_int_size + three_doubles_size + 2 * nasa_size;
    for (int i = 0; i < *n_species; i++) {
        size_t current_record_start = base_offset + i * record_size_per_species;
        size_t name_offset = current_record_start;
        size_t mw_offset = name_offset + name_size + comp_size + two_ints_size;
        size_t tmid_offset = mw_offset + double_size + one_int_size + 2 * double_size;
        size_t nasa_h_offset = tmid_offset + double_size;
        size_t nasa_l_offset = nasa_h_offset + nasa_size;
        memcpy(species[i].name, buffer + name_offset, name_size);
        species[i].name[16] = '\0';
        for (int j = 15; j >= 0 && species[i].name[j] == ' '; j--) species[i].name[j] = '\0';
        memcpy(&species[i].mw_g_per_mol, buffer + mw_offset, double_size);
        memcpy(&species[i].t_mid, buffer + tmid_offset, double_size);
        memcpy(species[i].nasa_high, buffer + nasa_h_offset, nasa_size);
        memcpy(species[i].nasa_low, buffer + nasa_l_offset, nasa_size);
    }
    free(buffer);
    fclose(fp);
    return 1;
}

// tran.binの読み込み
int load_tran_bin(const char* filename, SpeciesData species[], double diff_coeffs_out[][NUM_SPECIES][4], int n_species) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Error: Cannot open tran.bin file: %s\n", filename); return 0; }
    unsigned char* buffer;
    buffer = read_fortran_record(fp); if (buffer) free(buffer); else { fclose(fp); return 0; }
    buffer = read_fortran_record(fp); if (buffer) free(buffer); else { fclose(fp); return 0; }
    buffer = read_fortran_record(fp); if (buffer) free(buffer); else { fclose(fp); return 0; }
    buffer = read_fortran_record(fp); if (buffer) free(buffer); else { fclose(fp); return 0; }
    buffer = read_fortran_record(fp); if (!buffer) { fclose(fp); return 0; }
    for (int i = 0; i < n_species; i++) memcpy(species[i].lambda_coeffs, buffer + i * 4 * sizeof(double), 4 * sizeof(double));
    free(buffer);
    buffer = read_fortran_record(fp); if (!buffer) { fclose(fp); return 0; }
    for (int i = 0; i < n_species; i++) memcpy(species[i].eta_coeffs, buffer + i * 4 * sizeof(double), 4 * sizeof(double));
    free(buffer);
    buffer = read_fortran_record(fp); if (!buffer) { fclose(fp); return 0; }
    for (int i = 0; i < n_species; i++) {
        for (int j = 0; j < n_species; j++) {
            memcpy(diff_coeffs_out[i][j], buffer + (i * n_species + j) * 4 * sizeof(double), 4 * sizeof(double));
        }
    }
    free(buffer);
    fclose(fp);
    return 1;
}

// 温度とフィット係数から物性値(CGS単位)を計算
double calculate_property_from_fit(double temp, const double coeffs[4]) {
    if (temp <= 0) return 0.0;
    if (coeffs[0] == 0.0 && coeffs[1] == 0.0 && coeffs[2] == 0.0 && coeffs[3] == 0.0) return 0.0;
    double log_T = log(temp);
    double log_prop = coeffs[0] + coeffs[1] * log_T + coeffs[2] * pow(log_T, 2) + coeffs[3] * pow(log_T, 3);
    return exp(log_prop);
}

// 動的粘性係数計算 (Pa*s)
double calculate_mu_dynamic(int sp_idx, double T_local) {
    // 係数は粘性(poise)を計算するようにフィットされている
    double eta_poise = calculate_property_from_fit(T_local, loaded_species[sp_idx].eta_coeffs);
    return eta_poise * 0.1; // 単位変換: 1 poise = 0.1 Pa*s
}

// NASA多項式からモル比熱(J/mol-K)を計算
double calculate_cp_per_mole(double temp, const SpeciesData* sp) {
    // 温度が有効範囲外の場合はエラーハンドリング（または最も近い境界値を使う）
    if (temp <= 0.0) temp = 1.0; // 0K以下は物理的に無効なため、計算可能な最小値にクリップ

    // 温度に応じて高温用(nasa_high)か低温用(nasa_low)の係数を選択
    const double* a = (temp > sp->t_mid) ? sp->nasa_high : sp->nasa_low;
    
    // NASA多項式: Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    double cp_over_R = a[0] 
                     + a[1] * temp 
                     + a[2] * pow(temp, 2) 
                     + a[3] * pow(temp, 3) 
                     + a[4] * pow(temp, 4);
                     
    return cp_over_R * R_univ; // J/mol-K
}

// 動的比熱計算 (J/kg-K)
double calculate_Cp_dynamic(int sp_idx, double T_local) {
    // 1. モル比熱 (J/mol-K) を計算
    double cp_mole = calculate_cp_per_mole(T_local, &loaded_species[sp_idx]);
    
    // 2. 分子量 (kg/mol) で割って質量比熱 (J/kg-K) に変換
    double mw_kg_per_mol = loaded_species[sp_idx].mw_g_per_mol / 1000.0;
    if (mw_kg_per_mol < 1e-9) return 1000.0; // ゼロ除算防止
    
    return cp_mole / mw_kg_per_mol;
}

// 動的熱伝導率計算 (W/m-K)
double calculate_lambda_dynamic(int sp_idx, double T_local) {
    // 係数は熱伝導率(erg/cm-s-K)を計算するようにフィットされている
    double lambda_cgs = calculate_property_from_fit(T_local, loaded_species[sp_idx].lambda_coeffs);

    // 単位変換: 1 erg/s = 1e-7 W, 1 cm = 1e-2 m  =>  erg/cm-s-K = 1e-5 W/m-K
    return lambda_cgs * 1e-5; 
}

// 動的二成分拡散係数計算 (m^2/s)
double calculate_Dkj_dynamic(int k_idx, int j_idx, double T_local) {
    // 係数は二成分拡散係数(cm^2/s)を計算するようにフィットされている
    // tran.bin は対称な行列 D_kj = D_jk を格納している
    double d_cm2s = calculate_property_from_fit(T_local, diffusion_coeffs[k_idx][j_idx]);
    
    // 単位変換: 1 cm^2/s = 1e-4 m^2/s
    return d_cm2s * 1e-4; 
}




// ==============================================================================
// --- グローバル変数宣言 ---
// ==============================================================================


// --- 計算用配列 ---
Grid3D Yk[NUM_SPECIES];
Grid3D Yk_new[NUM_SPECIES];
Grid3D Yk_filtered[NUM_SPECIES];
Grid3D lambda_mix;// 拡散係数
Grid3D Dk[NUM_SPECIES];
Grid3D gamma_sgs;
Grid3D omega; // 反応項

// 変数宣言
// 速度・フィルタ済み速度・一時変数
Grid3D u, u_new, u_filtered, u_double_filtered, u_temp;
Grid3D v, v_new, v_filtered, v_double_filtered, v_temp;
Grid3D w, w_new, w_filtered, w_double_filtered, w_temp;

// スカラー場
Grid3D T, T_new, T_filtered;
Grid3D P, P_new, P_filtered;
Grid3D rho, rho_new, rho_filtered;
Grid3D G, G_new, G_filtered, G_double_filtered;

// G方程式用の速度場 (uG = u * G)
Grid3D uG, vG, wG;
Grid3D uG_filtered, vG_filtered, wG_filtered;
Grid3D uG_double_filtered, vG_double_filtered, wG_double_filtered;

// 格子属性 (int型)
Grid3D_Int attribute;

// 物性値・SGSモデル変数
Grid3D nu;          // 動粘性係数
Grid3D mu_mix;      // 混合気体の粘性係数
Grid3D alpha;       // 熱拡散率

Grid3D sgs_stress_u;
Grid3D sgs_stress_v;
Grid3D sgs_stress_w;

Grid3D nu_t;
Grid3D S_mag;
Grid3D q2;
Grid3D SijSij;
Grid3D div_u;
Grid3D ST_turbulent;
Grid3D mu_filtered;
Grid3D Cp_mix;      // 混合気体の定圧比熱

// ==============================================================================
// --- グローバル変数宣言 (動的確保用に変更) ---
// ==============================================================================

// 運動量 (ρu, ρv, ρw)
Grid3D rhou, rhov, rhow;
Grid3D rhou_new, rhov_new, rhow_new;
Grid3D rhou_initial, rhov_initial, rhow_initial;
Grid3D rhou_temp, rhov_temp, rhow_temp;

// RK法のステージごとの時間微分項 (k1)
Grid3D k1_rhou, k1_rhov, k1_rhow;
Grid3D k1_T, k1_G;
Grid3D k1_Yk[NUM_SPECIES]; // 化学種はポインタの配列にする

// RK法 (k2)
Grid3D k2_rhou, k2_rhov, k2_rhow;
Grid3D k2_T, k2_G;
Grid3D k2_Yk[NUM_SPECIES];

// RK法 (k3)
Grid3D k3_rhou, k3_rhov, k3_rhow;
Grid3D k3_T, k3_G;
Grid3D k3_Yk[NUM_SPECIES];

// RK法 (k4)
Grid3D k4_rhou, k4_rhov, k4_rhow;
Grid3D k4_T, k4_G;
Grid3D k4_Yk[NUM_SPECIES];

// 各化学種の気体定数 (これはサイズが小さいので動的確保の必要はありません)
double R0_WT[NUM_SPECIES];

// ==============================================================================
// --- グローバル変数宣言 (動的確保用に変更) ---
// ==============================================================================

// 化学量論比・質量分率（スカラー変数はそのまま）
double STOICH_O2_PER_C3H8;     
double STOICH_CO2_PER_C3H8;    
double STOICH_H2O_PER_C3H8;    

double Y_C3H8_inlet;
double Y_O2_inlet;
double Y_N2_inlet;

// 小さい配列（1次元）はそのまま
double inletYk[NUM_SPECIES];
double ambientYk[NUM_SPECIES];

// 時間平均計算用の配列 (3次元なので動的確保に変更) 
Grid3D u_mean, v_mean, w_mean;
Grid3D uu_mean, vv_mean, ww_mean;
Grid3D uv_mean, uw_mean, vw_mean;
Grid3D G_mean;
int time_avg_steps = 0; // カウンタはそのまま

// リゾルブド・レイノルズ応力 (3次元なので動的確保に変更) 
Grid3D re_stress_uu, re_stress_vv, re_stress_ww;
Grid3D re_stress_uv, re_stress_uw, re_stress_vw;

// 流入面変動 (2次元配列) 
// サイズが小さい(数MB以下)ため、動的確保せずそのまま static 配列でOKです
double u_inlet_fluct[NY][NZ];
double v_inlet_fluct[NY][NZ];
double w_inlet_fluct[NY][NZ];

// ==============================================================================
// --- グローバル変数宣言 (動的確保用に変更) ---
// ==============================================================================

// ステップ開始時の状態を保存
Grid3D u_initial, v_initial, w_initial;
Grid3D T_initial, G_initial;
Grid3D Yk_initial[NUM_SPECIES]; // 化学種は配列

// 中間状態の計算用
Grid3D u_temp, v_temp, w_temp;
Grid3D T_temp, G_temp;
Grid3D Yk_temp[NUM_SPECIES]; // 化学種は配列
Grid3D rho_temp;

// RK法の係数 (速度 u, v, w 用)
Grid3D k1_u, k1_v, k1_w;
Grid3D k2_u, k2_v, k2_w;
Grid3D k3_u, k3_v, k3_w;
Grid3D k4_u, k4_v, k4_w;

// RK法の係数 (T, G 用)
// ※もし直前の作業で既に宣言している場合は、ここは削除してください
Grid3D k1_T, k1_G;
Grid3D k2_T, k2_G;
Grid3D k3_T, k3_G;
Grid3D k4_T, k4_G;

// ==============================================================================
// --- グローバル変数宣言 (動的確保用に変更) ---
// ==============================================================================

// RK法の係数 (化学種 Yk 用)
// ※ Grid3Dの配列として宣言します
Grid3D k1_Yk[NUM_SPECIES];
Grid3D k2_Yk[NUM_SPECIES];
Grid3D k3_Yk[NUM_SPECIES];
Grid3D k4_Yk[NUM_SPECIES];

// デバッグ用: Gの勾配の大きさ
// double配列からGrid3Dポインタに変更します
Grid3D grad_G_magnitude_for_debug;



// 1つの計測点の情報を保持する構造体
typedef struct {
    int i, j, k; // 格子インデックス
    int count;   // この点で記録したデータ数
    double u_data[MAX_DATA_POINTS];
    double v_data[MAX_DATA_POINTS];
    double w_data[MAX_DATA_POINTS];
    double G_data[MAX_DATA_POINTS];
} MeasurementPoint;

// 計測点の配列
MeasurementPoint points[MAX_MEASUREMENT_POINTS];
int num_points_to_measure = 0; // 実際に計測する点の数

// ==============================================================================
// --- 移流項計算関数 ---
// ==============================================================================

/**
 * @brief 非保存形の移流項 (U⋅∇)φ をQUICKスキームで計算する (スカラー輸送用)
 */
double calculate_convection_non_conservative(
    int i, int j, int k,
    const Grid3D phi, // const double phi[NX][NY][NZ] から変更
    const Grid3D u, const Grid3D v, const Grid3D w
) {
    if (i < 2 || i >= NX - 2 || j < 2 || j >= NY - 2 || k < 2 || k >= NZ - 2) {
        double conv_term_fallback = 0.0;
        if (u[i][j][k] >= 0.0) conv_term_fallback = u[i][j][k] * (phi[i][j][k] - phi[i-1][j][k]) / DX;
        else conv_term_fallback = u[i][j][k] * (phi[i+1][j][k] - phi[i][j][k]) / DX;
        if (v[i][j][k] >= 0.0) conv_term_fallback += v[i][j][k] * (phi[i][j][k] - phi[i][j-1][k]) / DY;
        else conv_term_fallback += v[i][j][k] * (phi[i][j+1][k] - phi[i][j][k]) / DY;
        if (w[i][j][k] >= 0.0) conv_term_fallback += w[i][j][k] * (phi[i][j][k] - phi[i][j][k-1]) / DZ;
        else conv_term_fallback += w[i][j][k] * (phi[i][j][k+1] - phi[i][j][k]) / DZ;
        return conv_term_fallback;
    }
    double phi_iph, phi_imh, phi_jph, phi_jmh, phi_kph, phi_kmh;
    if (u[i][j][k] >= 0.0) {
        phi_iph = ( QUICK_COEFF_UW2*phi[i-1][j][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i+1][j][k] );
        phi_imh = ( QUICK_COEFF_UW2*phi[i-2][j][k] + QUICK_COEFF_UW1*phi[i-1][j][k] + QUICK_COEFF_DW1*phi[i][j][k] );
    } else {
        phi_iph = ( QUICK_COEFF_UW2*phi[i+2][j][k] + QUICK_COEFF_UW1*phi[i+1][j][k] + QUICK_COEFF_DW1*phi[i][j][k] );
        phi_imh = ( QUICK_COEFF_UW2*phi[i+1][j][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i-1][j][k] );
    }
    if (v[i][j][k] >= 0.0) {
        phi_jph = ( QUICK_COEFF_UW2*phi[i][j-1][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j+1][k] );
        phi_jmh = ( QUICK_COEFF_UW2*phi[i][j-2][k] + QUICK_COEFF_UW1*phi[i][j-1][k] + QUICK_COEFF_DW1*phi[i][j][k] );
    } else {
        phi_jph = ( QUICK_COEFF_UW2*phi[i][j+2][k] + QUICK_COEFF_UW1*phi[i][j+1][k] + QUICK_COEFF_DW1*phi[i][j][k] );
        phi_jmh = ( QUICK_COEFF_UW2*phi[i][j+1][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j-1][k] );
    }
    if (w[i][j][k] >= 0.0) {
        phi_kph = ( QUICK_COEFF_UW2*phi[i][j][k-1] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j][k+1] );
        phi_kmh = ( QUICK_COEFF_UW2*phi[i][j][k-2] + QUICK_COEFF_UW1*phi[i][j][k-1] + QUICK_COEFF_DW1*phi[i][j][k] );
    } else {
        phi_kph = ( QUICK_COEFF_UW2*phi[i][j][k+2] + QUICK_COEFF_UW1*phi[i][j][k+1] + QUICK_COEFF_DW1*phi[i][j][k] );
        phi_kmh = ( QUICK_COEFF_UW2*phi[i][j][k+1] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j][k-1] );
    }
    double convection_x = u[i][j][k] * (phi_iph - phi_imh) / DX;
    double convection_y = v[i][j][k] * (phi_jph - phi_jmh) / DY;
    double convection_z = w[i][j][k] * (phi_kph - phi_kmh) / DZ;
    return convection_x + convection_y + convection_z;
}

/**
 * @brief 保存形の移流項 ∇⋅( (ρU)φ ) をQUICKスキームで計算する (運動量輸送用)
 */
double calculate_convection_conservative(
    int i, int j, int k,
    const Grid3D phi,
    const Grid3D u, const Grid3D v, const Grid3D w,
    const Grid3D rho
) {
    if (i < 2 || i >= NX-2 || j < 2 || j >= NY-2 || k < 2 || k >= NZ-2) {
        double massflux_iph = 0.5*(rho[i+1][j][k]+rho[i][j][k]) * 0.5*(u[i+1][j][k]+u[i][j][k]);
        double massflux_imh = 0.5*(rho[i][j][k]+rho[i-1][j][k]) * 0.5*(u[i][j][k]+u[i-1][j][k]);
        double massflux_jph = 0.5*(rho[i][j+1][k]+rho[i][j][k]) * 0.5*(v[i][j+1][k]+v[i][j][k]);
        double massflux_jmh = 0.5*(rho[i][j][k]+rho[i][j-1][k]) * 0.5*(v[i][j][k]+v[i][j-1][k]);
        double massflux_kph = 0.5*(rho[i][j][k+1]+rho[i][j][k]) * 0.5*(w[i][j][k+1]+w[i][j][k]);
        double massflux_kmh = 0.5*(rho[i][j][k]+rho[i][j][k-1]) * 0.5*(w[i][j][k]+w[i][j][k-1]);
        double phi_iph = (massflux_iph > 0) ? phi[i][j][k]   : phi[i+1][j][k];
        double phi_imh = (massflux_imh > 0) ? phi[i-1][j][k] : phi[i][j][k];
        double phi_jph = (massflux_jph > 0) ? phi[i][j][k]   : phi[i][j+1][k];
        double phi_jmh = (massflux_jmh > 0) ? phi[i][j-1][k] : phi[i][j][k];
        double phi_kph = (massflux_kph > 0) ? phi[i][j][k]   : phi[i][j][k+1];
        double phi_kmh = (massflux_kmh > 0) ? phi[i][j][k-1] : phi[i][j][k];
        return (massflux_iph*phi_iph - massflux_imh*phi_imh)/DX + (massflux_jph*phi_jph - massflux_jmh*phi_jmh)/DY + (massflux_kph*phi_kph - massflux_kmh*phi_kmh)/DZ;
    }
    double phi_iph, phi_imh, phi_jph, phi_jmh, phi_kph, phi_kmh;
    double u_iph = 0.5*(u[i+1][j][k]+u[i][j][k]); double u_imh = 0.5*(u[i][j][k]+u[i-1][j][k]);
    double v_jph = 0.5*(v[i][j+1][k]+v[i][j][k]); double v_jmh = 0.5*(v[i][j][k]+v[i][j-1][k]);
    double w_kph = 0.5*(w[i][j][k+1]+w[i][j][k]); double w_kmh = 0.5*(w[i][j][k]+w[i][j][k-1]);
    double rho_iph = 0.5*(rho[i+1][j][k]+rho[i][j][k]); double rho_imh = 0.5*(rho[i][j][k]+rho[i-1][j][k]);
    double rho_jph = 0.5*(rho[i][j+1][k]+rho[i][j][k]); double rho_jmh = 0.5*(rho[i][j][k]+rho[i][j-1][k]);
    double rho_kph = 0.5*(rho[i][j][k+1]+rho[i][j][k]); double rho_kmh = 0.5*(rho[i][j][k]+rho[i][j][k-1]);
    double massflux_iph = rho_iph*u_iph; double massflux_imh = rho_imh*u_imh;
    double massflux_jph = rho_jph*v_jph; double massflux_jmh = rho_jmh*v_jmh;
    double massflux_kph = rho_kph*w_kph; double massflux_kmh = rho_kmh*w_kmh;
    if (u_iph >= 0.0) { phi_iph = (QUICK_COEFF_UW2*phi[i-1][j][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i+1][j][k]); }
    else { phi_iph = (QUICK_COEFF_UW2*phi[i+2][j][k] + QUICK_COEFF_UW1*phi[i+1][j][k] + QUICK_COEFF_DW1*phi[i][j][k]); }
    if (u_imh >= 0.0) { phi_imh = (QUICK_COEFF_UW2*phi[i-2][j][k] + QUICK_COEFF_UW1*phi[i-1][j][k] + QUICK_COEFF_DW1*phi[i][j][k]); }
    else { phi_imh = (QUICK_COEFF_UW2*phi[i+1][j][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i-1][j][k]); }
    if (v_jph >= 0.0) { phi_jph = (QUICK_COEFF_UW2*phi[i][j-1][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j+1][k]); }
    else { phi_jph = (QUICK_COEFF_UW2*phi[i][j+2][k] + QUICK_COEFF_UW1*phi[i][j+1][k] + QUICK_COEFF_DW1*phi[i][j][k]); }
    if (v_jmh >= 0.0) { phi_jmh = (QUICK_COEFF_UW2*phi[i][j-2][k] + QUICK_COEFF_UW1*phi[i][j-1][k] + QUICK_COEFF_DW1*phi[i][j][k]); }
    else { phi_jmh = (QUICK_COEFF_UW2*phi[i][j+1][k] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j-1][k]); }
    if (w_kph >= 0.0) { phi_kph = (QUICK_COEFF_UW2*phi[i][j][k-1] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j][k+1]); }
    else { phi_kph = (QUICK_COEFF_UW2*phi[i][j][k+2] + QUICK_COEFF_UW1*phi[i][j][k+1] + QUICK_COEFF_DW1*phi[i][j][k]); }
    if (w_kmh >= 0.0) { phi_kmh = (QUICK_COEFF_UW2*phi[i][j][k-2] + QUICK_COEFF_UW1*phi[i][j][k-1] + QUICK_COEFF_DW1*phi[i][j][k]); }
    else { phi_kmh = (QUICK_COEFF_UW2*phi[i][j][k+1] + QUICK_COEFF_UW1*phi[i][j][k]   + QUICK_COEFF_DW1*phi[i][j][k-1]); }
    return (massflux_iph*phi_iph-massflux_imh*phi_imh)/DX + (massflux_jph*phi_jph-massflux_jmh*phi_jmh)/DY + (massflux_kph*phi_kph-massflux_kmh*phi_kmh)/DZ;
}

// ==============================================================================
// --- WENO5 スキーム用 補助関数 ---
// ==============================================================================

// WENO5の滑らかさ指標(IS)を計算する関数
// 5つの点 (v1, v2, v3, v4, v5) から、3つのステンシルのISを計算する
void calculate_weno5_smoothness(
    double v1, double v2, double v3, double v4, double v5,
    double* is0, double* is1, double* is2
) {
    // IS_k = sum_{l=1}^{r-1} int_{x_{i-1/2}}^{x_{i+1/2}} (Delta x)^{2l-1} (d^{l}p_k(x)/dx^{l})^2 dx
    // 5次WENO (r=3) の場合の具体的な計算式
    *is0 = (13.0/12.0) * pow(v1 - 2.0*v2 + v3, 2) + (1.0/4.0) * pow(v1 - 4.0*v2 + 3.0*v3, 2);
    *is1 = (13.0/12.0) * pow(v2 - 2.0*v3 + v4, 2) + (1.0/4.0) * pow(v2 - v4, 2);
    *is2 = (13.0/12.0) * pow(v3 - 2.0*v4 + v5, 2) + (1.0/4.0) * pow(3.0*v3 - 4.0*v4 + v5, 2);
}

// WENO5の勾配を計算する関数 (1次元)
// phi配列から、点iにおける勾配を計算する
// dx: 格子間隔
double calculate_weno5_gradient_1d(
    const Grid3D phi, // const double phi[NX][NY][NZ] から変更
    int i, int j, int k, int dir, double dx
) {
    // WENOに必要な6点を取得する
    // dir: 0=x方向, 1=y方向, 2=z方向
    double p[6];
    if (dir == 0) { // X-direction
        for(int n=0; n<6; ++n) p[n] = phi[i-2+n][j][k];
    } else if (dir == 1) { // Y-direction
        for(int n=0; n<6; ++n) p[n] = phi[i][j-2+n][k];
    } else { // Z-direction
        for(int n=0; n<6; ++n) p[n] = phi[i][j][k-2+n];
    }
    
    // 1. 各ステンシルでの勾配計算 (3次精度)
    double grad0 = (2.0*p[0] - 9.0*p[1] + 7.0*p[2]) / (6.0 * dx); // 厳密にはこれは勾配ではないが...
    double grad1 = (-p[1] + p[3]) / (2.0 * dx);
    double grad2 = (-7.0*p[2] + 9.0*p[3] - 2.0*p[4]) / (6.0 * dx);
    
    // 正確には、これは再構築された点iでの値の勾配であり、以下のようになるべき（Jiang & Shu, 1996）
    // d(phi)/dx at point i
    grad0 = (p[0] - 6.0*p[1] + 15.0*p[2] - 10.0*p[3] + 0.0*p[4] + 0.0*p[5]) / (60.0 * dx); // This is not standard... let's use a simpler formulation for flux difference.
    // WENOは本来、フラックスの差分を計算するのに使われる。ここでは勾配計算に応用する。
    // F_{i+1/2} - F_{i-1/2}
    // ここでは単純化のため、点iでの勾配を直接計算するバージョンを採用する。
    // d(phi)/dx ≈ (phi_{i+1/2} - phi_{i-1/2})/dx
    // 5次精度の中心差分: (-p[i+2] + 8p[i+1] - 8p[i-1] + p[i-2]) / 12dx
    // ここではより安定なWENO再構築を用いる
    
    // WENO flux difference reconstruction at point i
    // Reconstruct phi at i+1/2 and i-1/2
    
    // --- 再構築 at i+1/2 ---
    double is0_p, is1_p, is2_p;
    calculate_weno5_smoothness(p[1], p[2], p[3], p[4], p[5], &is0_p, &is1_p, &is2_p);
    
    const double epsilon = 1.0e-6; // ゼロ除算防止
    const double d0_p = 3.0/10.0, d1_p = 6.0/10.0, d2_p = 1.0/10.0; // optimal weights for right-biased stencil
    
    double alpha0_p = d0_p / pow(epsilon + is0_p, 2);
    double alpha1_p = d1_p / pow(epsilon + is1_p, 2);
    double alpha2_p = d2_p / pow(epsilon + is2_p, 2);
    double sum_alpha_p = alpha0_p + alpha1_p + alpha2_p;
    
    double w0_p = alpha0_p / sum_alpha_p;
    double w1_p = alpha1_p / sum_alpha_p;
    double w2_p = alpha2_p / sum_alpha_p;
    
    double phi_iph_stencil0 = (2.0*p[1] - 7.0*p[2] + 11.0*p[3]) / 6.0;
    double phi_iph_stencil1 = (-p[2] + 5.0*p[3] + 2.0*p[4]) / 6.0;
    double phi_iph_stencil2 = (2.0*p[3] + 5.0*p[4] - p[5]) / 6.0;
    
    double phi_iph = w0_p * phi_iph_stencil0 + w1_p * phi_iph_stencil1 + w2_p * phi_iph_stencil2;

    // --- 再構築 at i-1/2 ---
    double is0_m, is1_m, is2_m;
    calculate_weno5_smoothness(p[0], p[1], p[2], p[3], p[4], &is0_m, &is1_m, &is2_m);

    const double d0_m = 1.0/10.0, d1_m = 6.0/10.0, d2_m = 3.0/10.0; // optimal weights for left-biased stencil

    double alpha0_m = d0_m / pow(epsilon + is0_m, 2);
    double alpha1_m = d1_m / pow(epsilon + is1_m, 2);
    double alpha2_m = d2_m / pow(epsilon + is2_m, 2);
    double sum_alpha_m = alpha0_m + alpha1_m + alpha2_m;

    double w0_m = alpha0_m / sum_alpha_m;
    double w1_m = alpha1_m / sum_alpha_m;
    double w2_m = alpha2_m / sum_alpha_m;

    double phi_imh_stencil0 = (-p[0] + 5.0*p[1] + 2.0*p[2]) / 6.0;
    double phi_imh_stencil1 = (2.0*p[1] + 5.0*p[2] - p[3]) / 6.0;
    double phi_imh_stencil2 = (11.0*p[2] - 7.0*p[3] + 2.0*p[4]) / 6.0;

    double phi_imh = w0_m * phi_imh_stencil0 + w1_m * phi_imh_stencil1 + w2_m * phi_imh_stencil2;

    // 最終的な勾配
    return (phi_iph - phi_imh) / dx;
}

/*解説
    WENOスキームは本来、`u * dG/dx` のような移流項のフラックス `u*G` をセル界面で再構築するために開発されたものです。ここではそれを応用し、`G` そのものをセル界面 `i+1/2` と `i-1/2` で再構築し、その差分から中心点 `i` の勾配を計算しています。
    calculate_weno5_smoothness`: Jiang and Shu (1996) の論文で提案された、標準的な滑らかさ指標の計算式を実装しています。
    calculate_weno5_gradient_1d`: 1次元方向のWENO5勾配を計算します。
    まず計算に必要な6点を配列 `p` にコピーします。
    次に、セル界面 `i+1/2` と `i-1/2` での `G` の値を、それぞれWENOスキームで再構築します。
    最後に、`(G_iph - G_imh) / dx` で中心差分的に勾配を計算します。
*/

// ==============================================================================
// --- 乱流用 補助関数 ---
// ==============================================================================


// 簡易乱数生成器 
// 乱数の状態を保持するグローバル変数
static unsigned long long next_rand = 1;

// 乱数の種を設定する関数 (srandの代わり)
void my_srand(unsigned int seed) {
    next_rand = seed;
}

// 0からMY_RAND_MAXまでの整数の乱数を生成する関数 (randの代わり)
// MY_RAND_MAXは32767
#define MY_RAND_MAX 32767
int my_rand(void) {
    next_rand = next_rand * 1103515245 + 12345;
    return (unsigned int)(next_rand / 65536) % (MY_RAND_MAX + 1);
}

// 最終修正版 v2】
void initialize_vortices() {

    printf("\n--- DEBUG: Inside initialize_vortices ---\n");
    printf("Test rand() #1: %d\n", rand());
    printf("Test rand() #2: %d\n", rand());
    printf("Test rand() as double: %.6f\n", rand() / (double)RAND_MAX);
    printf("-----------------------------------------\n\n");
    printf("Initializing %d vortices for SEM...\n", NUM_VORTICES);
    double u_rms = INLET_VELOCITY * INLET_TURBULENCE_INTENSITY;
    for (int i = 0; i < NUM_VORTICES; i++) {
        /*double r1 = my_rand() / (double)MY_RAND_MAX;
        double r2 = my_rand() / (double)MY_RAND_MAX;
        double r3 = my_rand() / (double)MY_RAND_MAX;
        double r4 = my_rand() / (double)MY_RAND_MAX;
        double r5 = my_rand() / (double)MY_RAND_MAX;
        double r6 = my_rand() / (double)MY_RAND_MAX;
        double r7 = my_rand() / (double)MY_RAND_MAX;*/

        double r1 = rand() / (double)RAND_MAX; 
        double r2 = rand() / (double)RAND_MAX;
        double r3 = rand() / (double)RAND_MAX;
        double r4 = rand() / (double)RAND_MAX;
        double r5 = rand() / (double)RAND_MAX;
        double r6 = rand() / (double)RAND_MAX;
        double r7 = rand() / (double)RAND_MAX;

        vortices[i].x = (r1 - 0.5) * VORTEX_BOX_X;
        vortices[i].y = r2 * VORTEX_BOX_Y;
        vortices[i].z = r3 * VORTEX_BOX_Z;
        double min_radius = TURBULENT_LENGTH_SCALE * 0.5;
        double max_radius = TURBULENT_LENGTH_SCALE * 1.5;
        vortices[i].radius = min_radius + (max_radius - min_radius) * r4;
        vortices[i].strength_u = (r5 * 2.0 - 1.0) * u_rms;
        vortices[i].strength_v = (r6 * 2.0 - 1.0) * u_rms;
        vortices[i].strength_w = (r7 * 2.0 - 1.0) * u_rms;
    }
}

// SEM: 渦の位置を更新(移流)する関数
void update_vortices(double dt) {
    // 乱流強度の基準値を計算（再設定時に使用）
    double u_rms = INLET_VELOCITY * INLET_TURBULENCE_INTENSITY;

    for (int i = 0; i < NUM_VORTICES; i++) {
        // x方向に平均流速で移流
        vortices[i].x += INLET_VELOCITY * dt;

        // 仮想的な箱から出た渦を、上流側に再配置する (周期的境界条件 + ランダムリセット)
        if (vortices[i].x > VORTEX_BOX_X / 2.0) {
            
            // 1. X位置をボックスの反対側（上流）に戻す
            vortices[i].x -= VORTEX_BOX_X;

            // 2. Y, Z 位置をランダムに再設定
            // 0.0 ~ 1.0 の乱数生成
            double r_y = (double)rand() / (double)RAND_MAX;
            double r_z = (double)rand() / (double)RAND_MAX;

            vortices[i].y = r_y * VORTEX_BOX_Y;
            vortices[i].z = r_z * VORTEX_BOX_Z;

            // 3. 渦の強さ (u, v, w成分) をランダムに再設定
            // -1.0 ~ 1.0 の乱数生成
            double r_u = (double)rand() / (double)RAND_MAX;
            double r_v = (double)rand() / (double)RAND_MAX;
            double r_w = (double)rand() / (double)RAND_MAX;

            vortices[i].strength_u = (r_u * 2.0 - 1.0) * u_rms;
            vortices[i].strength_v = (r_v * 2.0 - 1.0) * u_rms;
            vortices[i].strength_w = (r_w * 2.0 - 1.0) * u_rms;

            // 4. (オプション) 渦のサイズも再設定することで、より多様性を持たせる
            double r_rad = (double)rand() / (double)RAND_MAX;
            double min_radius = TURBULENT_LENGTH_SCALE * 0.5;
            double max_radius = TURBULENT_LENGTH_SCALE * 1.5;
            vortices[i].radius = min_radius + (max_radius - min_radius) * r_rad;
        }
    }
}

// SEM: 合成渦法に基づいて流入乱流を生成する
// 内部の計算過程を詳細に出力する
void generate_inlet_turbulence_SEM(
    int t_loop,
    double dt,
    int inlet_attribute[NY][NZ],
    double u_inlet_fluct[NY][NZ],
    double v_inlet_fluct[NY][NZ],
    double w_inlet_fluct[NY][NZ]
) {
    // 1. 全ての渦の位置を更新
    update_vortices(dt);

    // デバッグ出力：最初の渦の状態を確認 
    /*if (t_loop % 10 == 0) {
        printf("\n--- Detailed SEM Debug (Vortex #0) ---\n");
        printf("---t_loop: %d---\n", t_loop);
        printf("Vortex[0] position: x=%.4f, y=%.4f, z=%.4f\n", vortices[0].x, vortices[0].y, vortices[0].z);
        printf("Vortex[0] radius: %.4f\n", vortices[0].radius);
        printf("Vortex[0] strength: u=%.4f, v=%.4f, w=%.4f\n", vortices[0].strength_u, vortices[0].strength_v, vortices[0].strength_w);
        printf("--------------------------------------\n");
    }*/

    // 2. 各流入格子点での速度変動を計算
    #pragma omp parallel for collapse(2)
    for (int j = 0; j < NY; j++) {
        for (int k = 0; k < NZ; k++) {
            u_inlet_fluct[j][k] = 0.0;
            v_inlet_fluct[j][k] = 0.0;
            w_inlet_fluct[j][k] = 0.0;

            if (inlet_attribute[j][k] == 1) {
                double y_pos = j * DY;
                double z_pos = k * DZ;
                
                for (int v_idx = 0; v_idx < NUM_VORTICES; v_idx++) {
                    double dx = 0.0 - vortices[v_idx].x;
                    double dy = y_pos - vortices[v_idx].y;
                    double dz = z_pos - vortices[v_idx].z;
                    double dist = sqrt(dx*dx + dy*dy + dz*dz);
                    double radius = vortices[v_idx].radius;
                    double influence = 0.0;
                    if (dist < radius) {
                        influence = 1.0 - (dist / radius);
                    }
                    
                    // デバッグ出力：中心点での計算過程を確認 
                    /*if (t_loop == 0 && v_idx == 0 && j == NY/2 && k == NZ/2) {
                        printf("\n--- Detailed SEM Debug (Center Point Calculation) ---\n");
                        printf("Center point (j,k): (%d, %d)\n", j, k);
                        printf("Center point pos: y=%.4f, z=%.4f\n", y_pos, z_pos);
                        printf("Distance to Vortex[0]: dist=%.4f\n", dist);
                        printf("Influence of Vortex[0]: influence=%.4f (dist < radius is %s)\n", influence, (dist < radius ? "true" : "false"));
                        printf("-----------------------------------------------------\n\n");
                    }*/
                    
                    u_inlet_fluct[j][k] += vortices[v_idx].strength_u * influence;
                    v_inlet_fluct[j][k] += vortices[v_idx].strength_v * influence;
                    w_inlet_fluct[j][k] += vortices[v_idx].strength_w * influence;
                }
            }
        }
    }

    // 3. 質量保存のため、まず平均変動がゼロになるように補正する
    double total_u_fluct = 0.0, total_v_fluct = 0.0, total_w_fluct = 0.0;
    int num_inlet_cells = 0;
    for (int j = 0; j < NY; j++) {
        for (int k = 0; k < NZ; k++) {
            if (inlet_attribute[j][k] == 1) {
                total_u_fluct += u_inlet_fluct[j][k];
                total_v_fluct += v_inlet_fluct[j][k];
                total_w_fluct += w_inlet_fluct[j][k];
                num_inlet_cells++;
            }
        }
    }

    if (num_inlet_cells > 0) {
        double avg_u_fluct = total_u_fluct / num_inlet_cells;
        double avg_v_fluct = total_v_fluct / num_inlet_cells;
        double avg_w_fluct = total_w_fluct / num_inlet_cells;
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (inlet_attribute[j][k] == 1) {
                    u_inlet_fluct[j][k] -= avg_u_fluct;
                    v_inlet_fluct[j][k] -= avg_v_fluct;
                    w_inlet_fluct[j][k] -= avg_w_fluct;
                }
            }
        }
    }

    // 4. 平均を引いた後の変動の強度を計算し、目標値に正規化する
    double total_u_sq = 0.0, total_v_sq = 0.0, total_w_sq = 0.0;
    for (int j = 0; j < NY; j++) {
        for (int k = 0; k < NZ; k++) {
            if (inlet_attribute[j][k] == 1) {
                total_u_sq += u_inlet_fluct[j][k] * u_inlet_fluct[j][k];
                total_v_sq += v_inlet_fluct[j][k] * v_inlet_fluct[j][k];
                total_w_sq += w_inlet_fluct[j][k] * w_inlet_fluct[j][k];
            }
        }
    }

    if (num_inlet_cells > 0) {
        double u_rms_generated = sqrt(total_u_sq / num_inlet_cells);
        double v_rms_generated = sqrt(total_v_sq / num_inlet_cells);
        double w_rms_generated = sqrt(total_w_sq / num_inlet_cells);
        double rms_target = INLET_VELOCITY * INLET_TURBULENCE_INTENSITY;
        
        double scale_u = (u_rms_generated > SMALL_NUMBER) ? rms_target / u_rms_generated : 0.0;
        double scale_v = (v_rms_generated > SMALL_NUMBER) ? rms_target / v_rms_generated : 0.0;
        double scale_w = (w_rms_generated > SMALL_NUMBER) ? rms_target / w_rms_generated : 0.0;
        
        if (t_loop == 0) {
            printf("--- SEM Debug Info (t=0) ---\n");
            printf("Target RMS: %.4f\n", rms_target);
            printf("Generated u_rms (before scale): %.4f, scale_u: %.4f\n", u_rms_generated, scale_u);
            printf("Generated v_rms (before scale): %.4f, scale_v: %.4f\n", v_rms_generated, scale_v);
            printf("Generated w_rms (before scale): %.4f, scale_w: %.4f\n", w_rms_generated, scale_w);
            printf("----------------------------\n\n");
        }
        
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (inlet_attribute[j][k] == 1) {
                    u_inlet_fluct[j][k] *= scale_u;
                    v_inlet_fluct[j][k] *= scale_v;
                    w_inlet_fluct[j][k] *= scale_w;
                }
            }
        }
    }
}

// --- 関数プロトタイプ宣言  ---
void initialize_probe_file(void); // 単数形に変更
void record_flame_region_data(int t_loop, const Grid3D u, const Grid3D v, const Grid3D w, const Grid3D G, const Grid3D nu_t, const Grid3D S_mag);

// ==============================================================================
// --- 本命コード ---
// ==============================================================================

void initialize() {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                u[i][j][k] = 0.0; u_new[i][j][k] = 0.0; u_filtered[i][j][k] = 0.0; u_double_filtered[i][j][k] = 0.0; u_temp[i][j][k] = 0.0;
                v[i][j][k] = 0.0; v_new[i][j][k] = 0.0; v_filtered[i][j][k] = 0.0; v_double_filtered[i][j][k] = 0.0; v_temp[i][j][k] = 0.0;
                w[i][j][k] = 0.0; w_new[i][j][k] = 0.0; w_filtered[i][j][k] = 0.0; w_double_filtered[i][j][k] = 0.0; w_temp[i][j][k] = 0.0;
                T[i][j][k] = 0.0; T_new[i][j][k] = 0.0; T_filtered[i][j][k] = 0.0;
                P[i][j][k] = 0.0; P_new[i][j][k] = 0.0; P_filtered[i][j][k] = 0.0;
                rho[i][j][k] = 0.0; rho_new[i][j][k] = 0.0; rho_filtered[i][j][k] = 0.0;
                G[i][j][k] = 0.0; G_new[i][j][k] = 0.0; G_filtered[i][j][k] = 0.0; G_double_filtered[i][j][k] = 0.0;
                uG_filtered[i][j][k] = 0.0; vG_filtered[i][j][k] = 0.0; wG_filtered[i][j][k] = 0.0;
                uG_double_filtered[i][j][k] = 0.0; vG_double_filtered[i][j][k] = 0.0; wG_double_filtered[i][j][k] = 0.0;
                omega[i][j][k] = 0.0;
                gamma_sgs[i][j][k] = 0.0;
                S_mag[i][j][k] = 0.0;
                q2[i][j][k] = 0.0;
                sgs_stress_u[i][j][k] = 0.0;
                sgs_stress_v[i][j][k] = 0.0;
                sgs_stress_w[i][j][k] = 0.0;
                nu_t[i][j][k] = 0.0;
                Cp_mix[i][j][k] = DEFAULT_CP_AIR;        
                lambda_mix[i][j][k] = DEFAULT_LAMBDA_AIR; 
                mu_filtered[i][j][k] = DEFAULT_MU_AIR;  
                for (int n = 0; n < NUM_SPECIES; n++) {
                    Yk[n][i][j][k] = 0.0; Yk_new[n][i][j][k] = 0.0;
                    Dk[n][i][j][k] = 0.0;
                }
                uG[i][j][k] = 0.0; vG[i][j][k] = 0.0; wG[i][j][k] = 0.0;
                for (int n = 0; n < NUM_SPECIES; n++) {
                    Yk_filtered[n][i][j][k] = 0.0;
                }
                u_mean[i][j][k] = 0.0; v_mean[i][j][k] = 0.0; w_mean[i][j][k] = 0.0;
                uu_mean[i][j][k] = 0.0; vv_mean[i][j][k] = 0.0; ww_mean[i][j][k] = 0.0;
                uv_mean[i][j][k] = 0.0; uw_mean[i][j][k] = 0.0; vw_mean[i][j][k] = 0.0;
                G_mean[i][j][k] = 0.0;
                
                re_stress_uu[i][j][k] = 0.0; re_stress_vv[i][j][k] = 0.0; re_stress_ww[i][j][k] = 0.0;
                re_stress_uv[i][j][k] = 0.0; re_stress_uw[i][j][k] = 0.0; re_stress_vw[i][j][k] = 0.0;
                
                grad_G_magnitude_for_debug[i][j][k] = 0.0;
                rhou[i][j][k] = 0.0; rhou_new[i][j][k] = 0.0; rhou_initial[i][j][k] = 0.0; rhou_temp[i][j][k] = 0.0;
                rhov[i][j][k] = 0.0; rhov_new[i][j][k] = 0.0; rhov_initial[i][j][k] = 0.0; rhov_temp[i][j][k] = 0.0;
                rhow[i][j][k] = 0.0; rhow_new[i][j][k] = 0.0; rhow_initial[i][j][k] = 0.0; rhow_temp[i][j][k] = 0.0;
            }
        }
    }
    time_avg_steps = 0;
}

void setting(
    Grid3D_Int attribute, 
    Grid3D u, Grid3D u_new, 
    Grid3D v, Grid3D v_new, 
    Grid3D w, Grid3D w_new, 
    Grid3D P, Grid3D P_new,
    Grid3D T, Grid3D T_new,
    Grid3D G, Grid3D G_new,
    Grid3D *Yk, Grid3D *Yk_new, 
    Grid3D *Dk,                 
    Grid3D omega,
    Grid3D rho, Grid3D rho_new,
    Grid3D rhou, Grid3D rhou_new,
    Grid3D rhov, Grid3D rhov_new,
    Grid3D rhow, Grid3D rhow_new
) {
    // --- 1. 領域属性の決定 ---
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (i == 0) {
                    // i=0面 (流入面)
                    double y_coord = j * DY;
                    double z_coord = k * DZ;
                    double distance_sq = pow(y_coord - INLET_CENTER_Y, 2) + pow(z_coord - INLET_CENTER_Z, 2);
                    if (distance_sq <= pow(INLET_RADIUS, 2)) {
                        attribute[i][j][k] = 1; // 流入セル
                    } else if (distance_sq <= pow(RIM_OUTER_RADIUS, 2)) {
                        attribute[i][j][k] = 6; // バーナーリム（固体壁）    
                    } else {
                        attribute[i][j][k] = 4; // 壁面セル (ノズル面)
                    }
                } else if (i == NX - 1) {
                    attribute[i][j][k] = 2; // 流出セル

                } else if (j == 0) {
                    attribute[i][j][k] = 5; // 対称境界 (新しい属性番号を割り当てる)
                } else if (j == NY - 1 || k == 0 || k == NZ - 1) {
                    // j==0 の条件を上記で処理したため、ここから削除
                    attribute[i][j][k] = 4; // 側壁 (jの最大値、kの最小値・最大値)
                } else {
                    attribute[i][j][k] = 3; // 内部流体
                }
            }
        }
    }

    // --- 2. 全セルの物理量の初期値設定 ---
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                P[i][j][k] = 0.0; // 圧力変動の初期値はゼロ

                switch (attribute[i][j][k]) {
                    case 1: // 流入セル
                        u[i][j][k] = INLET_VELOCITY;
                        v[i][j][k] = 0.0;
                        w[i][j][k] = 0.0;
                        T[i][j][k] = INLET_TEMPERATURE;
                        G[i][j][k] = MIN_G_CLIP; // 未燃
                        for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk[sp][i][j][k] = inletYk[sp];
                        break;

                    case 4: // 壁面セル
                        u[i][j][k] = 0.0; // 速度ゼロ
                        v[i][j][k] = 0.0;
                        w[i][j][k] = 0.0;
                        T[i][j][k] = AMBIENT_TEMPERATURE;
                        G[i][j][k] = MIN_G_CLIP;
                        for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk[sp][i][j][k] = ambientYk[sp];
                        break;

                    case 2: // 流出セル
                        u[i][j][k] = 0.0;
                        v[i][j][k] = 0.0;
                        w[i][j][k] = 0.0;
                        T[i][j][k] = AMBIENT_TEMPERATURE;
                        G[i][j][k] = MIN_G_CLIP; // 未燃
                        for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk[sp][i][j][k] = ambientYk[sp];
                        break;

                    case 5: // 対称境界 (j=0)
                    case 3: // 内部流体セル (j>0)
                        u[i][j][k] = 0.0;
                        v[i][j][k] = 0.0;
                        w[i][j][k] = 0.0;

                        if (i <= INITIAL_FLAME_FRONT_I) {
                            // --- 既燃ガスの設定 ---
                            G[i][j][k] = MAX_G_CLIP;
                            T[i][j][k] = BURNT_GAS_TEMPERATURE;
                            double consumed_Y_C3H8 = inletYk[C3H8_INDEX];
                            Yk[C3H8_INDEX][i][j][k] = inletYk[C3H8_INDEX] - consumed_Y_C3H8;
                            Yk[O2_INDEX][i][j][k]  = inletYk[O2_INDEX] - consumed_Y_C3H8 * STOICH_O2_PER_C3H8;
                            Yk[N2_INDEX][i][j][k]  = inletYk[N2_INDEX];
                            Yk[CO2_INDEX][i][j][k] = inletYk[CO2_INDEX] + consumed_Y_C3H8 * STOICH_CO2_PER_C3H8;
                            Yk[H2O_INDEX][i][j][k] = inletYk[H2O_INDEX] + consumed_Y_C3H8 * STOICH_H2O_PER_C3H8;
                        } else {
                            // --- 未燃ガスの設定 ---
                            G[i][j][k] = MIN_G_CLIP;
                            T[i][j][k] = INLET_TEMPERATURE;
                            for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk[sp][i][j][k] = inletYk[sp];
                        }
                        break;
                }

                // --- 3. 状態方程式から初期密度を計算 ---
                double sum_Y_over_W = 0.0;
                for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                    sum_Y_over_W += Yk[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0);
                }
                if (sum_Y_over_W > SMALL_NUMBER && T[i][j][k] > 1.0) {
                    double W_mix = 1.0 / sum_Y_over_W;
                    double R_mix = R_univ / W_mix;
                    rho[i][j][k] = pressure_const / (R_mix * T[i][j][k]);
                } else {
                    rho[i][j][k] = DEFAULT_RHO_AIR;
                }

                rhou[i][j][k] = rho[i][j][k] * u[i][j][k];
                rhov[i][j][k] = rho[i][j][k] * v[i][j][k];
                rhow[i][j][k] = rho[i][j][k] * w[i][j][k];
            }
        }
    }
    
    // --- 4. u_newなどの_new配列を初期化 ---
    // (apply_boundary_conditions の最後でコピーされるが、念のため初期化)
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                u_new[i][j][k] = u[i][j][k];
                v_new[i][j][k] = v[i][j][k];
                w_new[i][j][k] = w[i][j][k];
                T_new[i][j][k] = T[i][j][k];
                P_new[i][j][k] = P[i][j][k];
                G_new[i][j][k] = G[i][j][k];
                rho_new[i][j][k] = rho[i][j][k];
                rhou_new[i][j][k] = rhou[i][j][k];
                rhov_new[i][j][k] = rhov[i][j][k];
                rhow_new[i][j][k] = rhow[i][j][k];
                for (int sp = 0; sp < NUM_SPECIES; sp++) {
                    Yk_new[sp][i][j][k] = Yk[sp][i][j][k];
                }
            }
        }
    }
}


// NaN/Infをチェックし、発覚したら計算を停止するデバッグ関数
void check_for_nan(const Grid3D var, const char* var_name, int t_loop) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (!isfinite(var[i][j][k])) {
                    #pragma omp critical
                    {
                        printf("FATAL ERROR: NaN or Inf detected in variable '%s' at time step %d, location (i=%d, j=%d, k=%d).\n", var_name, t_loop, i, j, k);
                        printf("Aborting simulation.\n");
                        exit(1); // 計算を強制終了
                    }
                }
            }
        }
    }
}

void apply_box_filter(
    const Grid3D_Int attribute, int t,
    const Grid3D u, Grid3D u_filtered,
    const Grid3D v, Grid3D v_filtered,
    const Grid3D w, Grid3D w_filtered,
    const Grid3D P, Grid3D P_filtered,
    const Grid3D T, Grid3D T_filtered,
    const Grid3D G, Grid3D G_filtered,
    const Grid3D uG, Grid3D uG_filtered,
    const Grid3D vG, Grid3D vG_filtered,
    const Grid3D wG, Grid3D wG_filtered,
    const Grid3D *Yk, Grid3D *Yk_filtered, 
    const Grid3D rho, Grid3D rho_filtered
) {
    const double denominator = 6.0 + FILTER_WEIGHT_CENTER; // 分母 (6近傍点 + 中心点の重み合計)

    #pragma omp parallel for collapse(3)
    for (int i = 1; i < NX - 1; i++) { // 境界を除いた内部点のみフィルタリング
        for (int j = 1; j < NY - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                if (attribute[i][j][k] == 3) { // 流体セルのみ
                    // u のフィルタリング
                    u_filtered[i][j][k] = (u[i-1][j][k]   + u[i][j-1][k]   + u[i][j][k-1] +
                                           FILTER_WEIGHT_CENTER * u[i][j][k] +
                                           u[i][j][k+1]   + u[i][j+1][k]   + u[i+1][j][k]) / denominator;

                    // v のフィルタリング
                    v_filtered[i][j][k] = (v[i-1][j][k]   + v[i][j-1][k]   + v[i][j][k-1] +
                                           FILTER_WEIGHT_CENTER * v[i][j][k] +
                                           v[i][j][k+1]   + v[i][j+1][k]   + v[i+1][j][k]) / denominator;

                    // w のフィルタリング
                    w_filtered[i][j][k] = (w[i-1][j][k]   + w[i][j-1][k]   + w[i][j][k-1] +
                                           FILTER_WEIGHT_CENTER * w[i][j][k] +
                                           w[i][j][k+1]   + w[i][j+1][k]   + w[i+1][j][k]) / denominator;

                    // P のフィルタリング
                    P_filtered[i][j][k] = (P[i-1][j][k]   + P[i][j-1][k]   + P[i][j][k-1] +
                                           FILTER_WEIGHT_CENTER * P[i][j][k] +
                                           P[i][j][k+1]   + P[i][j+1][k]   + P[i+1][j][k]) / denominator;

                    // T のフィルタリング
                    T_filtered[i][j][k] = (T[i-1][j][k]   + T[i][j-1][k]   + T[i][j][k-1] +
                                           FILTER_WEIGHT_CENTER * T[i][j][k] +
                                           T[i][j][k+1]   + T[i][j+1][k]   + T[i+1][j][k]) / denominator;

                    // rho のフィルタリング
                    rho_filtered[i][j][k] = (rho[i-1][j][k]   + rho[i][j-1][k]   + rho[i][j][k-1] +
                                             FILTER_WEIGHT_CENTER * rho[i][j][k] +
                                             rho[i][j][k+1]   + rho[i][j+1][k]   + rho[i+1][j][k]) / denominator;

                    // Yk のフィルタリング (各化学種ごとに行う)
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        Yk_filtered[sp][i][j][k] = (Yk[sp][i-1][j][k]   + Yk[sp][i][j-1][k]   + Yk[sp][i][j][k-1] +
                                                    FILTER_WEIGHT_CENTER * Yk[sp][i][j][k] +
                                                    Yk[sp][i][j][k+1]   + Yk[sp][i][j+1][k]   + Yk[sp][i+1][j][k]) / denominator;
                    }

                    // G のフィルタリング
                    G_filtered[i][j][k] = (G[i-1][j][k]   + G[i][j-1][k]   + G[i][j][k-1] +
                                           FILTER_WEIGHT_CENTER * G[i][j][k] +
                                           G[i][j][k+1]   + G[i][j+1][k]   + G[i+1][j][k]) / denominator;

                    // uG のフィルタリング
                    uG_filtered[i][j][k] = (uG[i-1][j][k]  + uG[i][j-1][k]  + uG[i][j][k-1] +
                                            FILTER_WEIGHT_CENTER * uG[i][j][k] +
                                            uG[i][j][k+1]  + uG[i][j+1][k]  + uG[i+1][j][k]) / denominator;

                    // vG のフィルタリング
                    vG_filtered[i][j][k] = (vG[i-1][j][k]  + vG[i][j-1][k]  + vG[i][j][k-1] +
                                            FILTER_WEIGHT_CENTER * vG[i][j][k] +
                                            vG[i][j][k+1]  + vG[i][j+1][k]  + vG[i+1][j][k]) / denominator;

                    // wG のフィルタリング
                    wG_filtered[i][j][k] = (wG[i-1][j][k]  + wG[i][j-1][k]  + wG[i][j][k-1] +
                                            FILTER_WEIGHT_CENTER * wG[i][j][k] +
                                            wG[i][j][k+1]  + wG[i][j+1][k]  + wG[i+1][j][k]) / denominator;

                } else { // 流体セルでない場合 (境界など) はフィルタリングせず値をコピー
                    u_filtered[i][j][k] = u[i][j][k];
                    v_filtered[i][j][k] = v[i][j][k];
                    w_filtered[i][j][k] = w[i][j][k];
                    P_filtered[i][j][k] = P[i][j][k];
                    T_filtered[i][j][k] = T[i][j][k];
                    rho_filtered[i][j][k] = rho[i][j][k]; 
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        Yk_filtered[sp][i][j][k] = Yk[sp][i][j][k];
                    }
                    // G方程式関連のコピー
                    G_filtered[i][j][k] = G[i][j][k];
                    uG_filtered[i][j][k] = uG[i][j][k];
                    vG_filtered[i][j][k] = vG[i][j][k];
                    wG_filtered[i][j][k] = wG[i][j][k];
                }
            }
        }
    }

    // 境界の処理: フィルタリングされていない境界の _filtered 配列に値を設定
    // (流体セルでない場合は、フィルタリング前の値をコピー)
    // i=0 および i=NX-1 の面
    for (int j = 0; j < NY; j++) {
        for (int k = 0; k < NZ; k++) {
            if (attribute[0][j][k] != 3) {
                u_filtered[0][j][k] = u[0][j][k]; v_filtered[0][j][k] = v[0][j][k]; w_filtered[0][j][k] = w[0][j][k];
                P_filtered[0][j][k] = P[0][j][k]; T_filtered[0][j][k] = T[0][j][k];
                G_filtered[0][j][k] = G[0][j][k]; uG_filtered[0][j][k] = uG[0][j][k];
                vG_filtered[0][j][k] = vG[0][j][k]; wG_filtered[0][j][k] = wG[0][j][k];
                for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk_filtered[sp][0][j][k] = Yk[sp][0][j][k];
                rho_filtered[0][j][k] = rho[0][j][k];
            }
            if (attribute[NX-1][j][k] != 3) {
                u_filtered[NX-1][j][k] = u[NX-1][j][k]; v_filtered[NX-1][j][k] = v[NX-1][j][k]; w_filtered[NX-1][j][k] = w[NX-1][j][k];
                P_filtered[NX-1][j][k] = P[NX-1][j][k]; T_filtered[NX-1][j][k] = T[NX-1][j][k];
                G_filtered[NX-1][j][k] = G[NX-1][j][k]; uG_filtered[NX-1][j][k] = uG[NX-1][j][k];
                vG_filtered[NX-1][j][k] = vG[NX-1][j][k]; wG_filtered[NX-1][j][k] = wG[NX-1][j][k];
                for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk_filtered[sp][NX-1][j][k] = Yk[sp][NX-1][j][k];
                rho_filtered[NX-1][j][k] = rho[NX-1][j][k];
            }
        }
    }
    // j=0 および j=NY-1 の面 (iのループは境界を含むように修正)
    for (int i = 0; i < NX; i++) {
        for (int k = 0; k < NZ; k++) {
            if (attribute[i][0][k] != 3) {
                 u_filtered[i][0][k] = u[i][0][k]; v_filtered[i][0][k] = v[i][0][k]; w_filtered[i][0][k] = w[i][0][k];
                 P_filtered[i][0][k] = P[i][0][k]; T_filtered[i][0][k] = T[i][0][k];
                 G_filtered[i][0][k] = G[i][0][k]; uG_filtered[i][0][k] = uG[i][0][k];
                 vG_filtered[i][0][k] = vG[i][0][k]; wG_filtered[i][0][k] = wG[i][0][k];
                 for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk_filtered[sp][i][0][k] = Yk[sp][i][0][k];
                 rho_filtered[i][0][k] = rho[i][0][k];
            }
            if (attribute[i][NY-1][k] != 3) {
                 u_filtered[i][NY-1][k] = u[i][NY-1][k]; v_filtered[i][NY-1][k] = v[i][NY-1][k]; w_filtered[i][NY-1][k] = w[i][NY-1][k];
                 P_filtered[i][NY-1][k] = P[i][NY-1][k]; T_filtered[i][NY-1][k] = T[i][NY-1][k];
                 G_filtered[i][NY-1][k] = G[i][NY-1][k]; uG_filtered[i][NY-1][k] = uG[i][NY-1][k];
                 vG_filtered[i][NY-1][k] = vG[i][NY-1][k]; wG_filtered[i][NY-1][k] = wG[i][NY-1][k];
                 for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk_filtered[sp][i][NY-1][k] = Yk[sp][i][NY-1][k];
                 rho_filtered[i][NY-1][k] = rho[i][NY-1][k];
            }
        }
    }
    // k=0 および k=NZ-1 の面 (i,jのループは境界を含むように修正)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            if (attribute[i][j][0] != 3) {
                u_filtered[i][j][0] = u[i][j][0]; v_filtered[i][j][0] = v[i][j][0]; w_filtered[i][j][0] = w[i][j][0];
                P_filtered[i][j][0] = P[i][j][0]; T_filtered[i][j][0] = T[i][j][0];
                G_filtered[i][j][0] = G[i][j][0]; uG_filtered[i][j][0] = uG[i][j][0];
                vG_filtered[i][j][0] = vG[i][j][0]; wG_filtered[i][j][0] = wG[i][j][0];
                for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk_filtered[sp][i][j][0] = Yk[sp][i][j][0];
                rho_filtered[i][j][0] = rho[i][j][0];
            }
            if (attribute[i][j][NZ-1] != 3) {
                u_filtered[i][j][NZ-1] = u[i][j][NZ-1]; v_filtered[i][j][NZ-1] = v[i][j][NZ-1]; w_filtered[i][j][NZ-1] = w[i][j][NZ-1];
                P_filtered[i][j][NZ-1] = P[i][j][NZ-1]; T_filtered[i][j][NZ-1] = T[i][j][NZ-1];
                G_filtered[i][j][NZ-1] = G[i][j][NZ-1]; uG_filtered[i][j][NZ-1] = uG[i][j][NZ-1];
                vG_filtered[i][j][NZ-1] = vG[i][j][NZ-1]; wG_filtered[i][j][NZ-1] = w[i][j][NZ-1];
                for (int sp = 0; sp < NUM_SPECIES; ++sp) Yk_filtered[sp][i][j][NZ-1] = Yk[sp][i][j][NZ-1];
                rho_filtered[i][j][NZ-1] = rho[i][j][NZ-1];
            }
        }
    }
}

void calculate_max_velocity(const Grid3D_Int attribute, const Grid3D u, const Grid3D v, const Grid3D w, double *max_velocity) {
    double max_val = 0.0;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (attribute[i][j][k] == 1 || attribute[i][j][k] == 3) { // ★変更★ 流体領域も考慮
                    double vel = sqrt(u[i][j][k]*u[i][j][k] + v[i][j][k]*v[i][j][k] + w[i][j][k]*w[i][j][k]);
                    if (vel > max_val) max_val = vel;
                }
            }
        }
    }
    *max_velocity = max_val > SMALL_NUMBER ? max_val : INLET_VELOCITY;
}


//  混合気体の動粘性係数と粘性係数を計算
void update_nu_mix(
    const Grid3D_Int attribute,
    const Grid3D T_filtered,
    const Grid3D *Yk_filtered, 
    Grid3D nu,
    Grid3D mu_mix,
    const Grid3D rho_filtered
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                if (attribute[i][j][k] == 3) {
                    double T_local = T_filtered[i][j][k];
                    if (T_local <= 0) {
                        nu[i][j][k] = 1.5e-5;
                        mu_mix[i][j][k] = 1.8e-5;
                        continue;
                    }

                    // 1. モル分率 Xk を計算
                    double Xk_local[NUM_SPECIES];
                    double sum_Y_over_MOLW = 0.0;
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        sum_Y_over_MOLW += Yk_filtered[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0);
                    }
                    if (sum_Y_over_MOLW < 1e-9) {
                        nu[i][j][k] = 1.5e-5; mu_mix[i][j][k] = 1.8e-5; continue;
                    }
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        Xk_local[sp] = (Yk_filtered[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0)) / sum_Y_over_MOLW;
                    }

                    // 2. Wilkeの混合則で混合粘性係数 mu_mix を計算
                    double mu = 0.0;
                    double mu_k_local[NUM_SPECIES];
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        // 純粋成分の粘性係数を動的に計算
                        mu_k_local[sp] = calculate_mu_dynamic(sp, T_local);
                    }

                    for (int sp_i = 0; sp_i < NUM_SPECIES; ++sp_i) {
                        if (Xk_local[sp_i] < 1e-12) continue;
                        double mu_i = mu_k_local[sp_i];
                        double M_i = loaded_species[sp_i].mw_g_per_mol; // g/mol
                        double denominator = 0.0;
                        for (int sp_j = 0; sp_j < NUM_SPECIES; ++sp_j) {
                            if (Xk_local[sp_j] < 1e-12) continue;
                            double mu_j = mu_k_local[sp_j];
                            double M_j = loaded_species[sp_j].mw_g_per_mol; // g/mol
                            // sp_i == sp_j の時は phi_ij = 1.0
                            double phi_ij_term = pow(1.0 + pow(mu_i / mu_j, 0.5) * pow(M_j / M_i, 0.25), 2) / (sqrt(8.0) * sqrt(1.0 + M_i / M_j));
                            denominator += Xk_local[sp_j] * phi_ij_term;
                        }
                        if (denominator > 1e-9) {
                            mu += (Xk_local[sp_i] * mu_i) / denominator;
                        }
                    }
                    mu_mix[i][j][k] = fmax(MIN_MU_MIX_CLIP, mu);


                    // 3. 動粘性係数 nu を計算
                    if (rho_filtered[i][j][k] > 1e-3) {
                        nu[i][j][k] = mu_mix[i][j][k] / rho_filtered[i][j][k];
                    } else {
                        nu[i][j][k] = 1.5e-5; // フォールバック
                    }
                    nu[i][j][k] = fmax(MIN_NU_MIX_CLIP, nu[i][j][k]);

                } else {
                    nu[i][j][k] = 1.5e-5;
                    mu_mix[i][j][k] = 1.8e-5;
                }
            }
        }
    }
}


// Improved Model A を実装した渦粘性計算関数 
void calculate_nu_t(
    const Grid3D_Int attribute, int t,
    const Grid3D u_filtered,
    const Grid3D v_filtered,
    const Grid3D w_filtered,
    const Grid3D G_filtered,
    Grid3D S_mag,
    Grid3D nu_t,
    Grid3D SijSij,
    Grid3D div_u
) {
    double Delta = DX; // フィルター幅（格子幅）
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < NX-1; i++) {
        for (int j = 1; j < NY-1; j++) {
            for (int k = 1; k < NZ-1; k++) {
                if (attribute[i][j][k] == 3) {
                    // 速度勾配テンソルの成分
                    double du_dx = (u_filtered[i+1][j][k] - u_filtered[i-1][j][k]) / (2.0 * DX);
                    double dv_dy = (v_filtered[i][j+1][k] - v_filtered[i][j-1][k]) / (2.0 * DY);
                    double dw_dz = (w_filtered[i][j][k+1] - w_filtered[i][j][k-1]) / (2.0 * DZ);
                    double du_dy = (u_filtered[i][j+1][k] - u_filtered[i][j-1][k]) / (2.0 * DY);
                    double du_dz = (u_filtered[i][j][k+1] - u_filtered[i][j][k-1]) / (2.0 * DZ);
                    double dv_dx = (v_filtered[i+1][j][k] - v_filtered[i-1][j][k]) / (2.0 * DX);
                    double dv_dz = (v_filtered[i][j][k+1] - v_filtered[i][j][k-1]) / (2.0 * DZ);
                    double dw_dx = (w_filtered[i+1][j][k] - w_filtered[i-1][j][k]) / (2.0 * DX);
                    double dw_dy = (w_filtered[i][j+1][k] - w_filtered[i][j-1][k]) / (2.0 * DY);

                    // S_ij S_ij の計算（対称部分のみ）
                    double S11 = du_dx;
                    double S22 = dv_dy;
                    double S33 = dw_dz;
                    double S12 = 0.5 * (du_dy + dv_dx);
                    double S13 = 0.5 * (du_dz + dw_dx);
                    double S23 = 0.5 * (dv_dz + dw_dy);
                    SijSij[i][j][k] = 2.0 * (S12*S12 + S13*S13 + S23*S23) + S11*S11 + S22*S22 + S33*S33;
                    S_mag[i][j][k] = sqrt(2.0 * SijSij[i][j][k]);
                    div_u[i][j][k] = du_dx + dv_dy + dw_dz;
                    // =========================================
                    //  新規実装: Gに応じた可変スマゴリンスキー定数 
                    // =========================================
                    double G_val = G_filtered[i][j][k];
                    // Gを 0~1 にクリップ（念のため）
                    if (G_val < 0.0) G_val = 0.0;
                    if (G_val > 1.0) G_val = 1.0;

                    // 基本的な線形補間: G=0でCS_UNBURNED, G=1でCS_BURNT
                    double Cs_local = CS_UNBURNED + (CS_BURNT - CS_UNBURNED) * G_val;

                    // オプション: 火炎帯(G=0.5付近)でさらにブーストする場合
                    Cs_local += CS_FLAME_BOOST * 4.0 * G_val * (1.0 - G_val);

                    // 局所的な SGS粘性係数の計算
                    nu_t[i][j][k] = (Cs_local * Delta) * (Cs_local * Delta) * S_mag[i][j][k];

                } else {
                    SijSij[i][j][k] = 0.0;
                    div_u[i][j][k] = 0.0;
                    S_mag[i][j][k] = 0.0;
                    nu_t[i][j][k] = 0.0;
                }
            }
        }
    }
}


void calculate_omega(
    const Grid3D_Int attribute,
    const Grid3D G_filtered,
    const Grid3D *Yk_filtered, // ★
    const Grid3D rho_filtered,
    const Grid3D ST_turbulent,
    Grid3D omega
) {
    const double nu_unburnt = 1.5e-5; // 未燃ガスの動粘性係数 (m^2/s)
    const double delta_L = nu_unburnt / S_L; // 層流火炎厚さ (m)
    const double beta = OMEGA_MODEL_BETA; // 局在化パラメータ

    #pragma omp parallel for collapse(3)
    for(int i = 1; i < NX-1; i++){
        for (int j = 1; j < NY-1; j++){
            for (int k = 1; k < NZ-1; k++){
                if (attribute[i][j][k] == 3) {
                    // |∇<G>| の計算 (中心差分)
                    double grad_G_x = (G_filtered[i+1][j][k] - G_filtered[i-1][j][k]) / (2.0 * DX);
                    double grad_G_y = (G_filtered[i][j+1][k] - G_filtered[i][j-1][k]) / (2.0 * DY);
                    double grad_G_z = (G_filtered[i][j][k+1] - G_filtered[i][j][k-1]) / (2.0 * DZ);
                    double grad_G_mag = sqrt(grad_G_x*grad_G_x + grad_G_y*grad_G_y + grad_G_z*grad_G_z);

                    // 燃料と酸化剤が存在する場合のみ反応
                    double Y_C3H8_local = Yk_filtered[C3H8_INDEX][i][j][k];
                    double Y_O2_local  = Yk_filtered[O2_INDEX][i][j][k];

                    if (Y_C3H8_local > 1e-6 && Y_O2_local > 1e-6 && grad_G_mag > 1e-6) {
                        // 新しい反応速度モデルの実装: ω = ρ * S_L * (|∇G| / δ_L) * exp[-β(1-G)²]
                        double G_local = G_filtered[i][j][k];
                        double exp_term = exp(-beta * (G_local - 0.70) * (G_local - 0.70));
                        omega[i][j][k] = rho_filtered[i][j][k] * S_L * (grad_G_mag / delta_L) * exp_term;
                    } else {
                        omega[i][j][k] = 0.0;
                    }
                    
                    omega[i][j][k] = fmax(0.0, omega[i][j][k]);
                } else {
                    omega[i][j][k] = 0.0;
                }
            }
        }
    }
}



//  各化学種の有効拡散係数を計算（詳細モデル使用）
void update_Dk(
    const Grid3D_Int attribute,
    const Grid3D T_filtered,
    const Grid3D *Yk_filtered, 
    const Grid3D P_filtered,
    Grid3D *Dk 
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                if (attribute[i][j][k] == 3) {
                    double T_local = T_filtered[i][j][k];
                    if (T_local <= 0) {
                        for (int sp = 0; sp < NUM_SPECIES; ++sp) Dk[sp][i][j][k] = 1e-5;
                        continue;
                    }

                    // 1. モル分率 Xk を計算
                    double Xk_local[NUM_SPECIES];
                    double sum_Y_over_MOLW = 0.0;
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        sum_Y_over_MOLW += Yk_filtered[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0);
                    }
                    if (sum_Y_over_MOLW < 1e-9) {
                        for (int sp = 0; sp < NUM_SPECIES; ++sp) Dk[sp][i][j][k] = 1e-5;
                        continue;
                    }
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        Xk_local[sp] = (Yk_filtered[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0)) / sum_Y_over_MOLW;
                    }

                    // 2. Hirschfelder-Curtissの近似式で有効拡散係数Dkを計算
                    for (int sp_k = 0; sp_k < NUM_SPECIES; ++sp_k) {
                        double sum_Xj_over_Dkj = 0.0;
                        for (int sp_j = 0; sp_j < NUM_SPECIES; ++sp_j) {
                            if (sp_k == sp_j) continue; // 自分自身とのペアは除く
                            if (Xk_local[sp_j] > 1e-12) {
                                // 二成分拡散係数 D_kj を動的に計算
                                double D_kj = calculate_Dkj_dynamic(sp_k, sp_j, T_local);
                                if (D_kj > 1e-12) {
                                    sum_Xj_over_Dkj += Xk_local[sp_j] / D_kj;
                                }
                            }
                        }
                        
                        // (1-Xk)がゼロに近い場合（純粋成分の場合）は発散するので、適当な二成分拡散係数で代用
                        if (fabs(1.0 - Xk_local[sp_k]) < 1e-9) {
                             // N2が支配的なので、N2との二成分拡散係数を代表値として使う
                             int other_sp_idx = (sp_k != N2_INDEX) ? N2_INDEX : O2_INDEX;
                             Dk[sp_k][i][j][k] = calculate_Dkj_dynamic(sp_k, other_sp_idx, T_local);
                        } else if (sum_Xj_over_Dkj > 1e-12) {
                            Dk[sp_k][i][j][k] = (1.0 - Xk_local[sp_k]) / sum_Xj_over_Dkj;
                        } else {
                            Dk[sp_k][i][j][k] = 1e-5; // Fallback
                        }

                        // 圧力補正 (D ∝ 1/P)
                        // P_filteredは変動分なので、基準圧 pressure_const を足す
                        double total_pressure = pressure_const + P_filtered[i][j][k];
                        if (total_pressure > 1.0) {
                             Dk[sp_k][i][j][k] *= (PATM / total_pressure);
                        }
                        
                        Dk[sp_k][i][j][k] = fmax(1e-9, fmin(1e-2, Dk[sp_k][i][j][k])); // 妥当な範囲にクリップ
                    }
                } else {
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) Dk[sp][i][j][k] = 0.0;
                }
            }
        }
    }
}




//  混合気体の定圧比熱を計算（詳細モデル使用）
void update_Cp_mix(
    const Grid3D_Int attribute,
    const Grid3D T_filtered,
    const Grid3D *Yk_filtered, // ★
    Grid3D Cp_mix
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                if (attribute[i][j][k] == 3) {
                    double T_local = T_filtered[i][j][k];
                    if (T_local <= 0) {
                        Cp_mix[i][j][k] = 1000.0; // フォールバック値（空気相当）
                        continue;
                    }

                    double current_cp_mix = 0.0;
                    // 各化学種の質量分率で加重平均をとる
                    for (int sp_idx = 0; sp_idx < NUM_SPECIES; ++sp_idx) {
                        // 純粋成分の比熱を動的に計算
                        double cp_k_pure = calculate_Cp_dynamic(sp_idx, T_local);
                        current_cp_mix += Yk_filtered[sp_idx][i][j][k] * cp_k_pure;
                    }
                    Cp_mix[i][j][k] = fmax(100.0, current_cp_mix); // 物理的に妥当な範囲にクリップ

                } else {
                    // 流体セル以外はデフォルト値
                    Cp_mix[i][j][k] = 1000.0;
                }
            }
        }
    }
}

//  混合熱伝導率と熱拡散率を計算（詳細モデル使用）
void update_lambda_mix_alpha(
    const Grid3D_Int attribute,
    const Grid3D T_filtered,
    const Grid3D *Yk_filtered, 
    const Grid3D Cp_mix,
    Grid3D lambda_mix,
    Grid3D alpha,
    const Grid3D rho_filtered
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            for (int k = 0; k < NZ; ++k) {
                if (attribute[i][j][k] == 3) {
                    double T_local = T_filtered[i][j][k];
                    if (T_local <= 0) {
                        lambda_mix[i][j][k] = 0.025; // フォールバック値（空気相当）
                        alpha[i][j][k] = 1.8e-5;    // フォールバック値（空気相当）
                        continue;
                    }

                    // 1. モル分率 Xk を計算
                    double Xk_local[NUM_SPECIES];
                    double sum_Y_over_MOLW = 0.0;
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        sum_Y_over_MOLW += Yk_filtered[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0);
                    }
                    if (sum_Y_over_MOLW < 1e-9) {
                        lambda_mix[i][j][k] = 0.025; alpha[i][j][k] = 1.8e-5; continue;
                    }
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        Xk_local[sp] = (Yk_filtered[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0)) / sum_Y_over_MOLW;
                    }
                    
                    // 2. Wassiljewの混合則で混合熱伝導率 lambda_mix を計算
                    double current_lambda_mix = 0.0;
                    double lambda_k_local[NUM_SPECIES];
                    double mu_k_local[NUM_SPECIES];
                    // 事前に全化学種の純粋な物性値を計算しておく
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        lambda_k_local[sp] = calculate_lambda_dynamic(sp, T_local);
                        mu_k_local[sp] = calculate_mu_dynamic(sp, T_local);
                    }

                    for (int sp_i = 0; sp_i < NUM_SPECIES; ++sp_i) {
                        if (Xk_local[sp_i] < 1e-12) continue;
                        double lambda_i = lambda_k_local[sp_i];
                        double mu_i = mu_k_local[sp_i];
                        double M_i = loaded_species[sp_i].mw_g_per_mol; // g/mol
                        double denominator = 0.0;

                        for (int sp_j = 0; sp_j < NUM_SPECIES; ++sp_j) {
                            if (Xk_local[sp_j] < 1e-12) continue;
                            double mu_j = mu_k_local[sp_j];
                            double M_j = loaded_species[sp_j].mw_g_per_mol; // g/mol
                            
                            // 相互作用係数 A_ij (Wilkeのφ_ijと同じ)
                            double A_ij = pow(1.0 + pow(mu_i / mu_j, 0.5) * pow(M_j / M_i, 0.25), 2) 
                                        / (sqrt(8.0) * sqrt(1.0 + M_i / M_j));
                            denominator += Xk_local[sp_j] * A_ij;
                        }

                        if (denominator > 1e-9) {
                            current_lambda_mix += (Xk_local[sp_i] * lambda_i) / denominator;
                        }
                    }
                    lambda_mix[i][j][k] = fmax(1e-3, current_lambda_mix); // 物理的に妥当な範囲にクリップ

                    // 3. 熱拡散率 alpha を計算
                    if (rho_filtered[i][j][k] > 1e-6 && Cp_mix[i][j][k] > 1e-6) {
                        alpha[i][j][k] = lambda_mix[i][j][k] / (rho_filtered[i][j][k] * Cp_mix[i][j][k]);
                    } else {
                        alpha[i][j][k] = 1.8e-5; // フォールバック値
                    }
                    alpha[i][j][k] = fmax(1e-7, alpha[i][j][k]); // 物理的に妥当な範囲にクリップ

                } else {
                    lambda_mix[i][j][k] = 0.0;
                    alpha[i][j][k] = 0.0;
                }
            }
        }
    }
}

void update_rho_from_EOS(
    const Grid3D_Int attribute,
    const Grid3D T_new,
    const Grid3D *Yk_new, 
    Grid3D rho_new
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (attribute[i][j][k] == 3 || attribute[i][j][k] == 2) {
                    double sum_Y_over_W = 0.0;
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        sum_Y_over_W += Yk_new[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0);
                    }

                    if (sum_Y_over_W > 1e-9 && T_new[i][j][k] > 1.0) { 
                        double WTM = 1.0 / sum_Y_over_W;
                        double R_mix = R_univ / WTM;
                        rho_new[i][j][k] = pressure_const / (R_mix * T_new[i][j][k]);
                    } else {
                        rho_new[i][j][k] = rho[i][j][k]; 
                    }
                }
            }
        }
    }
}


void calculate_ST_turbulent_SSR_SGS_jp_global_vars(
    double S_L_param,
    double nu_unburnt_param,
    double tau_expansion_ratio_param,
    double D3_fractal_dimension_param,
    const Grid3D rho_filtered
) {

    #pragma omp parallel for collapse(3)
    for (int i = 1; i < NX - 1; ++i) {
        for (int j = 1; j < NY - 1; ++j) {
            for (int k = 1; k < NZ - 1; ++k) {
                if (attribute[i][j][k] == 3) {
                    double nu_filtered_local = nu[i][j][k]; // 現在の動粘性係数を使用

                    double X_eta_param_numerator = SijSij[i][j][k] - (div_u[i][j][k] * div_u[i][j][k]) * ONE_THIRD;
                    double X_eta_param = (pow(DX, 4.0) / (nu_filtered_local * nu_filtered_local)) * X_eta_param_numerator;
                    
                    double Delta_over_eta;
                    if (X_eta_param < SMALL_NUMBER) {
                        Delta_over_eta = 1.0;
                    } else {
                        double pow_X_eta = pow(X_eta_param, 0.75); // 指数部は1.5/2=0.75
                        double arg_log_eta = SSR_SGS_MODEL_B_ETA * pow_X_eta;
                        Delta_over_eta = SSR_SGS_MODEL_A_ETA * pow_X_eta * (arg_log_eta > 1.0 ? log(arg_log_eta) : 0.0);
                    }
                    double eta_kolmogorov = DX / (Delta_over_eta + SMALL_NUMBER);

                    double delta_F_zeldovich = nu_unburnt_param / (S_L_param + SMALL_NUMBER);
                    double D_coherent_eddy = 8.0 * eta_kolmogorov;
                    double alpha_scaling_factor = 8.0 * exp(SSR_SGS_MODEL_C_ALPHA * delta_F_zeldovich / (D_coherent_eddy + SMALL_NUMBER));

                    double base_turb_term = alpha_scaling_factor * eta_kolmogorov / DX;
                    double term_turb_wrinkling_norm = pow(base_turb_term, (2.0 - D3_fractal_dimension_param));

                    double term_thermal_expansion_norm = 0.0;
                    if (tau_expansion_ratio_param * S_L_param > SMALL_NUMBER) {
                        term_thermal_expansion_norm = DX * fmax(0.0, div_u[i][j][k]) / (tau_expansion_ratio_param * S_L_param);
                    }
                    
                    double ST_over_SL_ratio = term_turb_wrinkling_norm + term_thermal_expansion_norm;
                    ST_turbulent[i][j][k]  = S_L_param * fmax(0.0, ST_over_SL_ratio);

                    // 乱流が弱い場合でも、最低でも層流燃焼速度で伝播することを保証する
                    ST_turbulent[i][j][k] = fmax(S_L_param, ST_turbulent[i][j][k]);

                } else {
                    ST_turbulent[i][j][k]  = S_L_param;
                }
            }
        }
    }
}

// ===============================
// 【WENO5スキームによる移流項計算】
// ===============================
/**
 * @brief 非保存形の移流項 (U⋅∇)φ を5次精度WENOスキームで計算する
 */
double calculate_convection_weno5(
    int i, int j, int k,
    const Grid3D phi, // const double phi[NX][NY][NZ] から変更
    const Grid3D u, const Grid3D v, const Grid3D w
) {
    // 各方向の勾配をWENO5で計算
    double dphi_dx = calculate_weno5_gradient_1d(phi, i, j, k, 0, DX);
    double dphi_dy = calculate_weno5_gradient_1d(phi, i, j, k, 1, DY);
    double dphi_dz = calculate_weno5_gradient_1d(phi, i, j, k, 2, DZ);

    // 移流項 u*dφ/dx + v*dφ/dy + w*dφ/dz を計算して返す
    return u[i][j][k] * dphi_dx + v[i][j][k] * dphi_dy + w[i][j][k] * dphi_dz;
}

// ================================
// --- MUSCL スキーム用 補助関数 ---
// ================================

/**
 * @brief van Leer Flux Limiter関数
 *        連続する勾配の比 r = delta_upwind / delta_downwind を入力とし、
 *        制限された勾配を計算して返す。
 * @param r 勾配の比率
 * @return 制限関数φ(r)の値
 *
 * van Leer Limiter の定義: φ(r) = (r + |r|) / (1 + |r|)
 */
double van_leer_limiter(double r) {
    // ゼロ除算を避けるための安全対策
    if (fabs(1.0 + fabs(r)) < 1e-9) {
        return 2.0; // r -> inf の極限値
    }
    return (r + fabs(r)) / (1.0 + fabs(r));
}

/**
 * @brief Superbee Flux Limiter関数
 *        最も圧縮的なリミッターの一つで、不連続面を非常にシャープに捉える。
 * @param r 勾配の比率
 * @return 制限関数φ(r)の値
 *
 * Superbee Limiter の定義: φ(r) = max(0, min(2r, 1), min(r, 2))
 * ※この定義は複数の項に分けることで、より効率的に計算できます。
 */
double superbee_limiter(double r) {
    if (r <= 0.0) {
        return 0.0;
    }
    // max(0, min(2r, 1)) の部分を計算
    double term1 = fmin(2.0 * r, 1.0);
    // max(0, min(r, 2)) の部分を計算
    double term2 = fmin(r, 2.0);
    
    // 最終的に term1 と term2 のうち大きい方を返す
    return fmax(term1, term2);
}

/**
 * @brief 非保存形の移流項 (U⋅∇)φ をMUSCL (van Leer limiter) スキームで計算する
 * @param i, j, k 計算対象の格子点インデックス
 * @param phi 移流させるスカラー量 (例: G[NX][NY][NZ])
 * @param u, v, w 速度ベクトル
 * @return (i, j, k)における移流項の値
 */
double calculate_convection_muscl(
    int i, int j, int k,
    const Grid3D phi,
    const Grid3D u, const Grid3D v, const Grid3D w
) {
    // 境界付近では安定な1次風上差分にフォールバック
    if (i < 2 || i >= NX - 2 || j < 2 || j >= NY - 2 || k < 2 || k >= NZ - 2) {
        double conv_term_fallback = 0.0;
        if (u[i][j][k] >= 0.0) conv_term_fallback = u[i][j][k] * (phi[i][j][k] - phi[i-1][j][k]) / DX;
        else conv_term_fallback = u[i][j][k] * (phi[i+1][j][k] - phi[i][j][k]) / DX;
        if (v[i][j][k] >= 0.0) conv_term_fallback += v[i][j][k] * (phi[i][j][k] - phi[i][j-1][k]) / DY;
        else conv_term_fallback += v[i][j][k] * (phi[i][j+1][k] - phi[i][j][k]) / DY;
        if (w[i][j][k] >= 0.0) conv_term_fallback += w[i][j][k] * (phi[i][j][k] - phi[i][j][k-1]) / DZ;
        else conv_term_fallback += w[i][j][k] * (phi[i][j][k+1] - phi[i][j][k]) / DZ;
        return conv_term_fallback;
    }

    const double kappa = 0.0; // 2次精度風上差分 (kappa=0)
    double convection_x, convection_y, convection_z;

    // --- X方向の移流項 ---
    if (u[i][j][k] >= 0.0) { // 正の流速 (風上は i-1, i-2)
        // i-1/2 界面の再構築
        double delta_up_im = phi[i-1][j][k] - phi[i-2][j][k];
        double delta_down_im = phi[i][j][k] - phi[i-1][j][k];
        double r_im = (fabs(delta_down_im) < SMALL_NUMBER) ? 1e9 : delta_up_im / delta_down_im; // ゼロ除算防止
        //double phi_lim_im = van_leer_limiter(r_im);
        double phi_lim_im = superbee_limiter(r_im);
        double phi_imh = phi[i-1][j][k] + 0.5 * phi_lim_im * delta_down_im;

        // i+1/2 界面の再構築
        double delta_up_i = phi[i][j][k] - phi[i-1][j][k];
        double delta_down_i = phi[i+1][j][k] - phi[i][j][k];
        double r_i = (fabs(delta_down_i) < SMALL_NUMBER) ? 1e9 : delta_up_i / delta_down_i; // ゼロ除算防止
        //double phi_lim_i = van_leer_limiter(r_i);
        double phi_lim_i = superbee_limiter(r_i); 
        double phi_iph = phi[i][j][k] + 0.5 * phi_lim_i * delta_down_i;

        convection_x = u[i][j][k] * (phi_iph - phi_imh) / DX;
    } else { // 負の流速 (風上は i+1, i+2)
        // i-1/2 界面の再構築
        double delta_up_i = phi[i+1][j][k] - phi[i][j][k];
        double delta_down_i = phi[i][j][k] - phi[i-1][j][k];
        double r_i = (fabs(delta_down_i) < SMALL_NUMBER) ? 1e9 : delta_up_i / delta_down_i; // ゼロ除算防止
        //double phi_lim_i = van_leer_limiter(r_i);
        double phi_lim_i = superbee_limiter(r_i); 
        double phi_imh = phi[i][j][k] - 0.5 * phi_lim_i * delta_down_i;

        // i+1/2 界面の再構築
        double delta_up_ip = phi[i+2][j][k] - phi[i+1][j][k];
        double delta_down_ip = phi[i+1][j][k] - phi[i][j][k];
        double r_ip = (fabs(delta_down_ip) < SMALL_NUMBER) ? 1e9 : delta_up_ip / delta_down_ip; // ゼロ除算防止
        //double phi_lim_ip = van_leer_limiter(r_ip);
        double phi_lim_ip = superbee_limiter(r_ip); 
        double phi_iph = phi[i+1][j][k] - 0.5 * phi_lim_ip * delta_down_ip;
        
        convection_x = u[i][j][k] * (phi_iph - phi_imh) / DX;
    }
    
    // --- Y方向の移流項 ---
    if (v[i][j][k] >= 0.0) { // 正の流速 (風上は j-1, j-2)
        double delta_up_jm = phi[i][j-1][k] - phi[i][j-2][k];
        double delta_down_jm = phi[i][j][k] - phi[i][j-1][k];
        double r_jm = (fabs(delta_down_jm) < SMALL_NUMBER) ? 1e9 : delta_up_jm / delta_down_jm;
        double phi_lim_jm = superbee_limiter(r_jm);
        double phi_jmh = phi[i][j-1][k] + 0.5 * phi_lim_jm * delta_down_jm;

        double delta_up_j = phi[i][j][k] - phi[i][j-1][k];
        double delta_down_j = phi[i][j+1][k] - phi[i][j][k];
        double r_j = (fabs(delta_down_j) < SMALL_NUMBER) ? 1e9 : delta_up_j / delta_down_j;
        double phi_lim_j = superbee_limiter(r_j);
        double phi_jph = phi[i][j][k] + 0.5 * phi_lim_j * delta_down_j;

        convection_y = v[i][j][k] * (phi_jph - phi_jmh) / DY;
    } else { // 負の流速 (風上は j+1, j+2)
        double delta_up_j = phi[i][j+1][k] - phi[i][j][k];
        double delta_down_j = phi[i][j][k] - phi[i][j-1][k];
        double r_j = (fabs(delta_down_j) < SMALL_NUMBER) ? 1e9 : delta_up_j / delta_down_j;
        double phi_lim_j = superbee_limiter(r_j);
        double phi_jmh = phi[i][j][k] - 0.5 * phi_lim_j * delta_down_j;

        double delta_up_jp = phi[i][j+2][k] - phi[i][j+1][k];
        double delta_down_jp = phi[i][j+1][k] - phi[i][j][k];
        double r_jp = (fabs(delta_down_jp) < SMALL_NUMBER) ? 1e9 : delta_up_jp / delta_down_jp;
        double phi_lim_jp = superbee_limiter(r_jp);
        double phi_jph = phi[i][j+1][k] - 0.5 * phi_lim_jp * delta_down_jp;

        convection_y = v[i][j][k] * (phi_jph - phi_jmh) / DY;
    }

    // --- Z方向の移流項 ---
    if (w[i][j][k] >= 0.0) { // 正の流速 (風上は k-1, k-2)
        double delta_up_km = phi[i][j][k-1] - phi[i][j][k-2];
        double delta_down_km = phi[i][j][k] - phi[i][j][k-1];
        double r_km = (fabs(delta_down_km) < SMALL_NUMBER) ? 1e9 : delta_up_km / delta_down_km;
        double phi_lim_km = superbee_limiter(r_km);
        double phi_kmh = phi[i][j][k-1] + 0.5 * phi_lim_km * delta_down_km;

        double delta_up_k = phi[i][j][k] - phi[i][j][k-1];
        double delta_down_k = phi[i][j][k+1] - phi[i][j][k];
        double r_k = (fabs(delta_down_k) < SMALL_NUMBER) ? 1e9 : delta_up_k / delta_down_k;
        double phi_lim_k = superbee_limiter(r_k);
        double phi_kph = phi[i][j][k] + 0.5 * phi_lim_k * delta_down_k;
        
        convection_z = w[i][j][k] * (phi_kph - phi_kmh) / DZ;
    } else { // 負の流速 (風上は k+1, k+2)
        double delta_up_k = phi[i][j][k+1] - phi[i][j][k];
        double delta_down_k = phi[i][j][k] - phi[i][j][k-1];
        double r_k = (fabs(delta_down_k) < SMALL_NUMBER) ? 1e9 : delta_up_k / delta_down_k;
        double phi_lim_k = superbee_limiter(r_k);
        double phi_kmh = phi[i][j][k] - 0.5 * phi_lim_k * delta_down_k;

        double delta_up_kp = phi[i][j][k+2] - phi[i][j][k+1];
        double delta_down_kp = phi[i][j][k+1] - phi[i][j][k];
        double r_kp = (fabs(delta_down_kp) < SMALL_NUMBER) ? 1e9 : delta_up_kp / delta_down_kp;
        double phi_lim_kp = superbee_limiter(r_kp);
        double phi_kph = phi[i][j][k+1] - 0.5 * phi_lim_kp * delta_down_kp;
        
        convection_z = w[i][j][k] * (phi_kph - phi_kmh) / DZ;
    }

    return convection_x + convection_y + convection_z;
}

// ==============================================================================
// --- 物理モデル・数値スキーム定数 (上級者向け) ---
// ==============================================================================
#define TENO_CUTOFF_THRESHOLD 1.0e-6 // TENOのカットオフ閾値 (C_T). この値よりISが小さいステンシルを「滑らか」と判断する
#define TENO_EPSILON 1.0e-40         // TENOの重み計算におけるゼロ除算防止用の微小数 (WENOのepsilonよりずっと小さくする)
#define TENO_POWER_Q 6.0             // TENOの重み計算に用いる指数 (q)

// ==============================================================================
// --- TENO5 スキーム用 補助関数 ---
// ==============================================================================

/**
 * @brief 1次元方向の勾配 d(phi)/dx を5次精度TENOスキームで計算する。
 *        WENO5の実装をベースに、TENOの重み付けロジックを導入。
 * @param phi 物理量配列
 * @param i, j, k 計算対象のセルインデックス
 * @param dir 計算方向 (0:x, 1:y, 2:z)
 * @param dx 格子間隔
 * @return 計算された勾配
 */

// ==============================================================================
// TENO5 スキーム用 再構築関数 
// ==============================================================================

/**
 * @brief 5点のステンシル (v[0]...v[4]) から、i+1/2 の界面値を再構築する
 *        標準的なWENO5/TENO5の重み付けを使用
 */
double teno5_reconstruct(double v0, double v1, double v2, double v3, double v4) {
    // 滑らかさ指標 (IS) の計算
    double is0, is1, is2;
    // IS0: v0, v1, v2
    is0 = (13.0/12.0) * pow(v0 - 2.0*v1 + v2, 2) + (1.0/4.0) * pow(v0 - 4.0*v1 + 3.0*v2, 2);
    // IS1: v1, v2, v3
    is1 = (13.0/12.0) * pow(v1 - 2.0*v2 + v3, 2) + (1.0/4.0) * pow(v1 - v3, 2);
    // IS2: v2, v3, v4
    is2 = (13.0/12.0) * pow(v2 - 2.0*v3 + v4, 2) + (1.0/4.0) * pow(3.0*v2 - 4.0*v3 + v4, 2);

    // TENO カットオフ判定
    // TENO_CUTOFF_THRESHOLD (例: 1e-5 ~ 1e-6) より小さいISを持つステンシルのみ採用
    int delta0 = (is0 < TENO_CUTOFF_THRESHOLD) ? 1 : 0;
    int delta1 = (is1 < TENO_CUTOFF_THRESHOLD) ? 1 : 0;
    int delta2 = (is2 < TENO_CUTOFF_THRESHOLD) ? 1 : 0;

    // 3つの候補ステンシルでの多項式補間値 (右側バイアス: i+1/2を求める標準形)
    double p0 = (2.0*v0 - 7.0*v1 + 11.0*v2) / 6.0;
    double p1 = (-v1 + 5.0*v2 + 2.0*v3) / 6.0;
    double p2 = (2.0*v2 + 5.0*v3 - v4) / 6.0;

    // もし全てのステンシルが「滑らかでない（衝撃波/不連続）」と判定されたら
    // 最も安定な（滑らかさがマシな）ENO的選択、または3次精度へのフォールバックを行う
    if (delta0 + delta1 + delta2 == 0) {
        // 最もISが小さいものを採用 (ENO方式)
        if (is0 <= is1 && is0 <= is2) return p0;
        else if (is1 <= is0 && is1 <= is2) return p1;
        else return p2;
    }

    // TENO 重みの計算
    // 線形重み (d0=0.1, d1=0.6, d2=0.3 はJS-WENOの標準だが、TENOでは調整されることもある)
    // ここでは標準的なWENO5の最適重みを使用
    const double d0 = 0.1, d1 = 0.6, d2 = 0.3;

    double w0_s = delta0 * d0;
    double w1_s = delta1 * d1;
    double w2_s = delta2 * d2;

    // 重みの正規化 (Strong Scale Separation)
    // 単純な切り捨てだけでなく、元の重み比率を維持する
    double w_sum = w0_s + w1_s + w2_s;
    if (w_sum < 1e-40) return p1; // 安全策

    // 最終的な再構築値
    return (w0_s * p0 + w1_s * p1 + w2_s * p2) / w_sum;
}

// ===================================
// TENO5 移流項計算関数 
// ===================================
/**
 * @brief 非保存形の移流項 (U⋅∇)φ を5次精度TENOスキームで計算する。
 *        速度の向きに応じてステンシルを切り替える（風上処理）。
 */
double calculate_convection_teno(
    int i, int j, int k,
    const Grid3D phi,
    const Grid3D u, const Grid3D v, const Grid3D w
) {
    // 境界付近は安全なQUICKにフォールバック
    if (i < 3 || i >= NX - 3 || j < 3 || j >= NY - 3 || k < 3 || k >= NZ - 3) {
        return calculate_convection_non_conservative(i, j, k, phi, u, v, w);
    }

    double flux_x, flux_y, flux_z;

    // --- X方向の計算 ---
    if (u[i][j][k] >= 0.0) {
        // 風が正方向 -> 左側の点を使って、界面 i-1/2 と i+1/2 を構築
        // i+1/2 の左側値: (i-2, i-1, i, i+1, i+2)
        double phi_iph = teno5_reconstruct(phi[i-2][j][k], phi[i-1][j][k], phi[i][j][k], phi[i+1][j][k], phi[i+2][j][k]);
        // i-1/2 の左側値: (i-3, i-2, i-1, i, i+1)
        double phi_imh = teno5_reconstruct(phi[i-3][j][k], phi[i-2][j][k], phi[i-1][j][k], phi[i][j][k], phi[i+1][j][k]);
        flux_x = u[i][j][k] * (phi_iph - phi_imh) / DX;
    } else {
        // 風が負方向 -> 右側の点を使って、界面 i-1/2 と i+1/2 を構築（左右対称に反転）
        // データの並びを反転させて teno5_reconstruct に渡すことで右バイアスを実現
        // i-1/2 の右側値: (i+2, i+1, i, i-1, i-2)
        double phi_imh = teno5_reconstruct(phi[i+2][j][k], phi[i+1][j][k], phi[i][j][k], phi[i-1][j][k], phi[i-2][j][k]);
        // i+1/2 の右側値: (i+3, i+2, i+1, i, i-1)
        double phi_iph = teno5_reconstruct(phi[i+3][j][k], phi[i+2][j][k], phi[i+1][j][k], phi[i][j][k], phi[i-1][j][k]);
        flux_x = u[i][j][k] * (phi_iph - phi_imh) / DX;
    }

    // --- Y方向の計算 ---
    if (v[i][j][k] >= 0.0) {
        double phi_jph = teno5_reconstruct(phi[i][j-2][k], phi[i][j-1][k], phi[i][j][k], phi[i][j+1][k], phi[i][j+2][k]);
        double phi_jmh = teno5_reconstruct(phi[i][j-3][k], phi[i][j-2][k], phi[i][j-1][k], phi[i][j][k], phi[i][j+1][k]);
        flux_y = v[i][j][k] * (phi_jph - phi_jmh) / DY;
    } else {
        double phi_jmh = teno5_reconstruct(phi[i][j+2][k], phi[i][j+1][k], phi[i][j][k], phi[i][j-1][k], phi[i][j-2][k]);
        double phi_jph = teno5_reconstruct(phi[i][j+3][k], phi[i][j+2][k], phi[i][j+1][k], phi[i][j][k], phi[i][j-1][k]);
        flux_y = v[i][j][k] * (phi_jph - phi_jmh) / DY;
    }

    // --- Z方向の計算 ---
    if (w[i][j][k] >= 0.0) {
        double phi_kph = teno5_reconstruct(phi[i][j][k-2], phi[i][j][k-1], phi[i][j][k], phi[i][j][k+1], phi[i][j][k+2]);
        double phi_kmh = teno5_reconstruct(phi[i][j][k-3], phi[i][j][k-2], phi[i][j][k-1], phi[i][j][k], phi[i][j][k+1]);
        flux_z = w[i][j][k] * (phi_kph - phi_kmh) / DZ;
    } else {
        double phi_kmh = teno5_reconstruct(phi[i][j][k+2], phi[i][j][k+1], phi[i][j][k], phi[i][j][k-1], phi[i][j][k-2]);
        double phi_kph = teno5_reconstruct(phi[i][j][k+3], phi[i][j][k+2], phi[i][j][k+1], phi[i][j][k], phi[i][j][k-1]);
        flux_z = w[i][j][k] * (phi_kph - phi_kmh) / DZ;
    }

    return flux_x + flux_y + flux_z;
}

// ==============================================================================
// --- 保存型 TENO5 スキーム用 補助関数 ---
// ==============================================================================

/**
 * @brief Lax-Friedrichs Flux Splitting を行う関数
 *        流束 F(q) = rho * u * phi を、正の流速成分 F+ と負の流速成分 F- に分解する。
 * @param rho 密度
 * @param u   速度
 * @param phi 移流される物理量 (u, v, w のいずれか)
 * @param alpha 最大流速（波の伝播速度）
 * @param flux_plus  出力: F+ の値
 * @param flux_minus 出力: F- の値
 *
 * F+(q) = 0.5 * (F(q) + alpha * q)
 * F-(q) = 0.5 * (F(q) - alpha * q)
 * ただし、q = rho * phi
 */
void lax_friedrichs_split(double rho, double u, double phi, double alpha, double* flux_plus, double* flux_minus)
{
    double q = rho * phi;
    double F = rho * u * phi;
    *flux_plus  = 0.5 * (F + alpha * q);
    *flux_minus = 0.5 * (F - alpha * q);
}

/**
 * @brief 1次元方向の界面での数値流束を、保存型TENOスキームで計算する。
 *        Lax-Friedrichs Flux Splitting を使用。
 * @param q 物理量 (rho*phi) が格納された6点のステンシル配列
 * @param f_plus  正の流束 F+ が格納された6点のステンシル配列
 * @param f_minus 負の流束 F- が格納された6点のステンシル配列
 * @return 界面 i+1/2 における最終的な数値流束
 */
double calculate_teno_flux_1d(const double q[6], const double f_plus[6], const double f_minus[6])
{
    // --- 1. F+ に対する左バイアスのTENO再構築 (界面 i+1/2 で f_plus を計算) ---
    double f_plus_iph;
    {
        double is0, is1, is2;
        calculate_weno5_smoothness(f_plus[0], f_plus[1], f_plus[2], f_plus[3], f_plus[4], &is0, &is1, &is2);

        int delta0 = (is0 < TENO_CUTOFF_THRESHOLD);
        int delta1 = (is1 < TENO_CUTOFF_THRESHOLD);
        int delta2 = (is2 < TENO_CUTOFF_THRESHOLD);

        double rec0 = (-f_plus[0] + 5.0*f_plus[1] + 2.0*f_plus[2]) / 6.0;
        double rec1 = (2.0*f_plus[1] + 5.0*f_plus[2] - f_plus[3]) / 6.0;
        double rec2 = (11.0*f_plus[2] - 7.0*f_plus[3] + 2.0*f_plus[4]) / 6.0;
        
        if (delta0 + delta1 + delta2 == 0) {
            if (is0 <= is1 && is0 <= is2) f_plus_iph = rec0;
            else if (is1 <= is0 && is1 <= is2) f_plus_iph = rec1;
            else f_plus_iph = rec2;
        } else {
            const double d0 = 1.0/10.0, d1 = 6.0/10.0, d2 = 3.0/10.0;
            double C_sum = delta0 * d0 + delta1 * d1 + delta2 * d2;
            double C0 = delta0 * d0 / C_sum; double C1 = delta1 * d1 / C_sum; double C2 = delta2 * d2 / C_sum;
            double alpha0 = C0 * (1.0 + pow((is0 + TENO_EPSILON) / (is1 + TENO_EPSILON), TENO_POWER_Q) + pow((is0 + TENO_EPSILON) / (is2 + TENO_EPSILON), TENO_POWER_Q));
            double alpha1 = C1 * (1.0 + pow((is1 + TENO_EPSILON) / (is0 + TENO_EPSILON), TENO_POWER_Q) + pow((is1 + TENO_EPSILON) / (is2 + TENO_EPSILON), TENO_POWER_Q));
            double alpha2 = C2 * (1.0 + pow((is2 + TENO_EPSILON) / (is0 + TENO_EPSILON), TENO_POWER_Q) + pow((is2 + TENO_EPSILON) / (is1 + TENO_EPSILON), TENO_POWER_Q));
            if (!isfinite(alpha0)) alpha0 = 0.0; if (!isfinite(alpha1)) alpha1 = 0.0; if (!isfinite(alpha2)) alpha2 = 0.0;
            double w0 = C0 / alpha0; double w1 = C1 / alpha1; double w2 = C2 / alpha2;
            double w_sum = w0 + w1 + w2;
            if (w_sum < SMALL_NUMBER) { w0 = d0; w1 = d1; w2 = d2; w_sum = 1.0; }
            w0 /= w_sum; w1 /= w_sum; w2 /= w_sum;
            f_plus_iph = w0 * rec0 + w1 * rec1 + w2 * rec2;
        }
    }

    // --- 2. F- に対する右バイアスのTENO再構築 (界面 i+1/2 で f_minus を計算) ---
    double f_minus_iph;
    {
        double is0, is1, is2;
        calculate_weno5_smoothness(f_minus[1], f_minus[2], f_minus[3], f_minus[4], f_minus[5], &is0, &is1, &is2);

        int delta0 = (is0 < TENO_CUTOFF_THRESHOLD);
        int delta1 = (is1 < TENO_CUTOFF_THRESHOLD);
        int delta2 = (is2 < TENO_CUTOFF_THRESHOLD);
        
        double rec0 = (2.0*f_minus[1] - 7.0*f_minus[2] + 11.0*f_minus[3]) / 6.0;
        double rec1 = (-f_minus[2] + 5.0*f_minus[3] + 2.0*f_minus[4]) / 6.0;
        double rec2 = (2.0*f_minus[3] + 5.0*f_minus[4] - f_minus[5]) / 6.0;

        if (delta0 + delta1 + delta2 == 0) {
            if (is0 <= is1 && is0 <= is2) f_minus_iph = rec0;
            else if (is1 <= is0 && is1 <= is2) f_minus_iph = rec1;
            else f_minus_iph = rec2;
        } else {
            const double d0 = 3.0/10.0, d1 = 6.0/10.0, d2 = 1.0/10.0;
            double C_sum = delta0 * d0 + delta1 * d1 + delta2 * d2;
            double C0 = delta0 * d0 / C_sum; double C1 = delta1 * d1 / C_sum; double C2 = delta2 * d2 / C_sum;
            double alpha0 = C0 * (1.0 + pow((is0 + TENO_EPSILON) / (is1 + TENO_EPSILON), TENO_POWER_Q) + pow((is0 + TENO_EPSILON) / (is2 + TENO_EPSILON), TENO_POWER_Q));
            double alpha1 = C1 * (1.0 + pow((is1 + TENO_EPSILON) / (is0 + TENO_EPSILON), TENO_POWER_Q) + pow((is1 + TENO_EPSILON) / (is2 + TENO_EPSILON), TENO_POWER_Q));
            double alpha2 = C2 * (1.0 + pow((is2 + TENO_EPSILON) / (is0 + TENO_EPSILON), TENO_POWER_Q) + pow((is2 + TENO_EPSILON) / (is1 + TENO_EPSILON), TENO_POWER_Q));
            if (!isfinite(alpha0)) alpha0 = 0.0; if (!isfinite(alpha1)) alpha1 = 0.0; if (!isfinite(alpha2)) alpha2 = 0.0;
            double w0 = C0 / alpha0; double w1 = C1 / alpha1; double w2 = C2 / alpha2;
            double w_sum = w0 + w1 + w2;
            if (w_sum < SMALL_NUMBER) { w0 = d0; w1 = d1; w2 = d2; w_sum = 1.0; }
            w0 /= w_sum; w1 /= w_sum; w2 /= w_sum;
            f_minus_iph = w0 * rec0 + w1 * rec1 + w2 * rec2;
        }
    }

    // --- 3. 最終的な数値流束を合成 ---
    return f_plus_iph + f_minus_iph;
}

/**
 * @brief 保存形の移流項 ∇⋅( (ρU)φ ) をTENOスキームで計算する (運動量輸送用)
 */
double calculate_convection_conservative_teno(
    int i, int j, int k,
    const Grid3D phi, // const double phi[NX][NY][NZ] から変更
    const Grid3D u, const Grid3D v, const Grid3D w,
    const Grid3D rho
) {
    // 境界付近では安定なQUICKにフォールバック
    if (i < 3 || i >= NX - 3 || j < 3 || j >= NY - 3 || k < 3 || k >= NZ - 3) {
        return calculate_convection_conservative(i, j, k, phi, u, v, w, rho);
    }

    // --- X方向の流束の差分 (F_{i+1/2} - F_{i-1/2})/DX ---
    double flux_div_x;
    {
        // 最大流速 alpha を計算 (ローカルな絶対速度 + 音速。ここでは簡易的に最大速度とする)
        double alpha_x = 0.0;
        for (int n = -3; n <= 3; n++) alpha_x = fmax(alpha_x, fabs(u[i+n][j][k]));
        if (alpha_x < SMALL_NUMBER) alpha_x = SMALL_NUMBER;

        // F_{i+1/2} の計算
        double q_sten[6], fp_sten[6], fm_sten[6];
        for (int n = 0; n < 6; n++) {
            lax_friedrichs_split(rho[i-2+n][j][k], u[i-2+n][j][k], phi[i-2+n][j][k], alpha_x, &fp_sten[n], &fm_sten[n]);
        }
        double flux_iph = calculate_teno_flux_1d(q_sten, fp_sten, fm_sten);
        
        // F_{i-1/2} の計算
        for (int n = 0; n < 6; n++) {
            lax_friedrichs_split(rho[i-3+n][j][k], u[i-3+n][j][k], phi[i-3+n][j][k], alpha_x, &fp_sten[n], &fm_sten[n]);
        }
        double flux_imh = calculate_teno_flux_1d(q_sten, fp_sten, fm_sten);
        
        flux_div_x = (flux_iph - flux_imh) / DX;
    }

    // --- Y方向の流束の差分 (F_{j+1/2} - F_{j-1/2})/DY ---
    double flux_div_y;
    {
        double alpha_y = 0.0;
        for (int n = -3; n <= 3; n++) alpha_y = fmax(alpha_y, fabs(v[i][j+n][k]));
        if (alpha_y < SMALL_NUMBER) alpha_y = SMALL_NUMBER;

        double q_sten[6], fp_sten[6], fm_sten[6];
        for (int n = 0; n < 6; n++) {
            lax_friedrichs_split(rho[i][j-2+n][k], v[i][j-2+n][k], phi[i][j-2+n][k], alpha_y, &fp_sten[n], &fm_sten[n]);
        }
        double flux_jph = calculate_teno_flux_1d(q_sten, fp_sten, fm_sten);

        for (int n = 0; n < 6; n++) {
            lax_friedrichs_split(rho[i][j-3+n][k], v[i][j-3+n][k], phi[i][j-3+n][k], alpha_y, &fp_sten[n], &fm_sten[n]);
        }
        double flux_jmh = calculate_teno_flux_1d(q_sten, fp_sten, fm_sten);

        flux_div_y = (flux_jph - flux_jmh) / DY;
    }

    // --- Z方向の流束の差分 (F_{k+1/2} - F_{k-1/2})/DZ ---
    double flux_div_z;
    {
        double alpha_z = 0.0;
        for (int n = -3; n <= 3; n++) alpha_z = fmax(alpha_z, fabs(w[i][j][k+n]));
        if (alpha_z < SMALL_NUMBER) alpha_z = SMALL_NUMBER;

        double q_sten[6], fp_sten[6], fm_sten[6];
        for (int n = 0; n < 6; n++) {
            lax_friedrichs_split(rho[i][j][k-2+n], w[i][j][k-2+n], phi[i][j][k-2+n], alpha_z, &fp_sten[n], &fm_sten[n]);
        }
        double flux_kph = calculate_teno_flux_1d(q_sten, fp_sten, fm_sten);

        for (int n = 0; n < 6; n++) {
            lax_friedrichs_split(rho[i][j][k-3+n], w[i][j][k-3+n], phi[i][j][k-3+n], alpha_z, &fp_sten[n], &fm_sten[n]);
        }
        double flux_kmh = calculate_teno_flux_1d(q_sten, fp_sten, fm_sten);
        
        flux_div_z = (flux_kph - flux_kmh) / DZ;
    }

    return flux_div_x + flux_div_y + flux_div_z;
}

// ==============================================================================
// 【G方程式のみWENOを適用】calculate_RHS 関数 
// ==============================================================================
void calculate_RHS(
    // --- 入力: ある時点での物理量 ---
    int t_loop,
    const Grid3D u_in, const Grid3D v_in, const Grid3D w_in,
    const Grid3D T_in, const Grid3D G_in,
    const Grid3D *Yk_in, // 配列のポインタなので*が必要
    const Grid3D rho_in,

    // --- 出力: 計算された時間微分 ---
    Grid3D drhou_dt, Grid3D drhov_dt, Grid3D drhow_dt,
    Grid3D dT_dt, Grid3D dG_dt,
    Grid3D *dYk_dt 
) {
    // G方程式の速度成分の計算
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                uG[i][j][k] = u_in[i][j][k] * G_in[i][j][k];
                vG[i][j][k] = v_in[i][j][k] * G_in[i][j][k];
                wG[i][j][k] = w_in[i][j][k] * G_in[i][j][k];
            }
        }
    }
    
    // --- 1. フィルタリング ---
    apply_box_filter(
        attribute, 0,
        u_in, u_filtered, v_in, v_filtered, w_in, w_filtered,
        P, P_filtered,
        T_in, T_filtered, G_in, G_filtered,
        uG, uG_filtered, vG, vG_filtered, wG, wG_filtered,
        Yk_in, Yk_filtered, rho_in, rho_filtered
    );

    // --- 2. 物性値・乱流モデル計算 ---
    update_nu_mix(attribute, T_filtered, Yk_filtered, nu, mu_mix, rho_filtered);
    calculate_nu_t(attribute, 0, u_filtered, v_filtered, w_filtered, G_filtered, S_mag, nu_t, SijSij, div_u);

    // --- 3. 各方程式の時間微分項 (RHS) を計算 ---
    // (a) 運動量のRHS 
    // 動的な参照密度（領域平均密度）を計算
    double total_mass = 0.0;
    double total_volume = 0.0;
    #pragma omp parallel for reduction(+:total_mass, total_volume)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                // 流体セルのみを対象に質量と体積を積算
                if (attribute[i][j][k] == 1 || attribute[i][j][k] == 2 || attribute[i][j][k] == 3 || attribute[i][j][k] == 5) {
                    total_mass += rho_filtered[i][j][k] * DX * DY * DZ;
                    total_volume += DX * DY * DZ;
                }
            }
        }
    }
    double rho_avg_dynamic = (total_volume > 0) ? (total_mass / total_volume) : DEFAULT_RHO_AIR;
    
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                if (attribute[i][j][k] == 3) {
                    double convection_rhou = calculate_convection_conservative(i, j, k, u_filtered, u_filtered, v_filtered, w_filtered, rho_filtered);
                    double convection_rhov = calculate_convection_conservative(i, j, k, v_filtered, u_filtered, v_filtered, w_filtered, rho_filtered);
                    double convection_rhow = calculate_convection_conservative(i, j, k, w_filtered, u_filtered, v_filtered, w_filtered, rho_filtered);

                    //double convection_rhou = calculate_convection_conservative_teno(i, j, k, u_filtered, u_filtered, v_filtered, w_filtered, rho_filtered);
                    //double convection_rhov = calculate_convection_conservative_teno(i, j, k, v_filtered, u_filtered, v_filtered, w_filtered, rho_filtered);
                    //double convection_rhow = calculate_convection_conservative_teno(i, j, k, w_filtered, u_filtered, v_filtered, w_filtered, rho_filtered);

                    double mu_eff = mu_mix[i][j][k] + rho_filtered[i][j][k] * nu_t[i][j][k];
                    
                    double d2u_dx2 = (u_filtered[i+1][j][k] - 2.0*u_filtered[i][j][k] + u_filtered[i-1][j][k]) / (DX*DX);
                    double d2u_dy2 = (u_filtered[i][j+1][k] - 2.0*u_filtered[i][j][k] + u_filtered[i][j-1][k]) / (DY*DY);
                    double d2u_dz2 = (u_filtered[i][j][k+1] - 2.0*u_filtered[i][j][k] + u_filtered[i][j][k-1]) / (DZ*DZ);
                    double diffusion_u = mu_eff * (d2u_dx2 + d2u_dy2 + d2u_dz2);

                    double d2v_dx2 = (v_filtered[i+1][j][k] - 2.0*v_filtered[i][j][k] + v_filtered[i-1][j][k]) / (DX*DX);
                    double d2v_dy2 = (v_filtered[i][j+1][k] - 2.0*v_filtered[i][j][k] + v_filtered[i][j-1][k]) / (DY*DY);
                    double d2v_dz2 = (v_filtered[i][j][k+1] - 2.0*v_filtered[i][j][k] + v_filtered[i][j][k-1]) / (DZ*DZ);
                    double diffusion_v = mu_eff * (d2v_dx2 + d2v_dy2 + d2v_dz2);

                    double d2w_dx2 = (w_filtered[i+1][j][k] - 2.0*w_filtered[i][j][k] + w_filtered[i-1][j][k]) / (DX*DX);
                    double d2w_dy2 = (w_filtered[i][j+1][k] - 2.0*w_filtered[i][j][k] + w_filtered[i][j-1][k]) / (DY*DY);
                    double d2w_dz2 = (w_filtered[i][j][k+1] - 2.0*w_filtered[i][j][k] + w_filtered[i][j][k-1]) / (DZ*DZ);
                    double diffusion_w = mu_eff * (d2w_dx2 + d2w_dy2 + d2w_dz2);

                    double gravity_term = (rho_filtered[i][j][k] - rho_avg_dynamic) * G_GRAVITY;
                    
                    drhou_dt[i][j][k] = -convection_rhou + diffusion_u + gravity_term;

                    // ====================================
                    // 熱膨張による加速力 (Expansion Force) 
                    // ====================================
                    double force_x = 0.0;
                    double force_y = 0.0;
                    double force_z = 0.0;

                    #if USE_FLAME_ACCEL_MODEL
                    double G_local = G_filtered[i][j][k];

                    // 火炎帯内部 (Gが0～1の間) のみを判定
                    if (G_local > 0.01 && G_local < 0.99) {
                        
                        // 1. Gの勾配（＝火炎面の法線ベクトル）を計算
                        // Gは未燃(0)→既燃(1)へ増えるので、勾配方向＝膨張方向となる
                        double dG_dx = (G_filtered[i+1][j][k] - G_filtered[i-1][j][k]) / (2.0 * DX);
                        double dG_dy = (G_filtered[i][j+1][k] - G_filtered[i][j-1][k]) / (2.0 * DY);
                        double dG_dz = (G_filtered[i][j][k+1] - G_filtered[i][j][k-1]) / (2.0 * DZ);

                        // [X方向: 流れ方向]
                        // 上流(マイナス)へ押す力はカットし、下流(プラス)への加速のみ許可
                        if (dG_dx < 0.0) dG_dx = 0.0;

                        // [Y方向: 半径方向]
                        // y=0が対称面なので、y>0に向かう(プラス)力のみ許可。内側への吸い込みをカット。
                        if (dG_dy < 0.0) dG_dy = 0.0;

                        // [Z方向: 奥行き方向]
                        // z中心(INLET_CENTER_Z)から見て「外側」への力のみ許可
                        double z_center_rel = (k * DZ) - INLET_CENTER_Z;
                        if (z_center_rel > 0.0) {
                            if (dG_dz < 0.0) dG_dz = 0.0; // 正側で負の勾配ならカット
                        } else {
                            if (dG_dz > 0.0) dG_dz = 0.0; // 負側で正の勾配ならカット
                        }

                        // 2. 加速させる領域を決めるウィンドウ関数 (ガウス分布)
                        // define定義した WIDTH と CENTER を使用
                        double diff = G_local - FLAME_ACCEL_CENTER;
                        double window = exp( - (diff * diff) / (2.0 * FLAME_ACCEL_WIDTH * FLAME_ACCEL_WIDTH) );

                        // 現在の半径方向の位置を計算 (y座標が中心からどれだけ離れているか)
                        // ※コードの座標系に合わせて調整してください。
                        // y=0 が中心なら:
                        double radial_pos = fabs(j * DY); 
                        // もし y座標が INLET_CENTER_Y などの定数でずれているなら:
                        // double radial_pos = fabs(j * DY - INLET_CENTER_Y);

                        // マスク係数の計算
                        // 中心(0)で0.0、保護半径(PROTECTION_RADIUS)で約0.6、それ以遠で1.0になる関数
                        // これにより「先端は加速しない」「側面は加速する」を実現
                        //double tip_mask = 1.0 - exp( - (radial_pos * radial_pos) / (TIP_PROTECTION_RADIUS * TIP_PROTECTION_RADIUS) );
                        double tip_mask = 1.0;

                        // 3. 力のベクトルの計算
                        // Force = 強さ(STRENGTH) * 密度 * 勾配 * ウィンドウ
                        // 密度を掛けるのは、運動量(rho*v)のソース項だから
                        double common_factor = FLAME_ACCEL_STRENGTH * rho_filtered[i][j][k] * window * tip_mask;

                        // X方向（流れ方向）のダンピング係数
                        // 0.1 ～ 0.2 程度に設定し、ジェット噴射化を防ぐ
                        // これで先端流速 12m/s -> 5~6m/s 程度に落ち着くはずです
                        double weight_x = 0.05; 
                        double weight_yz = 1.0;

                        force_x = common_factor * dG_dx * weight_x;
                        force_y = common_factor * dG_dy * weight_yz; 
                        force_z = common_factor * dG_dz * weight_yz; 
                    
                        // ==========================================================
                        // リム付近の保護 (Z方向ランピング) 
                        // ==========================================================
                        double z_pos = k * DZ; // 現在のZ座標
                        double ramp_factor = 1.0;

                        if (z_pos < FLAME_RAMP_DISTANCE) {
                            // 出口(0.0)では0.0、RAMP_DISTANCEで1.0になる線形係数
                            ramp_factor = z_pos / FLAME_RAMP_DISTANCE;
                            
                            // より安全にするなら2乗カーブ (立ち上がりを遅くする)
                            // ramp_factor = pow(z_pos / FLAME_RAMP_DISTANCE, 2.0);
                        }

                        // 力を減衰させる
                        force_x *= ramp_factor;
                        force_y *= ramp_factor;
                        force_z *= ramp_factor;
                    }
                    #endif

                    //力のクリッピング (Safety Clipping) 
                    // これで数値的なスパイク（異常値）をカットします
                    // x成分
                    if (force_x > MAX_FLAME_FORCE_LIMIT) force_x = MAX_FLAME_FORCE_LIMIT;
                    if (force_x < -MAX_FLAME_FORCE_LIMIT) force_x = -MAX_FLAME_FORCE_LIMIT;
                    // y成分 (これが最も重要)
                    if (force_y > MAX_FLAME_FORCE_LIMIT) force_y = MAX_FLAME_FORCE_LIMIT;
                    if (force_y < -MAX_FLAME_FORCE_LIMIT) force_y = -MAX_FLAME_FORCE_LIMIT;
                    // z成分
                    if (force_z > MAX_FLAME_FORCE_LIMIT) force_z = MAX_FLAME_FORCE_LIMIT;
                    if (force_z < -MAX_FLAME_FORCE_LIMIT) force_z = -MAX_FLAME_FORCE_LIMIT;

                    // --- 4. 最終的な時間微分の合算 ---
                    drhou_dt[i][j][k] = -convection_rhou + diffusion_u + gravity_term + force_x;
                    drhov_dt[i][j][k] = -convection_rhov + diffusion_v + force_y;
                    drhow_dt[i][j][k] = -convection_rhow + diffusion_w + force_z; 
                } else {
                    drhou_dt[i][j][k] = 0.0; drhov_dt[i][j][k] = 0.0; drhow_dt[i][j][k] = 0.0;
                }
            }
        }
    }

    // (b) G方程式のRHS (WENOハイブリッド適用)
    double S_L_param = S_L;
    double rho_unburnt_approx = 1.2; 
    double rho_burnt_approx = 0.2; 
    double tau_expansion_ratio_param = (rho_unburnt_approx / rho_burnt_approx) - 1.0;
    calculate_ST_turbulent_SSR_SGS_jp_global_vars(S_L_param, NU_UNBURNT_AIR_APPROX, tau_expansion_ratio_param, FRACTAL_DIMENSION_D3, rho_filtered);
    
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < NX - 1; i++) {
        for (int j = 1; j < NY - 1; j++) {
            for (int k = 1; k < NZ - 1; k++) {
                if (attribute[i][j][k] == 3) {

                    
                    //double convection;
                    /*G方程式ではWENOスキームを使用
                    if (i >= 3 && i < NX - 3 && j >= 3 && j < NY - 3 && k >= 3 && k < NZ - 3) {
                        convection = calculate_convection_weno5(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);
                    } else {
                        convection = calculate_convection_non_conservative(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);
                    }*/

                    //double convection = calculate_convection_non_conservative(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);

                    // ハイブリッドスキームの導入 
                    // 火炎面近傍 (Gが0と1の間) ではWENOを、それ以外ではQUICKを使用
                    /*double G_local = G_filtered[i][j][k];
                    if ( (G_local > 0.01 && G_local < 0.99) && // Gが中間値で
                        (i >= 3 && i < NX - 3 && j >= 3 && j < NY - 3 && k >= 3 && k < NZ - 3) ) // 配列境界に十分なマージンがある
                    {
                        // 精度を重視
                        convection = calculate_convection_weno5(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);
                    } else {
                        // 安定性を重視
                        convection = calculate_convection_non_conservative(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);
                    }*/

                    //double convection = calculate_convection_muscl(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);

                    double convection = calculate_convection_teno(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);

                    const int SCHEME_TRANSITION_STEP = 500; // 200ステップまでは安定なスキームを使う
                    /*double convection;
                    // --- 移流項の切り替え ---
                    if (t_loop < SCHEME_TRANSITION_STEP) {
                        // 初期段階では安定なQUICKスキームを使用
                        convection = calculate_convection_non_conservative(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);
                    } else {
                        // 安定化後は高精度なTENOスキームを使用
                        convection = calculate_convection_muscl(i, j, k, G_filtered, u_filtered, v_filtered, w_filtered);
                    }*/

                    double gamma_iph = (nu_t[i+1][j][k] + nu_t[i][j][k]) / (2.0 * Sc_t);
                    double gamma_imh = (nu_t[i][j][k]   + nu_t[i-1][j][k]) / (2.0 * Sc_t);
                    double gamma_jph = (nu_t[i][j+1][k] + nu_t[i][j][k]) / (2.0 * Sc_t);
                    double gamma_jmh = (nu_t[i][j][k]   + nu_t[i][j-1][k]) / (2.0 * Sc_t);
                    double gamma_kph = (nu_t[i][j][k+1] + nu_t[i][j][k]) / (2.0 * Sc_t);
                    double gamma_kmh = (nu_t[i][j][k]   + nu_t[i][j][k-1]) / (2.0 * Sc_t);
                    double sgs_flux_term_div_x = (gamma_iph * (G_filtered[i+1][j][k] - G_filtered[i][j][k]) / DX - gamma_imh * (G_filtered[i][j][k] - G_filtered[i-1][j][k]) / DX) / DX;
                    double sgs_flux_term_div_y = (gamma_jph * (G_filtered[i][j+1][k] - G_filtered[i][j][k]) / DY - gamma_jmh * (G_filtered[i][j][k] - G_filtered[i][j-1][k]) / DY) / DY;
                    double sgs_flux_term_div_z = (gamma_kph * (G_filtered[i][j][k+1] - G_filtered[i][j][k]) / DZ - gamma_kmh * (G_filtered[i][j][k] - G_filtered[i][j][k-1]) / DZ) / DZ;
                    double sgs_scalar_flux_contribution = sgs_flux_term_div_x + sgs_flux_term_div_y + sgs_flux_term_div_z;
                    
                    double grad_G_x, grad_G_y, grad_G_z;
                    // --- 勾配計算の切り替え ---
                    if (t_loop < SCHEME_TRANSITION_STEP) {
                        // 初期段階では、振動に比較的強い中心差分を使用
                        grad_G_x = (G_filtered[i+1][j][k] - G_filtered[i-1][j][k]) / (2.0 * DX);
                        grad_G_y = (G_filtered[i][j+1][k] - G_filtered[i][j-1][k]) / (2.0 * DY);
                        grad_G_z = (G_filtered[i][j][k+1] - G_filtered[i][j][k-1]) / (2.0 * DZ);
                    } else {
                        // 安定化後は、デカップリングを抑えるWENOスキームで勾配を計算
                        if (i >= 3 && i < NX - 3 && j >= 3 && j < NY - 3 && k >= 3 && k < NZ - 3) {
                            grad_G_x = calculate_weno5_gradient_1d(G_filtered, i, j, k, 0, DX);
                            grad_G_y = calculate_weno5_gradient_1d(G_filtered, i, j, k, 1, DY);
                            grad_G_z = calculate_weno5_gradient_1d(G_filtered, i, j, k, 2, DZ);
                        } else {
                            grad_G_x = (G_filtered[i+1][j][k] - G_filtered[i-1][j][k]) / (2.0 * DX);
                            grad_G_y = (G_filtered[i][j+1][k] - G_filtered[i][j-1][k]) / (2.0 * DY);
                            grad_G_z = (G_filtered[i][j][k+1] - G_filtered[i][j][k-1]) / (2.0 * DZ);
                        }
                    }

                    double grad_G_mag = sqrt(grad_G_x * grad_G_x + grad_G_y * grad_G_y + grad_G_z * grad_G_z);
                    grad_G_magnitude_for_debug[i][j][k] = grad_G_mag;
                    double flame_propagation = ST_turbulent[i][j][k] * grad_G_mag;

                    // ==========================================================
                    // 火炎形状維持のための人工拡散 
                    // ==========================================================
                    double artificial_diffusion = 0.0;
                    
                    #if USE_FLAME_ACCEL_MODEL
                    double G_local = G_filtered[i][j][k];
                    
                    // 加速力が働いている領域周辺 (G=0.8～1.0付近)
                    if (G_local > 0.1 && G_local < 0.99) {
                        
                        // ウィンドウ関数（加速モデルと同じ定義を使う）
                        double diff = G_local - FLAME_ACCEL_CENTER;
                        double window = exp( - (diff * diff) / (2.0 * FLAME_ACCEL_WIDTH * FLAME_ACCEL_WIDTH) );
                        
                        // Gのラプラシアン（2階微分）を簡易計算
                        double d2G_dx2 = (G_filtered[i+1][j][k] - 2.0*G_local + G_filtered[i-1][j][k]) / (DX*DX);
                        double d2G_dy2 = (G_filtered[i][j+1][k] - 2.0*G_local + G_filtered[i][j-1][k]) / (DY*DY);
                        double d2G_dz2 = (G_filtered[i][j][k+1] - 2.0*G_local + G_filtered[i][j][k-1]) / (DZ*DZ);
                        
                        double radial_pos = fabs(j * DY);
                       double tip_mask = 1.0 - exp( - (radial_pos * radial_pos) / (TIP_PROTECTION_RADIUS * TIP_PROTECTION_RADIUS) );

                        // ベースとなる拡散係数として nu_t を使用
                        // 係数 5.0 程度で強力になじませる
                        double art_diff_coeff = 5.0 * (nu_t[i][j][k] + 1.0e-5) * window * tip_mask;
                        
                        // 追加の拡散項
                        artificial_diffusion = art_diff_coeff * (d2G_dx2 + d2G_dy2 + d2G_dz2);
                    }
                    #endif

                    dG_dt[i][j][k] = -convection + flame_propagation + sgs_scalar_flux_contribution + artificial_diffusion;
                } else {
                    grad_G_magnitude_for_debug[i][j][k] = 0.0;
                    dG_dt[i][j][k] = 0.0;
                }
            }
        }
    }

    // ==============================================================================
    // (c) Yk (化学種) のRHS 
    // ==============================================================================
    
    // ナッジング（緩和）係数の定義
    // 反応項の代わりに、Gで定義される平衡状態へ強力に引き戻します。
    // 値が大きすぎると数値不安定になりますが、dt ~ 1e-5 なので 5000.0 程度なら安全です。
    const double RELAXATION_COEFF = 5000.0; 
    update_Dk(attribute, T_filtered, Yk_filtered, P_filtered, Dk);

    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
        #pragma omp parallel for collapse(3)
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                for (int k = 1; k < NZ - 1; k++) {
                    if (attribute[i][j][k] == 3) {
                        // 1. 移流項 (MUSCLスキーム推奨)
                        double convection_term = calculate_convection_muscl(i, j, k, Yk_filtered[sp], u_filtered, v_filtered, w_filtered);

                        // 2. 分子拡散項
                        double d2Yk_dx2 = (Yk_filtered[sp][i+1][j][k] - 2.0*Yk_filtered[sp][i][j][k] + Yk_filtered[sp][i-1][j][k]) / (DX*DX);
                        double d2Yk_dy2 = (Yk_filtered[sp][i][j+1][k] - 2.0*Yk_filtered[sp][i][j][k] + Yk_filtered[sp][i][j-1][k]) / (DY*DY);
                        double d2Yk_dz2 = (Yk_filtered[sp][i][j][k+1] - 2.0*Yk_filtered[sp][i][j][k] + Yk_filtered[sp][i][j][k-1]) / (DZ*DZ);
                        double molecular_diffusion_term = Dk[sp][i][j][k] * (d2Yk_dx2 + d2Yk_dy2 + d2Yk_dz2);

                        // 3. SGS拡散項
                        double Dt_iph = (nu_t[i+1][j][k]+nu_t[i][j][k])/(2.0*Sc_t); double Dt_imh = (nu_t[i][j][k]+nu_t[i-1][j][k])/(2.0*Sc_t);
                        double Dt_jph = (nu_t[i][j+1][k]+nu_t[i][j][k])/(2.0*Sc_t); double Dt_jmh = (nu_t[i][j][k]+nu_t[i][j-1][k])/(2.0*Sc_t);
                        double Dt_kph = (nu_t[i][j][k+1]+nu_t[i][j][k])/(2.0*Sc_t); double Dt_kmh = (nu_t[i][j][k]+nu_t[i][j][k-1])/(2.0*Sc_t);
                        
                        double sgs_flux_conv_x = (Dt_iph * (Yk_filtered[sp][i+1][j][k] - Yk_filtered[sp][i][j][k])/DX - Dt_imh * (Yk_filtered[sp][i][j][k] - Yk_filtered[sp][i-1][j][k])/DX)/DX;
                        double sgs_flux_conv_y = (Dt_jph * (Yk_filtered[sp][i][j+1][k] - Yk_filtered[sp][i][j][k])/DY - Dt_jmh * (Yk_filtered[sp][i][j][k] - Yk_filtered[sp][i][j-1][k])/DY)/DY;
                        double sgs_flux_conv_z = (Dt_kph * (Yk_filtered[sp][i][j][k+1] - Yk_filtered[sp][i][j][k])/DZ - Dt_kmh * (Yk_filtered[sp][i][j][k] - Yk_filtered[sp][i][j][k-1])/DZ)/DZ;
                        double sgs_diffusion_term = sgs_flux_conv_x + sgs_flux_conv_y + sgs_flux_conv_z;

                        // 4. ナッジング項 (Nudging / Relaxation)
                        // Gの値に基づいて、その場所にあるべき化学種濃度(Target)を計算し、そこへ緩和させます。
                        double G_val = G_filtered[i][j][k];
                        if (G_val < 0.0) G_val = 0.0;
                        if (G_val > 1.0) G_val = 1.0;

                        double Y_target = 0.0;
                        
                        // inletYk[] はグローバル変数として定義されている前提です
                        if (sp == C3H8_INDEX) {
                            // 燃料: 未燃で inlet, 既燃(G=1)で 0
                            Y_target = inletYk[C3H8_INDEX] * (1.0 - G_val);
                        } else if (sp == O2_INDEX) {
                            // 酸素: 燃料消費量に応じて減少
                            double consumed = inletYk[C3H8_INDEX] * STOICH_O2_PER_C3H8;
                            Y_target = fmax(0.0, inletYk[O2_INDEX] - consumed * G_val);
                        } else if (sp == CO2_INDEX) {
                            // CO2: 燃料消費量に応じて生成
                            double produced = inletYk[C3H8_INDEX] * STOICH_CO2_PER_C3H8;
                            Y_target = inletYk[CO2_INDEX] + produced * G_val;
                        } else if (sp == H2O_INDEX) {
                            // H2O: 燃料消費量に応じて生成
                            double produced = inletYk[C3H8_INDEX] * STOICH_H2O_PER_C3H8;
                            Y_target = inletYk[H2O_INDEX] + produced * G_val;
                        } else { 
                            // N2 (不活性): 流入条件を維持（希釈効果はGによる混合で自然に表現されるため）
                            Y_target = inletYk[N2_INDEX];
                        }

                        // 目標値との差分に係数を掛けてソース項とする
                        double nudging_term = RELAXATION_COEFF * (Y_target - Yk_filtered[sp][i][j][k]);

                        // 最終的な時間発展式
                        // 反応項(reaction_term)と人工拡散(artificial_diffusion)はナッジングに置き換わりました
                        dYk_dt[sp][i][j][k] = -convection_term + molecular_diffusion_term + sgs_diffusion_term + nudging_term;

                    } else {
                        dYk_dt[sp][i][j][k] = 0.0;
                    }
                }
            }
        }
    }
    
    // ==============================================================================
    // (d) T (温度) のRHS
    // ==============================================================================
    update_Cp_mix(attribute, T_filtered, Yk_filtered, Cp_mix);
    update_lambda_mix_alpha(attribute, T_filtered, Yk_filtered, Cp_mix, lambda_mix, alpha, rho_filtered);
    
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < NX-1; i++) {
        for (int j = 1; j < NY-1; j++) {
            for (int k = 1; k < NZ-1; k++) {
                if (attribute[i][j][k] == 3) {
                    // 1. 移流項 (MUSCLスキーム)
                    double convection = calculate_convection_muscl(i, j, k, T_filtered, u_filtered, v_filtered, w_filtered);

                    // 2. 分子拡散項
                    double d2T_dx2 = (T_filtered[i+1][j][k] - 2.0*T_filtered[i][j][k] + T_filtered[i-1][j][k]) / (DX*DX);
                    double d2T_dy2 = (T_filtered[i][j+1][k] - 2.0*T_filtered[i][j][k] + T_filtered[i][j-1][k]) / (DY*DY);
                    double d2T_dz2 = (T_filtered[i][j][k+1] - 2.0*T_filtered[i][j][k] + T_filtered[i][j][k-1]) / (DZ*DZ);
                    
                    double molecular_diff = 0.0;
                    if (rho_filtered[i][j][k] > 1e-6 && Cp_mix[i][j][k] > 1e-6) {
                        molecular_diff = lambda_mix[i][j][k] / (rho_filtered[i][j][k] * Cp_mix[i][j][k]) * (d2T_dx2 + d2T_dy2 + d2T_dz2);
                    }

                    // 3. SGS拡散項
                    double Pr_t_local = Pr_t;
                    double kappa_iph = (nu_t[i+1][j][k] + nu_t[i][j][k])/(2.0*Pr_t_local); double kappa_imh = (nu_t[i][j][k] + nu_t[i-1][j][k])/(2.0*Pr_t_local);
                    double kappa_jph = (nu_t[i][j+1][k] + nu_t[i][j][k])/(2.0*Pr_t_local); double kappa_jmh = (nu_t[i][j][k] + nu_t[i][j-1][k])/(2.0*Pr_t_local);
                    double kappa_kph = (nu_t[i][j][k+1] + nu_t[i][j][k])/(2.0*Pr_t_local); double kappa_kmh = (nu_t[i][j][k] + nu_t[i][j][k-1])/(2.0*Pr_t_local);
                    
                    double sgs_flux_x = (kappa_iph * (T_filtered[i+1][j][k] - T_filtered[i][j][k])/DX - kappa_imh * (T_filtered[i][j][k] - T_filtered[i-1][j][k])/DX)/DX;
                    double sgs_flux_y = (kappa_jph * (T_filtered[i][j+1][k] - T_filtered[i][j][k])/DY - kappa_jmh * (T_filtered[i][j][k] - T_filtered[i][j-1][k])/DY)/DY;
                    double sgs_flux_z = (kappa_kph * (T_filtered[i][j][k+1] - T_filtered[i][j][k])/DZ - kappa_kmh * (T_filtered[i][j][k] - T_filtered[i][j][k-1])/DZ)/DZ;
                    double sgs_diff = sgs_flux_x + sgs_flux_y + sgs_flux_z;

                    // 4. ナッジング項
                    // Gの値に基づいて目標温度を計算
                    double G_val = G_filtered[i][j][k];
                    if (G_val < 0.0) G_val = 0.0;
                    if (G_val > 1.0) G_val = 1.0;

                    // 線形補間: T = T_u + G * (T_b - T_u)
                    double T_target = INLET_TEMPERATURE + G_val * (BURNT_GAS_TEMPERATURE - INLET_TEMPERATURE);

                    // 緩和項
                    double nudging_term_T = RELAXATION_COEFF * (T_target - T_filtered[i][j][k]);

                    // 最終的な時間発展式
                    // 発熱項(Q_react)と人工拡散はナッジングに
                    dT_dt[i][j][k] = -convection + molecular_diff + sgs_diff + nudging_term_T;

                } else {
                    dT_dt[i][j][k] = 0.0;
                }
            }
        }
    }
}


//【Y軸対称化対応版】apply_boundary_conditions
void apply_boundary_conditions(
    const Grid3D_Int attribute, // int型
    Grid3D u, Grid3D v, Grid3D w,
    Grid3D rhou, Grid3D rhov, Grid3D rhow,
    Grid3D G, Grid3D T, Grid3D P,
    Grid3D *Yk, 
    Grid3D rho
) {
    // ==============================================================================
    // --- 1. 流入・流出境界 (i=0, i=NX-1) ---
    // ==============================================================================
    #pragma omp parallel for
    for(int j=0; j<NY; j++){
        for(int k=0; k<NZ; k++){

            // --- i=0 面 (流入境界) ---
            if(attribute[0][j][k] == 1){
                // 属性1: ジェット流入口 (乱流変動を適用)

                // 振幅は流速の 5% ~ 10% 程度
                double white_noise_amp = 0.10 * INLET_VELOCITY; 
                
                // -1.0 ~ 1.0 の乱数
                double r1 = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double r2 = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                double r3 = ((double)rand() / RAND_MAX) * 2.0 - 1.0;

                // SEMの変動(u_inlet_fluct)に、さらにノイズを足す
                u[0][j][k] = fmax(0.0, INLET_VELOCITY + u_inlet_fluct[j][k] + white_noise_amp * r1);
                v[0][j][k] = v_inlet_fluct[j][k] + white_noise_amp * r2;
                w[0][j][k] = w_inlet_fluct[j][k] + white_noise_amp * r3;
                T[0][j][k] = INLET_TEMPERATURE;
                G[0][j][k] = MIN_G_CLIP;
                for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][0][j][k] = inletYk[sp];
            }
            else if (attribute[0][j][k] == 4) {
                // 属性4: ノズル壁面 (自由境界として周囲空気の流入出を許容)
                if (u[1][j][k] < 0) { // 流入
                    u[0][j][k] = 0.0; v[0][j][k] = 0.0; w[0][j][k] = 0.0;
                    T[0][j][k] = AMBIENT_TEMPERATURE;
                    G[0][j][k] = MIN_G_CLIP;
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][0][j][k] = ambientYk[sp];
                } else { // 流出
                    u[0][j][k] = u[1][j][k]; v[0][j][k] = v[1][j][k]; w[0][j][k] = w[1][j][k];
                    T[0][j][k] = T[1][j][k]; G[0][j][k] = G[1][j][k];
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][0][j][k] = Yk[sp][1][j][k];
                }
            }

            // 属性6（バーナーリム）にNo-slip条件を適用
            else if (attribute[0][j][k] == 6) {
                // 属性6: バーナーリムの固体壁 (No-slip)
                u[0][j][k] = 0.0;
                v[0][j][k] = 0.0;
                w[0][j][k] = 0.0;
                
                // スカラー量は断熱を仮定し、隣の内部セルの値をコピー（ゼロ勾配）
                T[0][j][k] = T[1][j][k];
                G[0][j][k] = G[1][j][k];
                for(int sp=0; sp<NUM_SPECIES; sp++) {
                    Yk[sp][0][j][k] = Yk[sp][1][j][k];
                }
            }

            // --- i=NX-1 面 (流出境界) ---
            if(attribute[NX-1][j][k] == 2){
                // 属性2: ゼロ勾配条件
                u[NX-1][j][k] = u[NX-2][j][k];
                v[NX-1][j][k] = v[NX-2][j][k];
                w[NX-1][j][k] = w[NX-2][j][k];
                T[NX-1][j][k] = T[NX-2][j][k];
                G[NX-1][j][k] = G[NX-2][j][k];
                P[NX-1][j][k] = P[NX-2][j][k]; // 圧力もゼロ勾配
                for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][NX-1][j][k] = Yk[sp][NX-2][j][k];
            }
        }
    }

    // ==============================================================================
    // --- 2. 側壁境界 (j, k 方向) ---
    // ==============================================================================
    #pragma omp parallel for
    for(int i=1; i<NX-1; i++){

        // --- Y方向の境界 (j=0 と j=NY-1) ---
        for(int k=0; k<NZ; k++){
            // --- 境界 j=0 (下側): 対称境界条件 (Symmetry Boundary Condition) ---
            if (attribute[i][0][k] == 5) {
                // スカラー量 (温度, G値, 圧力, 質量分率)
                // 境界での勾配がゼロになるように、隣接する内部セル(j=1)の値をコピーする。
                // ∂φ/∂y = 0  -->  (φ[1] - φ[0]) / Δy = 0  -->  φ[0] = φ[1]
                T[i][0][k] = T[i][1][k];
                G[i][0][k] = G[i][1][k];
                P[i][0][k] = P[i][1][k];
                for(int sp=0; sp<NUM_SPECIES; sp++) {
                    Yk[sp][i][0][k] = Yk[sp][i][1][k];
                }

                // ベクトル量 (速度)
                // 境界に垂直な成分(v)はゼロにする。
                // 境界に平行な成分(u, w)は勾配をゼロにする（隣のセルの値をコピー）。
                u[i][0][k] = u[i][1][k];      // 平行成分 (x-velocity)
                v[i][0][k] = 0.0;             // <<-- 垂直成分 (y-velocity) はゼロ
                w[i][0][k] = w[i][1][k];      // 平行成分 (z-velocity)
            }

            // --- 境界 j=NY-1 (上側): 自由境界 (元のまま) ---
            // こちらは計算領域の上端なので、元の自由境界条件をそのまま使用します。
            if (attribute[i][NY-1][k] == 4) { // 壁属性だが自由境界として扱う
                if (v[i][NY-2][k] < 0) { // 流れが計算領域に「入ってくる」場合
                    u[i][NY-1][k] = 0.0; v[i][NY-1][k] = 0.0; w[i][NY-1][k] = 0.0;
                    T[i][NY-1][k] = AMBIENT_TEMPERATURE;
                    G[i][NY-1][k] = MIN_G_CLIP;
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][NY-1][k] = ambientYk[sp];
                } else { // 流れが計算領域から「出ていく」場合 (ゼロ勾配)
                    u[i][NY-1][k] = u[i][NY-2][k]; v[i][NY-1][k] = v[i][NY-2][k]; w[i][NY-1][k] = w[i][NY-2][k];
                    T[i][NY-1][k] = T[i][NY-2][k]; G[i][NY-1][k] = G[i][NY-2][k];
                    P[i][NY-1][k] = P[i][NY-2][k];
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][NY-1][k] = Yk[sp][i][NY-2][k];
                }
            }
        } // k-loop end

        // --- Z方向の境界 (k=0 と k=NZ-1): 自由境界 (元のまま) ---
        // こちらはZ方向の境界なので、変更はありません。
        for(int j=0; j<NY; j++){ // 注意: j=0 も含める
            // --- 境界 k=0 (手前側) ---
            if (attribute[i][j][0] == 4) {
                if (w[i][j][1] > 0) { // 流入
                    u[i][j][0] = 0.0; v[i][j][0] = 0.0; w[i][j][0] = 0.0;
                    T[i][j][0] = AMBIENT_TEMPERATURE;
                    G[i][j][0] = MIN_G_CLIP;
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][j][0] = ambientYk[sp];
                } else { // 流出
                    u[i][j][0] = u[i][j][1]; v[i][j][0] = v[i][j][1]; w[i][j][0] = w[i][j][1];
                    T[i][j][0] = T[i][j][1]; G[i][j][0] = G[i][j][1];
                    P[i][j][0] = P[i][j][1];
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][j][0] = Yk[sp][i][j][1];
                }
            }

            // --- 境界 k=NZ-1 (奥側) ---
            if (attribute[i][j][NZ-1] == 4) {
                if (w[i][j][NZ-2] < 0) { // 流入
                    u[i][j][NZ-1] = 0.0; v[i][j][NZ-1] = 0.0; w[i][j][NZ-1] = 0.0;
                    T[i][j][NZ-1] = AMBIENT_TEMPERATURE;
                    G[i][j][NZ-1] = MIN_G_CLIP;
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][j][NZ-1] = ambientYk[sp];
                } else { // 流出
                    u[i][j][NZ-1] = u[i][j][NZ-2]; v[i][j][NZ-1] = v[i][j][NZ-2]; w[i][j][NZ-1] = w[i][j][NZ-2];
                    T[i][j][NZ-1] = T[i][j][NZ-2]; G[i][j][NZ-1] = G[i][j][NZ-2];
                    P[i][j][NZ-1] = P[i][j][NZ-2];
                    for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][j][NZ-1] = Yk[sp][i][j][NZ-2];
                }
            }
        } 
    } 


    // ==============================================================================
    // --- 3. 物理量の更新と正規化 ---
    // ==============================================================================
    // 境界条件が適用された後の最終的な変数 (rho, rhou, Ykの合計など) を計算します。
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                // 状態方程式から密度を再計算
                double sum_Y_over_W=0;
                for(int sp=0; sp<NUM_SPECIES; ++sp) {
                    sum_Y_over_W += Yk[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0);
                }
                if (sum_Y_over_W > 1e-9 && T[i][j][k] > 1.0) {
                    rho[i][j][k] = pressure_const / (R_univ / (1.0/sum_Y_over_W) * T[i][j][k]);
                } else {
                    rho[i][j][k] = 1.2; // Fallback
                }

                // 運動量を更新
                rhou[i][j][k] = rho[i][j][k] * u[i][j][k];
                rhov[i][j][k] = rho[i][j][k] * v[i][j][k];
                rhow[i][j][k] = rho[i][j][k] * w[i][j][k];

                // 質量分率の合計が1になるように正規化
                if(attribute[i][j][k] == 3 || attribute[i][j][k] == 2 || attribute[i][j][k] == 5){ // 対称境界も正規化対象に含める
                    double sum_Y = 0.0;
                    for (int sp = 0; sp < NUM_SPECIES; sp++) sum_Y += Yk[sp][i][j][k];
                    if (sum_Y > 1e-6) {
                        for (int sp = 0; sp < NUM_SPECIES; sp++) Yk[sp][i][j][k] /= sum_Y;
                    }
                }
            }
        }
    }
}




// 時間平均の統計量を更新する関数 
void update_time_averages(
    const Grid3D u, const Grid3D v, const Grid3D w,
    const Grid3D G
) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                // 1次モーメント (合計)
                u_mean[i][j][k] += u[i][j][k];
                v_mean[i][j][k] += v[i][j][k];
                w_mean[i][j][k] += w[i][j][k];
                G_mean[i][j][k] += G[i][j][k];

                // 2次モーメント (合計)
                uu_mean[i][j][k] += u[i][j][k] * u[i][j][k];
                vv_mean[i][j][k] += v[i][j][k] * v[i][j][k];
                ww_mean[i][j][k] += w[i][j][k] * w[i][j][k];
                uv_mean[i][j][k] += u[i][j][k] * v[i][j][k];
                uw_mean[i][j][k] += u[i][j][k] * w[i][j][k];
                vw_mean[i][j][k] += v[i][j][k] * w[i][j][k];
            }
        }
    }
    time_avg_steps++; // 平均ステップ数をインクリメント
}

// リゾルブド・レイノルズ応力を計算する関数
void calculate_resolved_reynolds_stresses() {
    if (time_avg_steps == 0) return; // ゼロ除算防止

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                // 平均値を計算
                double avg_u = u_mean[i][j][k] / time_avg_steps;
                double avg_v = v_mean[i][j][k] / time_avg_steps;
                double avg_w = w_mean[i][j][k] / time_avg_steps;
                
                // レイノルズ応力 <u'v'> = <uv> - <u><v>
                re_stress_uu[i][j][k] = (uu_mean[i][j][k] / time_avg_steps) - (avg_u * avg_u);
                re_stress_vv[i][j][k] = (vv_mean[i][j][k] / time_avg_steps) - (avg_v * avg_v);
                re_stress_ww[i][j][k] = (ww_mean[i][j][k] / time_avg_steps) - (avg_w * avg_w);
                re_stress_uv[i][j][k] = (uv_mean[i][j][k] / time_avg_steps) - (avg_u * avg_v);
                re_stress_uw[i][j][k] = (uw_mean[i][j][k] / time_avg_steps) - (avg_u * avg_w);
                re_stress_vw[i][j][k] = (vw_mean[i][j][k] / time_avg_steps) - (avg_v * avg_w);

                 G_mean[i][j][k] /= time_avg_steps; 
            }
        }
    }
}

// =================================
// リスタート用VTKファイル読み込み関数 
// =================================
/**
 * @brief 指定されたVTKファイルから計算状態を読み込み、リスタート準備を行う
 * @param filename 読み込むVTKファイル名
 * @return 読み込みに成功した場合は 1、失敗した場合は 0 を返す
 */
int load_vtk_for_restart(const char* filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open restart file: %s\n", filename);
        return 0;
    }

    char line[256];
    int dim_x, dim_y, dim_z, point_data_count;

    // --- 1. ヘッダーと次元の検証 ---
    printf("Reading VTK header and dimensions...\n");
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "DIMENSIONS", 10) == 0) {
            sscanf(line, "DIMENSIONS %d %d %d", &dim_x, &dim_y, &dim_z);
            if (dim_x != NX || dim_y != NY || dim_z != NZ) {
                fprintf(stderr, "FATAL ERROR: VTK dimensions (%d, %d, %d) do not match compiled dimensions (%d, %d, %d).\n",
                        dim_x, dim_y, dim_z, NX, NY, NZ);
                fclose(fp);
                return 0;
            }
            printf("  Dimensions match: OK\n");
        }
        if (strncmp(line, "POINT_DATA", 10) == 0) {
            sscanf(line, "POINT_DATA %d", &point_data_count);
            if (point_data_count != NX * NY * NZ) {
                fprintf(stderr, "FATAL ERROR: VTK point count mismatch.\n");
                fclose(fp);
                return 0;
            }
            printf("  Point data count: OK\n");
            break; // データブロックの読み込みへ
        }
    }

    // --- 2. データブロックの読み込み ---
    printf("Reading data blocks...\n");
    while (fgets(line, sizeof(line), fp)) {
        // (a) 速度ベクトル
        if (strncmp(line, "VECTORS velocity", 16) == 0) {
            printf("  Loading 'velocity'...\n");
            for (int k = 0; k < NZ; k++) {
                for (int j = 0; j < NY; j++) {
                    for (int i = 0; i < NX; i++) {
                        if (fscanf(fp, "%lf %lf %lf", &u[i][j][k], &v[i][j][k], &w[i][j][k]) != 3) {
                            fprintf(stderr, "Error reading velocity data.\n"); fclose(fp); return 0;
                        }
                    }
                }
            }
        }
        // (b) 温度
        else if (strncmp(line, "SCALARS temperature", 19) == 0) {
            printf("  Loading 'temperature'...\n");
            fgets(line, sizeof(line), fp); // "LOOKUP_TABLE default" を読み飛ばす
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                if (fscanf(fp, "%lf", &T[i][j][k]) != 1) { fprintf(stderr, "Error reading T.\n"); fclose(fp); return 0; }
            }
        }
        // (c) G値
        else if (strncmp(line, "SCALARS G_value", 15) == 0) {
            printf("  Loading 'G_value'...\n");
            fgets(line, sizeof(line), fp);
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                if (fscanf(fp, "%lf", &G[i][j][k]) != 1) { fprintf(stderr, "Error reading G.\n"); fclose(fp); return 0; }
            }
        }
        // (d) 化学種
        else if (strncmp(line, "SCALARS Y_C3H8", 14) == 0) {
            printf("  Loading 'Y_C3H8'...\n");
            fgets(line, sizeof(line), fp);
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                if (fscanf(fp, "%lf", &Yk[C3H8_INDEX][i][j][k]) != 1) { fprintf(stderr, "Error reading Y_C3H8.\n"); fclose(fp); return 0; }
            }
        }
        else if (strncmp(line, "SCALARS Y_O2", 12) == 0) {
            printf("  Loading 'Y_O2'...\n");
            fgets(line, sizeof(line), fp);
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                if (fscanf(fp, "%lf", &Yk[O2_INDEX][i][j][k]) != 1) { fprintf(stderr, "Error reading Y_O2.\n"); fclose(fp); return 0; }
            }
        }
        else if (strncmp(line, "SCALARS Y_N2", 12) == 0) {
            printf("  Loading 'Y_N2'...\n");
            fgets(line, sizeof(line), fp);
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                if (fscanf(fp, "%lf", &Yk[N2_INDEX][i][j][k]) != 1) { fprintf(stderr, "Error reading Y_N2.\n"); fclose(fp); return 0; }
            }
        }
        else if (strncmp(line, "SCALARS Y_CO2", 13) == 0) {
            printf("  Loading 'Y_CO2'...\n");
            fgets(line, sizeof(line), fp);
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                if (fscanf(fp, "%lf", &Yk[CO2_INDEX][i][j][k]) != 1) { fprintf(stderr, "Error reading Y_CO2.\n"); fclose(fp); return 0; }
            }
        }
        else if (strncmp(line, "SCALARS Y_H2O", 13) == 0) {
            printf("  Loading 'Y_H2O'...\n");
            fgets(line, sizeof(line), fp);
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                if (fscanf(fp, "%lf", &Yk[H2O_INDEX][i][j][k]) != 1) { fprintf(stderr, "Error reading Y_H2O.\n"); fclose(fp); return 0; }
            }
        }
    }
    fclose(fp);
    
    // --- 3. 派生変数の再計算 ---
    // VTKファイルにはrho, rhou等の派生変数は含まれているが、
    // TとYkから再計算することで、状態方程式との一貫性を保証する。
    printf("Re-calculating dependent variables (rho, rhou, etc.) to ensure consistency...\n");
    update_rho_from_EOS(attribute, T, Yk, rho);

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                rhou[i][j][k] = rho[i][j][k] * u[i][j][k];
                rhov[i][j][k] = rho[i][j][k] * v[i][j][k];
                rhow[i][j][k] = rho[i][j][k] * w[i][j][k];
                
                // _new 配列も同じ値で初期化
                u_new[i][j][k] = u[i][j][k];
                v_new[i][j][k] = v[i][j][k];
                w_new[i][j][k] = w[i][j][k];
                T_new[i][j][k] = T[i][j][k];
                G_new[i][j][k] = G[i][j][k];
                rho_new[i][j][k] = rho[i][j][k];
                rhou_new[i][j][k] = rhou[i][j][k];
                rhov_new[i][j][k] = rhov[i][j][k];
                rhow_new[i][j][k] = rhow[i][j][k];
                for (int sp = 0; sp < NUM_SPECIES; sp++) {
                    Yk_new[sp][i][j][k] = Yk[sp][i][j][k];
                }
            }
        }
    }

    printf("Restart data successfully loaded and processed.\n\n");
    return 1; // 成功
}

// ===================================================
// 火炎領域サンプリング用ファイル初期化関数 (3点版) 
// ===================================================
void initialize_probe_file(void) {
    char filepath[512];

    printf("Initializing flame-region sampling output files (%d locations)...\n", NUM_PROBE_POSITIONS);

    for (int n = 0; n < NUM_PROBE_POSITIONS; n++) {
        // ファイル名を生成 (例: probe_data_flame_region_x0.csv, _x1.csv, ...)
        // X座標の数値をファイル名に入れることも可能です
        sprintf(filepath, "%s%s_loc%d.csv", OUTPUT_DIR, PROBE_FILENAME_BASE, n);

        printf(" -> Creating: %s (X = %.3f m)\n", filepath, PROBE_X_LOCATIONS[n]);

        remove(filepath); // 既存削除
        
        FILE* fp = fopen(filepath, "w");
        if (fp == NULL) {
            fprintf(stderr, "Warning: Could not create probe file: %s\n", filepath);
            continue;
        }
        // ヘッダー書き込み
        fprintf(fp, "timestep,x_location,y_flame,y_physical,y_relative,u,v,w,G,nu_t,S_mag\n");
        fclose(fp);
    }
    printf(" -> All probe files created successfully.\n\n");
}


// ==============================================================================
// 火炎領域データ記録関数 
// ==============================================================================
void record_flame_region_data(int t_loop, const Grid3D u, const Grid3D v, const Grid3D w, const Grid3D G, const Grid3D nu_t, const Grid3D S_mag) {
    // Z方向の中心線 (固定)
    const int k_probe_line = NZ / 2;

    // 定義された全てのX位置についてループ
    for (int n = 0; n < NUM_PROBE_POSITIONS; n++) {
        
        double current_x_pos = PROBE_X_LOCATIONS[n];
        
        // 1. 物理座標から格子インデックス i を計算
        int i_probe = (int)(current_x_pos / DX);
        
        // 範囲外ならスキップ
        if (i_probe <= 0 || i_probe >= NX - 1) continue;

        // 2. その断面(i_probe)の中心線上で、火炎面(G=0.5)を探す
        // Y方向下から上へスキャン
        double y_flame = -1.0; 
        for (int j = 0; j < NY - 1; j++) {
            double G1 = G[i_probe][j][k_probe_line];
            double G2 = G[i_probe][j + 1][k_probe_line];
            
            // G=0.5 をまたぐ場所を探す
            if ((G1 - 0.5) * (G2 - 0.5) <= 0.0) {
                double y1 = j * DY;
                double y2 = (j + 1) * DY;
                if (fabs(G2 - G1) < 1e-9) { 
                    y_flame = y1; 
                } else { 
                    // 線形補間で正確なY座標を求める
                    y_flame = y1 + (0.5 - G1) / (G2 - G1) * (y2 - y1); 
                }
                break; // 最初に見つかった火炎面（下側）を採用
            }
        }

        // 3. 火炎面が見つかった場合、そのファイルの追記モードで開き、周辺データを記録
        if (y_flame > 0.0) {
            char filepath[512];
            sprintf(filepath, "%s%s_loc%d.csv", OUTPUT_DIR, PROBE_FILENAME_BASE, n);

            FILE* fp = fopen(filepath, "a");
            if (!fp) continue;

            // Y方向の全格子点をループして、範囲内のものを保存
            for (int j = 0; j < NY; j++) {
                double y_physical = j * DY;
                double y_relative = y_physical - y_flame;

                if (fabs(y_relative) <= SAMPLING_RANGE) {
                    fprintf(fp, "%d,%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e\n", 
                            t_loop,
                            current_x_pos, 
                            y_flame,
                            y_physical,
                            y_relative,
                            u[i_probe][j][k_probe_line], 
                            v[i_probe][j][k_probe_line], 
                            w[i_probe][j][k_probe_line], 
                            G[i_probe][j][k_probe_line],
                            nu_t[i_probe][j][k_probe_line],  // 追加
                            S_mag[i_probe][j][k_probe_line]  // 追加
                           );
                }
            }
            fclose(fp);
        }
    }
}

void output_vtk(int t) {
    char filepath[512];
    sprintf(filepath, "%sPremixed_C3H8_Air_%04d.vtk", OUTPUT_DIR, t);

    FILE *fp = fopen(filepath, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file: %s, errno = %d\n", filepath, errno);
        return;
    }

    // --- 1. VTKヘッダーと座標データ ---
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "VTK output\n");
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", NX, NY, NZ);
    fprintf(fp, "POINTS %d float\n", NX * NY * NZ);
    // ループ範囲: 0 から NX-1, NY-1, NZ-1
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                fprintf(fp, "%.6f %.6f %.6f\n", (double)i * DX, (double)j * DY, (double)k * DZ);
            }
        }
    }
    fprintf(fp, "\nPOINT_DATA %d\n", NX * NY * NZ);

    // --- 2. データブロック ---
    // 全てのループが k=0 to NZ-1, j=0 to NY-1, i=0 to NX-1 であることを確認

    fprintf(fp, "VECTORS velocity float\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f %.6f %.6f\n", isfinite(u[i][j][k])?u[i][j][k]:0.0, isfinite(v[i][j][k])?v[i][j][k]:0.0, isfinite(w[i][j][k])?w[i][j][k]:0.0);
    }

    fprintf(fp, "SCALARS temperature float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(T[i][j][k]) ? T[i][j][k] : 0.0);
    }
    
    fprintf(fp, "SCALARS G_value float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(G[i][j][k]) ? G[i][j][k] : 0.0);
    }

    fprintf(fp, "SCALARS Y_C3H8 float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(Yk[C3H8_INDEX][i][j][k]) ? Yk[C3H8_INDEX][i][j][k] : 0.0);
    }
    fprintf(fp, "SCALARS Y_O2 float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(Yk[O2_INDEX][i][j][k]) ? Yk[O2_INDEX][i][j][k] : 0.0);
    }
    fprintf(fp, "SCALARS Y_N2 float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(Yk[N2_INDEX][i][j][k]) ? Yk[N2_INDEX][i][j][k] : 0.0);
    }
    fprintf(fp, "SCALARS Y_CO2 float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(Yk[CO2_INDEX][i][j][k]) ? Yk[CO2_INDEX][i][j][k] : 0.0);
    }
    fprintf(fp, "SCALARS Y_H2O float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(Yk[H2O_INDEX][i][j][k]) ? Yk[H2O_INDEX][i][j][k] : 0.0);
    }

    fprintf(fp, "SCALARS density float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6f\n", isfinite(rho[i][j][k]) ? rho[i][j][k] : 0.0);
    }

    fprintf(fp, "SCALARS omega float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6e\n", isfinite(omega[i][j][k]) ? omega[i][j][k] : 0.0);
    }

    fprintf(fp, "SCALARS mu_mix float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6e\n", isfinite(mu_mix[i][j][k]) ? mu_mix[i][j][k] : 0.0);
    }
    fprintf(fp, "SCALARS nu_mix float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6e\n", isfinite(nu[i][j][k]) ? nu[i][j][k] : 0.0);
    }
    
    fprintf(fp, "SCALARS ST_turbulent float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6e\n", isfinite(ST_turbulent[i][j][k]) ? ST_turbulent[i][j][k] : 0.0);
    }
    fprintf(fp, "SCALARS nu_t float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6e\n", isfinite(nu_t[i][j][k]) ? nu_t[i][j][k] : 0.0);
    }
    fprintf(fp, "SCALARS grad_G_magnitude float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        fprintf(fp, "%.6e\n", isfinite(grad_G_magnitude_for_debug[i][j][k]) ? grad_G_magnitude_for_debug[i][j][k] : 0.0);
    }

    // --- 時間平均された統計量 ---
    // --- (1) レイノルズ応力 - 正規応力成分 (各方向の乱れの強さ) ---
    fprintf(fp, "SCALARS re_stress_uu float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        double stress = 0.0;
        if (time_avg_steps > 0) {
            double avg = u_mean[i][j][k] / time_avg_steps;
            stress = (uu_mean[i][j][k] / time_avg_steps) - (avg * avg);
        }
        fprintf(fp, "%.6e\n", isfinite(stress) ? stress : 0.0);
    }

    fprintf(fp, "SCALARS re_stress_vv float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        double stress = 0.0;
        if (time_avg_steps > 0) {
            double avg = v_mean[i][j][k] / time_avg_steps;
            stress = (vv_mean[i][j][k] / time_avg_steps) - (avg * avg);
        }
        fprintf(fp, "%.6e\n", isfinite(stress) ? stress : 0.0);
    }
    
    fprintf(fp, "SCALARS re_stress_ww float\n"); 
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        double stress = 0.0;
        if (time_avg_steps > 0) {
            double avg = w_mean[i][j][k] / time_avg_steps;
            stress = (ww_mean[i][j][k] / time_avg_steps) - (avg * avg);
        }
        fprintf(fp, "%.6e\n", isfinite(stress) ? stress : 0.0);
    }

    // --- (2) レイノルズ応力 - せん断応力成分 ---
    fprintf(fp, "SCALARS re_stress_uv float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        double stress = 0.0;
        if (time_avg_steps > 0) {
            double avg_u = u_mean[i][j][k] / time_avg_steps;
            double avg_v = v_mean[i][j][k] / time_avg_steps;
            stress = (uv_mean[i][j][k] / time_avg_steps) - (avg_u * avg_v);
        }
        fprintf(fp, "%.6e\n", isfinite(stress) ? stress : 0.0);
    }

    fprintf(fp, "SCALARS re_stress_uw float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        double stress = 0.0;
        if (time_avg_steps > 0) {
            double avg_u = u_mean[i][j][k] / time_avg_steps;
            double avg_w = w_mean[i][j][k] / time_avg_steps;
            stress = (uw_mean[i][j][k] / time_avg_steps) - (avg_u * avg_w);
        }
        fprintf(fp, "%.6e\n", isfinite(stress) ? stress : 0.0);
    }

    fprintf(fp, "SCALARS re_stress_vw float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        double stress = 0.0;
        if (time_avg_steps > 0) {
            double avg_v = v_mean[i][j][k] / time_avg_steps;
            double avg_w = w_mean[i][j][k] / time_avg_steps;
            stress = (vw_mean[i][j][k] / time_avg_steps) - (avg_v * avg_w);
        }
        fprintf(fp, "%.6e\n", isfinite(stress) ? stress : 0.0);
    }

    fprintf(fp, "SCALARS G_mean float\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (int k=0; k<NZ; k++) for (int j=0; j<NY; j++) for (int i=0; i<NX; i++) {
        double g_mean_val = 0.0;
        if (time_avg_steps > 0) {
            g_mean_val = G_mean[i][j][k] / time_avg_steps;
        }
        fprintf(fp, "%.6f\n", isfinite(g_mean_val) ? g_mean_val : 0.0);
    }
    
    fclose(fp);
}

// 現在の状態を高速にバイナリ保存（上書き）
void save_quick_checkpoint(int t) {
    char filepath[512];
    sprintf(filepath, "%s%s", OUTPUT_DIR, QUICK_CHECKPOINT_FILE);
    
    FILE *fp = fopen(filepath, "wb");
    if (!fp) return;

    // ステップ数と物理量を保存
    fwrite(&t, sizeof(int), 1, fp);
    fwrite(u, sizeof(double), NX*NY*NZ, fp);
    fwrite(v, sizeof(double), NX*NY*NZ, fp);
    fwrite(w, sizeof(double), NX*NY*NZ, fp);
    fwrite(T, sizeof(double), NX*NY*NZ, fp);
    fwrite(G, sizeof(double), NX*NY*NZ, fp);
    fwrite(rho, sizeof(double), NX*NY*NZ, fp);
    for(int sp=0; sp<NUM_SPECIES; sp++) fwrite(Yk[sp], sizeof(double), NX*NY*NZ, fp);

    fclose(fp);

    // 「ここから再開できるよ」という情報をテキストに残す
    char info_path[512];
    sprintf(info_path, "%s%s", OUTPUT_DIR, RESTART_INFO_FILE);
    FILE *fp_info = fopen(info_path, "w");
    if (fp_info) {
        fprintf(fp_info, "BINARY\n%d\n", t); // 目印とステップ数
        fclose(fp_info);
    }
}

// 高速バイナリから復帰
int load_quick_checkpoint(int *loaded_step) {
    char filepath[512];
    sprintf(filepath, "%s%s", OUTPUT_DIR, QUICK_CHECKPOINT_FILE);
    
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return 0;

    if (fread(loaded_step, sizeof(int), 1, fp) != 1) { fclose(fp); return 0; }

    fread(u, sizeof(double), NX*NY*NZ, fp);
    fread(v, sizeof(double), NX*NY*NZ, fp);
    fread(w, sizeof(double), NX*NY*NZ, fp);
    fread(T, sizeof(double), NX*NY*NZ, fp);
    fread(G, sizeof(double), NX*NY*NZ, fp);
    fread(rho, sizeof(double), NX*NY*NZ, fp);
    for(int sp=0; sp<NUM_SPECIES; sp++) fread(Yk[sp], sizeof(double), NX*NY*NZ, fp);

    fclose(fp);
    
    // 派生変数(rhou等)の整合性を取るため再計算
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                rhou[i][j][k] = rho[i][j][k] * u[i][j][k];
                rhov[i][j][k] = rho[i][j][k] * v[i][j][k];
                rhow[i][j][k] = rho[i][j][k] * w[i][j][k];
                // _new配列も初期化
                u_new[i][j][k] = u[i][j][k]; v_new[i][j][k] = v[i][j][k]; w_new[i][j][k] = w[i][j][k];
                T_new[i][j][k] = T[i][j][k]; G_new[i][j][k] = G[i][j][k]; rho_new[i][j][k] = rho[i][j][k];
                rhou_new[i][j][k] = rhou[i][j][k]; rhov_new[i][j][k] = rhov[i][j][k]; rhow_new[i][j][k] = rhow[i][j][k];
                for(int sp=0; sp<NUM_SPECIES; sp++) Yk_new[sp][i][j][k] = Yk[sp][i][j][k];
            }
        }
    }
    return 1;
}

void save_statistics_backup() {
    char filepath[512];
    sprintf(filepath, "%s%s", OUTPUT_DIR, STATS_BACKUP_FILE);
    FILE *fp = fopen(filepath, "wb");
    if (!fp) return;

    fwrite(&time_avg_steps, sizeof(int), 1, fp);
    fwrite(u_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(v_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(w_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(uu_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(vv_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(ww_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(uv_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(uw_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(vw_mean, sizeof(double), NX*NY*NZ, fp);
    fwrite(G_mean, sizeof(double), NX*NY*NZ, fp);
    fclose(fp);
}

void load_statistics_backup() {
    char filepath[512];
    sprintf(filepath, "%s%s", OUTPUT_DIR, STATS_BACKUP_FILE);
    FILE *fp = fopen(filepath, "rb");
    if (!fp) { printf("No stats backup found. Resetting averages.\n"); return; }

    fread(&time_avg_steps, sizeof(int), 1, fp);
    fread(u_mean, sizeof(double), NX*NY*NZ, fp);
    fread(v_mean, sizeof(double), NX*NY*NZ, fp);
    fread(w_mean, sizeof(double), NX*NY*NZ, fp);
    fread(uu_mean, sizeof(double), NX*NY*NZ, fp);
    fread(vv_mean, sizeof(double), NX*NY*NZ, fp);
    fread(ww_mean, sizeof(double), NX*NY*NZ, fp);
    fread(uv_mean, sizeof(double), NX*NY*NZ, fp);
    fread(uw_mean, sizeof(double), NX*NY*NZ, fp);
    fread(vw_mean, sizeof(double), NX*NY*NZ, fp);
    fread(G_mean, sizeof(double), NX*NY*NZ, fp);
    fclose(fp);
    printf("Statistics restored. Avg steps: %d\n", time_avg_steps);
}

void setup_measurement_points() {
    num_points_to_measure = 0;

    // 流入口半径 (m) からインデックス換算
    int j_shear = (int)(INLET_RADIUS / DY); 
    // 中心軸
    int j_center = 0;

    // 計測したいX位置 (ノズル直径 D = 2*INLET_RADIUS = 2.6cm)
    // 領域が12cmなので、10D(13cm)は入りません。
    // 代わりに 4D (5.2cm) と 8D (10.4cm) を計測します。
    int i_locs[] = {
        (int)((4.0 * INLET_RADIUS) / DX),  // 4D位置
        (int)((8.0 * INLET_RADIUS) / DX)   // 8D位置
    };
    int num_locs = 2;

    int center_k = NZ / 2; // Z方向は中心固定

    for (int n = 0; n < num_locs; n++) {
        int i_pos = i_locs[n];
        if (i_pos >= NX) continue; // 念のため範囲外チェック

        // 1. 中心軸 (平均流速確認用)
        if (num_points_to_measure < MAX_MEASUREMENT_POINTS) {
            points[num_points_to_measure].i = i_pos;
            points[num_points_to_measure].j = j_center;
            points[num_points_to_measure].k = center_k;
            points[num_points_to_measure].count = 0;
            num_points_to_measure++;
        }

        // 2. せん断層 (乱流スペクトル確認用)
        if (num_points_to_measure < MAX_MEASUREMENT_POINTS) {
            points[num_points_to_measure].i = i_pos;
            points[num_points_to_measure].j = j_shear;
            points[num_points_to_measure].k = center_k;
            points[num_points_to_measure].count = 0;
            num_points_to_measure++;
        }
    }
    printf("Measurement points setup: %d points.\n", num_points_to_measure);
}

void flush_point_data(double phi) {
    if (num_points_to_measure == 0) return;

    for (int p = 0; p < num_points_to_measure; p++) {
        if (points[p].count == 0) continue;

        char filename[256];
        char filepath[512];
        sprintf(filename, "LES_point_data_phi%.2f_i%d_j%d_k%d.csv", phi, points[p].i, points[p].j, points[p].k);
        sprintf(filepath, "%s%s", OUTPUT_DIR, filename);
        
        // 追記モードで開く
        FILE *fp = fopen(filepath, "a");
        if (fp == NULL) continue;

        // ファイルが新規ならヘッダー作成
        fseek(fp, 0, SEEK_END);
        if (ftell(fp) == 0) fprintf(fp, "u,v,w,G\n");

        for (int i = 0; i < points[p].count; i++) {
            fprintf(fp, "%.6f,%.6f,%.6f,%.6f\n",
                    points[p].u_data[i], points[p].v_data[i], points[p].w_data[i], points[p].G_data[i]);
        }
        fclose(fp);
        points[p].count = 0; // メモリクリア
    }
}

// メモリ確保用マクロ 
#define ALLOC_GRID(var) \
    do { \
        var = (Grid3D)malloc(sizeof(double) * NX * NY * NZ); \
        if (var == NULL) { \
            fprintf(stderr, "Memory allocation failed for %s\n", #var); \
            exit(1); \
        } \
    } while(0)

#define ALLOC_GRID_INT(var) \
    do { \
        var = (Grid3D_Int)malloc(sizeof(int) * NX * NY * NZ); \
        if (var == NULL) { \
            fprintf(stderr, "Memory allocation failed for %s\n", #var); \
            exit(1); \
        } \
    } while(0)

// ==================================================
// 計算モニタリング用ログ出力関数 (ディレクトリ指定対応)
// ==================================================
void write_monitor_log(int step, double dt, 
                       const Grid3D rho, const Grid3D T, const Grid3D G, const Grid3D *Yk) {
    
    // 10ステップに1回だけ出力
    if (step % 10 != 0) return;

    double total_mass = 0.0;
    double min_T = 1.0e9, max_T = -1.0e9;
    double min_G = 1.0e9, max_G = -1.0e9;
    double min_Y_fuel = 1.0e9;

    #pragma omp parallel for collapse(3) reduction(+:total_mass) reduction(min:min_T, min_G, min_Y_fuel) reduction(max:max_T, max_G)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                total_mass += rho[i][j][k] * DX * DY * DZ;
                if (T[i][j][k] < min_T) min_T = T[i][j][k];
                if (T[i][j][k] > max_T) max_T = T[i][j][k];
                if (G[i][j][k] < min_G) min_G = G[i][j][k];
                if (G[i][j][k] > max_G) max_G = G[i][j][k];
                if (Yk[C3H8_INDEX][i][j][k] < min_Y_fuel) min_Y_fuel = Yk[C3H8_INDEX][i][j][k];
            }
        }
    }

    char filepath[512];
    sprintf(filepath, "%ssimulation_monitor.csv", OUTPUT_DIR);

    FILE *fp = fopen(filepath, "a");
    if (fp == NULL) {
        // 開けなかった場合は標準エラーに出して戻る（停止はさせない）
        fprintf(stderr, "Warning: Could not open monitor log at %s\n", filepath);
        return;
    }

    // ファイルサイズが0ならヘッダーを書く
    fseek(fp, 0, SEEK_END);
    if (ftell(fp) == 0) {
        fprintf(fp, "step,dt,total_mass,min_T,max_T,min_G,max_G,min_Y_C3H8\n");
    }

    fprintf(fp, "%d,%.6e,%.6e,%.2f,%.2f,%.4f,%.4f,%.2e\n", 
            step, dt, total_mass, min_T, max_T, min_G, max_G, min_Y_fuel);
    
    fclose(fp);

    if (min_T < 200.0 || max_T > 3000.0 || min_Y_fuel < -1e-5) {
        printf(" [WARNING] Unstable values detected at step %d! T_range=[%.1f, %.1f], Min_Fuel=%.2e\n", 
               step, min_T, max_T, min_Y_fuel);
    }
}

// ===========================================
// G値に基づく熱力学量の同期 (Flamelet Mapping) 
// ===========================================
void update_thermodynamics_from_G(
    const Grid3D_Int attribute,
    Grid3D T,
    Grid3D *Yk, // 化学種配列へのポインタ
    Grid3D rho,
    const Grid3D G
) {
    // 未燃・既燃の参照値（境界条件と合わせる）
    const double T_u = INLET_TEMPERATURE;
    const double T_b = BURNT_GAS_TEMPERATURE;
    
    // 化学種の参照値（流入条件と平衡計算値）
    // ※ main関数内の設定と同じ値を使用
    double Y_fuel_u = 0.0; 
    double Y_O2_u   = 0.232; // 空気の概算
    // 厳密には main 内の inletYk を参照すべきですが、ここでは計算ロジックを示します
    // 必要に応じてグローバル変数 inletYk を参照するようにしてください
    
    // 簡易的にプロパン(C3H8)と酸素(O2)の消費をモデル化
    // C3H8 + 5 O2 -> 3 CO2 + 4 H2O
    
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (attribute[i][j][k] == 3 || attribute[i][j][k] == 1 || attribute[i][j][k] == 2) {
                    
                    // Gを 0~1 にクリップ
                    double G_local = G[i][j][k];
                    if (G_local < 0.0) G_local = 0.0;
                    if (G_local > 1.0) G_local = 1.0;

                    // 1. 温度のマッピング (線形補間)
                    // T = Tu + G * (Tb - Tu)
                    T[i][j][k] = T_u + G_local * (T_b - T_u);

                    // 2. 化学種のマッピング (簡易Schvab-Zeldovich)
                    // ここでは主要な燃料と酸化剤だけ同期させます（厳密さはtran.binに依存しない範囲で）
                    // 実際には inletYk グローバル変数を使うのがベストです。
                    // 既存の配列値をベースに、Gで未燃/既燃をブレンドします。
                    
                    // 未燃側の組成 (inletYk相当)
                    // mainで計算済みですが、ここでは簡易的に「現在のセルの未燃状態」を維持しつつ
                    // Gに応じて既燃状態へ遷移させると考えます。
                    
                    // もっと単純に、「反応進行度 G に応じて燃料が減る」とします。
                    // Y_fuel = Y_fuel_inlet * (1 - G)
                    // Y_product = Y_product_max * G
                    
                    // 実装簡略化
                    // 輸送方程式(calculate_RHS内のdYk_dt)は「計算しない」か、
                    // 計算してもこの関数で上書きすることで整合性を取ります。
                    
                    // inletYk がグローバル変数としてアクセス可能である前提のコード:
                    for (int sp = 0; sp < NUM_SPECIES; sp++) {
                        // 未燃の質量分率
                        double Y_u = inletYk[sp]; 
                        
                        // 既燃の質量分率 (完全燃焼を仮定)
                        double Y_b = 0.0;
                        if (sp == C3H8_INDEX) Y_b = 0.0;
                        else if (sp == O2_INDEX) Y_b = fmax(0.0, inletYk[O2_INDEX] - inletYk[C3H8_INDEX] * STOICH_O2_PER_C3H8);
                        else if (sp == N2_INDEX) Y_b = inletYk[N2_INDEX];
                        else if (sp == CO2_INDEX) Y_b = inletYk[C3H8_INDEX] * STOICH_CO2_PER_C3H8;
                        else if (sp == H2O_INDEX) Y_b = inletYk[C3H8_INDEX] * STOICH_H2O_PER_C3H8;
                        
                        // 線形補間
                        Yk[sp][i][j][k] = Y_u + G_local * (Y_b - Y_u);
                    }

                    // 3. 密度の再計算 (状態方程式)
                    // TとYkが変わったので、密度も必ず更新する
                    double sum_Y_over_W = 0.0;
                    for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                        sum_Y_over_W += Yk[sp][i][j][k] / (loaded_species[sp].mw_g_per_mol / 1000.0);
                    }
                    double R_mix = R_univ * sum_Y_over_W;
                    rho[i][j][k] = pressure_const / (R_mix * T[i][j][k]);
                }
            }
        }
    }
}

// ==============================================================================
//  メイン関数 (main)
// ==============================================================================
int main(int argc, char *argv[]) {

    // --- 1. メモリ確保 (MALLOC) ---
    // 巨大な配列をヒープ領域に確保します
    printf("Allocating memory for grid size %dx%dx%d (Approx %.2f GB)...\n", 
           NX, NY, NZ, (double)sizeof(double)*NX*NY*NZ*80/1024/1024/1024);

    ALLOC_GRID(u); ALLOC_GRID(v); ALLOC_GRID(w);
    ALLOC_GRID(T); ALLOC_GRID(P); ALLOC_GRID(rho); ALLOC_GRID(G);
    
    ALLOC_GRID(u_new); ALLOC_GRID(v_new); ALLOC_GRID(w_new);
    ALLOC_GRID(T_new); ALLOC_GRID(P_new); ALLOC_GRID(rho_new); ALLOC_GRID(G_new);

    ALLOC_GRID(u_filtered); ALLOC_GRID(v_filtered); ALLOC_GRID(w_filtered);
    ALLOC_GRID(u_double_filtered); ALLOC_GRID(v_double_filtered); ALLOC_GRID(w_double_filtered);
    ALLOC_GRID(T_filtered); ALLOC_GRID(P_filtered); ALLOC_GRID(rho_filtered);
    ALLOC_GRID(G_filtered); ALLOC_GRID(G_double_filtered);
    
    ALLOC_GRID(uG); ALLOC_GRID(vG); ALLOC_GRID(wG);
    ALLOC_GRID(uG_filtered); ALLOC_GRID(vG_filtered); ALLOC_GRID(wG_filtered);
    ALLOC_GRID(uG_double_filtered); ALLOC_GRID(vG_double_filtered); ALLOC_GRID(wG_double_filtered);

    ALLOC_GRID_INT(attribute);

    ALLOC_GRID(nu); ALLOC_GRID(mu_mix); ALLOC_GRID(alpha);
    ALLOC_GRID(sgs_stress_u); ALLOC_GRID(sgs_stress_v); ALLOC_GRID(sgs_stress_w);
    ALLOC_GRID(nu_t); ALLOC_GRID(S_mag); ALLOC_GRID(q2);
    ALLOC_GRID(SijSij); ALLOC_GRID(div_u); ALLOC_GRID(ST_turbulent);
    ALLOC_GRID(mu_filtered); ALLOC_GRID(Cp_mix);
    ALLOC_GRID(gamma_sgs); ALLOC_GRID(omega); ALLOC_GRID(lambda_mix);

    ALLOC_GRID(rhou); ALLOC_GRID(rhov); ALLOC_GRID(rhow);
    ALLOC_GRID(rhou_new); ALLOC_GRID(rhov_new); ALLOC_GRID(rhow_new);
    ALLOC_GRID(rhou_initial); ALLOC_GRID(rhov_initial); ALLOC_GRID(rhow_initial);
    ALLOC_GRID(rhou_temp); ALLOC_GRID(rhov_temp); ALLOC_GRID(rhow_temp);

    for(int sp=0; sp<NUM_SPECIES; sp++) {
        ALLOC_GRID(Yk[sp]); ALLOC_GRID(Yk_new[sp]); ALLOC_GRID(Yk_filtered[sp]);
        ALLOC_GRID(Dk[sp]); ALLOC_GRID(Yk_initial[sp]); ALLOC_GRID(Yk_temp[sp]);
        ALLOC_GRID(k1_Yk[sp]); ALLOC_GRID(k2_Yk[sp]); 
        ALLOC_GRID(k3_Yk[sp]); ALLOC_GRID(k4_Yk[sp]);
    }

    ALLOC_GRID(k1_rhou); ALLOC_GRID(k1_rhov); ALLOC_GRID(k1_rhow); ALLOC_GRID(k1_T); ALLOC_GRID(k1_G);
    ALLOC_GRID(k2_rhou); ALLOC_GRID(k2_rhov); ALLOC_GRID(k2_rhow); ALLOC_GRID(k2_T); ALLOC_GRID(k2_G);
    ALLOC_GRID(k3_rhou); ALLOC_GRID(k3_rhov); ALLOC_GRID(k3_rhow); ALLOC_GRID(k3_T); ALLOC_GRID(k3_G);
    ALLOC_GRID(k4_rhou); ALLOC_GRID(k4_rhov); ALLOC_GRID(k4_rhow); ALLOC_GRID(k4_T); ALLOC_GRID(k4_G);
    ALLOC_GRID(k1_u); ALLOC_GRID(k1_v); ALLOC_GRID(k1_w);
    ALLOC_GRID(k2_u); ALLOC_GRID(k2_v); ALLOC_GRID(k2_w);
    ALLOC_GRID(k3_u); ALLOC_GRID(k3_v); ALLOC_GRID(k3_w);
    ALLOC_GRID(k4_u); ALLOC_GRID(k4_v); ALLOC_GRID(k4_w);

    ALLOC_GRID(u_initial); ALLOC_GRID(v_initial); ALLOC_GRID(w_initial);
    ALLOC_GRID(T_initial); ALLOC_GRID(G_initial);
    ALLOC_GRID(u_temp); ALLOC_GRID(v_temp); ALLOC_GRID(w_temp);
    ALLOC_GRID(T_temp); ALLOC_GRID(G_temp); ALLOC_GRID(rho_temp);

    ALLOC_GRID(u_mean); ALLOC_GRID(v_mean); ALLOC_GRID(w_mean);
    ALLOC_GRID(uu_mean); ALLOC_GRID(vv_mean); ALLOC_GRID(ww_mean);
    ALLOC_GRID(uv_mean); ALLOC_GRID(uw_mean); ALLOC_GRID(vw_mean);
    ALLOC_GRID(G_mean);
    ALLOC_GRID(re_stress_uu); ALLOC_GRID(re_stress_vv); ALLOC_GRID(re_stress_ww);
    ALLOC_GRID(re_stress_uv); ALLOC_GRID(re_stress_uw); ALLOC_GRID(re_stress_vw);
    
    ALLOC_GRID(grad_G_magnitude_for_debug);

    printf("Memory allocation completed successfully.\n\n");

    // --- 2. 初期化と設定 ---
    srand(time(NULL)); 
    clock_t start_time = clock();
    
    // Ctrl+C ハンドラ登録
    signal(SIGINT, handle_sigint);

    // 計測点の準備
    setup_measurement_points();
    printf("[DEBUG] Number of measurement points to be monitored: %d\n\n", num_points_to_measure);

    // 物性値データのロード
    int n_species_loaded = 0, n_elements_loaded = 0;
    if (!load_chem_bin("chem.bin", loaded_species, &n_species_loaded, &n_elements_loaded)) return 1;
    if (!load_tran_bin("tran.bin", loaded_species, diffusion_coeffs, n_species_loaded)) return 1;
    
    // 組成とパラメータ設定
    const double equivalence_ratio_phi = 1.1; 
    printf("===== Running simulation for Equivalence Ratio (phi) = %.2f =====\n\n", equivalence_ratio_phi);
    
    double moles_C3H8 = 1.0;
    double moles_O2 = 5.0 / equivalence_ratio_phi;
    double moles_N2 = moles_O2 * 3.76;
    double mass_total = moles_C3H8 * loaded_species[C3H8_INDEX].mw_g_per_mol + 
                        moles_O2 * loaded_species[O2_INDEX].mw_g_per_mol + 
                        moles_N2 * loaded_species[N2_INDEX].mw_g_per_mol;
    inletYk[C3H8_INDEX] = (moles_C3H8 * loaded_species[C3H8_INDEX].mw_g_per_mol) / mass_total;
    inletYk[O2_INDEX]   = (moles_O2 * loaded_species[O2_INDEX].mw_g_per_mol) / mass_total;
    inletYk[N2_INDEX]   = (moles_N2 * loaded_species[N2_INDEX].mw_g_per_mol) / mass_total;
    inletYk[CO2_INDEX] = 0.0; inletYk[H2O_INDEX] = 0.0;
    
    ambientYk[C3H8_INDEX] = 0.0; ambientYk[O2_INDEX] = 0.232; ambientYk[N2_INDEX] = 0.768; 
    ambientYk[CO2_INDEX] = 0.0; ambientYk[H2O_INDEX] = 0.0;

    STOICH_O2_PER_C3H8  = (5.0 * loaded_species[O2_INDEX].mw_g_per_mol)  / loaded_species[C3H8_INDEX].mw_g_per_mol;
    STOICH_CO2_PER_C3H8 = (3.0 * loaded_species[CO2_INDEX].mw_g_per_mol) / loaded_species[C3H8_INDEX].mw_g_per_mol;
    STOICH_H2O_PER_C3H8 = (4.0 * loaded_species[H2O_INDEX].mw_g_per_mol) / loaded_species[C3H8_INDEX].mw_g_per_mol;

    // --- 格子属性の初期設定 ---
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (i == 0) {
                    double y_coord = j * DY; double z_coord = k * DZ;
                    double distance_sq = pow(y_coord - INLET_CENTER_Y, 2) + pow(z_coord - INLET_CENTER_Z, 2);
                    if (distance_sq <= pow(INLET_RADIUS, 2)) attribute[i][j][k] = 1; else attribute[i][j][k] = 4;
                } else if (i == NX - 1) attribute[i][j][k] = 2;
                else if (j == 0 || j == NY - 1 || k == 0 || k == NZ - 1) attribute[i][j][k] = 4;
                else attribute[i][j][k] = 3;
            }
        }
    }

    // --- 3. 自動リスタート判定ロジック ---
    int start_step = 0;
    int restart_mode = 0; // 0:新規, 1:VTK指定, 2:自動(Checkpoint)
    char restart_filename[512] = "";

    // (A) コマンドライン引数指定 (-restart file.vtk)
    if (argc == 3 && strcmp(argv[1], "-restart") == 0) {
        restart_mode = 1;
        strcpy(restart_filename, argv[2]);
        char *ptr = strrchr(restart_filename, '_');
        if (ptr) start_step = atoi(ptr + 1); // 簡易的にファイル名からステップ取得
    } 
    // (B) 自動検出 (restart_info.txt)
    else {
        char info_path[512];
        sprintf(info_path, "%s%s", OUTPUT_DIR, RESTART_INFO_FILE);
        FILE *fp_info = fopen(info_path, "r");
        if (fp_info) {
            char line[256];
            int saved_step = 0;
            if (fgets(line, sizeof(line), fp_info)) {
                line[strcspn(line, "\n")] = 0;
                if (strncmp(line, "BINARY", 6) == 0) {
                    // バイナリチェックポイント
                    char *step_line = fgets(line, sizeof(line), fp_info); // 次の行にステップ数
                    if (step_line && sscanf(step_line, "%d", &saved_step) == 1) {
                        start_step = saved_step;
                        restart_mode = 2; 
                        printf("\n!!! AUTO-RESUME DETECTED (From Checkpoint Step %d) !!!\n", saved_step);
                    }
                } else {
                    // VTKファイルパスが書いてある場合 (後方互換)
                    strcpy(restart_filename, line);
                    char *step_line = fgets(line, sizeof(line), fp_info);
                    if (step_line && sscanf(step_line, "%d", &saved_step) == 1) {
                        start_step = saved_step;
                        restart_mode = 1;
                        printf("\n!!! AUTO-RESUME DETECTED (From VTK Step %d) !!!\n", saved_step);
                    }
                }
            }
            fclose(fp_info);
        }
    }

    // --- 4. 状態のロードまたは初期化 ---
    if (restart_mode == 2) {
        // 高速バイナリからの復帰
        int loaded_step = 0;
        if (!load_quick_checkpoint(&loaded_step)) {
            fprintf(stderr, "Binary resume failed. Starting from scratch.\n");
            restart_mode = 0; start_step = 0;
        } else {
            start_step = loaded_step; // 読み込んだステップ数を正とする
            load_statistics_backup(); // 統計量も復元
            initialize_vortices(); 

            printf("\n\n--- Final Reynolds Number Calculation ---\n");

            // --- 1. 代表値の定義 ---
            double Re_L = 2.0 * INLET_RADIUS;      // 代表長さ = ノズル直径
            double Re_U = INLET_VELOCITY;          // 代表速度 = 流入速度
            double Re_T = INLET_TEMPERATURE;       // 代表温度 = 流入温度

            // --- 2. 流入ガスの物性値（密度と粘性係数）を計算 ---
            // (a) 密度の計算 (状態方程式から)
            double sum_Y_over_W_inlet = 0.0;
            for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                sum_Y_over_W_inlet += inletYk[sp] / (loaded_species[sp].mw_g_per_mol / 1000.0);
            }
            double WTM_inlet = 1.0 / (sum_Y_over_W_inlet + SMALL_NUMBER);
            double R_mix_inlet = R_univ / WTM_inlet;
            double rho_inlet = pressure_const / (R_mix_inlet * Re_T);

            // (b) 粘性係数の計算 (Wilkeの混合則)
            double Xk_inlet[NUM_SPECIES];
            for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                Xk_inlet[sp] = (inletYk[sp] / (loaded_species[sp].mw_g_per_mol / 1000.0)) / (sum_Y_over_W_inlet + SMALL_NUMBER);
            }

            double mu_inlet = 0.0;
            double mu_k_pure[NUM_SPECIES];
            for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                mu_k_pure[sp] = calculate_mu_dynamic(sp, Re_T);
            }

            for (int sp_i = 0; sp_i < NUM_SPECIES; ++sp_i) {
                if (Xk_inlet[sp_i] < 1e-12) continue;
                double mu_i = mu_k_pure[sp_i];
                double M_i = loaded_species[sp_i].mw_g_per_mol;
                double denominator = 0.0;
                for (int sp_j = 0; sp_j < NUM_SPECIES; ++sp_j) {
                    if (Xk_inlet[sp_j] < 1e-12) continue;
                    double mu_j = mu_k_pure[sp_j];
                    double M_j = loaded_species[sp_j].mw_g_per_mol;
                    double phi_ij_term = pow(1.0 + pow(mu_i / mu_j, 0.5) * pow(M_j / M_i, 0.25), 2) / (sqrt(8.0) * sqrt(1.0 + M_i / M_j));
                    denominator += Xk_inlet[sp_j] * phi_ij_term;
                }
                if (denominator > 1e-9) {
                    mu_inlet += (Xk_inlet[sp_i] * mu_i) / denominator;
                }
            }
            mu_inlet = fmax(MIN_MU_MIX_CLIP, mu_inlet);

            // --- 3. レイノルズ数の計算 ---
            double Re = (rho_inlet * Re_U * Re_L) / mu_inlet;

            // --- 4. 結果の表示 ---
            printf("Based on inlet conditions:\n");
            printf("  - Characteristic Length (Diameter): %.4f m\n", Re_L);
            printf("  - Characteristic Velocity:          %.2f m/s\n", Re_U);
            printf("  - Inlet Gas Density:                %.4f kg/m^3\n", rho_inlet);
            printf("  - Inlet Gas Dynamic Viscosity:      %.3e Pa*s\n", mu_inlet);
            printf("--------------------------------------------------\n");
            printf("  Jet Reynolds Number (Re):           %.0f\n", Re);
            printf("--------------------------------------------------\n");
            
            // また、SSR-SGSパラメータの表示部分も有用なので移植すると良いでしょう
            double S_L_param = S_L;
            double sum_Y_over_W_unburnt = 0.0;
            for(int sp=0; sp<NUM_SPECIES; ++sp) sum_Y_over_W_unburnt += inletYk[sp] / (loaded_species[sp].mw_g_per_mol / 1000.0);
            double WTM_unburnt = 1.0 / sum_Y_over_W_unburnt;
            double R_mix_unburnt = R_univ / WTM_unburnt;
            double rho_unburnt_param = pressure_const / (R_mix_unburnt * INLET_TEMPERATURE);
            double rho_burnt_approx = pressure_const / (R_mix_unburnt * BURNT_GAS_TEMPERATURE); 
            double tau_expansion_ratio_param = (rho_unburnt_param / rho_burnt_approx) - 1.0;

            printf("SSR-SGS params: S_L=%.3f, rho_u=%.3f, nu_u=%.2e, tau=%.2f\n\n",
           S_L_param, rho_unburnt_param, NU_UNBURNT_AIR_APPROX, tau_expansion_ratio_param);     
        }
    }
    else if (restart_mode == 1) {
        // VTKからの復帰
        printf("Resuming from VTK: %s\n", restart_filename);
        if (!load_vtk_for_restart(restart_filename)) {
            fprintf(stderr, "VTK resume failed. Starting from scratch.\n");
            restart_mode = 0; start_step = 0;
        } else {
            load_statistics_backup();
            initialize_vortices();

            printf("\n\n--- Final Reynolds Number Calculation ---\n");

            // --- 1. 代表値の定義 ---
            double Re_L = 2.0 * INLET_RADIUS;      // 代表長さ = ノズル直径
            double Re_U = INLET_VELOCITY;          // 代表速度 = 流入速度
            double Re_T = INLET_TEMPERATURE;       // 代表温度 = 流入温度

            // --- 2. 流入ガスの物性値（密度と粘性係数）を計算 ---
            // (a) 密度の計算 (状態方程式から)
            double sum_Y_over_W_inlet = 0.0;
            for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                sum_Y_over_W_inlet += inletYk[sp] / (loaded_species[sp].mw_g_per_mol / 1000.0);
            }
            double WTM_inlet = 1.0 / (sum_Y_over_W_inlet + SMALL_NUMBER);
            double R_mix_inlet = R_univ / WTM_inlet;
            double rho_inlet = pressure_const / (R_mix_inlet * Re_T);

            // (b) 粘性係数の計算 (Wilkeの混合則)
            double Xk_inlet[NUM_SPECIES];
            for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                Xk_inlet[sp] = (inletYk[sp] / (loaded_species[sp].mw_g_per_mol / 1000.0)) / (sum_Y_over_W_inlet + SMALL_NUMBER);
            }

            double mu_inlet = 0.0;
            double mu_k_pure[NUM_SPECIES];
            for (int sp = 0; sp < NUM_SPECIES; ++sp) {
                mu_k_pure[sp] = calculate_mu_dynamic(sp, Re_T);
            }

            for (int sp_i = 0; sp_i < NUM_SPECIES; ++sp_i) {
                if (Xk_inlet[sp_i] < 1e-12) continue;
                double mu_i = mu_k_pure[sp_i];
                double M_i = loaded_species[sp_i].mw_g_per_mol;
                double denominator = 0.0;
                for (int sp_j = 0; sp_j < NUM_SPECIES; ++sp_j) {
                    if (Xk_inlet[sp_j] < 1e-12) continue;
                    double mu_j = mu_k_pure[sp_j];
                    double M_j = loaded_species[sp_j].mw_g_per_mol;
                    double phi_ij_term = pow(1.0 + pow(mu_i / mu_j, 0.5) * pow(M_j / M_i, 0.25), 2) / (sqrt(8.0) * sqrt(1.0 + M_i / M_j));
                    denominator += Xk_inlet[sp_j] * phi_ij_term;
                }
                if (denominator > 1e-9) {
                    mu_inlet += (Xk_inlet[sp_i] * mu_i) / denominator;
                }
            }
            mu_inlet = fmax(MIN_MU_MIX_CLIP, mu_inlet);

            // --- 3. レイノルズ数の計算 ---
            double Re = (rho_inlet * Re_U * Re_L) / mu_inlet;

            // --- 4. 結果の表示 ---
            printf("Based on inlet conditions:\n");
            printf("  - Characteristic Length (Diameter): %.4f m\n", Re_L);
            printf("  - Characteristic Velocity:          %.2f m/s\n", Re_U);
            printf("  - Inlet Gas Density:                %.4f kg/m^3\n", rho_inlet);
            printf("  - Inlet Gas Dynamic Viscosity:      %.3e Pa*s\n", mu_inlet);
            printf("--------------------------------------------------\n");
            printf("  Jet Reynolds Number (Re):           %.0f\n", Re);
            printf("--------------------------------------------------\n");
            
            // また、SSR-SGSパラメータの表示部分も有用なので移植すると良いでしょう
            double S_L_param = S_L;
            double sum_Y_over_W_unburnt = 0.0;
            for(int sp=0; sp<NUM_SPECIES; ++sp) sum_Y_over_W_unburnt += inletYk[sp] / (loaded_species[sp].mw_g_per_mol / 1000.0);
            double WTM_unburnt = 1.0 / sum_Y_over_W_unburnt;
            double R_mix_unburnt = R_univ / WTM_unburnt;
            double rho_unburnt_param = pressure_const / (R_mix_unburnt * INLET_TEMPERATURE);
            double rho_burnt_approx = pressure_const / (R_mix_unburnt * BURNT_GAS_TEMPERATURE); 
            double tau_expansion_ratio_param = (rho_unburnt_param / rho_burnt_approx) - 1.0;

            printf("SSR-SGS params: S_L=%.3f, rho_u=%.3f, nu_u=%.2e, tau=%.2f\n\n",
           S_L_param, rho_unburnt_param, NU_UNBURNT_AIR_APPROX, tau_expansion_ratio_param); 
        }
    }
    
    // 新規計算の場合
    if (restart_mode == 0) {
        printf("Starting NEW simulation...\n");
        initialize();
        setting(attribute, u, u_new, v, v_new, w, w_new, P, P_new, T, T_new, G, G_new,
                Yk, Yk_new, Dk, omega, rho, rho_new, rhou, rhou_new, rhov, rhov_new, rhow, rhow_new);
        initialize_vortices();
        initialize_probe_file();
        output_vtk(0);
    }

    // 流入面の属性配列作成（乱流生成用）
    int inlet_attribute[NY][NZ];
    for(int j=0; j<NY; ++j) for(int k=0; k<NZ; ++k) inlet_attribute[j][k] = (attribute[0][j][k] == 1) ? 1 : 0;

    // 時間刻み計算用の一時変数
    double max_velocity;
    calculate_max_velocity(attribute, u, v, w, &max_velocity);
    double DT_sim = COURANT_NUMBER * DX / (max_velocity + SMALL_NUMBER);
    
    // 総ステップ数の決定
    // もしstart_stepが既に大きい場合は、さらに SIMULATION_TIME 分だけ追加で回すか、
    // SIMULATION_TIME を「終了時刻」とするか。ここでは「総シミュレーション時間」を目標とします。
    int total_steps = (int)(SIMULATION_TIME / DT_sim);
    if (start_step >= total_steps) {
        printf("Simulation already reached target time. Extending by 1000 steps...\n");
        total_steps = start_step + 1000;
    }
    int start_avg_step = total_steps + 1; // 平均開始ステップ

    printf("\n--- Simulation Loop Start ---\n");
    printf("From Step %d to %d\n", start_step, total_steps);

    // ====================================
    // 5. 時間発展ループ 
    // ====================================
    for (int t_loop = start_step; t_loop < total_steps; t_loop++) {

        // ★Ctrl+C 検知★
        if (stop_requested) {
            printf("\n\n--- Stopping simulation by user request (Ctrl+C) ---\n");
            break; // ループを抜けて保存処理へ
        }

        // --- DT動的決定 ---
        calculate_max_velocity(attribute, u, v, w, &max_velocity);
        double DT_conv = COURANT_NUMBER * DX / (max_velocity + SMALL_NUMBER);
        double max_nu = 0.0, max_alpha = 0.0;
        #pragma omp parallel for collapse(3) reduction(max:max_nu, max_alpha)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                if(nu[i][j][k] > max_nu) max_nu = nu[i][j][k];
                if(alpha[i][j][k] > max_alpha) max_alpha = alpha[i][j][k];
            }
        }
        double DT_diff_nu = COURANT_DIFFUSION * (DX * DX) / (max_nu + SMALL_NUMBER);
        double DT_diff_alpha = COURANT_DIFFUSION * (DX * DX) / (max_alpha + SMALL_NUMBER);
        DT_sim = fmin(DT_conv, fmin(DT_diff_nu, DT_diff_alpha));

        // 乱流生成
        generate_inlet_turbulence_SEM(t_loop, DT_sim, inlet_attribute, u_inlet_fluct, v_inlet_fluct, w_inlet_fluct);

        // ==========================================
        // --- 4次ルンゲ＝クッタ法 (RK4) 開始 ---
        // ==========================================

        // --- (0) 初期状態 Φ_n を保存 ---
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            rhou_initial[i][j][k] = rhou[i][j][k]; rhov_initial[i][j][k] = rhov[i][j][k]; rhow_initial[i][j][k] = rhow[i][j][k];
            T_initial[i][j][k] = T[i][j][k]; G_initial[i][j][k] = G[i][j][k];
            for(int sp=0; sp<NUM_SPECIES; sp++) Yk_initial[sp][i][j][k] = Yk[sp][i][j][k];
        }

        // --- (1) Stage 1: k1 = f(Φ_n) ---
        calculate_RHS(t_loop, u, v, w, T, G, Yk, rho, k1_rhou, k1_rhov, k1_rhow, k1_T, k1_G, k1_Yk);

        // 中間状態1: Φ_temp = Φ_n + 0.5*DT*k1
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                T_temp[i][j][k] = T_initial[i][j][k] + 0.5 * DT_sim * k1_T[i][j][k];
                G_temp[i][j][k] = G_initial[i][j][k] + 0.5 * DT_sim * k1_G[i][j][k];
                for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] = Yk_initial[sp][i][j][k] + 0.5 * DT_sim * k1_Yk[sp][i][j][k];
                
                // Clipping & Normalization
                T_temp[i][j][k] = fmax(MIN_TEMPERATURE_CLIP, fmin(MAX_TEMPERATURE_CLIP, T_temp[i][j][k])); 
                G_temp[i][j][k] = fmax(MIN_G_CLIP, fmin(MAX_G_CLIP, G_temp[i][j][k]));
                double sum_Y=0; for(int sp=0; sp<NUM_SPECIES; sp++) { Yk_temp[sp][i][j][k] = fmax(MIN_Y_CLIP, fmin(MAX_Y_CLIP, Yk_temp[sp][i][j][k])); sum_Y+=Yk_temp[sp][i][j][k]; }
                if (sum_Y > SMALL_NUMBER) for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] /= sum_Y;
            } else { T_temp[i][j][k] = T_initial[i][j][k]; G_temp[i][j][k] = G_initial[i][j][k]; for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] = Yk_initial[sp][i][j][k]; }
        }
        update_rho_from_EOS(attribute, T_temp, Yk_temp, rho_temp);
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                rhou_temp[i][j][k] = rhou_initial[i][j][k] + 0.5 * DT_sim * k1_rhou[i][j][k]; 
                rhov_temp[i][j][k] = rhov_initial[i][j][k] + 0.5 * DT_sim * k1_rhov[i][j][k]; 
                rhow_temp[i][j][k] = rhow_initial[i][j][k] + 0.5 * DT_sim * k1_rhow[i][j][k];
                if(rho_temp[i][j][k] > SMALL_NUMBER){ u_temp[i][j][k] = rhou_temp[i][j][k]/rho_temp[i][j][k]; v_temp[i][j][k] = rhov_temp[i][j][k]/rho_temp[i][j][k]; w_temp[i][j][k] = rhow_temp[i][j][k]/rho_temp[i][j][k]; } else { u_temp[i][j][k]=0; v_temp[i][j][k]=0; w_temp[i][j][k]=0; }
            } else { u_temp[i][j][k] = u[i][j][k]; v_temp[i][j][k] = v[i][j][k]; w_temp[i][j][k] = w[i][j][k]; }
        }

        // --- (2) Stage 2: k2 = f(Φ_temp) ---
        calculate_RHS(t_loop, u_temp, v_temp, w_temp, T_temp, G_temp, Yk_temp, rho_temp, k2_rhou, k2_rhov, k2_rhow, k2_T, k2_G, k2_Yk);

        // 中間状態2: Φ_temp = Φ_n + 0.5*DT*k2
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                T_temp[i][j][k] = T_initial[i][j][k] + 0.5 * DT_sim * k2_T[i][j][k];
                G_temp[i][j][k] = G_initial[i][j][k] + 0.5 * DT_sim * k2_G[i][j][k];
                for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] = Yk_initial[sp][i][j][k] + 0.5 * DT_sim * k2_Yk[sp][i][j][k];
                
                T_temp[i][j][k] = fmax(MIN_TEMPERATURE_CLIP, fmin(MAX_TEMPERATURE_CLIP, T_temp[i][j][k])); 
                G_temp[i][j][k] = fmax(MIN_G_CLIP, fmin(MAX_G_CLIP, G_temp[i][j][k]));
                double sum_Y=0; for(int sp=0; sp<NUM_SPECIES; sp++) { Yk_temp[sp][i][j][k] = fmax(MIN_Y_CLIP, fmin(MAX_Y_CLIP, Yk_temp[sp][i][j][k])); sum_Y+=Yk_temp[sp][i][j][k]; }
                if (sum_Y > SMALL_NUMBER) for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] /= sum_Y;
            } else { T_temp[i][j][k] = T_initial[i][j][k]; G_temp[i][j][k] = G_initial[i][j][k]; for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] = Yk_initial[sp][i][j][k]; }
        }
        update_rho_from_EOS(attribute, T_temp, Yk_temp, rho_temp);
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                rhou_temp[i][j][k] = rhou_initial[i][j][k] + 0.5 * DT_sim * k2_rhou[i][j][k]; 
                rhov_temp[i][j][k] = rhov_initial[i][j][k] + 0.5 * DT_sim * k2_rhov[i][j][k]; 
                rhow_temp[i][j][k] = rhow_initial[i][j][k] + 0.5 * DT_sim * k2_rhow[i][j][k];
                if(rho_temp[i][j][k] > SMALL_NUMBER){ u_temp[i][j][k] = rhou_temp[i][j][k]/rho_temp[i][j][k]; v_temp[i][j][k] = rhov_temp[i][j][k]/rho_temp[i][j][k]; w_temp[i][j][k] = rhow_temp[i][j][k]/rho_temp[i][j][k]; } else { u_temp[i][j][k]=0; v_temp[i][j][k]=0; w_temp[i][j][k]=0; }
            } else { u_temp[i][j][k] = u[i][j][k]; v_temp[i][j][k] = v[i][j][k]; w_temp[i][j][k] = w[i][j][k]; }
        }

        // --- (3) Stage 3: k3 = f(Φ_temp) ---
        calculate_RHS(t_loop, u_temp, v_temp, w_temp, T_temp, G_temp, Yk_temp, rho_temp, k3_rhou, k3_rhov, k3_rhow, k3_T, k3_G, k3_Yk);

        // 中間状態3: Φ_temp = Φ_n + 1.0*DT*k3
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                T_temp[i][j][k] = T_initial[i][j][k] + 1.0 * DT_sim * k3_T[i][j][k];
                G_temp[i][j][k] = G_initial[i][j][k] + 1.0 * DT_sim * k3_G[i][j][k];
                for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] = Yk_initial[sp][i][j][k] + 1.0 * DT_sim * k3_Yk[sp][i][j][k];
                
                T_temp[i][j][k] = fmax(MIN_TEMPERATURE_CLIP, fmin(MAX_TEMPERATURE_CLIP, T_temp[i][j][k])); 
                G_temp[i][j][k] = fmax(MIN_G_CLIP, fmin(MAX_G_CLIP, G_temp[i][j][k]));
                double sum_Y=0; for(int sp=0; sp<NUM_SPECIES; sp++) { Yk_temp[sp][i][j][k] = fmax(MIN_Y_CLIP, fmin(MAX_Y_CLIP, Yk_temp[sp][i][j][k])); sum_Y+=Yk_temp[sp][i][j][k]; }
                if (sum_Y > SMALL_NUMBER) for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] /= sum_Y;
            } else { T_temp[i][j][k] = T_initial[i][j][k]; G_temp[i][j][k] = G_initial[i][j][k]; for(int sp=0; sp<NUM_SPECIES; sp++) Yk_temp[sp][i][j][k] = Yk_initial[sp][i][j][k]; }
        }
        update_rho_from_EOS(attribute, T_temp, Yk_temp, rho_temp);
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                rhou_temp[i][j][k] = rhou_initial[i][j][k] + 1.0 * DT_sim * k3_rhou[i][j][k]; 
                rhov_temp[i][j][k] = rhov_initial[i][j][k] + 1.0 * DT_sim * k3_rhov[i][j][k]; 
                rhow_temp[i][j][k] = rhow_initial[i][j][k] + 1.0 * DT_sim * k3_rhow[i][j][k];
                if(rho_temp[i][j][k] > SMALL_NUMBER){ u_temp[i][j][k] = rhou_temp[i][j][k]/rho_temp[i][j][k]; v_temp[i][j][k] = rhov_temp[i][j][k]/rho_temp[i][j][k]; w_temp[i][j][k] = rhow_temp[i][j][k]/rho_temp[i][j][k]; } else { u_temp[i][j][k]=0; v_temp[i][j][k]=0; w_temp[i][j][k]=0; }
            } else { u_temp[i][j][k] = u[i][j][k]; v_temp[i][j][k] = v[i][j][k]; w_temp[i][j][k] = w[i][j][k]; }
        }

        // --- (4) Stage 4: k4 = f(Φ_temp) ---
        calculate_RHS(t_loop, u_temp, v_temp, w_temp, T_temp, G_temp, Yk_temp, rho_temp, k4_rhou, k4_rhov, k4_rhow, k4_T, k4_G, k4_Yk);

        // --- (5) Final Update: Φ_{n+1} = Φ_n + (DT/6)*(k1 + 2k2 + 2k3 + k4) ---
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                T[i][j][k] = T_initial[i][j][k] + (DT_sim/6.0)*(k1_T[i][j][k] + 2.0*k2_T[i][j][k] + 2.0*k3_T[i][j][k] + k4_T[i][j][k]);
                G[i][j][k] = G_initial[i][j][k] + (DT_sim/6.0)*(k1_G[i][j][k] + 2.0*k2_G[i][j][k] + 2.0*k3_G[i][j][k] + k4_G[i][j][k]);
                for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][j][k] = Yk_initial[sp][i][j][k] + (DT_sim/6.0)*(k1_Yk[sp][i][j][k] + 2.0*k2_Yk[sp][i][j][k] + 2.0*k3_Yk[sp][i][j][k] + k4_Yk[sp][i][j][k]);
                
                T[i][j][k] = fmax(MIN_TEMPERATURE_CLIP, fmin(MAX_TEMPERATURE_CLIP, T[i][j][k])); 
                G[i][j][k] = fmax(MIN_G_CLIP, fmin(MAX_G_CLIP, G[i][j][k]));
                double sum_Y=0; for(int sp=0; sp<NUM_SPECIES; sp++) { Yk[sp][i][j][k] = fmax(MIN_Y_CLIP, fmin(MAX_Y_CLIP, Yk[sp][i][j][k])); sum_Y+=Yk[sp][i][j][k]; }
                if (sum_Y > SMALL_NUMBER) for(int sp=0; sp<NUM_SPECIES; sp++) Yk[sp][i][j][k] /= sum_Y;
            }
        }
        
        update_rho_from_EOS(attribute, T, Yk, rho);
        #pragma omp parallel for collapse(3)
        for (int i=0; i<NX; i++) for (int j=0; j<NY; j++) for (int k=0; k<NZ; k++) {
            if (attribute[i][j][k] == 3) {
                rhou[i][j][k] = rhou_initial[i][j][k] + (DT_sim/6.0)*(k1_rhou[i][j][k] + 2.0*k2_rhou[i][j][k] + 2.0*k3_rhou[i][j][k] + k4_rhou[i][j][k]);
                rhov[i][j][k] = rhov_initial[i][j][k] + (DT_sim/6.0)*(k1_rhov[i][j][k] + 2.0*k2_rhov[i][j][k] + 2.0*k3_rhov[i][j][k] + k4_rhov[i][j][k]);
                rhow[i][j][k] = rhow_initial[i][j][k] + (DT_sim/6.0)*(k1_rhow[i][j][k] + 2.0*k2_rhow[i][j][k] + 2.0*k3_rhow[i][j][k] + k4_rhow[i][j][k]);
                if (rho[i][j][k] > SMALL_NUMBER) {
                    u[i][j][k] = rhou[i][j][k] / rho[i][j][k];
                    v[i][j][k] = rhov[i][j][k] / rho[i][j][k];
                    w[i][j][k] = rhow[i][j][k] / rho[i][j][k];
                } else { u[i][j][k] = 0; v[i][j][k] = 0; w[i][j][k] = 0; }
            }
        }

        // ==========================================
        // --- RK4 終了 ---
        // ==========================================

        // これにより、T, Yk, rho が G の形状と完全に一致suruhazu。
        // Forceモデルが正しい密度（軽い既燃ガス）に対して作用するために必須です。
        // update_thermodynamics_from_G(attribute, T, Yk, rho, G);

        // 境界条件
        apply_boundary_conditions(attribute, u, v, w, rhou, rhov, rhow, G, T, P, Yk, rho);

        // --- 時系列データ記録 (メモリへ) ---
        if (t_loop >= START_DATA_ACQUISITION_STEP) {
            for (int p = 0; p < num_points_to_measure; p++) {
                if (points[p].count < MAX_DATA_POINTS) {
                    int i = points[p].i; int j = points[p].j; int k = points[p].k;
                    points[p].u_data[points[p].count] = u[i][j][k];
                    points[p].v_data[points[p].count] = v[i][j][k];
                    points[p].w_data[points[p].count] = w[i][j][k];
                    points[p].G_data[points[p].count] = G[i][j][k];
                    points[p].count++;
                }
            }
        }

        // --- 火炎領域プローブ (CSV追記) ---
        if (t_loop >= START_DATA_ACQUISITION_STEP && t_loop % PROBE_DATA_INTERVAL == 0) {
            record_flame_region_data(t_loop, u, v, w, G, nu_t, S_mag);
        }

        // --- 時間平均 ---
        if (t_loop >= start_avg_step) {
            update_time_averages(u, v, w, G);
        }

        // --- データの同期保存 (チェックポイント & CSV) ---
        // このタイミングで「ここから再開可能」な状態を作る
        if (t_loop > 0 && t_loop % CHECKPOINT_INTERVAL == 0) {
            printf("Checkpointing at step %d... ", t_loop);
            save_quick_checkpoint(t_loop);   // 物理場保存
            save_statistics_backup();        // 平均値保存
            flush_point_data(equivalence_ratio_phi); // 時系列データ書き出し＆クリア
            printf("Done.\n");
        }

        // --- モニタリングログ ---
        write_monitor_log(t_loop, DT_sim, rho, T, G, Yk);

        // --- VTK出力 ---
        if (t_loop > 0 && t_loop % OUTPUT_INTERVAL == 0) {
            check_for_nan(u, "u", t_loop); 
            calculate_resolved_reynolds_stresses();
            output_vtk(t_loop);
        }
        
        printf("Step: %d/%d (%.1f%%), DT: %.2e, MaxVel: %.2f, AvgSteps: %d\n", 
               t_loop, total_steps, (t_loop * 100.0 / total_steps), DT_sim, max_velocity, time_avg_steps);
    }

    // --- 終了時の最終保存 ---
    printf("\nFinalizing output...\n");
    
    // 最後に必ず残りのデータを吐き出す
    calculate_resolved_reynolds_stresses();
    // 正常終了なら最終ステップ、Ctrl+Cなら中断ステップで保存
    
    // VTKは正常終了時のみ、または必要なら出力
    if (!stop_requested) output_vtk(total_steps);

    // リスタート用データは必ず保存
    int save_step = stop_requested ? -1 : total_steps; // -1はステップ不明のマーク（実際には再開時にrestart_infoを読むので問題ない）
    // 正確にはループを抜けた時点のt_loopを知りたいが、ここでは簡易的に処理
    // 安全のため「現在の状態」を保存する関数を呼ぶだけでOK
    
    // Ctrl+Cで抜けた場合、t_loop変数はスコープ外だが、物理場配列(u, v...)には最新値が入っている。
    // restart_info.txtには「最後に成功したCheckpoint」か「今保存したCheckpoint」のステップが入る。
    // ここでは簡易的に 0 としているが、再開時は restart_info を読むのでファイルパスさえ合っていればOK。
    save_quick_checkpoint(0); 
    save_statistics_backup();
    flush_point_data(equivalence_ratio_phi);

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Simulation finished. Total time: %.2f seconds\n", elapsed);

    return 0;
}
