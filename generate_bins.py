import cantera as ct
import struct
import numpy as np
import sys
import warnings
import traceback
from io import BytesIO

# ==============================================================================
# ヘルパー関数
# ==============================================================================

def fit_transport_coeffs(gas_object, property_name, temp_points):
    """粘性係数または熱伝導率をフィットする"""
    if gas_object.n_species != 1:
        raise ValueError("This function requires a gas object with a single species.")

    prop_values = []
    log_T = np.log(temp_points)

    for T in temp_points:
        gas_object.TP = T, ct.one_atm
        if property_name == 'viscosity':
            prop_values.append(gas_object.viscosity * 10.0)  # Pa-s -> poise (g/cm-s)
        elif property_name == 'thermal_conductivity':
            prop_values.append(gas_object.thermal_conductivity * 1e5)  # W/m-K -> erg/cm-s-K

    log_property = np.log(np.array(prop_values))
    coeffs = np.polyfit(log_T, log_property, 3)
    return np.flip(coeffs)

def fit_diffusion_coeffs(gas_full, sp_name_i, sp_name_j, temp_points):
    """二成分拡散係数をフィットする"""
    sp_i = gas_full.species(sp_name_i)
    sp_j = gas_full.species(sp_name_j)

    gas_binary = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                             species=[sp_i, sp_j],
                             transport_model='mixture-averaged')

    d_ij_values = []
    log_T = np.log(temp_points)

    for T in temp_points:
        gas_binary.TP = T, ct.one_atm
        d_ij_values.append(gas_binary.binary_diff_coeffs[0, 1] * 1e4)  # m^2/s -> cm^2/s

    log_d_ij = np.log(np.array(d_ij_values))
    coeffs = np.polyfit(log_T, log_d_ij, 3)
    return np.flip(coeffs)

def write_record(f, data_bytes):
    """Fortranのレコード構造（レコード長、データ、レコード長）で書き込む"""
    record_len = len(data_bytes)
    f.write(struct.pack('<i', record_len))
    f.write(data_bytes)
    f.write(struct.pack('<i', record_len))

def read_record(f):
    """Fortranのレコード構造を読み込む (検証用)"""
    try:
        len_bytes = f.read(4)
        if not len_bytes: return None
        record_len_start = struct.unpack('<i', len_bytes)[0]
        
        current_pos = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(current_pos)
        if record_len_start > file_size:
             warnings.warn(f"Record length {record_len_start} seems invalid.")
             return None

        data = f.read(record_len_start)
        record_len_end_bytes = f.read(4)
        if not record_len_end_bytes: return data

        record_len_end = struct.unpack('<i', record_len_end_bytes)[0]
        if record_len_start != record_len_end:
            warnings.warn(f"Record length mismatch: {record_len_start} vs {record_len_end}")
        return data
    except (struct.error, IOError):
        return None

# ==============================================================================
# メイン処理
# ==============================================================================
try:
    print("Generating Fortran-compatible binary files...")

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # --- 設定 ---
    mechanism_file = 'gri30.yaml'
    species_names = ["CH4", "O2", "N2", "CO2", "H2O"]
    
    # --- 初期化 ---
    gas_full = ct.Solution(mechanism_file)
    KK = len(species_names)
    elements_in_species = set()
    for name in species_names:
        for elem in gas_full.species(name).composition:
            elements_in_species.add(elem)
    elements_in_mech = sorted(list(elements_in_species)) # 順序を固定
    MM = len(elements_in_mech)

    species_objects_full = [gas_full.species(name) for name in species_names]
    temp_points_fit = np.linspace(300.0, 3500.0, 20)
    
    # ==============================================================================
    # chem.bin の生成
    # ==============================================================================
    print(f"Writing chem.bin for {KK} species and {MM} elements...")
    with open('chem.bin', 'wb') as f_out:
        # 1. ヘッダーレコード
        buf = BytesIO(); buf.write("CHEM_GEN_PYTHON".encode('ascii').ljust(16, b'\0')); buf.write("V_FINAL".encode('ascii').ljust(16, b'\0')); buf.write("LITTLE_ENDIAN".encode('ascii').ljust(16, b'\0')); write_record(f_out, buf.getvalue())
        # 2. サイズ情報レコード
        buf = BytesIO(); buf.write(struct.pack('<iii', 0, 0, 0)); buf.write(struct.pack('<ii', MM, KK)); buf.write(b'\0' * 15 * 4); buf.write(struct.pack('<d', 0.0)); write_record(f_out, buf.getvalue())
        # 3. 元素情報レコード
        buf = BytesIO(); buf.write("ELEMENTS".encode('ascii').ljust(8, b'\0')); [ (buf.write(name.encode('ascii').ljust(16, b'\0')), buf.write(struct.pack('<d', gas_full.atomic_weight(name)))) for name in elements_in_mech]; write_record(f_out, buf.getvalue())
        # 4. 化学種情報レコード
        buf = BytesIO(); buf.write("SPECIES ".encode('ascii').ljust(8, b'\0'))
        for sp_obj in species_objects_full:
            buf.write(sp_obj.name.encode('ascii').ljust(16, b'\0'))
            composition = [int(sp_obj.composition.get(elem, 0)) for elem in elements_in_mech]; buf.write(struct.pack(f'<{MM}i', *composition))
            buf.write(struct.pack('<ii', 0, 0))
            # MWは g/mol で書き込む
            buf.write(struct.pack('<d', sp_obj.molecular_weight)) 
            buf.write(struct.pack('<i', 0))
            # T_midは通常1000.0 K (GRI-Mech 3.0の場合)
            t_mid = 1000.0
            buf.write(struct.pack('<ddd', 0.0, 0.0, t_mid))
            # NASA 7-coeff polynomials.
            high_coeffs = sp_obj.thermo.coeffs[1:8]
            low_coeffs = sp_obj.thermo.coeffs[8:15]
            if len(high_coeffs) != 7 or len(low_coeffs) != 7:
                 raise ValueError(f"Unexpected NASA coefficient format for {sp_obj.name}")
            buf.write(struct.pack('<7d', *high_coeffs))
            buf.write(struct.pack('<7d', *low_coeffs))
        write_record(f_out, buf.getvalue())
        # 5. 反応情報レコード (ダミー)
        buf = BytesIO(); buf.write("REACTIONS".encode('ascii').ljust(8, b'\0')); buf.write(struct.pack('<iiddd', 1, 0, 0.0, 0.0, 1.3e8)); buf.write(b'\0' * 12 * 4); write_record(f_out, buf.getvalue())
    print("'chem.bin' generated successfully.")

    # ==============================================================================
    # tran.bin の生成
    # ==============================================================================
    print(f"Calculating and writing tran.bin for {KK} species...")
    with open('tran.bin', 'wb') as f_out:
        buf = BytesIO(); buf.write("TRAN_GEN_PYTHON".encode('ascii').ljust(16, b'\0')); buf.write("V_FINAL".encode('ascii').ljust(16, b'\0')); buf.write("LITTLE_ENDIAN".encode('ascii').ljust(16, b'\0')); write_record(f_out, buf.getvalue())
        buf = BytesIO(); buf.write(struct.pack('<iii', 0, 0, 0)); buf.write(struct.pack('<i', KK)); buf.write(struct.pack('<i', 0)); write_record(f_out, buf.getvalue())
        buf = BytesIO(); buf.write("PRESSURE".encode('ascii').ljust(8, b'\0')); buf.write(struct.pack('<d', ct.one_atm)); write_record(f_out, buf.getvalue())
        buf = BytesIO()
        for sp_obj in species_objects_full:
            buf.write(struct.pack('<d', 0.0)); buf.write(struct.pack('<ddddd', 0.0, 0.0, 0.0, 0.0, 0.0)); buf.write(struct.pack('<i', 0))
        write_record(f_out, buf.getvalue())
        buf = BytesIO()
        for sp_obj in species_objects_full:
            gas_single = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=[sp_obj], transport_model='mixture-averaged')
            coeffs = fit_transport_coeffs(gas_single, 'thermal_conductivity', temp_points_fit)
            buf.write(struct.pack('<4d', *coeffs))
        write_record(f_out, buf.getvalue())
        buf = BytesIO()
        for sp_obj in species_objects_full:
            gas_single = ct.Solution(thermo='IdealGas', kinetics='GasKinetics', species=[sp_obj], transport_model='mixture-averaged')
            coeffs_eta = fit_transport_coeffs(gas_single, 'viscosity', temp_points_fit)
            buf.write(struct.pack('<4d', *coeffs_eta))
        write_record(f_out, buf.getvalue())
        buf = BytesIO()
        for i in range(KK):
            for j in range(KK):
                if i == j:
                    buf.write(struct.pack('<4d', 0.0, 0.0, 0.0, 0.0)); continue
                sp_i_name, sp_j_name = species_names[i], species_names[j]
                coeffs_d = fit_diffusion_coeffs(gas_full, sp_i_name, sp_j_name, temp_points_fit)
                buf.write(struct.pack('<4d', *coeffs_d))
        write_record(f_out, buf.getvalue())
    print("'tran.bin' generated successfully.")

    # ==============================================================================
    # 検証コード
    # ==============================================================================
    print("\nVerifying binary files and writing to text files...")
    with open('chem.bin', 'rb') as f_in, open('chem_out.txt', 'w') as f_txt:
        f_txt.write("--- Contents of chem.bin ---\n\n")
        data = read_record(f_in); f_txt.write(f"Header: {data[:16].decode('ascii', 'ignore').strip()}\n")
        data = read_record(f_in); mm_chk, kk_chk = struct.unpack_from('<ii', data, 12); f_txt.write(f"MM (Elements): {mm_chk}\nKK (Species):  {kk_chk}\n\n")
        data = read_record(f_in); f_txt.write("Elements Data:\n")
        for i in range(mm_chk):
            offset = 8 + i * 24; name = struct.unpack_from('<16s', data, offset)[0].decode('ascii','ignore').strip('\0'); weight = struct.unpack_from('<d', data, offset + 16)[0]; f_txt.write(f"  {name:<10s} Weight: {weight:.4f}\n")
        
        data = read_record(f_in)
        f_txt.write("\n--- Species Data ---\n")
        base_offset, name_size, comp_size, int_size, double_size, nasa_size = 8, 16, 4 * mm_chk, 4, 8, 7 * 8
        record_size_per_species = name_size + comp_size + 2*int_size + double_size + int_size + 3*double_size + 2*nasa_size
        
        for i in range(kk_chk):
            offset = base_offset + i * record_size_per_species
            name_offset = offset
            mw_offset = offset + name_size + comp_size + 2 * int_size
            tmid_offset = mw_offset + int_size + 2 * double_size
            nasa_h_offset = tmid_offset + double_size
            nasa_l_offset = nasa_h_offset + nasa_size
            name = struct.unpack_from('<16s', data, name_offset)[0].decode('ascii','ignore').strip('\0')
            mw = struct.unpack_from('<d', data, mw_offset)[0]
            t_mid = struct.unpack_from('<d', data, tmid_offset)[0]
            
            f_txt.write(f"\nSpecies #{i+1}: {name}\n")
            f_txt.write(f"  MW: {mw:.4f} g/mol,  T_mid: {t_mid:.1f} K\n")
            
            high_coeffs = struct.unpack_from('<7d', data, nasa_h_offset)
            f_txt.write("  NASA High-T Coeffs (a1-a7):\n")
            f_txt.write("    " + " ".join(f"{c:12.6e}" for c in high_coeffs) + "\n")
            
            low_coeffs = struct.unpack_from('<7d', data, nasa_l_offset)
            f_txt.write("  NASA Low-T Coeffs (a1-a7):\n")
            f_txt.write("    " + " ".join(f"{c:12.6e}" for c in low_coeffs) + "\n")

        data = read_record(f_in); p1_val = struct.unpack_from('<d', data, 24)[0]; f_txt.write(f"\nReactions Data (dummy):\n  P1 (pre-exp factor): {p1_val:e}\n")
    print("'chem_out.txt' generated.")

    with open('tran.bin', 'rb') as f_in, open('tran_out.txt', 'w') as f_txt:
        f_txt.write("--- Contents of tran.bin ---\n\n")
        read_record(f_in)
        read_record(f_in)
        data = read_record(f_in); patm = struct.unpack_from('<d', data, 8)[0]; f_txt.write(f"PATM (Reference Pressure) [Pa]: {patm:.1f}\n\n")
        read_record(f_in)
        f_txt.write("Fitted Polynomial Coefficients for Transport Properties\nFormat: A0, A1, A2, A3\n\n")
        data = read_record(f_in); f_txt.write("--- COFLAM (Thermal Conductivity) ---\n")
        for i in range(KK):
            coeffs = struct.unpack_from('<4d', data, i * 32); f_txt.write(f"  {species_names[i]:<10s}" + " ".join(f"{c:12.6f}" for c in coeffs) + "\n")
        data = read_record(f_in); f_txt.write("\n--- COFETA (Viscosity) ---\n")
        for i in range(KK):
            coeffs = struct.unpack_from('<4d', data, i * 32); f_txt.write(f"  {species_names[i]:<10s}" + " ".join(f"{c:12.6f}" for c in coeffs) + "\n")
        data = read_record(f_in); f_txt.write("\n--- COFD (Binary Diffusion Coefficients) ---\n")
        for i in range(KK):
            for j in range(KK):
                coeffs = struct.unpack_from('<4d', data, (i * KK + j) * 32); f_txt.write(f"  D_{species_names[i]}-{species_names[j]:<10s}" + " ".join(f"{c:12.6f}" for c in coeffs) + "\n")
    print("'tran_out.txt' generated.")

except Exception as e:
    print(f"\nAn error occurred: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)