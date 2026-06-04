import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_path, output_dir):
    """
    Fungsi produksi untuk preprocessing dataset Pima Indians Diabetes.
    """
    print("[INFO] Memulai proses preprocessing...")
    
    # 1. Data Loading
    if not os.path.exists(input_path):
        print(f"[ERROR] File {input_path} tidak ditemukan!")
        return
    
    df = pd.read_csv(input_path)
    print(f"[INFO] Berhasil memuat data dari {input_path}")

    # 2. Data Cleaning (Handling zero values as NaN)
    impute_cols = ['Glucose', 'BloodPressure', 'BMI', 'SkinThickness', 'Insulin']
    df[impute_cols] = df[impute_cols].replace(0, np.nan)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Imputasi Median
    print("[INFO] Melakukan imputasi median pada kolom critical...")
    imputer = SimpleImputer(strategy='median')
    X_imputed_data = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed_data, columns=X.columns)

    # 3. Feature Scaling
    print("[INFO] Melakukan standardisasi fitur...")
    scaler = StandardScaler()
    X_scaled_data = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled_data, columns=X.columns)

    # 4. Export Artefak & Penyimpanan Clean Data (Dinamis berbasis output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Menyimpan joblib di dalam folder preprocessing utama agar rapi
    base_preprocessing_dir = os.path.dirname(output_dir)
    joblib.dump(imputer, os.path.join(base_preprocessing_dir, 'imputer_Adithya.joblib'))
    joblib.dump(scaler, os.path.join(base_preprocessing_dir, 'scaler_Adithya.joblib'))
    print(f"[INFO] Artefak imputer dan scaler telah disimpan di: {base_preprocessing_dir}")

    # 5. Penyimpanan Clean Data
    final_df = pd.concat([X_scaled, y], axis=1)
    output_path = os.path.join(output_dir, 'diabetes_clean.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"[INFO] Selesai! Data bersih disimpan di: {output_path}")

if __name__ == "__main__":
    # Menggunakan absolute path berbasis lokasi file agar aman dijalankan dari direktori mana pun
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    
    INPUT_FILE = os.path.join(BASE_DIR, 'namadataset_raw', 'diabetes.csv')
    OUTPUT_FOLDER = os.path.join(BASE_DIR, 'preprocessing', 'namadataset_preprocessing')
    
    run_preprocessing(INPUT_FILE, OUTPUT_FOLDER)