# 04 — Machine Learning: Prediksi ISPU (Materi Inti Skripsi)

## Judul Skripsi

> **PEMODELAN PREDIKTIF KUALITAS UDARA MENGGUNAKAN RANDOM FOREST DAN ADAPTIVE
> NEURO-FUZZY INFERENCE SYSTEM DENGAN OPTIMASI PARTICLE SWARM OPTIMIZATION**
> *(Studi Kasus: Provinsi Bengkulu)*

---

## Tujuan implementasi

Membuktikan bahwa tiga komponen algoritmik inti skripsi dapat dijalankan
dalam satu pipeline yang terintegrasi dan bebas data leakage:

| Komponen | Peran |
|---|---|
| **Random Forest (RF)** | Baseline regresi + pemilihan fitur berdasarkan *feature importance* |
| **ANFIS Sugeno orde-1** | Model nonlinear dengan MF Gaussian; parameter konsekuen diestimasi Least Squares |
| **PSO** | Mengoptimasi parameter premis (pusat dan lebar MF) menggunakan RMSE validasi sebagai fitness |

---

## Arsitektur pipeline

```
CSV input
   ↓ validasi kolom, konversi tipe, buang baris NaN target
   ↓ buat fitur lag ISPU (ISPU_lag1) — temporal, bukan leakage
   ↓
Pembagian kronologis 70% / 15% / 15% (tanpa shuffle)
   train       → fit semua transformer dan model
   validation  → PSO fitness; tidak digunakan oleh RF
   test        → evaluasi final satu kali; tidak menyentuh tuning
   ↓
Imputasi median (fit pada train saja)
   ↓
RF feature importance (fit pada train saja)
   → top-k fitur dipilih untuk ANFIS (default: 3 fitur)
   ↓
StandardScaler fitur ANFIS (fit pada train saja)
   ↓
RF Baseline (fit pada full feature set, train saja)
   ↓
PSO (n_particles=20, max_iter=50, patience=10)
   setiap evaluasi partikel:
     • set premise params ANFIS
     • fit consequent LS pada train
     • hitung RMSE pada validation → fitness
   ↓
ANFIS-PSO final:
   • set premise params terbaik dari PSO
   • fit consequent LS pada train+validation
   ↓
Evaluasi test set: MAE, RMSE, R²  (RF vs ANFIS-PSO)
   ↓
Output: predictions.csv, metrics.json, feature_importance.csv,
        pso_convergence.csv, anfis_params.npz, random_forest_ispu.joblib,
        hasil_prediksi_dashboard.csv (kompatibel folder 05)
```

---

## Skema ANFIS Sugeno orde-1

- **MF**: Gaussian — `μ(x) = exp(-0.5 * ((x - c) / σ)²)`
- **Aturan**: produk antar MF semua fitur
- **Default**: 3 fitur × 2 MF/fitur = **8 aturan fuzzy**
- **Konsekuen**: `f_r(x) = a_r0 + a_r1·x1 + ... + a_rk·xk` (Least Squares)
- **Output**: `ŷ = Σ_r w̄_r · f_r(x)` (w̄_r = kekuatan aturan ternormalisasi)

---

## Struktur data input

| Kolom | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `Tanggal` | datetime | **Ya** | Format `YYYY-MM-DD` |
| `ISPU` | float | **Ya** | Nilai ISPU target (observasi ke-t+lag) |
| `PM25` | float | Disarankan | μg/m³ |
| `PM10` | float | Disarankan | μg/m³ |
| `SO2` | float | Opsional | μg/m³ |
| `CO` | float | Opsional | mg/m³ |
| `NO2` | float | Opsional | μg/m³ |
| `O3` | float | Opsional | μg/m³ |
| `suhu` | float | Opsional | °C |
| `kelembapan` | float | Opsional | % |
| `kecepatan_angin` | float | Opsional | m/s |

Kolom yang tidak tersedia akan diabaikan secara otomatis.
Fitur lag ISPU dibuat otomatis oleh pipeline.

---

## Cara menjalankan

### 1. Siapkan lingkungan

```bash
# Buat virtual environment
python -m venv .venv

# Aktifkan:
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Jalankan dengan data sintetis (smoke test)

```bash
# Langkah 1: buat (atau perbarui) data sintetis
python src/generate_demo_data.py

# Langkah 2: jalankan pipeline penuh
python src/pipeline.py

# Opsi kecil/cepat untuk testing
python src/pipeline.py --pso-particles 10 --pso-iters 20 --anfis-features 2
```

### 3. Jalankan dengan data CSV milik Anda

```bash
python src/pipeline.py --input path/ke/data_anda.csv
```

### 4. Opsi CLI lengkap

```
python src/pipeline.py --help

  --input PATH          File CSV input (default: data/synthetic_air_quality.csv)
  --output DIR          Direktori output (default: output/)
  --lag N               Horizon prediksi lag ISPU (default: 1)
  --train-frac F        Proporsi train (default: 0.70)
  --val-frac F          Proporsi validation (default: 0.15)
  --anfis-features N    Jumlah fitur top untuk ANFIS (default: 3)
  --n-mf N              Jumlah MF per fitur (default: 2)
  --pso-particles N     Jumlah partikel PSO (default: 20)
  --pso-iters N         Maksimum iterasi PSO (default: 50)
  --pso-patience N      Berhenti awal jika tidak ada perbaikan (default: 10)
  --seed N              Random seed (default: 42)
  --verbose             Tampilkan log detail
```

### 5. Jalankan unit tests

```bash
pytest tests/test_pipeline.py -v
# atau tanpa pytest:
python tests/test_pipeline.py
```

---

## Output yang dihasilkan

| File | Isi |
|---|---|
| `output/predictions.csv` | Tanggal, ISPU_Aktual, ISPU_Prediksi_RF, ISPU_Prediksi_ANFIS_PSO |
| `output/hasil_prediksi_dashboard.csv` | Format kompatibel dengan dashboard folder 05 |
| `output/metrics.json` | MAE, RMSE, R² untuk RF dan ANFIS-PSO; info split dan PSO |
| `output/feature_importance.csv` | Kepentingan fitur RF (urut menurun) |
| `output/pso_convergence.csv` | Riwayat RMSE terbaik per iterasi PSO |
| `output/random_forest_ispu.joblib` | Model RF tersimpan (joblib) |
| `output/anfis_params.npz` | Parameter premis dan konsekuen ANFIS terbaik |

---

## Batasan dan catatan penelitian

- **Data sintetis ≠ data resmi**: `data/synthetic_air_quality.csv` dibuat oleh
  komputer dengan pola AR(1) + seasonal. Metrik dari data ini tidak boleh dikutip
  sebagai hasil penelitian skripsi.
- **Gunakan data resmi**: untuk skripsi, gantikan dengan data ISPU tervalidasi dari
  KLHK / BMKG / BPPD Bengkulu, pastikan satuan, horizon prediksi, dan definisi ISPU
  konsisten dengan metodologi.
- **Target leakage dilarang**: kolom `ISPU` pada timestamp yang sama tidak boleh
  menjadi prediktor. Pipeline ini membuat `ISPU_lag1` (atau lag-N) agar temporal sah.
- **Test set satu kali**: jangan menggunakan test set untuk tuning — ini dilakukan
  oleh pipeline secara otomatis (PSO hanya melihat validation set).
- **R² dapat negatif**: jika model lebih buruk dari mean baseline, R² < 0 dan
  ditampilkan apa adanya; jangan disembunyikan.

---

## Hubungan dengan folder lain

| Folder | Hubungan |
|---|---|
| `05-hci-data-visualization/` | Baca `output/hasil_prediksi_dashboard.csv` untuk dashboard |
| Folder lain | Tidak ada hubungan algoritmik; implementasi terpisah |
