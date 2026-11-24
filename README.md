# Klasifikasi Komentar Spam dan Promosi Judi Online Berbahasa Indonesia Menggunakan Skema Group Split Anti-Leakage

**UAS Kelompok 5 - Natural Language Processing (Spam Detection Track)**

## Struktur Proyek
- ambil_data.py: Pengambilan data ulasan mentah (data/raw/reviews_raw_id.csv).
- bersihkan_data.py: Pembersihan teks, normalisasi (norm_text), pelabelan, dan hashing dokumen (data/processed/dataset_judol_clean.csv).
- split_data.py: Pembagian dataset (train.csv, val.csv, test.csv) menggunakan GroupShuffleSplit berbasis group_id dengan audit anti-leakage (group_id dan doc_hash).
- latih_model.py: Ekstraksi fitur TF-IDF (ngram_range=(1, 2), max_features=5000), pelatihan Multinomial Naive Bayes dan baseline Logistic Regression, serta penyimpanan model dan visualisasi ke artifacts/.

## Ringkasan Dataset & Audit Anti-Leakage
- Total data bersih: 1.241 baris (kelas 0 non-spam: 1.239, kelas 1 spam/judol: 2).
- Overlap group_id antar split (train-val, train-test, val-test): 0 (bebas kebocoran grup).
- Overlap doc_hash antar split (train-val, train-test, val-test): 0 (bebas duplikasi teks).

## Hasil Evaluasi Model (Test Set)
- Multinomial Naive Bayes (alpha=0.5): Accuracy 1.00, Macro F1 0.499 (kelas 0 F1 = 1.00, kelas 1 F1 = 0.00 karena jumlah sampel kelas positif di dataset sangat terbatas yaitu 2 baris total dan 1 baris pada test set).
- Logistic Regression (class_weight='balanced'): Accuracy 1.00, Macro F1 0.499.

## Cara Menjalankan
python bersihkan_data.py
python split_data.py
python latih_model.py
