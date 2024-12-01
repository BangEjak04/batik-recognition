# Batik Recognition with Xception

Proyek ini adalah aplikasi klasifikasi motif batik menggunakan model **Xception** sebagai feature extractor dan lapisan tambahan untuk klasifikasi. Model dikembangkan dengan framework **TensorFlow** dan **Keras**, menggunakan dataset gambar motif batik.

## Fitur
- Klasifikasi gambar motif batik dengan akurasi tinggi.
- Visualisasi hasil training (loss dan akurasi) melalui grafik.
- Confusion matrix dan laporan klasifikasi.
- Penyimpanan model terbaik secara otomatis.

## Dataset
Dataset memiliki struktur sebagai berikut:

batik_dataset/
├── Batik Dayak/
├── Batik Geblek Renteng/
├── Batik Ikat Celup/
├── Batik Insang/
├── Batik Kawung/
├── Batik Lasem/
├── Batik Megamendung/
├── Batik Pala/
├── Batik Parang/
├── Batik Sekar Jagad/
└── Batik Tambal/

Setiap folder berisi gambar dengan label sesuai nama folder.

## Teknologi dan Library yang Digunakan
- **Python**
- **TensorFlow** & **Keras**
- **Pandas** dan **NumPy** untuk manipulasi data
- **Matplotlib** dan **Seaborn** untuk visualisasi
- **scikit-learn** untuk evaluasi model

## Instalasi
1. Clone repositori ini:
   ```bash
   git clone https://github.com/BangEjak04/batik-recognition.git
   cd batik-recognition
   ```
2. Buat virtual environment dan aktifkan:
    ```
    python -m venv .venv
    source .venv/bin/activate   # Untuk Linux/Mac
    .venv\Scripts\activate      # Untuk Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Cara Menjalankan
1. Jalankan server Flask:
    ```bash
    python main.py
    ```
2. Buka browser dan akses:
    ```bash
    http://127.0.0.1:5001
    ```

## Cara Menjalankan
1. Jalankan skrip utama untuk melatih model:
    ```bash
    python main.py
    ```
2. Model akan dilatih menggunakan data train, dan hasil validasi ditampilkan.
3. Model terbaik akan disimpan dalam format .keras.

## Visualisasi Hasil
1. Grafik Loss dan Akurasi:
    - Grafik menunjukkan perubahan loss dan accuracy selama pelatihan.
2. Confusion Matrix:
    - Menunjukkan performa model dalam memprediksi kelas.
3. Laporan Klasifikasi:
    - Termasuk metrik seperti precision, recall, dan F1-score.

## Hasil Model
Setelah pelatihan, akurasi pada test set akan dicetak dan disimpan.

## Lisensi
Proyek ini menggunakan lisensi MIT.

## Kontak
Jika ada pertanyaan atau saran, silakan hubungi:

- Nama: Reza Andyah Wijaya
- Email: rezaandyahw@gmail.com