# Scholarship Eligibility Prediction using Machine Learning

## Deskripsi Proyek

Proyek ini bertujuan untuk membangun model Machine Learning yang dapat memprediksi kelayakan penerima beasiswa berdasarkan data akademik dan kondisi sosial ekonomi mahasiswa. Model dikembangkan menggunakan algoritma Random Forest, Support Vector Machine (SVM), dan Extreme Gradient Boosting (XGBoost), kemudian dibandingkan untuk memperoleh algoritma dengan performa terbaik.

Penelitian ini merupakan implementasi tahapan Machine Learning mulai dari data understanding, data preparation, modeling, evaluasi model, interpretasi menggunakan SHAP, hingga deployment aplikasi berbasis Streamlit sebagai bagian dari Proyek Akhir Mata Kuliah Pembelajaran Mesin.

---

## Latar Belakang

Proses seleksi penerima beasiswa umumnya masih dilakukan secara manual sehingga membutuhkan waktu yang lama dan berpotensi menimbulkan subjektivitas dalam pengambilan keputusan. Melalui pendekatan Machine Learning, proses seleksi diharapkan dapat dilakukan secara lebih cepat, objektif, dan konsisten berdasarkan pola dari data historis mahasiswa.

---

## Rumusan Masalah

- Bagaimana membangun model Machine Learning untuk mengklasifikasikan kelayakan penerima beasiswa?
- Algoritma manakah yang memberikan performa terbaik dalam memprediksi kelayakan penerima beasiswa?
- Bagaimana mengimplementasikan model terbaik ke dalam aplikasi berbasis web?

---

## Tujuan Penelitian

- Membangun model klasifikasi penerima beasiswa menggunakan Machine Learning.
- Membandingkan performa algoritma Random Forest, Support Vector Machine (SVM), dan XGBoost.
- Menentukan model terbaik berdasarkan hasil evaluasi.
- Mengimplementasikan model terbaik ke dalam aplikasi berbasis Streamlit.

---

## Dataset

Dataset yang digunakan merupakan **Dataset Seleksi Beasiswa** yang diperoleh dari Kaggle.

Dataset terdiri dari **300 data mahasiswa** dengan **8 atribut** dan **1 atribut target**.

### Feature

- IPK
- Semester
- Penghasilan Orang Tua
- Tanggungan Keluarga
- Prestasi
- Aktif Organisasi
- Status Rumah
- Jenis Kelamin

### Target

- Diterima Beasiswa
  - Ya
  - Tidak

---

## Metodologi

Penelitian menggunakan metodologi **CRISP-DM (Cross Industry Standard Process for Data Mining)** yang terdiri atas:

- Business Understanding
- Data Understanding
- Data Preparation
- Modeling
- Evaluation
- Deployment

---

## Algoritma

- Random Forest
- Support Vector Machine (SVM)
- Extreme Gradient Boosting (XGBoost)

---

## Library yang Digunakan

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

---

## Evaluasi Model

Metrik evaluasi yang digunakan meliputi:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

Hasil perbandingan model menunjukkan bahwa Random Forest memberikan performa terbaik.

| Model | Accuracy |
|--------|----------|
| Random Forest | **73.33%** |
| Support Vector Machine (SVM) | **70.00%** |
| XGBoost | **68.33%** |

---

## Struktur Repository

```text
uas-ml/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset/
│
├── notebook/
│
├── images/
│
└── laporan/
```

---

## Cara Menjalankan

Clone repository.

```bash
git clone https://github.com/rfpmaa/uas-ml.git
```

Masuk ke folder project.

```bash
cd uas-ml
```

Install seluruh library.

```bash
pip install -r requirements.txt
```

Jalankan aplikasi Streamlit.

```bash
streamlit run app.py
```

---

## Hasil

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

### Confusion Matrix Random Forest

![RF](images/confusion_matrix_rf.png)

### Confusion Matrix Support Vector Machine

![SVM](images/confusion_matrix_svm.png)

### Confusion Matrix XGBoost

![XGBoost](images/confusion_matrix_xgb.png)

### ROC Curve Random Forest

![ROC RF](images/roc_curve_rf.png)

### ROC Curve Support Vector Machine

![ROC SVM](images/roc_curve_svm.png)

### ROC Curve XGBoost

![ROC XGB](images/roc_curve_xgb.png)

### SHAP Summary Plot

![SHAP](images/shap_summary.png)

---

## Deployment

- **GitHub Repository:** https://github.com/rfpmaa/uas-ml
- **Streamlit:** *(isi link deployment kamu nanti)*
- **YouTube Presentation:** *(isi link video nanti)*

---

## Penulis

**Rafania Putri Mahendra**

Program Studi Teknik Informatika  
Fakultas Ilmu Komputer  
Universitas Dian Nuswantoro

Mata Kuliah Pembelajaran Mesin
