import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Sistem Prediksi Beasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CSS CUSTOM 
# ==========================
st.markdown("""
<style>
h1 { text-align: center; color: #0068C9; }
h2, h3 { color: #0068C9; }

[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
    color: #FF4B4B !important; 
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATASET & MODEL
# ==========================
@st.cache_data
def load_data():
    return pd.read_csv("dataset/beasiswa.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_model.pkl")

df = load_data()
model = load_model()

# ==========================
# SIDEBAR NAVIGATION
# ==========================
st.sidebar.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135810.png" width="120">
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.title("Menu Utama")

menu = st.sidebar.radio(
    "Navigasi Halaman:",
    [
        "Dashboard EDA", 
        "Model Demo", 
        "Evaluasi Model", 
        "Interpretasi Hasil",
        "Dokumentasi"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Machine Learning ini dibuat untuk memenuhi Tugas Akhir (UAS).")

# ==========================
# MENU 1: DASHBOARD EDA
# ==========================
if menu == "Dashboard EDA":
    st.title("Dashboard Exploratory Data Analysis")
    st.markdown("---")
    
    st.write("Visualisasi interaktif dan analisis dari dataset historis pendaftar beasiswa.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", f"{df.shape[0]} Baris")
    with col2:
        st.metric("Total Fitur", f"{df.shape[1]-1} Kolom")
    with col3:
        st.metric("Diterima (1)", len(df[df['Diterima_Beasiswa'] == 1]))
    with col4:
        st.metric("Ditolak (0)", len(df[df['Diterima_Beasiswa'] == 0]))

    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Distribusi Target (Diterima vs Ditolak)")
        st.write("Grafik interaktif jumlah mahasiswa berdasarkan status penerimaan.")
        st.bar_chart(df['Diterima_Beasiswa'].value_counts())
        
    with col_chart2:
        st.subheader("Analisis Korelasi (Heatmap)")
        st.write("Hubungan antar variabel. Semakin mendekati 1 atau -1, semakin kuat pengaruhnya.")
        st.image("images/correlation_heatmap.png", use_container_width=True)

    st.subheader("Distribusi IPK Berdasarkan Status Beasiswa")

    fig = px.histogram(
        df,
        x="IPK",
        color="Diterima_Beasiswa",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cuplikan Dataset")
    st.dataframe(df, use_container_width=True)

# ==========================
# MENU 2: MODEL DEMO (PREDIKSI)
# ==========================
elif menu == "Model Demo":
    st.title("Model Demo (Prediksi Kelayakan)")
    st.markdown("---")
    st.write("Silakan masukkan data mahasiswa pada form di bawah ini untuk mendapatkan prediksi dari model **Random Forest**.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Data Akademik & Ekonomi")
            ipk = st.number_input("IPK (Skala 4.00)", min_value=0.00, max_value=4.00, value=3.00, step=0.01)
            semester = st.selectbox("Semester Saat Ini", [1,2,3,4,5,6,7,8])
            penghasilan_input = st.text_input("Penghasilan Orang Tua (Rp)", value="3.000.000")
            penghasilan = int(penghasilan_input.replace(".", "").replace(",", ""))
            tanggungan = st.number_input("Jumlah Tanggungan Keluarga", min_value=1, max_value=10, value=3)

        with col2:
            st.markdown("#### Data Non-Akademik")
            prestasi = st.selectbox("Tingkat Prestasi Tertinggi", ["Internasional", "Kabupaten", "Nasional", "Provinsi", "Tidak Ada"])
            organisasi = st.selectbox("Aktif Berorganisasi?", ["Tidak", "Ya"])
            status_rumah = st.selectbox("Status Kepemilikan Tempat Tinggal", ["Kontrak", "Kos/Asrama", "Menumpang", "Milik Sendiri"])
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["L", "P"])

    prestasi_map = {"Internasional":0, "Kabupaten":1, "Nasional":2, "Provinsi":3, "Tidak Ada":4}
    organisasi_map = {"Tidak":0, "Ya":1}
    rumah_map = {"Kontrak":0, "Kos/Asrama":1, "Menumpang":2, "Milik Sendiri":3}
    jk_map = {"L":0, "P":1}

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Prediksi", use_container_width=True):
        data_input = [[
            ipk, semester, penghasilan, tanggungan, 
            prestasi_map[prestasi], organisasi_map[organisasi], 
            rumah_map[status_rumah], jk_map[jenis_kelamin]
        ]]

        data_df = pd.DataFrame(data_input, columns=[
            "IPK", "Semester", "Penghasilan_Ortu", "Tanggungan_Keluarga", 
            "Prestasi", "Aktif_Organisasi", "Status_Rumah", "Jenis_Kelamin"
        ])

        hasil = model.predict(data_df)
        prob = model.predict_proba(data_df)

        st.markdown("---")
        st.subheader("Hasil Prediksi Model")

        if hasil[0] == 1:
            st.success("Mahasiswa diprediksi **LAYAK** menerima beasiswa.")
            st.metric("Probabilitas/Kepercayaan Model", f"{prob[0][1]*100:.2f}%")
        else:
            st.error("Mahasiswa diprediksi **TIDAK LAYAK** menerima beasiswa.")
            st.metric("Probabilitas/Kepercayaan Model", f"{prob[0][0]*100:.2f}%")

# ==========================
# MENU 3: EVALUASI MODEL
# ==========================
elif menu == "Evaluasi Model":
    st.title("Evaluasi Model Machine Learning")
    st.markdown("---")
    
    st.subheader("1. Metrik Evaluasi Klasifikasi")
    st.info("""
    Berikut adalah arti dari masing-masing metrik yang digunakan untuk mengevaluasi model klasifikasi:
    * **Accuracy:** Persentase tebakan model yang benar secara keseluruhan.
    * **Precision:** Ketepatan model saat memprediksi mahasiswa 'Layak' (seberapa kecil error salah sasaran).
    * **Recall:** Kemampuan model menemukan/mendeteksi semua mahasiswa yang benar-benar 'Layak'.
    * **F1-Score:** Rata-rata harmonis yang menyeimbangkan nilai Precision dan Recall.
    """)
    
    hasil_model = pd.DataFrame({
        "Algoritma": ["Random Forest", "Support Vector Machine (SVM)"],
        "Accuracy": ["73.33%", "70.00%"],
        "Precision": ["75.12%", "71.45%"],
        "Recall": ["72.50%", "68.20%"],
        "F1-Score": ["73.79%", "69.78%"]
    })
    st.table(hasil_model)
    
    st.markdown("---")
    
    st.subheader("2. Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        st.image("images/confusion_matrix_rf.png", use_container_width=True)
    with col2:
        st.markdown("**SVM**")
        st.image("images/confusion_matrix_svm.png", use_container_width=True)

    st.markdown("---")
    
    st.subheader("3. ROC Curve")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Random Forest**")
        st.image("images/roc_curve_rf.png", use_container_width=True)
    with col4:
        st.markdown("**SVM**")
        st.image("images/roc_curve_svm.png", use_container_width=True)

# ==========================
# MENU 4: INTERPRETASI HASIL
# ==========================
elif menu == "Interpretasi Hasil":
    st.title("Interpretasi Hasil & Business Insights")
    st.markdown("---")
    
    st.subheader("Penjelasan Model")
    st.write("""
    Berdasarkan tahapan evaluasi, **Random Forest** terbukti sebagai algoritma terbaik dengan akurasi **73.33%**. 
    Algoritma ini bekerja dengan cara membangun banyak pohon keputusan (decision trees) dan menggabungkan hasilnya 
    untuk menghindari bias dan overfitting.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Feature Importance (Interpretasi SHAP)")
    st.image("images/shap_summary.png", width=550)
    st.info("""
    **Cara membaca visualisasi SHAP di atas:**
    * Semakin tinggi letak suatu fitur (berada di urutan teratas), semakin besar pengaruhnya terhadap keputusan model.
    * Berdasarkan hasil interpretasi, fitur seperti **IPK**, **Penghasilan Orang Tua**, dan **Jumlah Tanggungan** menjadi tiga faktor utama yang paling dipertimbangkan sistem.
    """)
    
    st.markdown("---")
    st.subheader("Business Insights (Dampak Bisnis/Institusi)")
    st.success("""
    Penerapan model ini memberikan keuntungan utama sebagai berikut:
    1. **Efisiensi:** Mampu menyeleksi kandidat dalam waktu singkat.
    2. **Objektivitas Tinggi:** Meminimalisir favoritisme atau human error.
    3. **Tepat Sasaran:** Memastikan alokasi dana beasiswa diberikan kepada kandidat yang membutuhkan dan berprestasi.
    """)

    
    st.title("Interpretasi Hasil & Business Insights")
    st.markdown("---")
    
    st.subheader("Penjelasan Model")
    st.write("""
    Berdasarkan tahapan evaluasi, **Random Forest** terbukti sebagai algoritma terbaik dengan akurasi **73.33%**. 
    Algoritma ini bekerja dengan cara membangun banyak pohon keputusan (*decision trees*) dan menggabungkan hasilnya. 
    Hal ini membuat Random Forest sangat baik dalam menangani kombinasi data numerik (seperti IPK dan Penghasilan) 
    serta data kategorikal (seperti Status Rumah dan Prestasi) tanpa mudah terjebak pada *overfitting*.
    """)

    st.markdown("<br>", unsafe_allow_html=True) 
    st.subheader("Feature Importance (Interpretasi SHAP)")
    
    st.image("images/shap_summary.png", width=350)
    
    st.info("""
    **Cara membaca visualisasi SHAP di atas:**
    * Semakin tinggi letak suatu fitur (berada di urutan teratas), semakin besar pengaruhnya terhadap keputusan model.
    * Berdasarkan hasil interpretasi model, fitur seperti **IPK**, **Penghasilan Orang Tua**, dan **Jumlah Tanggungan** menjadi tiga faktor utama yang paling dipertimbangkan oleh sistem dalam menentukan kelayakan penerima beasiswa.
    """)
    st.markdown("---")
    
    st.subheader("Business Insights (Dampak Bisnis/Institusi)")
    st.success("""
    Penerapan model Machine Learning ini memberikan 3 keuntungan utama bagi institusi/kampus:
    
    1. **Efisiensi Waktu & Biaya:** Panitia seleksi tidak perlu lagi mengecek kelayakan ratusan atau ribuan pelamar secara manual satu per satu. Sistem dapat menyeleksi kandidat dalam hitungan detik.
    2. **Objektivitas Tinggi (Menghindari Bias):** Keputusan didasarkan murni pada pola data (IPK, Tanggungan, Penghasilan, dll), meminimalisir unsur favoritisme atau kesalahan manusia (*human error*).
    3. **Tepat Sasaran:** Model membantu memastikan bahwa alokasi dana beasiswa benar-benar jatuh ke tangan mahasiswa yang secara statistik dan historis memiliki profil yang paling membutuhkan dan berprestasi.
    """)

# ==========================
# MENU 5: DOKUMENTASI
# ==========================
elif menu == "Dokumentasi":
    st.title("Dokumentasi Proyek")
    st.markdown("---")
    
    st.subheader("Metodologi (Alur CRISP-DM)")
    st.write("Pengembangan aplikasi ini mematuhi standar CRISP-DM yang terdiri dari:")
    st.markdown("""
    * **Business Understanding:** Menentukan tujuan pembuatan sistem SPK kelayakan beasiswa.
    * **Data Understanding:** Mengeksplorasi dataset 300 mahasiswa.
    * **Data Preparation:** Melakukan penanganan missing values, Label Encoding, dan membagi dataset (80:20).
    * **Modeling:** Melakukan tuning (GridSearchCV) dan pelatihan pada model Random Forest dan SVM.
    * **Evaluation:** Membandingkan nilai performa (Akurasi, ROC-AUC) untuk menentukan algoritma final.
    * **Deployment:** Mengintegrasikan model ke dalam Streamlit Web App.
    """)
    
    st.subheader("Library yang Digunakan")
    st.code("Streamlit, Pandas, NumPy, Scikit-Learn, Joblib, Matplotlib, Seaborn, Plotly, SHAP", language="python")

    st.subheader("Informasi Dataset")
    st.write("""
    Dataset yang digunakan merupakan data pendaftar beasiswa yang terdiri dari **300 baris data** dan **9 kolom** (8 Atribut + 1 Target). 
    * **Atribut (Fitur):** IPK, Semester, Penghasilan Orang Tua, Tanggungan Keluarga, Prestasi, Aktif Organisasi, Status Rumah, Jenis Kelamin.
    * **Target (Label):** `Diterima_Beasiswa` (1 = Layak/Diterima, 0 = Tidak Layak/Ditolak).
    """)
    
    st.subheader("Penjelasan Fitur Input")
    st.write("""
    * **IPK:** Indeks Prestasi Kumulatif terakhir mahasiswa.
    * **Semester:** Tingkat semester mahasiswa saat mendaftar.
    * **Penghasilan Ortu:** Total penghasilan gabungan orang tua (dalam mata uang Rupiah).
    * **Tanggungan Keluarga:** Jumlah tanggungan anak dari orang tua.
    * **Prestasi:** Tingkat pencapaian lomba tertinggi yang pernah diraih.
    * **Aktif Organisasi:** Keterlibatan dalam kegiatan kampus/BEM/HIMA.
    * **Status Rumah:** Kepemilikan hunian tetap keluarga.
    * **Jenis Kelamin:** L = Laki-laki, P = Perempuan.
    """)
    
    st.subheader("Kesimpulan")
    st.write("""
    Model Machine Learning berhasil dibangun dan di-deploy. Pendekatan Random Forest terbukti unggul dibandingkan SVM 
    dalam mengatasi masalah klasifikasi dengan campuran tipe data. Sistem ini siap diuji coba secara lebih luas 
    sebagai Sistem Pendukung Keputusan (SPK) di lingkungan institusi pendidikan.
    """)
    
    st.subheader("Cara Penggunaan Aplikasi")
    st.info("""
    1. Buka menu **Model Demo** pada navigasi di sebelah kiri.
    2. Masukkan data mahasiswa pelamar beasiswa ke dalam form yang tersedia.
    3. Pastikan format angka untuk penghasilan sudah benar.
    4. Klik tombol **Prediksi Sekarang**.
    5. Sistem akan mengeluarkan hasil (Layak/Tidak Layak) beserta nilai probabilitasnya.
    """)

# ==========================
# FOOTER 
# ==========================
st.markdown("---")
st.caption("Dibuat menggunakan Streamlit, Scikit-learn, XGBoost, SHAP, dan Plotly.")
