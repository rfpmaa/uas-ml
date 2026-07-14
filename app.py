import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================
# PAGE CONFIG (WAJIB PALING ATAS)
# ==========================
st.set_page_config(
    page_title="Sistem Prediksi Beasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CSS CUSTOM UNTUK TAMPILAN
# ==========================
st.markdown("""
<style>
/* Styling Umum */
h1 { text-align: center; color: #0068C9; }
h2, h3 { color: #0068C9; }

/* Menghilangkan radio button default di sidebar */
[data-testid="stSidebar"] div[role="radiogroup"] div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Mengubah warna teks menu yang sedang aktif/dipilih */
[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] label[aria-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] div[data-baseweb="radio"] input[type="radio"]:checked + div p {
    color: #0068C9 !important;
    font-weight: 800 !important;
    font-size: 17px !important;
}

/* Hover effect */
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
        "Beranda Utama",
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
# STEP 9: MENU BERANDA
# ==========================
if menu == "Beranda Utama":
    st.title("🎓 Sistem Prediksi Penerima Beasiswa")
    
    # Gambar Ilustrasi Beasiswa
    st.image("https://images.unsplash.com/photo-1523050854058-8df90110c9f1?q=80&w=1200&auto=format&fit=crop", 
             use_container_width=True, caption="Ilustrasi Pendidikan & Beasiswa")
    
    st.markdown("---")
    
    st.subheader("🎯 Tujuan Penelitian")
    st.write("""
    Penelitian ini bertujuan untuk mengotomatisasi proses seleksi penerima beasiswa secara objektif, 
    tepat sasaran, dan efisien menggunakan teknik Machine Learning. Model yang dibangun diharapkan 
    mampu menggantikan metode manual yang memakan waktu dan meminimalisir bias dalam penentuan kandidat.
    """)
    
    st.subheader("💡 Ringkasan Proyek")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Dataset", "300 Data")
    with col2:
        st.metric("Jumlah Fitur", "8 Atribut")
    with col3:
        st.metric("Model Terbaik", "Random Forest")
    with col4:
        st.metric("Akurasi Model", "73.33%")

# ==========================
# STEP 10: DASHBOARD EDA
# ==========================
elif menu == "Dashboard EDA":
    st.title("📊 Dashboard Exploratory Data Analysis")
    st.markdown("---")
    
    st.subheader("Cuplikan Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Visualisasi Interaktif (Plotly)")
    
    col_chart1, col_chart2 = st.columns(2)
    
    # Pie Chart
    with col_chart1:
        fig_pie = px.pie(
            df, 
            names="Diterima_Beasiswa", 
            title="Distribusi Status Beasiswa",
            color_discrete_sequence=px.colors.sequential.RdBu,
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # Box Plot IPK
    with col_chart2:
        # Konversi tipe data agar label di boxplot rapi
        df_plot = df.copy()
        df_plot['Status'] = df_plot['Diterima_Beasiswa'].map({1: 'Diterima', 0: 'Ditolak'})
        
        fig_box = px.box(
            df_plot, 
            x="Status", 
            y="IPK", 
            color="Status",
            title="Pengaruh IPK terhadap Status Beasiswa"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("Analisis Korelasi (Heatmap)")
    st.image("images/correlation_heatmap.png", width=650)


# ==========================
# MENU 2: MODEL DEMO (PREDIKSI)
# ==========================
elif menu == "Model Demo":
    st.title("🔮 Model Demo (Prediksi Kelayakan)")
    st.markdown("---")
    st.write("Silakan masukkan data mahasiswa pada form di bawah ini untuk mendapatkan prediksi.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎓 Data Akademik & Ekonomi")
            ipk = st.number_input("IPK (Skala 4.00)", min_value=0.00, max_value=4.00, value=3.00, step=0.01)
            semester = st.selectbox("Semester Saat Ini", [1,2,3,4,5,6,7,8])
            penghasilan_input = st.text_input("Penghasilan Orang Tua (Rp)", value="3000000")
            penghasilan = int(penghasilan_input.replace(".", "").replace(",", ""))
            tanggungan = st.number_input("Jumlah Tanggungan Keluarga", min_value=1, max_value=10, value=3)

        with col2:
            st.markdown("#### 🏆 Data Non-Akademik")
            prestasi = st.selectbox("Tingkat Prestasi Tertinggi", ["Internasional", "Kabupaten", "Nasional", "Provinsi", "Tidak Ada"])
            organisasi = st.selectbox("Aktif Berorganisasi?", ["Tidak", "Ya"])
            status_rumah = st.selectbox("Status Kepemilikan Tempat Tinggal", ["Kontrak", "Kos/Asrama", "Menumpang", "Milik Sendiri"])
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["L", "P"])

    prestasi_map = {"Internasional":0, "Kabupaten":1, "Nasional":2, "Provinsi":3, "Tidak Ada":4}
    organisasi_map = {"Tidak":0, "Ya":1}
    rumah_map = {"Kontrak":0, "Kos/Asrama":1, "Menumpang":2, "Milik Sendiri":3}
    jk_map = {"L":0, "P":1}

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Prediksi Sekarang", use_container_width=True):
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
        st.subheader("🎯 Hasil Prediksi Model")

        if hasil[0] == 1:
            st.success("🎉 Mahasiswa diprediksi **LAYAK** menerima beasiswa.")
            st.metric("Probabilitas/Kepercayaan Model", f"{prob[0][1]*100:.2f}%")
        else:
            st.error("⚠️ Mahasiswa diprediksi **TIDAK LAYAK** menerima beasiswa.")
            st.metric("Probabilitas/Kepercayaan Model", f"{prob[0][0]*100:.2f}%")


# ==========================
# STEP 11: EVALUASI MODEL
# ==========================
elif menu == "Evaluasi Model":
    st.title("📈 Evaluasi Model Machine Learning")
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
    
    st.subheader("2. Confusion Matrix & ROC Curve")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        st.image("images/confusion_matrix_rf.png", use_container_width=True)
        st.image("images/roc_curve_rf.png", use_container_width=True)
    with col2:
        st.markdown("**SVM**")
        st.image("images/confusion_matrix_svm.png", use_container_width=True)
        st.image("images/roc_curve_svm.png", use_container_width=True)


# ==========================
# MENU 4: INTERPRETASI HASIL
# ==========================
elif menu == "Interpretasi Hasil":
    st.title("💡 Interpretasi Hasil & Business Insights")
    st.markdown("---")
    
    st.subheader("Penjelasan Model")
    st.write("""
    Berdasarkan tahapan evaluasi, **Random Forest** terbukti sebagai algoritma terbaik dengan akurasi **73.33%**. 
    Algoritma ini bekerja dengan cara membangun banyak pohon keputusan (*decision trees*) dan menggabungkan hasilnya 
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
    Penerapan model ini memberikan 3 keuntungan utama:
    1. **Efisiensi:** Mampu menyeleksi kandidat dalam hitungan detik.
    2. **Objektivitas Tinggi:** Meminimalisir favoritisme atau human error.
    3. **Tepat Sasaran:** Memastikan alokasi dana beasiswa diberikan kepada kandidat yang membutuhkan dan berprestasi.
    """)


# ==========================
# STEP 12: DOKUMENTASI
# ==========================
elif menu == "Dokumentasi":
    st.title("📖 Dokumentasi Proyek")
    st.markdown("---")
    
    st.subheader("1. Metodologi (Alur CRISP-DM)")
    st.write("Pengembangan aplikasi ini mematuhi standar **CRISP-DM** yang terdiri dari:")
    st.markdown("""
    * **Business Understanding:** Menentukan tujuan pembuatan sistem SPK kelayakan beasiswa.
    * **Data Understanding:** Mengeksplorasi dataset 300 mahasiswa.
    * **Data Preparation:** Melakukan *handling missing values*, *Label Encoding*, dan membagi dataset (80:20).
    * **Modeling:** Melakukan tuning (GridSearchCV) dan pelatihan pada model Random Forest dan SVM.
    * **Evaluation:** Membandingkan nilai performa (Akurasi, ROC-AUC) untuk menentukan algoritma final.
    * **Deployment:** Mengintegrasikan model ke dalam *Streamlit Web App*.
    """)
    
    st.subheader("2. Library yang Digunakan")
    st.code("Streamlit, Pandas, NumPy, Scikit-Learn, Joblib, Matplotlib, Seaborn, Plotly, SHAP", language="python")
    
    st.subheader("3. Penjelasan Fitur Input")
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
    
    st.subheader("4. Kesimpulan")
    st.write("""
    Model Machine Learning berhasil dibangun dan dideploy. Pendekatan *Random Forest* terbukti unggul dibandingkan SVM 
    dalam mengatasi masalah klasifikasi dengan campuran tipe data. Sistem ini siap diuji coba secara lebih luas 
    sebagai Sistem Pendukung Keputusan (SPK) di lingkungan institusi pendidikan.
    """)

# ==========================
# STEP 13: FOOTER
# ==========================
st.markdown("---")
st.caption("🚀 Dibuat menggunakan Streamlit, Scikit-learn, XGBoost, SHAP, dan Plotly.")
