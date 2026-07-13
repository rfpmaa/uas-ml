import streamlit as st
import pandas as pd
import joblib

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
h1 { text-align: center; color: #0068C9; }
h2 { color: #0068C9; }
.css-1d391kg { padding-top: 2rem; }
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
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
st.sidebar.title("Navigasi Menu")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Beranda", "📊 Dashboard EDA", "🔮 Prediksi Beasiswa", "📈 Evaluasi & Dokumentasi"]
)

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Machine Learning ini dibuat untuk memenuhi Tugas Akhir (UAS).")

# ==========================
# HALAMAN 1: BERANDA
# ==========================
if menu == "🏠 Beranda":
    st.title("🎓 Sistem Prediksi Penerima Beasiswa")
    st.markdown("---")
    
    st.header("Latar Belakang Proyek")
    st.write("""
    Selamat datang di Aplikasi Prediksi Kelayakan Penerima Beasiswa. 
    Aplikasi ini dikembangkan sebagai solusi berbasis *Artificial Intelligence* (AI) 
    untuk membantu institusi pendidikan dalam menyeleksi kandidat penerima beasiswa secara objektif, 
    cepat, dan akurat berdasarkan data historis.
    
    Proses seleksi manual seringkali memakan waktu dan rentan terhadap bias subjektif. 
    Oleh karena itu, pendekatan *Machine Learning* digunakan untuk memetakan pola dari 
    data akademik (seperti IPK dan Semester) serta data non-akademik (seperti Penghasilan Orang Tua, 
    Tanggungan, dan Prestasi) guna memprediksi kelayakan seorang mahasiswa.
    """)
    
    st.header("Metodologi Penelitian (CRISP-DM)")
    st.write("Penelitian dan pengembangan aplikasi ini mematuhi standar industri **CRISP-DM** *(Cross-Industry Standard Process for Data Mining)* yang terdiri dari 6 tahapan utama:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        1. **Business Understanding:** Memahami masalah seleksi beasiswa dan menentukan tujuan pembuatan model prediktif.
        2. **Data Understanding:** Mengumpulkan **300 data mahasiswa** dan melakukan analisis awal untuk melihat karakteristik dari 8 atribut yang digunakan.
        3. **Data Preparation:** Membersihkan data, melakukan transformasi (seperti *label encoding* untuk data kategorikal), dan membagi data menjadi *Training* dan *Testing set*.
        """)
    with col2:
        st.markdown("""
        4. **Modeling:** Melatih algoritma Machine Learning. Pada eksperimen ini, digunakan **Random Forest** dan **Support Vector Machine (SVM)**.
        5. **Evaluation:** Mengukur performa model menggunakan *Confusion Matrix* dan akurasi untuk menentukan model terbaik yang akan di-deploy.
        6. **Deployment:** Mengimplementasikan model terbaik (Random Forest) ke dalam aplikasi interaktif berbasis web menggunakan antarmuka **Streamlit** (aplikasi yang sedang Anda gunakan saat ini).
        """)

# ==========================
# HALAMAN 2: DASHBOARD EDA
# ==========================
elif menu == "📊 Dashboard EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("---")
    
    st.write("""
    Halaman ini menampilkan ringkasan data yang digunakan untuk melatih model Machine Learning. 
    Tahap EDA sangat penting untuk menemukan *insight*, melihat sebaran data, dan memahami korelasi antar variabel.
    """)
    
    st.subheader("Cuplikan Dataset Mahasiswa")
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Statistik Utama Dataset")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Baris Data", f"{df.shape[0]} Data")
    with col2:
        st.metric("Total Fitur (X)", f"{df.shape[1]-1} Kolom")
    with col3:
        st.metric("Target Prediksi (Y)", "Diterima_Beasiswa")
    with col4:
        st.metric("Algoritma Final", "Random Forest")
        
    st.markdown("---")
    
    st.subheader("Analisis Korelasi (Heatmap)")
    st.write("""
    Heatmap di bawah ini menunjukkan seberapa kuat hubungan antar fitur dalam dataset. 
    Nilai yang mendekati 1 atau -1 menunjukkan korelasi yang kuat terhadap peluang mahasiswa diterima beasiswanya.
    """)
    # Pastikan file gambar ini benar-benar ada di GitHub di folder 'images'
    st.image("images/correlation_heatmap.png", caption="Korelasi Antar Variabel", width=600)

# ==========================
# HALAMAN 3: PREDIKSI
# ==========================
elif menu == "🔮 Prediksi Beasiswa":
    st.title("🔮 Form Prediksi Beasiswa")
    st.markdown("---")
    st.write("Silakan masukkan data mahasiswa yang akan dievaluasi ke dalam formulir di bawah ini. Pastikan data yang dimasukkan akurat.")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎓 Data Akademik & Ekonomi")
            ipk = st.number_input("IPK (Skala 4.00)", min_value=0.00, max_value=4.00, value=3.00, step=0.01)
            semester = st.selectbox("Semester Saat Ini", [1,2,3,4,5,6,7,8])
            penghasilan_input = st.text_input("Penghasilan Orang Tua (Rp)", value="3.000.000", help="Ketik menggunakan angka dan titik. Contoh: 3.500.000")
            # Membersihkan input penghasilan dari titik/koma
            penghasilan = int(penghasilan_input.replace(".", "").replace(",", ""))
            tanggungan = st.number_input("Jumlah Tanggungan Keluarga", min_value=1, max_value=10, value=3)

        with col2:
            st.markdown("#### 🏆 Data Non-Akademik")
            prestasi = st.selectbox("Tingkat Prestasi Tertinggi", ["Internasional", "Kabupaten", "Nasional", "Provinsi", "Tidak Ada"])
            organisasi = st.selectbox("Aktif Berorganisasi?", ["Tidak", "Ya"])
            status_rumah = st.selectbox("Status Kepemilikan Tempat Tinggal", ["Kontrak", "Kos/Asrama", "Menumpang", "Milik Sendiri"])
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["L", "P"])

    # Mapping nilai inputan
    prestasi_map = {"Internasional":0, "Kabupaten":1, "Nasional":2, "Provinsi":3, "Tidak Ada":4}
    organisasi_map = {"Tidak":0, "Ya":1}
    rumah_map = {"Kontrak":0, "Kos/Asrama":1, "Menumpang":2, "Milik Sendiri":3}
    jk_map = {"L":0, "P":1}

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Proses Prediksi Kelayakan", use_container_width=True):
        data_input = [[
            ipk,
            semester,
            penghasilan,
            tanggungan,
            prestasi_map[prestasi],
            organisasi_map[organisasi],
            rumah_map[status_rumah],
            jk_map[jenis_kelamin]
        ]]

        # Nama kolom sesuai dataset beasiswa.csv
        data_df = pd.DataFrame(data_input, columns=[
            "IPK", "Semester", "Penghasilan_Ortu", "Tanggungan_Keluarga", 
            "Prestasi", "Aktif_Organisasi", "Status_Rumah", "Jenis_Kelamin"
        ])

        hasil = model.predict(data_df)
        prob = model.predict_proba(data_df)

        st.markdown("---")
        st.subheader("🎯 Kesimpulan Hasil Prediksi")

        if hasil[0] == 1:
            st.success("🎉 Berdasarkan analisis pola data, Mahasiswa ini diprediksi **LAYAK** menerima beasiswa.")
            st.metric("Tingkat Kepercayaan Model (Probabilitas)", f"{prob[0][1]*100:.2f}%")
        else:
            st.error("⚠️ Berdasarkan analisis pola data, Mahasiswa ini diprediksi **TIDAK LAYAK** menerima beasiswa.")
            st.metric("Tingkat Kepercayaan Model (Probabilitas)", f"{prob[0][0]*100:.2f}%")
            
        st.info("💡 **Catatan:** Model ini berfungsi sebagai Sistem Pendukung Keputusan (SPK). Keputusan final pencairan beasiswa tetap berada di tangan panitia seleksi/institusi.")

# ==========================
# HALAMAN 4: EVALUASI
# ==========================
elif menu == "📈 Evaluasi & Dokumentasi":
    st.title("📈 Evaluasi Performa Model")
    st.markdown("---")
    
    st.write("""
    Halaman ini mendokumentasikan hasil komparasi antara algoritma yang diuji selama fase *Training*.
    Perbandingan dilakukan untuk memastikan bahwa model yang diimplementasikan ke dalam aplikasi 
    adalah model yang paling optimal dalam mengenali kandidat mahasiswa.
    """)
    
    st.subheader("1. Komparasi Akurasi Algoritma")
    hasil_model = pd.DataFrame({
        "Algoritma Machine Learning": ["Random Forest Classifier", "Support Vector Machine (SVM)"],
        "Skor Akurasi (Accuracy)": ["73.33%", "70.00%"],
        "Status": ["✅ Model Terpilih", "❌ Tereliminasi"]
    })
    st.table(hasil_model)
    
    st.write("""
    Berdasarkan hasil eksperimen komprehensif, algoritma **Random Forest** memperoleh akurasi sebesar **73.33%**, 
    unggul dibandingkan **Support Vector Machine (SVM)** yang memperoleh **70.00%**. 
    Random Forest terbukti lebih tangguh menangani data tabular dengan kombinasi fitur numerik dan kategorikal.
    """)
    
    st.markdown("---")
    
    st.subheader("2. Analisis Confusion Matrix")
    st.write("Confusion matrix menunjukkan seberapa baik model dalam membedakan kelas Positif (Layak) dan Negatif (Tidak Layak).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest (Terpilih)**")
        st.image("images/confusion_matrix_rf.png", use_container_width=True)
    with col2:
        st.markdown("**SVM (Pembanding)**")
        st.image("images/confusion_matrix_svm.png", use_container_width=True)

    st.markdown("---")
    
    st.subheader("3. Analisis ROC Curve")
    st.write("Kurva ROC (Receiver Operating Characteristic) mengukur kemampuan model dalam membedakan antar kelas. Area Under Curve (AUC) yang lebih tinggi menunjukkan performa yang lebih baik.")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Random Forest (Terpilih)**")
        st.image("images/roc_curve_rf.png", use_container_width=True)
    with col4:
        st.markdown("**SVM (Pembanding)**")
        st.image("images/roc_curve_svm.png", use_container_width=True)
