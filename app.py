import streamlit as st
import pandas as pd
import joblib

# ==========================
# LOAD DATASET & MODEL
# ==========================
df = pd.read_csv("dataset/beasiswa.csv")
model = joblib.load("models/random_forest_model.pkl")

st.write("Feature model:")
st.write(list(model.feature_names_in_))

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Prediksi Penerima Beasiswa",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
h1{
    text-align:center;
}

h2{
    color:#0068C9;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Sistem Prediksi Penerima Beasiswa")

st.markdown("---")

# ==========================
# DESKRIPSI
# ==========================

st.header("Deskripsi Proyek")

st.write("""
Aplikasi ini dibuat sebagai implementasi Machine Learning untuk
memprediksi kelayakan penerima beasiswa berdasarkan data akademik
dan non-akademik mahasiswa.

Model terbaik yang digunakan adalah **Random Forest** karena
memiliki performa lebih baik dibandingkan Support Vector Machine (SVM).

Penelitian dilakukan menggunakan metodologi **CRISP-DM**
mulai dari Business Understanding hingga Evaluation.
""")

st.markdown("---")

# ==========================
# DASHBOARD EDA
# ==========================

st.header("Dashboard EDA")

st.subheader("Dataset")

st.dataframe(df)

st.subheader("Statistik Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", df.shape[0])

with col2:
    st.metric("Jumlah Fitur", df.shape[1]-1)

with col3:
    st.metric("Target", "Diterima_Beasiswa")

st.markdown("---")

st.subheader("Correlation Heatmap")

st.image("images/correlation_heatmap.png", width=500)

st.markdown("---")

# ==========================
# PREDIKSI
# ==========================

st.header("Prediksi Penerima Beasiswa")

# ==========================
# INPUT DATA
# ==========================

col1, col2 = st.columns(2)

with col1:

    ipk = st.number_input(
        "IPK",
        min_value=0.00,
        max_value=4.00,
        value=3.00,
        step=0.01
    )

    semester = st.selectbox(
        "Semester",
        [1,2,3,4,5,6,7,8]
    )

    penghasilan_input = st.text_input(
        "Penghasilan Orang Tua (contoh: 5.000.000)",
        value="3.000.000"
    )

    penghasilan = int(
        penghasilan_input.replace(".", "").replace(",", "")
    )

    tanggungan = st.number_input(
        "Jumlah Tanggungan",
        min_value=1,
        max_value=10,
        value=3
    )

with col2:

    prestasi = st.selectbox(
        "Prestasi",
        [
            "Internasional",
            "Kabupaten",
            "Nasional",
            "Provinsi",
            "Tidak Ada"
        ]
    )

    organisasi = st.selectbox(
        "Aktif Organisasi",
        [
            "Tidak",
            "Ya"
        ]
    )

    status_rumah = st.selectbox(
        "Status Rumah",
        [
            "Kontrak",
            "Kos/Asrama",
            "Menumpang",
            "Milik Sendiri"
        ]
    )

    jenis_kelamin = st.selectbox(
        "Jenis Kelamin",
        [
            "L",
            "P"
        ]
    )

prestasi_map = {
    "Internasional":0,
    "Kabupaten":1,
    "Nasional":2,
    "Provinsi":3,
    "Tidak Ada":4
}

organisasi_map = {
    "Tidak":0,
    "Ya":1
}

rumah_map = {
    "Kontrak":0,
    "Kos/Asrama":1,
    "Menumpang":2,
    "Milik Sendiri":3
}

jk_map = {
    "L":0,
    "P":1
}

if st.button("Prediksi"):

    data = [[
        ipk,
        semester,
        penghasilan,
        tanggungan,
        prestasi_map[prestasi],
        organisasi_map[organisasi],
        rumah_map[status_rumah],
        jk_map[jenis_kelamin]
    ]]


    import pandas as pd

    data = pd.DataFrame(data, columns=[
        "IPK",
        "Semester",
        "Penghasilan_Orang_Tua",
        "Jumlah_Tanggungan",
        "Prestasi",
        "Aktif_Organisasi",
        "Status_Rumah",
        "Jenis_Kelamin"
    ])

    st.write(data)
    st.write(data.dtypes)

    hasil = model.predict(data)
    prob = model.predict_proba(data)

    st.markdown("## Hasil Prediksi")

    if hasil[0] == 1:
        st.success("Mahasiswa diprediksi **LAYAK** menerima beasiswa.")
        st.metric(
            "Probabilitas Layak",
            f"{prob[0][1]*100:.2f}%"
        )
    else:
        st.error("Mahasiswa diprediksi **TIDAK LAYAK** menerima beasiswa.")
        st.metric(
            "Probabilitas Tidak Layak",
            f"{prob[0][0]*100:.2f}%"
        )

st.markdown("---")

st.header("Evaluasi Model")

hasil_model = pd.DataFrame({
    "Model":[
        "Random Forest",
        "Support Vector Machine (SVM)"
    ],
    "Accuracy":[
        "73.33%",
        "70.00%"
    ]
})

st.table(hasil_model)

st.subheader("Confusion Matrix Random Forest")
st.image("images/confusion_matrix_rf.png", width=350)

st.subheader("Confusion Matrix SVM")
st.image("images/confusion_matrix_svm.png", width=350)

st.subheader("ROC Curve Random Forest")
st.image("images/roc_curve_rf.png", width=350)

st.subheader("ROC Curve SVM")
st.image("images/roc_curve_svm.png", width=350)

st.markdown("---")


st.header("Interpretasi Hasil")

st.write("""
Berdasarkan hasil eksperimen, algoritma Random Forest memperoleh
akurasi sebesar **73,33%**, lebih tinggi dibandingkan Support Vector Machine (SVM)
yang memperoleh akurasi **70,00%**.

Hal ini menunjukkan bahwa Random Forest lebih baik dalam mengenali
pola data pada dataset penerima beasiswa sehingga dipilih sebagai model terbaik.

Prediksi dilakukan berdasarkan kombinasi beberapa faktor seperti:

- IPK
- Semester
- Penghasilan Orang Tua
- Jumlah Tanggungan
- Prestasi
- Keaktifan Organisasi
- Status Rumah
- Jenis Kelamin

Model ini dapat digunakan sebagai alat bantu dalam proses seleksi
penerima beasiswa, namun keputusan akhir tetap perlu mempertimbangkan
kebijakan institusi yang berlaku.
""")

st.markdown("---")

st.header("Dokumentasi")

st.subheader("Dataset")

st.write("""
Dataset yang digunakan merupakan dataset klasifikasi penerima
beasiswa yang terdiri dari **300 data mahasiswa**
dengan **8 atribut** dan **1 target**.
""")

st.subheader("Metodologi")

st.write("""
Tahapan penelitian menggunakan metode **CRISP-DM**, yaitu:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
""")

st.subheader("Cara Menggunakan Aplikasi")

st.write("""
1. Masukkan data mahasiswa pada form prediksi.
2. Klik tombol **Prediksi**.
3. Sistem akan menampilkan hasil prediksi beserta probabilitasnya.
4. Evaluasi model dapat dilihat pada bagian bawah aplikasi.
""")
