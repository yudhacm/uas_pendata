import streamlit as st
import pandas as pd
import joblib

# Load model Decision Tree (atau model yang kamu pilih)
model = joblib.load('decision_tree_model.pkl')  # Ganti dengan model yang sesuai jika perlu

# Label encoder (jika kamu menyimpan dan perlu load)
label_encoder = joblib.load('label_encoder.pkl')  # Pastikan kamu menyimpan encoder juga jika digunakan

# Daftar kolom fitur sesuai urutan pelatihan
feature_columns = ['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
                   'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
                   'itching', 'irritability', 'delayed_healing', 'partial_paresis',
                   'muscle_stiffness', 'alopecia', 'obesity']

# Judul aplikasi
st.title("Prediksi Risiko Diabetes Dini 🧬")
st.markdown("Aplikasi ini memprediksi apakah seseorang memiliki risiko diabetes berdasarkan gejala awal.")

# Pilih metode input
menu = st.radio("Pilih metode input:", ["Input Manual", "Upload CSV"])

if menu == "Input Manual":
    st.subheader("Masukkan Nilai Fitur:")
    age = st.number_input("Umur", min_value=1, max_value=120)

    def binary_input(label):
        return st.selectbox(label, ["Yes", "No"])

    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    polyuria = binary_input("Polyuria")
    polydipsia = binary_input("Polydipsia")
    sudden_weight_loss = binary_input("Sudden Weight Loss")
    weakness = binary_input("Weakness")
    polyphagia = binary_input("Polyphagia")
    genital_thrush = binary_input("Genital Thrush")
    visual_blurring = binary_input("Visual Blurring")
    itching = binary_input("Itching")
    irritability = binary_input("Irritability")
    delayed_healing = binary_input("Delayed Healing")
    partial_paresis = binary_input("Partial Paresis")
    muscle_stiffness = binary_input("Muscle Stiffness")
    alopecia = binary_input("Alopecia")
    obesity = binary_input("Obesity")

    if st.button("Prediksi"):
        input_data = pd.DataFrame([[age, gender, polyuria, polydipsia, sudden_weight_loss,
                                    weakness, polyphagia, genital_thrush, visual_blurring,
                                    itching, irritability, delayed_healing, partial_paresis,
                                    muscle_stiffness, alopecia, obesity]], columns=feature_columns)
        
        # Encoding sama seperti saat training
        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                input_data[col] = input_data[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

        prediction = model.predict(input_data)[0]
        label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"Hasil Prediksi: **{label}**")

elif menu == "Upload CSV":
    st.subheader("Upload File CSV")
    uploaded_file = st.file_uploader("Upload file .csv", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data yang diupload:")
        st.dataframe(df)

        try:
            # Encoding kolom kategorikal
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

            prediction = model.predict(df)
            df['Prediksi'] = label_encoder.inverse_transform(prediction)

            st.subheader("Hasil Prediksi")
            st.dataframe(df)

            # Tombol unduh hasil
            csv = df.to_csv(index=False).encode()
            st.download_button("Download hasil prediksi", csv, "hasil_prediksi.csv", "text/csv")
        except Exception as e:
            st.error("Terjadi kesalahan saat prediksi: " + str(e))

st.markdown("---")
st.caption("Dibuat dengan Streamlit dan Decision Tree Classifier")
