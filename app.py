# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load model
model = load_model("model_produksi_tanaman.h5")
model.save("model_saved_format.h5")
model = tf.keras.models.load_model("model_saved_format.h5")

# Judul aplikasi
st.title("Prediksi Produksi Tanaman Pangan di Sumatera")

st.markdown("""
Aplikasi ini memprediksi **Produksi Tanaman** berdasarkan fitur-fitur input seperti Provinsi, Item, dan variabel lainnya.
""")

# Load dan olah dataset dummy (untuk ambil fitur one-hot)
@st.cache_data
def load_data():
    df = pd.read_csv("Data_tanaman_pangan_Sumatera.csv")
    df_one = pd.get_dummies(df, columns=['Provinsi', 'Item'], prefix=['Provinsi', 'Item'])
    df_one = df_one.drop(columns=['Tahun', 'Produksi'])  # Drop target dan tahun
    return df, df_one

df_raw, df_one_template = load_data()
provinsi_options = df_raw['Provinsi'].unique()
item_options = df_raw['Item'].unique()

# Input pengguna
provinsi = st.selectbox("Pilih Provinsi", provinsi_options)
item = st.selectbox("Pilih Komoditas", item_options)
luas_panen = st.number_input("Masukkan Luas Panen (ha):", min_value=0.0)
hasil_per_ha = st.number_input("Masukkan Hasil per Hektar (ku/ha):", min_value=0.0)

# Tombol prediksi
if st.button("Prediksi Produksi"):
    # Siapkan dataframe input
    input_df = pd.DataFrame(columns=df_one_template.columns)
    input_data = {
        'Luas Panen (Ha)': luas_panen,
        'Hasil (Ku/Ha)': hasil_per_ha
    }

    for col in df_one_template.columns:
        if col.startswith("Provinsi_"):
            input_data[col] = 1 if col == f"Provinsi_{provinsi}" else 0
        elif col.startswith("Item_"):
            input_data[col] = 1 if col == f"Item_{item}" else 0
        elif col not in input_data:
            input_data[col] = 0  # default 0

    input_df.loc[0] = input_data

    # Scaling data (gunakan scaler yang sama dengan pelatihan)
    scaler_X = StandardScaler()
    scaler_X.fit(df_one_template)  # Fit ke seluruh template fitur

    X_scaled = scaler_X.transform(input_df)

    # Prediksi
    pred_scaled = model.predict(X_scaled)

    # Kembalikan skala jika target juga diskalakan
    scaler_y = StandardScaler()
    y_raw = df_raw['Produksi'].values.reshape(-1, 1)
    scaler_y.fit(y_raw)
    y_pred = scaler_y.inverse_transform(pred_scaled)

    # Tampilkan hasil prediksi
    st.success(f"Prediksi Produksi: {y_pred[0][0]:,.2f} kuintal")

    # Ambil data aktual untuk provinsi dan komoditas
    df_filtered = df_raw[(df_raw['Provinsi'] == provinsi) & (df_raw['Item'] == item)]

    if not df_filtered.empty:
        produksi_aktual = df_filtered['Produksi'].mean()  # Ambil rata-rata produksi sebagai representasi

        # Plot aktual vs prediksi
        fig, ax = plt.subplots()
        ax.bar(['Aktual', 'Prediksi'], [produksi_aktual, y_pred[0][0]], color=['blue', 'green'])
        ax.set_ylabel('Produksi (kuintal)')
        ax.set_title(f"Produksi Aktual vs Prediksi\n{item} di {provinsi}")

        for i, v in enumerate([produksi_aktual, y_pred[0][0]]):
            ax.text(i, v + max([produksi_aktual, y_pred[0][0]]) * 0.01, f"{v:,.2f}", ha='center')

        st.pyplot(fig)
    else:
        st.warning("Data aktual tidak tersedia untuk kombinasi Provinsi dan Komoditas ini.")