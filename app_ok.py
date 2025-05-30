import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import datetime

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Aplikasi Peramalan Penumpang")
st.title("Peramalan Jumlah Penumpang dengan XGBoost")

# --- Fungsi Pembantu ---

@st.cache_data
def load_data(uploaded_file):
    """Memuat data dari file CSV yang diunggah dan memproses kolom tanggal."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Mengubah kolom 'Date' menjadi datetime dan menjadikannya indeks
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            df = df.set_index('Date')
            df.sort_index(inplace=True)
            # Memastikan kolom 'Jumlah Penumpang' adalah numerik
            df['Jumlah Penumpang'] = pd.to_numeric(df['Jumlah Penumpang'])
            return df
        except Exception as e:
            st.error(f"Gagal memuat atau memproses data: {e}")
            return None
    return None

@st.cache_data
def create_features(df):
    """Membuat fitur-fitur rekayasa dari indeks deret waktu."""
    df_copy = df.copy() # Bekerja pada salinan untuk menghindari SettingWithCopyWarning
    df_copy['year'] = df_copy.index.year
    df_copy['month'] = df_copy.index.month
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['dayofyear'] = df_copy.index.dayofyear
    df_copy['weekofyear'] = df_copy.index.isocalendar().week.astype(int)
    df_copy['quarter'] = df_copy.index.quarter

    # Menambahkan fitur lag
    df_copy['lag_1'] = df_copy['Jumlah Penumpang'].shift(1)
    df_copy['lag_2'] = df_copy['Jumlah Penumpang'].shift(2)
    df_copy['lag_3'] = df_copy['Jumlah Penumpang'].shift(3)

    # Menambahkan fitur rata-rata bergerak
    df_copy['rolling_mean_3'] = df_copy['Jumlah Penumpang'].shift(1).rolling(window=3).mean()
    df_copy['rolling_mean_7'] = df_copy['Jumlah Penumpang'].shift(1).rolling(window=7).mean()

    # Menghapus baris yang mengandung NaN setelah pembuatan fitur (misalnya, karena lag)
    return df_copy.dropna()

# --- Inisialisasi State Sesi Streamlit ---
# Digunakan untuk menyimpan variabel antar-jalankan (run) aplikasi
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'future_forecast_df' not in st.session_state:
    st.session_state.future_forecast_df = None
if 'df_processed_for_training' not in st.session_state:
    st.session_state.df_processed_for_training = None


# --- Sidebar untuk Pengaturan Parameter ---
st.sidebar.header("Pengaturan Model XGBoost")
n_estimators = st.sidebar.slider("Jumlah Estimator (n_estimators)", 100, 2000, 1000, 50)
learning_rate = st.sidebar.slider("Tingkat Pembelajaran (learning_rate)", 0.001, 0.1, 0.01, 0.001)
max_depth = st.sidebar.slider("Kedalaman Maksimum (max_depth)", 3, 10, 5, 1)
subsample = st.sidebar.slider("Subsample (subsample)", 0.5, 1.0, 0.7, 0.05)
colsample_bytree = st.sidebar.slider("Colsample By Tree (colsample_bytree)", 0.5, 1.0, 0.7, 0.05)
early_stopping_rounds = st.sidebar.slider("Early Stopping Rounds", 10, 200, 50, 10)

st.sidebar.header("Pengaturan Peramalan")
forecast_months = st.sidebar.slider("Jumlah bulan ke depan untuk diramalkan:", 1, 24, 6, 1)


# --- Bagian Utama Aplikasi ---

# 1. Upload Data
st.header("1. Unggah Data")
uploaded_file = st.file_uploader("Unggah file 'dataset_penumpang.csv' Anda", type="csv")

if uploaded_file is not None:
    st.session_state.df = load_data(uploaded_file)
    if st.session_state.df is not None:
        st.success("Data berhasil dimuat!")
        st.write("5 Baris Pertama Data:")
        st.write(st.session_state.df.head())
    else:
        st.error("Terjadi kesalahan saat memuat data. Pastikan format file CSV benar.")

# 2. Tampilan Plot Deret Waktu
st.header("2. Plot Deret Waktu")
if st.session_state.df is not None:
    fig = px.line(st.session_state.df, x=st.session_state.df.index, y='Jumlah Penumpang',
                  title='Jumlah Penumpang dari Waktu ke Waktu')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Silakan unggah data untuk melihat plot deret waktu.")

# 3. Pelatihan Model
st.header("3. Pelatihan Model")
if st.button("Latih Model", disabled=st.session_state.df is None):
    with st.spinner("Model sedang dilatih..."):
        # Buat fitur-fitur dari data
        df_processed = create_features(st.session_state.df.copy())
        st.session_state.df_processed_for_training = df_processed # Simpan untuk referensi fitur

        TARGET = 'Jumlah Penumpang'
        # Dapatkan semua kolom kecuali target sebagai fitur
        FEATURES = [col for col in df_processed.columns if col != TARGET]

        if not FEATURES:
            st.error("Tidak ada fitur yang dibuat. Pastikan data Anda memiliki kolom 'Jumlah Penumpang' dan cukup baris untuk pembuatan fitur.")
            st.session_state.model = None
            st.session_state.X_test = None
            st.session_state.y_test = None
            st.stop()


        # Pemisahan data: menggunakan 80% data untuk pelatihan, 20% untuk pengujian
        split_index = int(len(df_processed) * 0.8)
        train_df = df_processed.iloc[:split_index]
        test_df = df_processed.iloc[split_index:]

        X_train, y_train = train_df[FEATURES], train_df[TARGET]
        X_test, y_test = test_df[FEATURES], test_df[TARGET]

        # Perbaikan: Periksa apakah X_test/y_test kosong atau terlalu kecil sebelum memasukkannya ke eval_set
        eval_set_list = [(X_train, y_train)]
        if not X_test.empty and not y_test.empty:
            # Periksa apakah jumlah baris di X_test cukup untuk early_stopping_rounds
            # Jika X_test terlalu kecil, early stopping mungkin tidak berfungsi dengan baik atau memicu error
            if len(X_test) >= early_stopping_rounds: # Sebaiknya lebih besar atau sama
                eval_set_list.append((X_test, y_test))
            else:
                st.warning(f"Data pengujian terlalu kecil ({len(X_test)} baris) untuk early stopping rounds ({early_stopping_rounds}). Evaluasi pada test set akan dilewatkan.")
        else:
            st.warning("Data pengujian kosong. Evaluasi pada test set akan dilewatkan.")

        # Inisialisasi dan latih XGBoost Regressor dengan parameter dari sidebar
        model = xgb.XGBRegressor(
            objective='reg:squarederror', # Tujuan regresi
            n_estimators=n_estimators,           # Jumlah pohon
            learning_rate=learning_rate,          # Tingkat pembelajaran
            max_depth=max_depth,                 # Kedalaman maksimum pohon
            subsample=subsample,               # Rasio subsample baris
            colsample_bytree=colsample_bytree,        # Rasio subsample kolom saat membuat setiap pohon
            random_state=42,             # Untuk reproduktifitas
            n_jobs=-1                    # Menggunakan semua inti CPU yang tersedia
        )

        model.fit(X_train, y_train,
                  eval_set=eval_set_list, # Menggunakan list yang sudah diperiksa
                  early_stopping_rounds=early_stopping_rounds,
                  verbose=False)

        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.success("Model berhasil dilatih!")
        st.write(f"Ukuran data pelatihan: {len(X_train)} baris")
        st.write(f"Ukuran data pengujian: {len(X_test)} baris")
        st.write(f"Fitur yang digunakan: {FEATURES}")

# 4. Pengujian Model
st.header("4. Pengujian Model")
if st.button("Uji Model", disabled=st.session_state.model is None or st.session_state.X_test is None or st.session_state.X_test.empty):
    with st.spinner("Model sedang diuji..."):
        if st.session_state.model is not None and st.session_state.X_test is not None and not st.session_state.X_test.empty:
            y_pred = st.session_state.model.predict(st.session_state.X_test)
            st.session_state.y_pred = y_pred

            mae = mean_absolute_error(st.session_state.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
            r2 = r2_score(st.session_state.y_test, y_pred)

            st.write(f"**Metrik Evaluasi pada Data Pengujian:**")
            st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
            st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
            st.write(f"**R-squared (R2):** {r2:.2f}")

            # Plot Aktual vs. Prediksi
            results_df = pd.DataFrame({
                'Aktual': st.session_state.y_test,
                'Prediksi': y_pred
            }, index=st.session_state.y_test.index)

            fig_test = px.line(results_df, x=results_df.index, y=['Aktual', 'Prediksi'],
                               title='Jumlah Penumpang Aktual vs. Prediksi (Set Pengujian)')
            st.plotly_chart(fig_test, use_container_width=True)
        else:
            st.warning("Model belum dilatih atau data pengujian kosong. Silakan latih model terlebih dahulu.")

# 5. Peramalan Masa Depan
st.header("5. Peramalan Masa Depan")
# forecast_months sudah diambil dari sidebar

if st.button("Ramalkan Masa Depan", disabled=st.session_state.model is None):
    with st.spinner(f"Melakukan peramalan untuk {forecast_months} bulan..."):
        if st.session_state.model is not None and st.session_state.df_processed_for_training is not None:
            last_date = st.session_state.df.index.max() # Tanggal terakhir dari data asli
            # Buat rentang tanggal untuk masa depan
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                         periods=forecast_months,
                                         freq='MS') # 'MS' untuk awal bulan

            future_df_template = pd.DataFrame(index=future_dates)
            future_df_template['Jumlah Penumpang'] = np.nan # Placeholder untuk target

            # Gabungkan data historis yang sudah diproses dengan tanggal masa depan untuk membuat fitur
            # Ini penting agar fitur lag dan rolling mean dapat dihitung dengan benar untuk periode peramalan
            combined_df_for_features = pd.concat([st.session_state.df_processed_for_training, future_df_template])
            combined_df_for_features = create_features(combined_df_for_features)

            # Saring hanya baris untuk tanggal masa depan
            X_future = combined_df_for_features.loc[future_dates]

            # Pastikan kolom X_future cocok dengan kolom yang digunakan saat melatih model
            # Ini sangat penting agar model dapat membuat prediksi yang benar
            # Ambil kolom fitur dari X_test yang digunakan saat pelatihan
            if st.session_state.X_test is not None and not st.session_state.X_test.empty:
                training_features_cols = st.session_state.X_test.columns
            else:
                # Fallback: jika X_test tidak tersedia, ambil fitur dari df_processed_for_training
                training_features_cols = [col for col in st.session_state.df_processed_for_training.columns if col != 'Jumlah Penumpang']
                if not training_features_cols:
                    st.error("Tidak dapat menentukan fitur pelatihan. Harap latih model terlebih dahulu.")
                    st.stop()

            # Pastikan X_future hanya memiliki kolom yang sama dengan training_features_cols
            X_future = X_future[training_features_cols]


            if X_future.empty:
                st.error("Tidak dapat membuat fitur untuk peramalan. Mungkin ada masalah dengan tanggal atau data historis tidak cukup.")
                st.stop()
            # Cek apakah ada NaN di fitur masa depan yang akan digunakan untuk prediksi
            if X_future.isnull().values.any():
                st.warning("Fitur untuk peramalan mengandung nilai NaN. Ini mungkin karena data historis tidak cukup untuk membuat fitur lag/rolling mean untuk semua periode peramalan yang diminta. Peramalan mungkin tidak akurat atau tidak lengkap.")
                original_future_dates = X_future.index
                X_future = X_future.dropna() # Hanya ramalkan baris yang fiturnya lengkap.
                if X_future.empty:
                    st.error("Setelah menghilangkan NaN, tidak ada data yang tersisa untuk diramalkan. Coba kurangi jumlah bulan yang diramalkan.")
                    st.stop()
                st.info(f"Hanya {len(X_future)} dari {len(original_future_dates)} bulan yang dapat diramalkan karena keterbatasan fitur historis.")


            future_predictions = st.session_state.model.predict(X_future)
            st.session_state.future_forecast_df = pd.DataFrame({
                'Tanggal': X_future.index,
                'Jumlah Penumpang Ramalan': future_predictions.round(0).astype(int) # Pembulatan dan ubah ke integer
            }).set_index('Tanggal')

            st.write("Peramalan Masa Depan:")
            st.write(st.session_state.future_forecast_df)

            # Plot data historis dan peramalan
            # Gabungkan data asli dan hasil peramalan untuk plot
            combined_plot_df = pd.DataFrame(index=st.session_state.df.index.union(st.session_state.future_forecast_df.index))
            combined_plot_df['Jumlah Penumpang Aktual'] = st.session_state.df['Jumlah Penumpang']
            combined_plot_df['Jumlah Penumpang Ramalan'] = st.session_state.future_forecast_df['Jumlah Penumpang Ramalan']

            fig_forecast = px.line(combined_plot_df, x=combined_plot_df.index,
                                   y=['Jumlah Penumpang Aktual', 'Jumlah Penumpang Ramalan'],
                                   title='Jumlah Penumpang Historis dan Ramalan')

            # Tandai awal periode peramalan
            fig_forecast.add_vline(x=st.session_state.df.index.max(), line_dash="dash", line_color="red",
                                   annotation_text="Awal Peramalan")

            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.warning("Model belum dilatih. Silakan latih model terlebih dahulu sebelum melakukan peramalan.")
