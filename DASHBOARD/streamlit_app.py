"""
Dashboard Interaktif — Peramalan Nilai Tukar USD/IDR (ARIMA vs LSTM)
=====================================================================
Cara menjalankan:
    1. Pastikan folder `dashboard_data/` (hasil ekspor dari notebook SKRIPSI)
       berada satu direktori dengan file ini.
    2. Install dependensi:
           pip install streamlit plotly pandas tensorflow joblib scikit-learn
    3. Jalankan:
           streamlit run streamlit_app.py

Pembaruan versi ini:
    - Pengguna dapat memasukkan TANGGAL BERAPA PUN ke depan (tidak lagi
      dibatasi pada rentang tetap 30 hari bursa), karena dashboard kini
      memuat model LSTM terlatih beserta scaler-nya, lalu menghitung
      peramalan secara rekursif langsung dari tanggal terakhir data
      historis hingga tanggal yang diminta pengguna.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Prediksi Kurs USD/IDR",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = "dashboard_data"

# Batas horizon peramalan dinamis (hari kalender) agar ekstrapolasi tidak
# menjadi tidak realistis. Boleh diubah sesuai kebutuhan.
MAX_HORIZON_DAYS = 365


# --------------------------------------------------------------------------
# LOAD DATA (hasil ekspor dari notebook, tidak mengubah pipeline aslinya)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df_hist = pd.read_csv(f"{DATA_DIR}/dashboard_actual.csv", parse_dates=["Date"])
    df_test = pd.read_csv(f"{DATA_DIR}/dashboard_test_predictions.csv", parse_dates=["Date"])
    df_future = pd.read_csv(f"{DATA_DIR}/dashboard_future_forecast.csv", parse_dates=["Date"])
    df_compare = pd.read_csv(f"{DATA_DIR}/dashboard_model_comparison.csv")
    meta = pd.read_csv(f"{DATA_DIR}/dashboard_meta.csv").iloc[0]
    return df_hist, df_test, df_future, df_compare, meta


@st.cache_resource
def load_model_and_scaler():
    """Model & scaler dimuat sekali saja (cache_resource) karena ukurannya
    relatif besar dan tidak berubah selama aplikasi berjalan."""
    from tensorflow.keras.models import load_model
    import joblib

    model = load_model(f"{DATA_DIR}/lstm_model.keras")
    scaler = joblib.load(f"{DATA_DIR}/scaler.pkl")
    return model, scaler


try:
    df_hist, df_test, df_future, df_compare, meta = load_data()
except FileNotFoundError:
    st.error(
        "File data belum ditemukan. Jalankan seluruh sel di notebook `SKRIPSI.ipynb` "
        "(termasuk sel ekspor data dashboard di bagian paling akhir) terlebih dahulu, "
        "lalu pastikan folder `dashboard_data/` berada di direktori yang sama dengan "
        "`streamlit_app.py` ini."
    )
    st.stop()

best_order = meta["best_order"]
best_model_name = meta["best_model_name"]
LOOKBACK = int(meta["lookback"]) if "lookback" in meta else 30

try:
    lstm_model, scaler = load_model_and_scaler()
    model_ready = True
except Exception as e:
    model_ready = False
    st.warning(
        "Model LSTM/scaler belum ditemukan atau gagal dimuat, sehingga peramalan "
        "dinamis untuk tanggal di luar data yang sudah diekspor tidak dapat dihitung. "
        f"Detail teknis: {e}"
    )


# --------------------------------------------------------------------------
# FUNGSI PERAMALAN DINAMIS (rekursif, sesuai cara kerja LSTM univariat)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dynamic_forecast(target_date_str: str):
    """Menghitung peramalan LSTM secara rekursif dari tanggal terakhir data
    historis sampai target_date. Setiap prediksi dimasukkan kembali sebagai
    bagian dari input 30 hari terbaru untuk memprediksi langkah berikutnya --
    persis mekanisme yang dipakai saat notebook menghasilkan FORECAST_DAYS.
    """
    target_ts = pd.Timestamp(target_date_str)
    last_hist_date = df_hist["Date"].max()

    future_bdates = pd.bdate_range(start=last_hist_date + pd.Timedelta(days=1), end=target_ts)
    n_steps = len(future_bdates)
    if n_steps == 0:
        return pd.DataFrame(columns=["Date", "Forecast"])

    hist_sorted = df_hist.sort_values("Date")
    last_window_actual = hist_sorted["USD_IDR"].values[-LOOKBACK:].reshape(-1, 1)
    window_scaled = scaler.transform(last_window_actual).flatten().tolist()

    preds_scaled = []
    for _ in range(n_steps):
        x_input = np.array(window_scaled[-LOOKBACK:], dtype="float32").reshape(1, LOOKBACK, 1)
        pred_scaled = lstm_model.predict(x_input, verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)
        window_scaled.append(pred_scaled)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return pd.DataFrame({"Date": future_bdates, "Forecast": preds})


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("📈 Dashboard Peramalan Kurs USD/IDR")
st.caption("Perbandingan Model ARIMA vs LSTM — kini mendukung peramalan untuk tanggal berapa pun")

min_date = df_hist["Date"].min().date()
last_hist_date = df_hist["Date"].max().date()
max_selectable_date = (df_hist["Date"].max() + pd.Timedelta(days=MAX_HORIZON_DAYS)).date()

# --------------------------------------------------------------------------
# SIDEBAR — KONTROL INTERAKTIF
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Tampilan")

default_start = max(min_date, (df_hist["Date"].max() - pd.Timedelta(days=180)).date())
date_range = st.sidebar.date_input(
    "Rentang tanggal grafik",
    value=(default_start, last_hist_date),
    min_value=min_date,
    max_value=max_selectable_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, last_hist_date

st.sidebar.markdown("**Tampilkan garis:**")
show_actual = st.sidebar.checkbox("Data Aktual", value=True)
show_arima = st.sidebar.checkbox("Prediksi ARIMA (data uji)", value=True)
show_lstm = st.sidebar.checkbox("Prediksi LSTM (data uji)", value=True)
show_future = st.sidebar.checkbox("Peramalan ke Depan", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Cek Nilai pada Tanggal Tertentu")
st.sidebar.caption(
    f"Boleh diisi tanggal apa pun, termasuk lebih dari {MAX_HORIZON_DAYS} hari "
    "setelah data terakhir. Model akan menghitung peramalan secara langsung."
)
selected_date = st.sidebar.date_input(
    "Pilih tanggal",
    value=last_hist_date,
    min_value=min_date,
    max_value=max_selectable_date,
)

is_future_query = selected_date > last_hist_date

dynamic_forecast_df = pd.DataFrame(columns=["Date", "Forecast"])
if is_future_query and model_ready:
    with st.spinner(f"Menghitung peramalan LSTM secara rekursif hingga {selected_date.strftime('%d %B %Y')}..."):
        dynamic_forecast_df = generate_dynamic_forecast(str(selected_date))

st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Model Terbaik")
st.sidebar.success(f"{best_model_name}\n\nMAPE terkecil pada data uji")


# --------------------------------------------------------------------------
# FILTER DATA SESUAI RENTANG TANGGAL
# --------------------------------------------------------------------------
def filter_range(df):
    return df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]


hist_f = filter_range(df_hist)
test_f = filter_range(df_test)

# Gabungkan cache forecast bawaan notebook (df_future) dengan hasil dinamis
# (kalau pengguna sedang menanyakan tanggal di masa depan), lalu filter sesuai rentang.
future_combined = df_future.copy()
if not dynamic_forecast_df.empty:
    future_combined = (
        pd.concat([future_combined, dynamic_forecast_df])
        .drop_duplicates(subset="Date", keep="last")
        .sort_values("Date")
    )
future_f = filter_range(future_combined)


# --------------------------------------------------------------------------
# GRAFIK UTAMA (interaktif — zoom, hover, pan)
# --------------------------------------------------------------------------
fig = go.Figure()

if show_actual:
    fig.add_trace(go.Scatter(
        x=hist_f["Date"], y=hist_f["USD_IDR"],
        name="Aktual", line=dict(color="black", width=2),
    ))
if show_arima:
    fig.add_trace(go.Scatter(
        x=test_f["Date"], y=test_f["ARIMA"],
        name=f"ARIMA{best_order}", line=dict(color="orange", width=2, dash="dash"),
    ))
if show_lstm:
    fig.add_trace(go.Scatter(
        x=test_f["Date"], y=test_f["LSTM"],
        name="LSTM (univariat)", line=dict(color="green", width=2, dash="dash"),
    ))
if show_future:
    fig.add_trace(go.Scatter(
        x=future_f["Date"], y=future_f["Forecast"],
        name="Peramalan ke Depan", line=dict(color="red", width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=4),
    ))

sel_ts = pd.Timestamp(selected_date)
if start_d <= selected_date <= end_d:
    fig.add_vline(x=sel_ts, line_width=1, line_dash="dot", line_color="gray")

fig.update_layout(
    height=520,
    xaxis_title="Tanggal",
    yaxis_title="Kurs USD/IDR",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# NILAI PADA TANGGAL TERPILIH (mencari tanggal bursa terdekat / hitung dinamis)
# --------------------------------------------------------------------------
st.subheader(f"📅 Detail Nilai — {selected_date.strftime('%d %B %Y')}")

if is_future_query:
    if model_ready and not dynamic_forecast_df.empty:
        forecast_value = dynamic_forecast_df.iloc[-1]["Forecast"]
        n_steps_ahead = len(dynamic_forecast_df)
        st.info(
            f"Tanggal ini berada **{n_steps_ahead} hari bursa** setelah data historis terakhir "
            f"({last_hist_date.strftime('%d %b %Y')}), sehingga nilai berikut dihitung secara "
            "dinamis oleh model LSTM saat ini juga (bukan dari cache statis)."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Aktual", "—")
        c2.metric("Prediksi ARIMA", "—")
        c3.metric("Prediksi LSTM", "—")
        c4.metric("Peramalan Dinamis", f"{forecast_value:,.0f}")

        if n_steps_ahead > 90:
            st.caption(
                "⚠️ Tanggal yang dipilih cukup jauh dari data terakhir. Karena LSTM meramalkan "
                "secara rekursif (memakai prediksinya sendiri sebagai input langkah berikutnya), "
                "semakin jauh horizonnya, potensi akumulasi galat semakin besar. Gunakan angka ini "
                "sebagai gambaran tren, bukan kepastian nilai."
            )
    else:
        st.error(
            "Peramalan dinamis tidak dapat dihitung karena model LSTM/scaler belum berhasil dimuat. "
            "Pastikan `lstm_model.keras` dan `scaler.pkl` ada di folder `dashboard_data/`."
        )
else:
    all_dates = pd.concat([df_hist["Date"], df_test["Date"]]).drop_duplicates()
    nearest_date = all_dates.iloc[(all_dates - sel_ts).abs().argsort().iloc[0]]

    if nearest_date != sel_ts:
        st.caption(
            f"Tanggal {selected_date.strftime('%d %b %Y')} bukan hari bursa. "
            f"Menampilkan tanggal terdekat: **{nearest_date.strftime('%d %b %Y')}**."
        )

    row_hist = df_hist[df_hist["Date"] == nearest_date]
    row_test = df_test[df_test["Date"] == nearest_date]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aktual", f"{row_hist['USD_IDR'].values[0]:,.0f}" if not row_hist.empty else "—")
    c2.metric("Prediksi ARIMA", f"{row_test['ARIMA'].values[0]:,.0f}" if not row_test.empty and not np.isnan(row_test['ARIMA'].values[0]) else "—")
    c3.metric("Prediksi LSTM", f"{row_test['LSTM'].values[0]:,.0f}" if not row_test.empty and not np.isnan(row_test['LSTM'].values[0]) else "—")
    c4.metric("Peramalan ke Depan", "—")


# --------------------------------------------------------------------------
# TABEL PERBANDINGAN MODEL
# --------------------------------------------------------------------------
st.subheader("🏆 Perbandingan Performa Model (Data Uji)")
st.dataframe(df_compare, use_container_width=True, hide_index=True)

st.caption(
    "Pedoman interpretasi MAPE (Lewis, 1982): <10% sangat akurat · 10–20% baik · "
    "20–50% layak/cukup · >50% tidak akurat."
)

# --------------------------------------------------------------------------
# DATA MENTAH (opsional, bisa dilihat & diunduh)
# --------------------------------------------------------------------------
with st.expander("📄 Lihat data mentah pada rentang tanggal terpilih"):
    tab1, tab2, tab3 = st.tabs(["Aktual", "Prediksi Data Uji", "Peramalan ke Depan"])
    with tab1:
        st.dataframe(hist_f, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(test_f, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(future_f, use_container_width=True, hide_index=True)
