import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # Import for ACF/PACF

st.set_page_config(layout="wide")

st.title("XGBoost Time Series Forecasting App")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

    st.sidebar.header("Data Settings")
    if df is not None:
        # Let user select the timestamp column
        timestamp_columns = df.columns.tolist()
        selected_timestamp_column = st.sidebar.selectbox(
            "Select Timestamp Column", timestamp_columns
        )

        # Let user select the data column for forecasting
        data_columns = [col for col in df.columns if col != selected_timestamp_column]
        if not data_columns:
            st.sidebar.warning("No suitable data columns found after selecting timestamp.")
            st.stop() # Stop execution if no data columns

        selected_data_column = st.sidebar.selectbox(
            "Select Data Column for Forecasting", data_columns
        )

        # Process the dataframe based on user selections
        try:
            df[selected_timestamp_column] = pd.to_datetime(df[selected_timestamp_column])
            df = df.set_index(selected_timestamp_column)
            df = df.sort_index()

            # --- Display Raw Data ---
            st.subheader("Raw Data Preview")
            st.write(df) # Display entire DataFrame

            # --- Time Series Plot ---
            st.subheader(f"Time Series Plot of {selected_data_column}")
            fig_ts, ax_ts = plt.subplots(figsize=(12, 6))
            ax_ts.plot(df.index, df[selected_data_column])
            ax_ts.set_title(f'{selected_data_column} Over Time')
            ax_ts.set_xlabel("Timestamp")
            ax_ts.set_ylabel(selected_data_column)
            ax_ts.grid(True)
            st.pyplot(fig_ts)

            # --- ACF and PACF Plots ---
            st.subheader("Autocorrelation (ACF) and Partial Autocorrelation (PACF) Plots")

            # Ensure there's enough data for ACF/PACF plots
            if len(df[selected_data_column]) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    fig_acf = plot_acf(df[selected_data_column], lags=min(20, len(df[selected_data_column]) - 1))
                    st.pyplot(fig_acf)
                with col2:
                    fig_pacf = plot_pacf(df[selected_data_column], lags=min(20, len(df[selected_data_column]) - 1))
                    st.pyplot(fig_pacf)
            else:
                st.warning("Not enough data points to generate ACF/PACF plots.")

        except Exception as e:
            st.sidebar.error(f"Error processing timestamp column or plotting: {e}")
            df = None # Invalidate df if there's an error
else:
    st.info("Please upload a CSV file to begin.")

# --- Forecasting Parameters (only show if data is loaded) ---
if df is not None:
    st.sidebar.header("Model Parameters")
    look_back = st.sidebar.slider("Look-back Window (Months)", 1, 12, 3)
    test_size_ratio = st.sidebar.slider("Test Set Size Ratio", 0.1, 0.4, 0.2, 0.05)

    st.sidebar.subheader("XGBoost Hyperparameters")
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, 10)
    max_depth = st.sidebar.slider("max_depth", 3, 10, 5)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.05)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)

    st.sidebar.header("Forecast Horizons")
    k_months = st.sidebar.number_input("Recursive Forecast Months (k)", 1, 24, 6)

    # --- Data Preparation ---
    data_column_name = selected_data_column
    data = df[data_column_name].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data, look_back)

    # Split into training and testing sets
    test_size = int(len(X) * test_size_ratio)
    train_size = len(X) - test_size
    X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

    # --- XGBoost Model Training ---
    st.subheader("Model Training")
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1
    )

    if st.button("Train Model"):
        with st.spinner("Training XGBoost model..."):
            model.fit(X_train, Y_train)
        st.success("Model training complete!")

        # --- Evaluation ---
        st.subheader("Model Evaluation")

        train_predict_scaled = model.predict(X_train)
        test_predict_scaled = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict_scaled.reshape(-1, 1))
        Y_train_actual = scaler.inverse_transform(Y_train.reshape(-1, 1))
        test_predict = scaler.inverse_transform(test_predict_scaled.reshape(-1, 1))
        Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

        train_mape = mean_absolute_percentage_error(Y_train_actual, train_predict) * 100
        test_mape = mean_absolute_percentage_error(Y_test_actual, test_predict) * 100

        st.write(f"**Training MAPE:** {train_mape:.2f}%")
        st.write(f"**Testing MAPE:** {test_mape:.2f}%")

        st.subheader("Training and Testing Predictions")
        fig_predict, ax_predict = plt.subplots(figsize=(12, 6))

        train_plot_index = df.index[look_back : look_back + len(train_predict)]
        test_plot_index = df.index[look_back + len(train_predict) : look_back + len(train_predict) + len(test_predict)]
        full_plot_index = df.index[look_back:]

        ax_predict.plot(full_plot_index, scaler.inverse_transform(Y.reshape(-1,1)), label='Actual Data', color='blue')
        ax_predict.plot(train_plot_index, train_predict, label='Training Prediction', color='green', linestyle='--')
        ax_predict.plot(test_plot_index, test_predict, label='Testing Prediction', color='red', linestyle='--')

        ax_predict.set_title("XGBoost Training and Testing Predictions")
        ax_predict.set_xlabel("Timestamp")
        ax_predict.set_ylabel(data_column_name)
        ax_predict.legend()
        ax_predict.grid(True)
        st.pyplot(fig_predict)

        # --- Forecasting ---
        st.subheader("Forecasting Future Values")

        last_sequence = scaled_data[-look_back:].reshape(1, -1)
        forecasted_values = []

        for _ in range(k_months):
            predicted_value_scaled = model.predict(last_sequence)
            forecasted_values.append(predicted_value_scaled[0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1] = predicted_value_scaled[0]

        forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))

        st.write(f"**Recursive Forecast for next {k_months} months:**")
        st.write(pd.DataFrame(forecasted_values, columns=['Forecast']))

        last_timestamp = df.index[-1]
        future_timestamps_recursive = pd.date_range(start=last_timestamp, periods=k_months + 1, freq='MS')[1:]

        st.subheader("Forecasted Future Values")
        fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
        ax_forecast.plot(df.index[-100:], df[data_column_name].tail(100), label='Historical Data', color='blue')
        ax_forecast.plot(future_timestamps_recursive, forecasted_values, label=f'Recursive Forecast ({k_months} months)', color='purple', linestyle='--')

        ax_forecast.set_title(f'XGBoost Time Series Forecast')
        ax_forecast.set_xlabel('Timestamp')
        ax_forecast.set_ylabel(data_column_name)
        ax_forecast.legend()
        ax_forecast.grid(True)
        st.pyplot(fig_forecast)
