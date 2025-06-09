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

        # Let user select the forecasting frequency
        forecast_frequency_option = st.sidebar.selectbox(
            "Select Forecasting Frequency", ['Daily', 'Monthly', 'Yearly']
        )

        # Map selected frequency to pandas frequency string
        freq_map = {
            'Daily': 'D',
            'Monthly': 'MS', # Month Start
            'Yearly': 'YS'   # Year Start
        }
        selected_freq = freq_map[forecast_frequency_option]

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
            
            # Resample data based on selected frequency to ensure consistent time series
            # Using .asfreq() instead of .resample().mean() for simplicity if we expect no gaps
            # and want to explicitly fill, or just take the first value if multiple exist in a period.
            # However, .resample().mean().ffill() is generally more robust for aggregation.
            # Let's stick to the robust resampling for now.
            df = df[[selected_data_column]].resample(selected_freq).mean().fillna(method='ffill') # Forward fill missing values
            
            st.subheader("Raw Data Preview (Resampled)")
            st.write(df.head())

            # --- Display Full Data ---
            st.subheader("Full Data")
            st.write(df)

        except Exception as e:
            st.sidebar.error(f"Error processing timestamp column or resampling: {e}")
            df = None # Invalidate df if there's an error
else:
    st.info("Please upload a CSV file to begin.")

# --- Forecasting Parameters (only show if data is loaded) ---
if df is not None:
    st.sidebar.header("Model Parameters")
    
    # Adjust look-back slider label and range based on frequency
    if forecast_frequency_option == 'Daily':
        look_back_label = "Look-back Window (Days)"
        look_back_min = 7
        look_back_max = 90
        look_back_default = 30
    elif forecast_frequency_option == 'Monthly':
        look_back_label = "Look-back Window (Months)"
        look_back_min = 1
        look_back_max = 12
        look_back_default = 3
    else: # Yearly
        look_back_label = "Look-back Window (Years)"
        look_back_min = 1
        look_back_max = 5
        look_back_default = 1

    look_back = st.sidebar.slider(look_back_label, look_back_min, look_back_max, look_back_default)
    test_size_ratio = st.sidebar.slider("Test Set Size Ratio", 0.1, 0.4, 0.2, 0.05)

    st.sidebar.subheader("XGBoost Hyperparameters")
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, 10)
    max_depth = st.sidebar.slider("max_depth", 3, 10, 5)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.05)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)

    st.sidebar.header("Forecast Horizons")
    
    # Adjust forecast horizon labels and ranges based on frequency
    if forecast_frequency_option == 'Daily':
        k_label = "Recursive Forecast Days (k)"
        k_direct_label = "Direct Forecast Days (k_direct)"
        k_min = 1
        k_max = 30
        k_default = 7
    elif forecast_frequency_option == 'Monthly':
        k_label = "Recursive Forecast Months (k)"
        k_direct_label = "Direct Forecast Months (k_direct)"
        k_min = 1
        k_max = 24
        k_default = 6
    else: # Yearly
        k_label = "Recursive Forecast Years (k)"
        k_direct_label = "Direct Forecast Years (k_direct)"
        k_min = 1
        k_max = 5
        k_default = 1

    k_months = st.sidebar.number_input(k_label, k_min, k_max, k_default)
    k_months_direct = st.sidebar.number_input(k_direct_label, k_min, k_max, k_default)

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

    # --- ACF and PACF Plots ---
    st.subheader("ACF and PACF Plots")
    st.write("These plots help in understanding the autocorrelation and partial autocorrelation in the series.")
    if st.button("Generate ACF/PACF Plots"):
        if len(df[selected_data_column]) > 1: # Need at least 2 data points for ACF/PACF
            fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
            plot_acf(df[selected_data_column], ax=ax_acf, lags=min(20, len(df) - 1)) # Max lags is length of series - 1
            ax_acf.set_title(f'Autocorrelation Function (ACF) for {data_column_name}')
            st.pyplot(fig_acf)

            fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
            plot_pacf(df[selected_data_column], ax=ax_pacf, lags=min(20, len(df) - 1))
            ax_pacf.set_title(f'Partial Autocorrelation Function (PACF) for {data_column_name}')
            st.pyplot(fig_pacf)
        else:
            st.warning("Not enough data points to generate ACF/PACF plots. Please upload more data.")


    # --- XGBoost Model Training ---
    st.subheader("Model Training")
    # Initialize and train the XGBoost regressor
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1 # Use all available cores
    )

    if st.button("Train Model"):
        with st.spinner("Training XGBoost model..."):
            model.fit(X_train, Y_train)
        st.success("Model training complete!")

        # --- Evaluation ---
        st.subheader("Model Evaluation")

        # Make predictions on training and test data
        train_predict_scaled = model.predict(X_train)
        test_predict_scaled = model.predict(X_test)

        # Invert predictions to original scale
        train_predict = scaler.inverse_transform(train_predict_scaled.reshape(-1, 1))
        Y_train_actual = scaler.inverse_transform(Y_train.reshape(-1, 1))
        test_predict = scaler.inverse_transform(test_predict_scaled.reshape(-1, 1))
        Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

        # Calculate MAPE
        train_mape = mean_absolute_percentage_error(Y_train_actual, train_predict) * 100
        test_mape = mean_absolute_percentage_error(Y_test_actual, test_predict) * 100

        st.write(f"**Training MAPE:** {train_mape:.2f}%")
        st.write(f"**Testing MAPE:** {test_mape:.2f}%")

        # Plotting Training and Testing Predictions
        st.subheader("Training and Testing Predictions")
        fig_predict, ax_predict = plt.subplots(figsize=(12, 6))

        # Adjust indices for plotting
        train_plot_index = df.index[look_back : look_back + len(train_predict)]
        test_plot_index = df.index[look_back + len(train_predict) : look_back + len(train_predict) + len(test_predict)]
        full_plot_index = df.index[look_back:]

        # Plot actual values
        ax_predict.plot(full_plot_index, scaler.inverse_transform(Y.reshape(-1,1)), label='Actual Data', color='blue')
        # Plot training predictions
        ax_predict.plot(train_plot_index, train_predict, label='Training Prediction', color='green', linestyle='--')
        # Plot testing predictions
        ax_predict.plot(test_plot_index, test_predict, label='Testing Prediction', color='red', linestyle='--')

        ax_predict.set_title("XGBoost Training and Testing Predictions")
        ax_predict.set_xlabel("Timestamp")
        ax_predict.set_ylabel(data_column_name)
        ax_predict.legend()
        ax_predict.grid(True)
        st.pyplot(fig_predict)

        # --- Forecasting ---
        st.subheader("Forecasting Future Values")

        # Recursive Forecast
        last_sequence = scaled_data[-look_back:].reshape(1, -1)
        forecasted_values = []

        for _ in range(k_months):
            predicted_value_scaled = model.predict(last_sequence)
            forecasted_values.append(predicted_value_scaled[0])
            # Update last_sequence by dropping the first element and adding the new prediction
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[0, -1] = predicted_value_scaled[0]

        forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))

        st.write(f"**Recursive Forecast for next {k_months} {forecast_frequency_option.lower()}s:**")
        
        # Create future index for recursive forecast table
        last_timestamp_recursive = df.index[-1]
        future_index_recursive = pd.date_range(start=last_timestamp_recursive, periods=k_months + 1, freq=selected_freq)[1:]
        st.write(pd.DataFrame(forecasted_values, index=future_index_recursive, columns=['Forecast']))

        # Direct Forecast (No retraining)
        if k_months_direct > 0:
            direct_forecasted_values_recursive_path = []
            temp_last_sequence = scaled_data[-look_back:].reshape(1, -1)

            for _ in range(k_months_direct):
                predicted_value_scaled_temp = model.predict(temp_last_sequence)
                direct_forecasted_values_recursive_path.append(predicted_value_scaled_temp[0])
                temp_last_sequence = np.roll(temp_last_sequence, -1)
                temp_last_sequence[0, -1] = predicted_value_scaled_temp[0]

            direct_forecasted_values = scaler.inverse_transform(np.array(direct_forecasted_values_recursive_path).reshape(-1, 1))

            st.write(f"**Direct Forecast (using recursive application of 1-step model) for next {k_months_direct} {forecast_frequency_option.lower()}s:**")
            
            # Create future index for direct forecast table
            last_timestamp_direct = df.index[-1]
            future_index_direct = pd.date_range(start=last_timestamp_direct, periods=k_months_direct + 1, freq=selected_freq)[1:]
            st.write(pd.DataFrame(direct_forecasted_values, index=future_index_direct, columns=['Forecast']))

            # Generate future timestamps for plotting
            last_timestamp_plot = df.index[-1]
            future_timestamps_recursive_plot = pd.date_range(start=last_timestamp_plot, periods=k_months + 1, freq=selected_freq)[1:]
            future_timestamps_direct_plot = pd.date_range(start=last_timestamp_plot, periods=k_months_direct + 1, freq=selected_freq)[1:]

            # Plotting all forecasts
            st.subheader("Combined Forecasts")
            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
            
            # Plot a reasonable portion of historical data for context
            # Dynamically adjust historical plot length based on frequency
            if forecast_frequency_option == 'Daily':
                plot_historical_length = min(len(df), 120) # Up to 120 days
            elif forecast_frequency_option == 'Monthly':
                plot_historical_length = min(len(df), 36) # Up to 36 months (3 years)
            else: # Yearly
                plot_historical_length = min(len(df), 10) # Up to 10 years

            ax_forecast.plot(df.index[-plot_historical_length:], df[data_column_name].tail(plot_historical_length), label='Historical Data', color='blue') 
            
            ax_forecast.plot(future_timestamps_recursive_plot, forecasted_values, label=f'Recursive Forecast ({k_months} {forecast_frequency_option.lower()}s)', color='purple', linestyle='--')
            ax_forecast.plot(future_timestamps_direct_plot, direct_forecasted_values, label=f'Direct Forecast (recursive path for {k_months_direct} {forecast_frequency_option.lower()}s)', color='red', linestyle='--')

            ax_forecast.set_title(f'XGBoost Time Series Forecast ({forecast_frequency_option} Frequency)')
            ax_forecast.set_xlabel("Timestamp")
            ax_forecast.set_ylabel(data_column_name)
            ax_forecast.legend()
            ax_forecast.grid(True)
            st.pyplot(fig_forecast)
        else:
            st.warning(f"Please set '{k_direct_label}' to a value greater than 0.")

# Add footnote
st.markdown("---")
st.markdown("Created by Muhammad Ahsan copyright by ITS")
