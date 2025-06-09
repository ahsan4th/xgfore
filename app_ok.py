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
            
            # --- Display Full Original Data (before resampling) ---
            st.subheader("Full Original Data")
            st.write(df)

            # Resample data based on selected frequency to ensure consistent time series
            df_resampled = df[[selected_data_column]].resample(selected_freq).mean().fillna(method='ffill') # Forward fill missing values
            
        except Exception as e:
            st.sidebar.error(f"Error processing timestamp column or resampling: {e}")
            df = None # Invalidate df if there's an error
else:
    st.info("Please upload a CSV file to begin.")

# --- Forecasting Parameters (only show if data is loaded) ---
if df is not None: # Use the original df for displaying, but df_resampled for forecasting logic
    # --- ACF and PACF Plots (display immediately after data processing) ---
    st.subheader("ACF and PACF Plots (on Full Original Data)") # Changed subheader
    st.write("These plots help in understanding the autocorrelation and partial autocorrelation in the original time series data.")
    
    # Use the original df for ACF/PACF plots
    # Ensure the series for ACF/PACF is not all NaNs and has enough points
    temp_series_original = df[selected_data_column].dropna()
    
    if len(temp_series_original) > 1: # Need at least 2 data points for ACF/PACF
        fig_acf, ax_acf = plt.subplots(figsize=(12, 6))
        plot_acf(temp_series_original, ax=ax_acf, lags=min(20, len(temp_series_original) - 1)) # Max lags is length of series - 1
        ax_acf.set_title(f'Autocorrelation Function (ACF) for {selected_data_column} (Original Data)')
        st.pyplot(fig_acf)

        fig_pacf, ax_pacf = plt.subplots(figsize=(12, 6))
        plot_pacf(temp_series_original, ax=ax_pacf, lags=min(20, len(temp_series_original) - 1))
        ax_pacf.set_title(f'Partial Autocorrelation Function (PACF) for {selected_data_column} (Original Data)')
        st.pyplot(fig_pacf)
    else:
        st.warning("Not enough valid data points in the original series to generate ACF/PACF plots.")


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

    st.sidebar.header("Forecast Horizon") # Changed from Horizons (plural)
    
    # Adjust forecast horizon labels and ranges based on frequency
    if forecast_frequency_option == 'Daily':
        k_label = "Recursive Forecast Days (k)"
        k_min = 1
        k_max = 30
        k_default = 7
    elif forecast_frequency_option == 'Monthly':
        k_label = "Recursive Forecast Months (k)"
        k_min = 1
        k_max = 24
        k_default = 6
    else: # Yearly
        k_label = "Recursive Forecast Years (k)"
        k_min = 1
        k_max = 5
        k_default = 1

    k_months = st.sidebar.number_input(k_label, k_min, k_max, k_default)
    # Removed k_months_direct input

    # --- Data Preparation ---
    # Use the resampled data for model training and forecasting
    data_column_name = selected_data_column
    data = df_resampled[data_column_name].values.reshape(-1, 1)

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
        if len(X_train) == 0 or len(Y_train) == 0 or len(X_test) == 0 or len(Y_test) == 0:
            st.error("Not enough data to create valid training and test sets with the current parameters. Please adjust look-back window or test set ratio, or ensure enough historical data is present after resampling.")
        else:
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

            # Adjust indices for plotting (using df_resampled for plotting indices)
            train_plot_index = df_resampled.index[look_back : look_back + len(train_predict)]
            test_plot_index = df_resampled.index[look_back + len(train_predict) : look_back + len(train_predict) + len(test_predict)]
            full_plot_index = df_resampled.index[look_back:]

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
            st.subheader("Forecasting Future Values (Recursive)") # Updated subheader

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
            last_timestamp_recursive = df_resampled.index[-1]
            future_index_recursive = pd.date_range(start=last_timestamp_recursive, periods=k_months + 1, freq=selected_freq)[1:]
            st.write(pd.DataFrame(forecasted_values, index=future_index_recursive, columns=['Forecast']))

            # Generate future timestamps for plotting
            last_timestamp_plot = df_resampled.index[-1]
            future_timestamps_recursive_plot = pd.date_range(start=last_timestamp_plot, periods=k_months + 1, freq=selected_freq)[1:]

            # Plotting only recursive forecast
            st.subheader("Recursive Forecast Plot")
            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
            
            # Plot *full* historical data
            ax_forecast.plot(df_resampled.index, df_resampled[data_column_name], label='Historical Data', color='blue') 
            
            ax_forecast.plot(future_timestamps_recursive_plot, forecasted_values, label=f'Recursive Forecast ({k_months} {forecast_frequency_option.lower()}s)', color='purple', linestyle='--')

            ax_forecast.set_title(f'XGBoost Time Series Recursive Forecast ({forecast_frequency_option} Frequency)')
            ax_forecast.set_xlabel("Timestamp")
            ax_forecast.set_ylabel(data_column_name)
            ax_forecast.legend()
            ax_forecast.grid(True)
            st.pyplot(fig_forecast)
    else:
        st.info("Please train the model first to see forecasts.")

# Add footnote
st.markdown("---")
st.markdown("Created by Muhammad Ahsan. Copyright © Institut Teknologi Sepuluh Nopember")
