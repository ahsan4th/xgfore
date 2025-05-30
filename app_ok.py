import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

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
            st.subheader("Raw Data Preview")
            st.write(df.head())
        except Exception as e:
            st.sidebar.error(f"Error processing timestamp column: {e}")
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
    k_months_direct = st.sidebar.number_input("Direct Forecast Months (k_direct)", 1, 24, 6)

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
    if st.button("Train Model"):
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

        st.write(f"**Recursive Forecast for next {k_months} months:**")
        st.write(pd.DataFrame(forecasted_values, columns=['Forecast']))

        # Direct Forecast (Re-create create_dataset function for direct approach)
        def create_dataset_direct(dataset, look_back=1, forecast_horizon=1):
            X, Y = [], []
            for i in range(len(dataset) - look_back - forecast_horizon + 1):
                a = dataset[i:(i + look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back + forecast_horizon - 1, 0])
            return np.array(X), np.array(Y)

        # Prepare data for direct forecasting model
        X_direct, Y_direct = create_dataset_direct(scaled_data, look_back, k_months_direct)

        # Split data for direct model
        if len(X_direct) > 0: # Ensure there's enough data for direct forecast
            test_size_direct = int(len(X_direct) * test_size_ratio)
            train_size_direct = len(X_direct) - test_size_direct
            X_train_direct, X_test_direct = X_direct[0:train_size_direct,:], X_direct[train_size_direct:len(X_direct),:]
            Y_train_direct, Y_test_direct = Y_direct[0:train_size_direct], Y_direct[train_size_direct:len(Y_direct)]

            # Train a new model for direct forecast
            model_direct = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42,
                n_jobs=-1
            )
            with st.spinner(f"Training Direct Forecast Model for {k_months_direct} months..."):
                model_direct.fit(X_train_direct, Y_train_direct)
            st.success("Direct forecast model training complete!")

            # Predict using the direct model
            # The input for direct forecast is the last 'look_back' values
            last_sequence_direct = scaled_data[-look_back:].reshape(1, -1)
            direct_forecasted_value_scaled = model_direct.predict(last_sequence_direct)
            direct_forecasted_values = scaler.inverse_transform(direct_forecasted_value_scaled.reshape(-1, 1))

            st.write(f"**Direct Forecast for next {k_months_direct} month(s):**")
            st.write(pd.DataFrame(direct_forecasted_values, columns=['Forecast']))

            # Generate future timestamps for plotting
            last_timestamp = df.index[-1]
            future_timestamps_recursive = pd.date_range(start=last_timestamp, periods=k_months + 1, freq='MS')[1:]
            future_timestamps_direct = pd.date_range(start=last_timestamp, periods=k_months_direct + 1, freq='MS')[1:]

            # Plotting all forecasts
            st.subheader("Combined Forecasts")
            fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
            ax_forecast.plot(df.index[-100:], df[data_column_name].tail(100), label='Historical Data', color='blue') # Last 100 points for context
            ax_forecast.plot(future_timestamps_recursive, forecasted_values, label=f'Recursive Forecast ({k_months} months)', color='purple', linestyle='--')
            ax_forecast.plot(future_timestamps_direct, direct_forecasted_values, label=f'Direct Forecast ({k_months_direct} months)', color='red', linestyle='--')

            ax_forecast.set_title(f'XGBoost Time Series Forecast')
            ax_forecast.set_xlabel('Timestamp')
            ax_forecast.set_ylabel(data_column_name)
            ax_forecast.legend()
            ax_forecast.grid(True)
            st.pyplot(fig_forecast)
        else:
            st.warning("Not enough data to perform direct forecasting with the selected 'look_back' and 'forecast_horizon'.")
