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
    # Initialize and train the XGBoost regressor (This model will be used for both recursive and direct if no retraining is desired)
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

        st.write(f"**Recursive Forecast for next {k_months} months:**")
        st.write(pd.DataFrame(forecasted_values, columns=['Forecast']))

        # Direct Forecast (No retraining)
        # The create_dataset_direct function will still define how the data would be prepared
        # if a separate model were trained for each horizon, but we're reusing the 'model' here.
        def create_dataset_direct(dataset, look_back=1, forecast_horizon=1):
            X, Y = [], []
            for i in range(len(dataset) - look_back - forecast_horizon + 1):
                a = dataset[i:(i + look_back), 0]
                X.append(a)
                # Y is the value at forecast_horizon steps ahead
                Y.append(dataset[i + look_back + forecast_horizon - 1, 0])
            return np.array(X), np.array(Y)

        # For direct forecasting without retraining, we simply use the already trained 'model'.
        # The 'model' was trained to predict the next step (1-step ahead).
        # To use it for a direct k_months_direct forecast, we need to ensure the input
        # to the model corresponds to the 'look_back' features to predict the k-th month directly.
        # However, a single model trained for 1-step-ahead prediction will not directly
        # give an accurate k-step-ahead prediction in a "direct" fashion without being trained
        # specifically for that k-step horizon.

        # To adhere strictly to "no training again" and still show *something* for direct forecast,
        # we will apply the *already trained model* (which was trained for 1-step ahead)
        # to the most recent 'look_back' data. This is a simplification and not a true
        # multi-step direct forecasting strategy, which typically involves training separate
        # models for each horizon or a single model with multi-output capabilities.

        # For the purpose of this request, we will use the trained 'model' to predict
        # the 'k_months_direct' steps. This implies the 'model' is being asked to predict
        # further than it was explicitly trained for if k_months_direct > 1, but without
        # retraining.

        if k_months_direct > 0:
            last_sequence_for_direct_forecast = scaled_data[-look_back:].reshape(1, -1)
            direct_forecasted_value_scaled = model.predict(last_sequence_for_direct_forecast)

            # Important: The model was trained to predict 1-step ahead.
            # If k_months_direct is greater than 1, this direct_forecasted_value_scaled
            # is still a 1-step ahead prediction based on the last sequence.
            # A true "direct" forecast for k_months_direct without retraining the original model
            # would require a model specifically trained for that k_months_direct horizon.
            #
            # For the purpose of this problem's constraint ("tidak ada training lagi"),
            # we will take this single 1-step prediction and present it as the
            # k_months_direct forecast, acknowledging its limitation.
            # If the intent was a true direct multi-step forecast,
            # a different modeling strategy (e.g., training a separate model for each k,
            # or a multi-output model) would be required.

            # If k_months_direct is used as a specific future point for the model,
            # the original `create_dataset_direct` prepared the `Y` correctly for a model
            # to learn that specific horizon. However, since we are not retraining,
            # we must use the already trained `model` (which was for 1-step ahead).
            #
            # To make *some* sense of "direct forecast" without retraining, and sticking
            # to the current `model`, we will assume k_months_direct refers to the
            # next 'k_months_direct' values predicted *recursively* by the trained model,
            # but only showing the *last* one as the "direct" forecast. This is a compromise.
            # A more accurate interpretation of "direct forecast without training" would be
            # to use a model already trained for that specific horizon, but such a model
            # is not present here.

            # Let's adjust this to make more sense for "direct forecast" without retraining.
            # If we want a direct forecast for `k_months_direct` ahead, and we *don't* retrain,
            # the only way to get a value for `k_months_direct` ahead using the existing
            # 1-step-ahead trained model is to *recursively* apply it `k_months_direct` times
            # and take the last forecast. This essentially makes the "direct" forecast
            # a recursive one, but we are *not training a new model* for it.

            direct_forecasted_values_recursive_path = []
            temp_last_sequence = scaled_data[-look_back:].reshape(1, -1)

            for _ in range(k_months_direct):
                predicted_value_scaled_temp = model.predict(temp_last_sequence)
                direct_forecasted_values_recursive_path.append(predicted_value_scaled_temp[0])
                temp_last_sequence = np.roll(temp_last_sequence, -1)
                temp_last_sequence[0, -1] = predicted_value_scaled_temp[0]

            direct_forecasted_values = scaler.inverse_transform(np.array(direct_forecasted_values_recursive_path).reshape(-1, 1))

            st.write(f"**Direct Forecast (using recursive application of 1-step model) for next {k_months_direct} month(s):**")
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
            # Plot the "direct" forecast which is essentially the recursive forecast for k_months_direct
            ax_forecast.plot(future_timestamps_direct, direct_forecasted_values, label=f'Direct Forecast (recursive path for {k_months_direct} months)', color='red', linestyle='--')

            ax_forecast.set_title(f'XGBoost Time Series Forecast')
            ax_forecast.set_xlabel('Timestamp')
            ax_forecast.set_ylabel(data_column_name)
            ax_forecast.legend()
            ax_forecast.grid(True)
            st.pyplot(fig_forecast)
        else:
            st.warning("Please set 'Direct Forecast Months (k_direct)' to a value greater than 0.")
