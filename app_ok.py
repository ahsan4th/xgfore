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

# Initialize df, selected_timestamp_column, selected_data_column outside the if block
# This ensures they are always defined, even if no file is uploaded yet.
df = None
selected_timestamp_column = None
selected_data_column = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")

        st.sidebar.header("Data Settings")
        # Let user select the timestamp column
        timestamp_columns = df.columns.tolist()
        selected_timestamp_column = st.sidebar.selectbox(
            "Select Timestamp Column", timestamp_columns
        )

        # Let user select the data column for forecasting
        data_columns = [col for col in df.columns if col != selected_timestamp_column]
        if not data_columns:
            st.sidebar.warning("No suitable data columns found after selecting timestamp.")
            # If no data columns, invalidate df to prevent further processing
            df = None
            st.stop() # Stop execution if no data columns

        selected_data_column = st.sidebar.selectbox(
            "Select Data Column for Forecasting", data_columns
        )

        # Process the dataframe based on user selections
        df[selected_timestamp_column] = pd.to_datetime(df[selected_timestamp_column])
        df = df.set_index(selected_timestamp_column)
        df = df.sort_index()
        st.subheader("Raw Data Preview")
        st.write(df.head())

    except Exception as e:
        st.sidebar.error(f"Error processing file or columns: {e}")
        # Invalidate df and selected columns if there's an error during processing
        df = None
        selected_timestamp_column = None
        selected_data_column = None
else:
    st.info("Please upload a CSV file to begin.")

# The main application logic will only run if a file has been successfully uploaded
# and the timestamp and data columns have been selected.
if df is not None and selected_data_column is not None:
    # --- Forecasting Parameters ---
    st.sidebar.header("Model Parameters")
    look_back = st.sidebar.slider("Look-back Window (Months)", 1, 12, 3)
    test_size_ratio = st.sidebar.slider("Test Set Size Ratio", 0.1, 0.4, 0.2, 0.05)

    st.sidebar.subheader("XGBoost Hyperparameters")
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 100, 10)
    max_depth = st.sidebar.slider("max_depth", 3, 10, 5)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1, 0.01)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.05)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)

    st.sidebar.header("Forecast Horizon")
    # Only recursive forecast months are needed now
    k_months = st.sidebar.number_input("Recursive Forecast Months (k)", 1, 24, 6)

    # --- Data Preparation ---
    data_column_name = selected_data_column
    data = df[data_column_name].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    def create_dataset(dataset, look_back=1):
        """
        Creates a dataset for time series forecasting using a sliding window.
        X will be the look_back sequence, Y will be the next value.
        """
        X, Y = [], []
        # Ensure there's enough data for at least one window
        if len(dataset) <= look_back:
            return np.array([]), np.array([])

        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data, look_back)

    # Split into training and testing sets
    if len(X) == 0:
        st.error("Not enough data to create a dataset with the specified 'look_back' window. Please reduce 'look_back' or provide more data.")
    else:
        test_size = int(len(X) * test_size_ratio)
        # Ensure test_size is at least 1 if there's data to split, and not more than half the data
        if test_size == 0 and len(X) > 0:
            test_size = 1
        if len(X) < test_size + 1: # Ensure there's at least one sample for training
            st.error("Not enough data for the specified train-test split ratio. Please reduce 'Test Set Size Ratio' or provide more data.")
        else:
            train_size = len(X) - test_size
            X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
            Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

            # --- XGBoost Model Training ---
            st.subheader("Model Training")
            # Only show the train button if data is ready
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
                # Corrected line: test_predict should use test_predict_scaled
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
                actual_data_for_plot = scaler.inverse_transform(Y.reshape(-1, 1))
                plot_indices = df.index[look_back : look_back + len(actual_data_for_plot)]

                train_plot_index = plot_indices[0 : len(train_predict)]
                test_plot_index = plot_indices[len(train_predict) : len(train_predict) + len(test_predict)]

                # Plot actual values
                ax_predict.plot(plot_indices, actual_data_for_plot, label='Actual Data', color='blue')
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
                # Ensure scaled_data has enough elements for look_back
                if len(scaled_data) < look_back:
                    st.error("Not enough historical data to perform recursive forecast with the specified 'look_back'. Please reduce 'look_back' or provide more data.")
                else:
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
                    # Create a DataFrame for better display with timestamps
                    last_timestamp = df.index[-1]
                    future_timestamps_recursive = pd.date_range(start=last_timestamp, periods=k_months + 1, freq='MS')[1:]
                    st.write(pd.DataFrame(forecasted_values, index=future_timestamps_recursive, columns=['Forecast']))

                    # Plotting recursive forecast
                    st.subheader("Recursive Forecast Plot") # Changed title for clarity
                    fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))
                    # Plot the last 100 actual points for context, or fewer if less than 100 available
                    num_points_to_plot = min(100, len(df))
                    ax_forecast.plot(df.index[-num_points_to_plot:], df[data_column_name].tail(num_points_to_plot), label='Historical Data', color='blue')

                    ax_forecast.plot(future_timestamps_recursive, forecasted_values, label=f'Recursive Forecast ({k_months} months)', color='purple', linestyle='--')

                    ax_forecast.set_title(f'XGBoost Recursive Time Series Forecast')
                    ax_forecast.set_xlabel('Timestamp')
                    ax_forecast.set_ylabel(data_column_name)
                    ax_forecast.legend()
                    ax_forecast.grid(True)
                    st.pyplot(fig_forecast)
