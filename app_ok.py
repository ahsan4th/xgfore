import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Main Streamlit App
st.title("XGBoost Time Series Forecasting App")

st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df is not None:
        st.subheader("Original Data Head")
        st.write(df.head())

        # User inputs
        available_columns = df.columns.tolist()
        target_column = st.sidebar.selectbox("Select Target Column for Forecasting", available_columns)

        if target_column:
            # Ensure the target column is numeric
            if not pd.api.types.is_numeric_dtype(df[target_column]):
                st.error(f"The selected target column '{target_column}' is not numeric. Please select a numeric column.")
            else:
                window_size = st.sidebar.slider("Sliding Window Size (Lag Features)", min_value=1, max_value=20, value=5)
                test_size = st.sidebar.slider("Test Data Size (%)", min_value=10, max_value=50, value=20) / 100
                num_forecast_steps = st.sidebar.slider("Number of Forecast Steps", min_value=1, max_value=100, value=10)

                st.sidebar.subheader("XGBoost Parameters")
                n_estimators = st.sidebar.number_input("n_estimators", min_value=10, max_value=1000, value=100, step=10)
                learning_rate = st.sidebar.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
                max_depth = st.sidebar.number_input("max_depth", min_value=3, max_value=15, value=6, step=1)
                subsample = st.sidebar.slider("subsample", min_value=0.1, max_value=1.0, value=0.8, step=0.05)
                colsample_bytree = st.sidebar.slider("colsample_bytree", min_value=0.1, max_value=1.0, value=0.8, step=0.05)

                xgb_params = {
                    'objective': 'reg:squarederror',
                    'n_estimators': n_estimators,
                    'learning_rate': learning_rate,
                    'max_depth': max_depth,
                    'subsample': subsample,
                    'colsample_bytree': colsample_bytree,
                    'random_state': 42,
                    'n_jobs': -1
                }

                # Data Preparation
                st.subheader("Data Preparation")
                try:
                    X, y, processed_df = prepare_data(df, target_column, window_size)
                    st.write("Features (X) and Target (y) created using sliding window.")
                    st.write("X head:")
                    st.write(X.head())
                    st.write("y head:")
                    st.write(y.head())

                    # Train-Test Split
                    st.subheader("Training and Testing")
                    # Ensure there's enough data after creating lags for train/test split
                    if len(X) > window_size: # Need at least window_size + 1 data points
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False) # Use shuffle=False for time series
                        st.write(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

                        # Model Training
                        st.write("Training XGBoost model...")
                        model = train_model(X_train, y_train, xgb_params)
                        st.write("Model training complete.")

                        # Testing and Evaluation
                        st.subheader("Model Evaluation")
                        y_pred = model.predict(X_test)

                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)

                        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

                        # Plotting Actual vs Predicted
                        st.subheader("Actual vs Predicted (Test Set)")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(y_test.index, y_test, label="Actual", marker='o')
                        ax.plot(y_test.index, y_pred, label="Predicted", marker='x')
                        ax.set_xlabel("Index") # Or time-based index if available
                        ax.set_ylabel(target_column)
                        ax.set_title("Actual vs Predicted Values on Test Set")
                        ax.legend()
                        st.pyplot(fig)

                        # Forecasting
                        st.subheader("Forecasting")
                        st.write(f"Forecasting the next {num_forecast_steps} steps...")

                        # Get the last 'window_size' actual data points from the *original* processed dataframe
                        # This ensures we have the correct lag features for the start of the forecast
                        last_n_data_for_forecast = processed_df.tail(window_size +1) # Need window_size lags + 1 target for the first forecast step

                        forecast_results = forecast(model, last_n_data_for_forecast, window_size, target_column, num_forecast_steps)

                        st.write("Forecasted Values:")
                        forecast_df = pd.DataFrame(forecast_results, columns=["Forecast"])
                        # Optionally, create an index for the forecast (e.g., based on the last index of the training data)
                        # The last index of processed_df is the last point used for training/testing
                        last_data_index = processed_df.index[-1]
                        forecast_indices = pd.RangeIndex(start=last_data_index + 1, stop=last_data_index + 1 + num_forecast_steps)
                        forecast_df.index = forecast_indices
                        st.write(forecast_df)

                        # Plotting Forecast
                        st.subheader("Forecast Visualization")
                        fig_forecast, ax_forecast = plt.subplots(figsize=(12, 6))

                        # Plot the historical data that was used for training/testing
                        # Show the last portion of the processed_df including the data used for the last prediction
                        plot_history_length = max(len(y_test), window_size * 2) # Show at least test data length or double window
                        historical_plot_data = processed_df[target_column].tail(plot_history_length)
                        ax_forecast.plot(historical_plot_data.index, historical_plot_data, label="Historical Data", color='blue')

                        # Plot the forecasted data
                        ax_forecast.plot(forecast_df.index, forecast_df['Forecast'], label="Forecast", color='red', linestyle='--')

                        ax_forecast.set_xlabel("Index") # Or time-based index if available
                        ax_forecast.set_ylabel(target_column)
                        ax_forecast.set_title(f"Historical Data and {num_forecast_steps}-Step Forecast")
                        ax_forecast.legend()
                        st.pyplot(fig_forecast)

                        st.subheader("Download Forecast Results")
                        @st.cache_data
                        def convert_df_to_csv(df):
                            return df.to_csv().encode('utf-8')

                        csv_file = convert_df_to_csv(forecast_df)
                        st.download_button(
                            label="Download Forecast CSV",
                            data=csv_file,
                            file_name=f'{target_column}_forecast.csv',
                            mime='text/csv',
                        )

                    else:
                         st.warning(f"Not enough data after creating lags for training and testing. Need at least {window_size + 1} data points.")


                except Exception as e:
                    st.error(f"An error occurred during data preparation or model training: {e}")


        else:
            st.warning("Please select a target column to proceed.")

    else:
        st.warning("Could not load data from the uploaded file.")

else:
    st.info("Please upload a CSV file to start.")

