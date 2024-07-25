import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# Title of the app
st.title("Sales Forecasting")

# File uploader for sales data
uploaded_file = st.file_uploader("Upload Sales Data", type=['csv'])

if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Ensure that the date column is in datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Display the data
    st.subheader("Sales Data")
    st.write(data)
    
    # Prepare the data for modeling
    data['day'] = (data['date'] - data['date'].min()).dt.days
    X = data[['day']]
    y = data['sales']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    
    # Forecast future sales
    days_to_forecast = st.slider("Days to Forecast", min_value=1, max_value=365, value=30)
    last_day = data['day'].max()
    future_days = np.arange(last_day + 1, last_day + days_to_forecast + 1).reshape(-1, 1)
    future_dates = pd.date_range(start=data['date'].max() + timedelta(days=1), periods=days_to_forecast)
    future_sales = model.predict(future_days)
    
    # Combine the historical and forecasted data
    forecast_data = pd.DataFrame({'date': future_dates, 'sales': future_sales})
    combined_data = pd.concat([data[['date', 'sales']], forecast_data])
    
    # Plot the sales data
    st.subheader("Sales Forecast")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['date'], data['sales'], label='Historical Sales')
    ax.plot(future_dates, future_sales, label='Forecasted Sales', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    # Analyze and provide explanations for trends
    st.subheader("Trend Analysis")
    if future_sales[-1] > future_sales[0]:
        st.write("The forecast shows an upward trend in sales.")
        st.write("Possible reasons for this trend could be:")
        st.write("- Seasonal increases in demand.")
        st.write("- Successful marketing campaigns.")
        st.write("- New product launches or updates.")
    else:
        st.write("The forecast shows a downward trend in sales.")
        st.write("Possible reasons for this trend could be:")
        st.write("- Seasonal decreases in demand.")
        st.write("- Increased competition in the market.")
        st.write("- Market saturation or changing customer preferences.")
else:
    st.write("Please upload a sales data CSV file.")
