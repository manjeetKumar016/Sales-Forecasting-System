# ==============================
# Sales Forecasting System
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

# ---------- STEP 1: CREATE SAMPLE DATASET ----------
file_name = "sales_data.csv"

data_text = """Date,Product,Region,Sales
2023-01-05,Mobile,North,1200
2023-01-15,Laptop,South,1800
2023-02-10,Mobile,East,1500
2023-02-25,TV,West,2200
2023-03-12,TV,West,3000
2023-03-25,Mobile,North,2800
2023-04-05,Laptop,South,2500
2023-04-18,Mobile,East,2700
2023-05-01,TV,North,3200
2023-05-20,Laptop,South,3500
2023-06-10,Mobile,West,3600
2023-06-25,TV,East,4000
"""

with open(file_name, "w") as file:
    file.write(data_text)

print("sales_data.csv file created successfully!\n")

# ---------- STEP 2: LOAD DATA ----------
data = pd.read_csv(file_name)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# ---------- STEP 3: DATA CLEANING ----------
# Remove missing values if any
data.dropna(inplace=True)

# ---------- STEP 4: FEATURE ENGINEERING ----------
# Extract Month and Year
data['Month'] = data['Date'].dt.month_name()
data['Year'] = data['Date'].dt.year

# ---------- STEP 5: EXPLORATORY DATA ANALYSIS ----------
# Monthly Sales
monthly_sales = data.groupby(data['Date'].dt.to_period('M'))['Sales'].sum()

print("----- Monthly Sales -----")
print(monthly_sales)

# Best & Worst Month
best_month = monthly_sales.idxmax()
worst_month = monthly_sales.idxmin()

print("\nBest Month :", best_month, "=> Sales =", monthly_sales.max())
print("Worst Month:", worst_month, "=> Sales =", monthly_sales.min())

# ---------- STEP 6: VISUALIZATION ----------
plt.figure()
monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)
plt.show()

# ---------- STEP 7: SIMPLE FORECASTING (MOVING AVERAGE) ----------
# Using 3-month moving average as a basic forecasting method
forecast = monthly_sales.rolling(3).mean()

print("\n----- Forecasted Sales (3-Month Moving Average) -----")
print(forecast)

# ---------- STEP 8: FORECAST VISUALIZATION ----------
plt.figure()
monthly_sales.plot(label="Actual Sales", marker='o')
forecast.plot(label="Forecast (Moving Avg)", marker='o')
plt.title("Sales Forecasting")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
