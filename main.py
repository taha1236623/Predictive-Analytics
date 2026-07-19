import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the raw data sheet (Sheet1)
df = pd.read_excel("Online Retail.xlsx", sheet_name="Sheet1")

# Remove missing values
df = df.dropna()

# Create TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Convert InvoiceDate
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Group by date
daily_sales = df.groupby(df["InvoiceDate"].dt.strftime("%Y-%m-%d"))["TotalPrice"].sum().reset_index()
daily_sales.rename(columns={"InvoiceDate": "Date"}, inplace=True)
daily_sales["Day"] = range(1, len(daily_sales) + 1)
# Create day numbers
daily_sales["Day"] = range(1, len(daily_sales) + 1)

X = daily_sales[["Day"]]
y = daily_sales["TotalPrice"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
daily_sales["Prediction"] = model.predict(X)

# Plot
plt.figure(figsize=(8,5))

plt.scatter(daily_sales["Day"], y, color="blue", label="Actual Sales")
plt.plot(daily_sales["Day"], daily_sales["Prediction"], color="red", label="Predicted Sales")

plt.xlabel("Day")
plt.ylabel("Sales")
plt.title("Sales Prediction")
plt.legend()
plt.grid(True)

plt.show()

print(df.shape)
print(df["InvoiceDate"].head(10))
print("Number of rows in daily_sales:", len(daily_sales))
print("Number of rows in daily_sales:", len(daily_sales))
print(daily_sales.tail(10))