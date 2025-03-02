import requests
import pandas as pd
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Initialize an empty DataFrame to store data
data = pd.DataFrame(columns=["time", "date", "soil_moisture"])

# URL to fetch JSON data
url = "https://techvegan.in/pdeu-project/soil-data-ml.php"  # Replace with your actual URL

# Function to fetch and update data
def fetch_data():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            json_data = response.json()  # Assuming JSON is a list of objects
            global data

            # Iterate through the list of JSON objects
            for record in json_data:
                soil_moisture = record.get("soil_moisture", None)
                time_str = record.get("time", None)
                date_str = record.get("date", None)

                if soil_moisture is not None and time_str is not None and date_str is not None:
                    data = pd.concat([
                        data,
                        pd.DataFrame({"time": [time_str], "date": [date_str], "soil_moisture": [soil_moisture]})
                    ], ignore_index=True)

            print(f"Fetched {len(json_data)} records.")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching data: {e}")

# Function to preprocess and train model
def train_and_predict():
    global data

    if len(data) > 5:  # Ensure enough data for training
        # Combine time and date into a single datetime field for processing
        # Use format='%d-%m-%Y %I:%M %p' to handle 12-hour time with am/pm and errors='coerce' to handle invalid datetimes
        data["datetime"] = pd.to_datetime(data["date"] + " " + data["time"], format='%d-%m-%Y %I:%M %p', errors='coerce')  
        
        # Drop rows with invalid datetime values
        data.dropna(subset=['datetime'], inplace=True)

        # Convert datetime to numeric for ML
        data["time_numeric"] = data["datetime"].astype(int) // 10**9

        # Feature and target
        X = data[["time_numeric"]]
        y = data["soil_moisture"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained. MSE: {mse:.2f}")

        # Predict next value
        next_time = data["time_numeric"].iloc[-1] + 600  # Predict for the next 10 minutes
        next_time_df = pd.DataFrame([[next_time]], columns=["time_numeric"])
        predicted_moisture = model.predict(next_time_df)[0]

        # Determine soil condition
        if predicted_moisture < 20:
            condition = "Dry"
        elif predicted_moisture <= 50:
            condition = "Slightly Wet"
        else:
            condition = "Wet"

        print(f"Predicted soil moisture for next time: {predicted_moisture:.2f}%. Condition: {condition}")

# Main loop to fetch data and train model
try:
    while True:
        fetch_data()
        train_and_predict()
        time.sleep(10)  # Wait for 10 seconds
except KeyboardInterrupt:
    print("Process interrupted.")