import pandas as pd
from pymongo import MongoClient
import json

# Load the file
df = pd.read_csv("data/PRSA_Data_Wanshouxigong_20130301-20170228.csv")
df.dropna(inplace=True)

# Create 'time' column
df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])

# Drop split time columns, but keep station name
df = df.drop(columns=['No', 'year', 'month', 'day', 'hour'])

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["air_quality"]
collection = db["beijing_data"]
#collection.delete_many({})  # Clear previous runs

# Insert into MongoDB
data_json = json.loads(df.to_json(orient='records'))
collection.insert_many(data_json)

print(f"Inserted {len(data_json)} records into MongoDB.")

