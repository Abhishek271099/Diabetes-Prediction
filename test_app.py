from urllib import request
import requests

url = 'http://localhost:9696/predict'

data = {"Pregnancies": 0,
        "Glucose": 150,
        "BloodPressure": 85,
        "SkinThickness": 50,
        "Insulin": 200,
        "BMI": 30,
        "DiabetesPedigreeFunction": 1.188,
        "Age": 60}

response = requests.post(url, json=data)
        
print(response.json())