import requests

url = 'http://localhost:8185/predict'

visitor = {
    'age': 25.0,
    'gender': 'Female',
    'education': 0,
    'introversion_score': 8.86186,
    'sensing_score': 4.5106399620302735,
    'thinking_score': 2.80342,
    'judging_score': 4.77769000642399,
    'interest': 'Others',
}

result = requests.post(url, json=visitor).json()

print(result)
