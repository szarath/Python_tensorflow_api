import requests

# Sample data for predictions
data = {
    "examples": [
        [0],   # Example 1 with a single feature
        [1],   # Example 2 with a single feature
        [2]    # Example 3 with a single feature
    ]
}

# API URL for prediction
url = 'http://localhost:5000/predict'

# Send a POST request to the prediction endpoint
response = requests.post(url, json=data)

# Check the response status code
if response.status_code == 200:
    # If successful, get the predictions
    predictions = response.json()["predictions"]
    print("Predictions:", predictions)
else:
    # If not successful, print the error message
    print("Error:", response.json())
