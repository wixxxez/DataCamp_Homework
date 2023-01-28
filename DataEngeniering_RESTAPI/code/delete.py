import requests

user_id = 1
response = requests.delete(f"http://127.0.0.1:5000/api/delete/{user_id}")

print(response.text)