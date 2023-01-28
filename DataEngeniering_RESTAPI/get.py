import  requests

response = requests.get(url = "http://localhost:5000/api/last_10")

print(response.text)