import  requests

response = requests.get(url = "http://localhost:80/api/last_10")

print(response.text)