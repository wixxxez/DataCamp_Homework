import requests

### curl -i -H "Content-Type: application/json" -X POST -d "{""title"":""Read a book""}"  http://127.0.0.1:5000/api/add_record
headers = {'Content-Type': 'application/json'}
user_data = {
    'id': 1,
    'login': 'KafkaChampion',
    'email': 'champion@gmail.com',
    'isonline': True
}


r = requests.post("http://127.0.0.1:5000/api/add_record", json=user_data, headers=headers)
print(r.text)