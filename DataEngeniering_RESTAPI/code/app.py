from flask import Flask
import userStorage
from flask import  request
app = Flask(__name__)


@app.route("/api/add_record", methods=['POST'])
def addUserController():
    response = user_repository.addUser(request.json)
    return response

@app.route("/api/last_10", methods=['GET'])
def getLast10UsersController():
    users = user_repository.getLast10()
    response = {
        "last 10 users": users
    }
    return response

@app.route("/api/delete/<int:id>", methods=["DELETE"])
def DeleteUserController(id: int):
    users = user_repository.userStorage
    user = list(filter(lambda x: x['id'] == id, users))
    if(len(user) == 0):
        return  "Not found"
    users.remove(user[0])
    return f"Removed record {user[0]}"

if __name__ == "__main__":
    user_repository = userStorage.UserRepository()
    app.run(host='0.0.0.0', port=8080 , debug=True)