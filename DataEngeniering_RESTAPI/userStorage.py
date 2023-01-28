# Custom user storage
# Let's imaginate what we downloaded this data from db

class UserRepositoryMeta(type):

    _instances = {}

    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            instance = super().__call__(*args, **kwargs)
            self._instances[self] = instance
        return self._instances[self]

class UserRepository(metaclass=UserRepositoryMeta):

    _userStorage = [
        {
            'id': 1,
            'login': 'Ragnar the red',
            'email': 'ragnar@valhalla.com',
            'isonline': False
        },
        {
            'id': 2,
            'login': 'Soler from Astora',
            'email': 'praisethesun@sun.com',
            'isonline': True
        }
    ]

    def addUser(self, userParams: dict) -> str:
        last_record = self._userStorage[-1]
        userParams['id'] = last_record['id'] + 1
        user = UserEntity(userParams)
        user_record = {
            "id": user.id,
            "login": user.login,
            "email": user.email,
            "isonline": user.isonline
        }
        self._userStorage.append(user_record)

        return f"New record {user_record} was added!"

    def getLast10(self):

        last10Users = self._userStorage
        last10Users.reverse()
        return last10Users[:10]

    @property
    def userStorage(self):

        return self._userStorage

class UserEntity:

    def __init__(self, params: dict)-> None:

        self._id = params['id']
        self._login = params['login']
        self._email = params['email']
        self._isonline = params['isonline']

    @property
    def id(self)->int:
        return self._id

    @property
    def login(self)->str:
        return self._login

    @property
    def email(self)->str:
        return self._email

    @property
    def isonline(self):
        return self._isonline


