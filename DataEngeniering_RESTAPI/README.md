## Install guide
Build your image first

    docker build -t restapi .

Run container 

    docker run -d -ti -p 80:8080 --name=restapi restapi 

Check <a href="https://documenter.getpostman.com/view/15952878/2s935hPS4N">documentation</a> about API.
Try to run your container on local machine and run requests from Postman.