## Kohonen Challenge

Please see [kohonen.ipynb](kohonen.ipynb)

## running the docker container for the notebook
- from the command line, go to the root directory of the project and run:
```
docker build . -t mymodel
```
- after building has completed, run the command:
```
docker run -p 8888:8888 mymodel
```
- You will be granted an access token on the console
paste in:
http://localhost:8888/?token=your-token
This will grant you access to the notebook

## running the flask server locally
- run the below command in the root directory
```
python3 -m flask run
```
- Copy in the url to run the flask server
