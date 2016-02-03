# Devfest CNN Tutorial
## Docker install
###Create Image:
```
docker build -t your_image_name .
```
###Run container:
```
docker run -d --name your_container_name -p 8080:8888 -v /your_path/notebook:/notebook your_image_name
```
## Run model
###Run Script
```
python CNN_mnist.py 
```
After finishing runing the script:

```
tensorboard --logdir=/tmp/mnist_logs
```

