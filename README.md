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
Now you can use tensorflow in notebook at port 8080. Use ```docker-machine ip``` to get the IP address of your machine.
## Run model
###Run Script
```
python CNN_mnist.py 
```
After finishing runing the script:

```
tensorboard --logdir=/tmp/mnist_logs
```
##Presentation
The presentation is available at: [https://docs.google.com/presentation/d/1ch4YiKD83wERmmEFRvFIQ98Mtz65aGsLP3uWiTElY2I/edit?usp=sharing](https://docs.google.com/presentation/d/1ch4YiKD83wERmmEFRvFIQ98Mtz65aGsLP3uWiTElY2I/edit?usp=sharing)
