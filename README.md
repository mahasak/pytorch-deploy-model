# Simple Prediction API backed by PyTorch with deploy model example

## Before run server
If you don't have your model, Please download model from: https://download.pytorch.org/models/resnet50-19c8e357.pth
See more: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py 

## How to start api server

```bash
python api_server.py 
```
![Run Server](/images/run-server.png)
* API will available at `http://localhost:5000/predict`

## How to request to Prediction API

```bash
curl -X POST -F image=@path-to-image.jpg 'http://localhost:5000/predict'
```
![Request Prediction](/images/request.png)
