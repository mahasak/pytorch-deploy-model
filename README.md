# Simple Prediction API backed by PyTorch with deploy model example

## How to start api server

```bash
python api_server.py 
```
* API will available at `http://localhost:5000/predict`

## How to request to Prediction API

```bash
curl -X POST -F image=@path-to-image.jpg 'http://localhost:5000/predict'
```