import io
import json
import flask
import torch
import torch.nn.functional as F

from PIL import Image
from torch import nn
from torchvision import transforms as T
from torchvision.models import resnet50

app = flask.Flask(__name__)
model = None

with open('imagenet_class.txt', 'r') as f:
    classsified_label = eval(f.read())

def load_model():
    global model
    model = resnet50(pretrained=False)
    model_path = "./models/resnet50-19c8e357.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()


def prepare_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    with torch.no_grad():
        return torch.autograd.Variable(image[None])


@app.route("/predict", methods=["POST"])
def predict():
    # Predefine request status field as unsuccessful
    data = {"success": False}

    # Only available to HTTP POST method
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read image from POST body
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # Preprocessing the image
            image = prepare_image(image, target_size=(224, 224))

            # Get prediction results
            preds = F.softmax(model(image), dim=1)
            # Prepare first k items, from preditions result
            results = torch.topk(preds.cpu().data, k=4, dim=1)

            data['predictions'] = list()

            # Prepare predition result payload
            for prob, label in zip(results[0][0], results[1][0]):
                # Convert Tenser to Python Scalar with .item()
                label_name = classsified_label[label.item()]
                r = {"label": label_name, "probability": float(prob.item())}
                data['predictions'].append(r)

            # Update request status field
            data["success"] = True
    return flask.jsonify(data)

if __name__ == '__main__':
    print("Starting Prediction API Server ...")
    load_model()
    app.run()
