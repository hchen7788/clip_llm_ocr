from transformers import CLIPProcessor, CLIPModel
import torch
import clip
import os
from torchvision.datasets import MNIST
import numpy as np
import utils

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
prompt_templates = ['a photo of the number: "{}".']

model, preprocess = clip.load('RN50', device)
dataset = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=False)

image_features = []
image_labels = []
for image, class_id in dataset:
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    image_feature /= image_feature.norm()
    image_features.append(image_feature)
    image_labels.append(class_id)
image_features = torch.stack(image_features, dim=1).to(device)
image_features = image_features.squeeze()

# extract text feature
text_features = utils.extract_text_features()

# compute top-1 accuracy
logits = (100. * image_features @ text_features).softmax(dim=-1)
image_labels = torch.tensor(image_labels).unsqueeze(dim=1).to(device)
top1_acc, top3_acc, top5_acc = utils.evaluate(logits, image_labels, topk=(1, 3, 5))
# print(f'top-1 accuracy for MNIST dataset: {top1_acc[0]:.3f}')
# print(f'top-5 accuracy for MNIST dataset: {top5_acc[0]:.3f}')
print(f'top-1 accuracy for MNIST dataset: {top1_acc:.3f}')
print(f'top-3 accuracy for MNIST dataset: {top3_acc:.3f}')
print(f'top-5 accuracy for MNIST dataset: {top5_acc:.3f}')
