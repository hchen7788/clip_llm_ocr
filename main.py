from transformers import CLIPProcessor, CLIPModel
import torch
import clip
import os
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm
import utils

labels = ['0','1','2','3','4','5','6','7','8','9',]
prompt_templates = ['a photo of the number: "{}".',]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MNIST(root=os.path.expanduser("~/.cache"), download=True, train=False)
model, preprocess = clip.load('RN50', device)
model.to(device)

image_features = []
image_labels = []
for image, class_id in tqdm(dataset):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    image_feature /= image_feature.norm()
    image_features.append(image_feature)
    image_labels.append(class_id)
image_features = torch.stack(image_features, dim=1).to(device)
image_features = image_features.squeeze()

# extract text feature
text_features = utils.extract_text_features(labels, prompt_templates)

# compute accuracy
clip_logits = (100. * image_features @ text_features).softmax(dim=-1)
clip_image_labels = torch.tensor(image_labels).unsqueeze(dim=1).to(device)
clip_accuracies, clip_ranks = utils.evaluate_with_ranks(clip_logits, clip_image_labels)
print(f'top-1 accuracy for MNIST dataset: {clip_accuracies[0]:.3f}')