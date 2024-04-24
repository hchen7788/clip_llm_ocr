from transformers import CLIPProcessor, CLIPModel
import torch
import clip
import os
from torchvision.datasets import MNIST
import numpy as np
from tqdm import tqdm
import utils
import sys

arguments = sys.argv[1:]
print("evaluating the following architectures:", arguments)

labels = ['0','1','2','3','4','5','6','7','8','9',]
prompt_templates = ['a photo of the number: "{}".',]

# GPT descriptions
descriptions_l = {'0': "The number 0 is typically represented as a circle or oval shape. It has a closed loop without any line segments protruding from the shape. The circle is uniform in thickness around its perimeter. There are no internal features within the circle; it is a solid shape.",
                '1': "The number 1 is characterized by a single vertical line. The line is usually straight, extending from the top to the bottom of the visual field. It may have a slight slant or tilt depending on the style of handwriting. The line is typically uniform in thickness from top to bottom.",
                '2': "The number 2 consists of two curved shapes. The top part of the number resembles a backward C or a flattened semicircle. The bottom part extends from the midpoint of the top curve and curves outward to the right, resembling a mirrored S shape.",
                '3': "The number 3 is characterized by a combination of curved and straight lines. It begins with a rounded shape similar to a backward C or a semicircle, but with a slight protrusion to the right. From the midpoint of this shape, a diagonal line extends downward and curves slightly to the right, resembling a mirrored 'L' shape.",
                '4': "The number 4 is defined by a combination of straight and curved lines. It begins with a vertical line that extends downward from the top of the visual field. At the midpoint of this line, a horizontal line extends to the right, forming a right angle with the vertical line.",
                '5': "The number 5 consists of a combination of curved and straight lines. It begins with a vertical line that extends downward from the top of the visual field. At the midpoint of this line, a curved line extends outward to the right, resembling a mirrored S shape.",
                '6': "The number 6 is characterized by a combination of curved and straight lines. It begins with a vertical line that extends downward from the top of the visual field. At the midpoint of this line, a curved line extends outward to the right and then curves back inward, resembling a backward C shape.",
                '7': "The number 7 is defined by a combination of straight and curved lines. It begins with a short horizontal line that extends to the right from the top of the visual field. From the midpoint of this line, a diagonal line extends downward and curves slightly to the right, resembling a mirrored L shape.",
                '8': "The number 8 is typically represented as a pair of symmetrical circles or ovals. The circles are closed loops without any line segments protruding from the shapes. They are uniform in thickness around their perimeters and are typically of the same size. There are no internal features within the circles; they are solid shapes.",
                '9': "The number 9 is characterized by a combination of curved and straight lines. It begins with a short horizontal line that extends to the right from the top of the visual field. From the midpoint of this line, a diagonal line extends downward and curves slightly to the left, resembling a mirrored L shape."}

descriptions_s = {'0': "Number 0: A closed loop shape with uniform thickness around its perimeter.",
                  '1': "Number 1: A single vertical line with uniform thickness from top to bottom.",
                  '2': "Number 2: Two curved shapes, resembling a backward C and a mirrored S, with smooth and uniform curves.",
                  '3': "Number 3: A rounded shape resembling a backward C with a diagonal line extending downward and curving slightly to the right.",
                  '4': "Number 4: A vertical line extending downward with a horizontal line extending to the right, forming a right angle.",
                  '5': "Number 5: A vertical line with a curved line extending outward to the right, resembling a mirrored S, with smooth curves.",
                  '6': "Number 6: A vertical line with a curved line extending outward to the right and curving back inward, resembling a backward C.",
                  '7': "Number 7: A short horizontal line extending to the right with a diagonal line extending downward and curving slightly to the right.",
                  '8': "Number 8: A pair of symmetrical closed loops or circles with uniform thickness around their perimeters.",
                  '9': "Number 9: A short horizontal line extending to the right with a diagonal line extending downward and curving slightly to the left."}

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
text_features_clip = utils.extract_text_features(labels, prompt_templates)
text_features_llm_short = utils.extract_text_features_llm(labels, descriptions_s)
text_features_llm_long = utils.extract_text_features_llm(labels, descriptions_l)

image_labels = torch.tensor(image_labels).unsqueeze(dim=1).to(device)
# compute accuracy
if "clip" in arguments:
    clip_logits = (100. * image_features @ text_features_clip).softmax(dim=-1)
    clip_accuracies, clip_ranks = utils.evaluate_with_ranks(clip_logits, image_labels)
    print(f'top-1 accuracy for MNIST dataset: {clip_accuracies[0]:.3f}')
    utils.plot_ranks(clip_ranks, "CLIP")
if "clip_llm_short" in arguments:
    llm_logits = (100. * image_features @ text_features_llm_short).softmax(dim=-1)
    llm_accuracies, llm_ranks = utils.evaluate_with_ranks(llm_logits, image_labels)
    print(f'top-1 accuracy for MNIST dataset: {llm_accuracies[0]:.3f}')
    utils.plot_ranks(llm_ranks, "CLIP + GPT-3.5 (short prompts)")
if "clip_llm_long" in arguments:
    llm_logits = (100. * image_features @ text_features_llm_long).softmax(dim=-1)
    llm_accuracies, llm_ranks = utils.evaluate_with_ranks(llm_logits, image_labels)
    print(f'top-1 accuracy for MNIST dataset: {llm_accuracies[0]:.3f}')
    utils.plot_ranks(llm_ranks, "CLIP + GPT-3.5 (long prompts)")