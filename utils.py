import torch
import clip
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load('RN50', device)
model.to(device)

def plot_ranks(ranks, model):
  counter = Counter(ranks)

  numbers = list(counter.keys())
  frequencies = list(counter.values())
  total = sum(counter.values())
  percentages = [f"{(count / total) * 100:.2f}%" for count in frequencies]

  plt.figure(figsize=(10, 5))  # Optional: specifies the figure size
  bars = plt.bar(numbers, frequencies)  # Creates a bar chart
  plt.xlabel('Distribution')  # Label for the x-axis
  plt.ylabel('Frequency')  # Label for the y-axis
  plt.title("Prediction Frequency Distribution of " + model)
  plt.xticks(numbers)

  for bar, percentage in zip(bars, percentages):
      yval = bar.get_height()
      plt.text(bar.get_x() + bar.get_width()/2, yval, percentage, va='bottom')

  plt.show()

@torch.no_grad()
def evaluate(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())

    return res

@torch.no_grad()
def evaluate_with_ranks(output, target, topk=(1,)):
    # Number of classes could be determined from the size of the second dimension of output
    num_classes = output.shape[1]
    batch_size = target.size(0)

    # Get the top 'num_classes' indices of the predictions (sorted by highest probability)
    _, pred_full = output.topk(num_classes, 1, True, True)
    pred_full = pred_full.t()

    # Calculate the top 'maxk' accuracies as before
    maxk = max(topk)
    _, pred_topk = output.topk(maxk, 1, True, True)
    pred_topk = pred_topk.t()
    correct_topk = pred_topk.eq(target.view(1, -1).expand_as(pred_topk))

    res = []
    full_ranks = []
    for i in range(batch_size):
        # Finding the full rank of the correct class for each sample from the full predictions
        full_rank = (pred_full[:, i] == target[i]).nonzero(as_tuple=True)[0] + 1
        full_ranks.append(full_rank.item())

    for k in topk:
        # Calculate correct predictions within top k from the limited topk predictions
        correct_k = correct_topk[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())

    return res, full_ranks

@torch.no_grad()
def extract_text_features(labels, prompt_templates):
    model.to(device)
    model.eval()

    zeroshot_weights = []
    for label in labels:
        texts = [prompt.format(label) for prompt in prompt_templates]
        print(texts)
        texts = clip.tokenize(texts).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights