#!/bin/bash

# install dependencies
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install transformers
pip install --upgrade transformers
pip install sentence_transformers

# download CLIP model
pip install git+https://github.com/openai/CLIP.git
