import sys
# import os
# import re
# import json
import numpy as np
import pandas as pd
# from tqdm import tqdm
# import tensorflow as tf
# import torch
from transformers import BertTokenizer, NezhaModel

# sys.path.append('.')

nezha_gpt_dialog_dir = '/data/PLM/nezha_gpt_dialog'
tokenizer = BertTokenizer.from_pretrained(nezha_gpt_dialog_dir)
model = NezhaModel.from_pretrained(nezha_gpt_dialog_dir)
text = "我爱北京天安门"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
