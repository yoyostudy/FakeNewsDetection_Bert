import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
import pandas as pd
import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import FinetuneTagger
    

finetuner = FinetuneTagger(train_file='data/fake_news_validation_set.csv', ## this is only a placeholder, because we are guided not to upload the training data
                            val_file='data/fake_news_validation_set.csv')

## to load our trained model
finetuner.load('model/fine_tuned_model.pt')
## to validate our result:
finetuner.validate()   

