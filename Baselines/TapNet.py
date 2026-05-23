# TapNet is from the AEON library:

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def train_tapnet(train_data, train_label, test_data, test_label, input_size,args):
  # Import lazily so this module can still be loaded on aeon versions
  # where TapNetClassifier has been removed.
  from aeon.classification.deep_learning import TapNetClassifier
  from aeon.datasets import load_unit_test
  tapnet = TapNetClassifier(n_epochs=args.nEpoch,batch_size=8)
  tapnet.fit(train_data, train_label)
  y_pred = tapnet.predict(test_data)

  acc = accuracy_score(test_label, y_pred)
  print("Final Accuracy: ",acc)
  return acc

import torch
