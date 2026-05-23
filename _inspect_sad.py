import numpy as np

base = 'Datasets/SpokenArabicDigits'
Xtr = np.load(f'{base}/10-shot/X_train.npy')
ytr = np.load(f'{base}/10-shot/y_train.npy')
Xte = np.load(f'{base}/X_test.npy')
yte = np.load(f'{base}/y_test.npy')

print('train X:', Xtr.shape, 'dtype:', Xtr.dtype)
print('train y:', ytr.shape, np.unique(ytr, return_counts=True))
print('test  X:', Xte.shape)
print('test  y:', yte.shape, np.unique(yte, return_counts=True))
