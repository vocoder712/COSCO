import numpy as np

Xtr = np.load('Datasets/BasicMotions/1-shot/X_train.npy')
ytr = np.load('Datasets/BasicMotions/1-shot/y_train.npy')
Xte = np.load('Datasets/BasicMotions/X_test.npy')
yte = np.load('Datasets/BasicMotions/y_test.npy')

print('train X:', Xtr.shape, 'dtype:', Xtr.dtype)
print('train y:', ytr.shape, ytr.ravel())
print('test  X:', Xte.shape)
print('test  y:', yte.shape, 'per-class:', np.unique(yte, return_counts=True))

# How separable are classes from raw signal alone?
# Compute per-class mean of the training (1-shot) sample, then nearest-centroid on raw test data.
classes = np.unique(ytr)
centroids = np.stack([Xtr[ytr.ravel() == c].mean(axis=0) for c in classes])  # (C, T, F)
flat_c = centroids.reshape(len(classes), -1)
flat_te = Xte.reshape(Xte.shape[0], -1)
d = np.linalg.norm(flat_te[:, None, :] - flat_c[None, :, :], axis=-1)
pred = classes[d.argmin(axis=1)]
acc = (pred == yte.ravel()).mean()
print(f'Raw-signal 1-NN-centroid accuracy: {acc:.3f}')
