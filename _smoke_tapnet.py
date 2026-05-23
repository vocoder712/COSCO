import os
os.environ['AEON_DEPRECATION_WARNING'] = 'False'

import numpy as np
from aeon.classification.deep_learning import TapNetClassifier

clf = TapNetClassifier(n_epochs=1, batch_size=8)
X = np.random.rand(8, 6, 100).astype(np.float32)  # aeon expects (n, channels, time)
y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
clf.fit(X, y)
print('TapNet fit OK')
print('pred:', clf.predict(X[:2]))
