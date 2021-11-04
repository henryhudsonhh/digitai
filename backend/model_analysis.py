from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()

(xtr, xte, ytr, yte) = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

ks = np.arange(2, 10)
scores = []

for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(model, xtr, ytr, cv=15)
    score.mean()
    scores.append(score.mean())

plt.plot(ks, scores)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()