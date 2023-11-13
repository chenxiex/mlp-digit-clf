from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from joblib import dump
import matplotlib.pyplot as plt
X,y=load_digits(return_X_y=True)
clf=MLPClassifier()
clf.fit(X,y)
dump(clf,'digit.joblib')