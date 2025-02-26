# model.py
from sklearn.linear_model import LogisticRegression  # example ML model
def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model
