import pytest
from src.model import FakeNewsModel

def test_training_and_prediction():
    texts = ["real news", "fake news"]
    labels = [1, 0]

    clf = FakeNewsModel()
    clf.train(texts, labels)

    preds = clf.predict(["real news", "fake news"])
    assert preds.shape[0] == 2
