from model import FakeNewsModel

if __name__ == "__main__":
    clf = FakeNewsModel()
    clf.load()

    test_texts = ["Breaking: AI is amazing", "Flat earth confirmed"]
    predictions = clf.predict(test_texts)

    for text, pred in zip(test_texts, predictions):
        print(f"{text} → {'Real' if pred == 1 else 'Fake'}")
