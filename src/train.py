from model import FakeNewsModel

if __name__ == "__main__":
    # Dummy dataset for demo
    texts = ["This is real news", "This is fake news", "True story", "Totally fake"]
    labels = [1, 0, 1, 0]

    clf = FakeNewsModel()
    clf.train(texts, labels)
    clf.save()
    print("✅ Model trained and saved.")
