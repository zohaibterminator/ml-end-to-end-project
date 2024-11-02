import pickle

def save_model(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)

    return model