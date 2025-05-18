from src.data_preprocessing import load_and_preprocess

def test_data_split():
X_train, X_test, y_train, y_test = load_and_preprocess()
assert len(X_train) > 0
assert len(X_test) > 0

def test_data_shapes():
X_train, X_test, y_train, y_test = load_and_preprocess()
assert X_train.shape[1] == 4
