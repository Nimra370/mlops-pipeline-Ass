from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_and_preprocess

def train_and_evaluate():
X_train, X_test, y_train, y_test = load_and_preprocess()
model = RandomForestClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model Accuracy: {acc:.2f}")
joblib.dump(model, 'models/model.joblib')
return acc

if name == "main":
train_and_evaluate()
