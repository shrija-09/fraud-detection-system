import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

# Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced')
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, preds))

joblib.dump(model, "model/model.pkl")