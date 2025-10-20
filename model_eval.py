import sys, json, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def compute_fairness(df, preds, sensitive_col="genere"):
    groups = df[sensitive_col].unique()
    rates = [preds[df[sensitive_col]==g].mean() for g in groups]
    return abs(rates[0] - rates[1]) if len(rates) == 2 else 0.0

args = json.loads(sys.argv[1])
features = args.get("features", [])
model_type = args.get("model_type", "logreg")
threshold = float(args.get("threshold", 0.5))

df = pd.read_csv("dataset.csv")
for col in ["genere", "zona", "titolo"]:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df[features]
y = df["assunto"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if model_type == "tree":
    model = DecisionTreeClassifier(max_depth=5)
elif model_type == "svm":
    model = SVC(probability=True)
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= threshold).astype(int)

acc = accuracy_score(y_test, preds)
fair_gap = compute_fairness(df.iloc[y_test.index], preds)

result = {
    "accuracy": round(float(acc), 3),
    "fairness_gap": round(float(fair_gap), 3),
    "ethical_score": round(acc * (1 - fair_gap) * 100, 2)
}

print(json.dumps(result))
