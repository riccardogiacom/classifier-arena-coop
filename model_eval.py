import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    try:
        args = json.loads(sys.argv[1])
    except Exception as e:
        print(json.dumps({"error": f"Errore JSON: {e}"}))
        return

    features = args.get("features", [])
    model_type = args.get("model_type", "tree")
    threshold = args.get("threshold", 0.5)

    # === Carica dataset ===
    df = pd.read_csv("dataset.csv")

    # One-hot encoding coerente (senza drop)
    df = pd.get_dummies(df, columns=["genere", "zona", "titolo"], drop_first=False)

    # Assicuriamoci che tutte le colonne previste esistano
    for col in [
        "genere_M",
        "zona_Centro", "zona_Periferia", "zona_Suburbio",
        "titolo_Nessuno", "titolo_Diploma", "titolo_Laurea", "titolo_Master"
    ]:
        if col not in df.columns:
            df[col] = 0

    # Normalizza i nomi delle feature per sicurezza
    normalize = lambda x: x.lower().replace("à", "a").replace("è", "e").replace("é", "e")

    # Mappa tra feature “umane” e colonne reali
    feature_map = {
        "eta": ["eta"],
        "reddito": ["reddito"],
        "esperienza": ["esperienza"],
        "genere": ["genere_M"],
        "zona": ["zona_Centro", "zona_Periferia", "zona_Suburbio"],
        "titolo": ["titolo_Nessuno", "titolo_Diploma", "titolo_Laurea", "titolo_Master"]
    }

    # Trova le colonne da usare
    selected_features = []
    for f in features:
        key = normalize(f)
        cols = feature_map.get(key, [])
        selected_features.extend([c for c in cols if c in df.columns])

    if not selected_features:
        print(json.dumps({
            "error": f"Nessuna feature valida selezionata. Ricevute: {features}"
        }))
        return

    X = df[selected_features]
    y = df["assunto"]

    # === Split dati ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Selezione modello ===
    if model_type == "logreg":
        model = LogisticRegression(max_iter=2000)
    elif model_type == "forest":
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=3,
            random_state=42
        )
    elif model_type == "tree":
        model = DecisionTreeClassifier(max_depth=None, random_state=42)
    else:
        model = DecisionTreeClassifier(random_state=42)

    # === Addestramento ===
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # === Fairness gap (se "genere" è presente) ===
    fair_gap = 0
    if "genere_M" in X_test.columns:
        males = X_test[X_test["genere_M"] == 1]
        females = X_test[X_test["genere_M"] == 0]
        if len(males) > 0 and len(females) > 0:
            fair_gap = abs(model.predict(males).mean() - model.predict(females).mean())

    ethical_score = max(0, (acc * 100) - (fair_gap * 100))

    # === Salva modello e feature usate ===
    joblib.dump(model, "current_model.pkl")
    with open("current_features.json", "w") as f:
        json.dump(selected_features, f)

    # === Logging leggibile ===
    print(json.dumps({
        "accuracy": round(acc * 100, 2),
        "fairness_gap": round(fair_gap * 100, 2),
        "ethical_score": round(ethical_score, 2),
        "model_type": model_type,
        "features": selected_features
    }))

if __name__ == "__main__":
    main()
