import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ======================
#  FUNZIONE PRINCIPALE
# ======================
def main():
    try:
        # 1️⃣ Lettura argomenti dal comando Node
        if len(sys.argv) < 2:
            raise ValueError("Nessun argomento fornito al modello.")

        args = json.loads(sys.argv[1])
        features = args.get("features", [])
        model_type = args.get("model_type", "logreg")
        threshold = float(args.get("threshold", 0.5))

        # 2️⃣ Carica dataset
        df = pd.read_csv("dataset.csv")

        # 3️⃣ Encoding variabili categoriche
        for col in ["genere", "zona", "titolo"]:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # 4️⃣ Definisci X e y
        y = df["assunto"].astype(int)
        if not features:
            features = ["eta", "genere", "esperienza", "zona", "titolo"]
        X = df[features]

        # 5️⃣ Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 6️⃣ Normalizzazione
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 7️⃣ Seleziona modello
        if model_type == "logreg":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "tree":
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        elif model_type == "svm":
            model = SVC(probability=True, kernel="rbf", random_state=42)
        else:
            raise ValueError(f"Modello sconosciuto: {model_type}")

        # 8️⃣ Addestramento
        model.fit(X_train, y_train)

        # 9️⃣ Predizioni
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_test, preds)

        # 🔹 Calcolo fairness gap (es. centro vs periferia)
        if "zona" in features:
            df_test = pd.DataFrame(X_test, columns=features)
            df_test["y_true"] = y_test.values
            df_test["y_pred"] = preds
            zona_idx = features.index("zona")
            centro_mask = df_test["zona"] > df_test["zona"].median()
            mean_centro = df_test.loc[centro_mask, "y_pred"].mean()
            mean_perif = df_test.loc[~centro_mask, "y_pred"].mean()
            fairness_gap = abs(mean_centro - mean_perif)
        else:
            fairness_gap = 0.0

        # 🔹 Calcola punteggio etico
        ethical_score = round((acc * (1 - fairness_gap)) * 100, 2)

        # ✅ Restituisci output come JSON
        result = {
            "accuracy": float(acc),
            "fairness_gap": float(fairness_gap),
            "ethical_score": float(ethical_score),
        }

        print(json.dumps(result))

    except Exception as e:
        # ❌ In caso di errore, restituisci un JSON valido
        import traceback
        error_message = {
            "error": str(e),
            "trace": traceback.format_exc()
        }
        print(json.dumps(error_message))
        sys.exit(0)


if __name__ == "__main__":
    main()
