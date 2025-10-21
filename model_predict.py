import sys
import json
import pandas as pd
import joblib
import numpy as np

def main():
    try:
        args = json.loads(sys.argv[1])
    except Exception as e:
        print(json.dumps({"error": f"Errore JSON: {e}"}))
        return

    eta = args.get("eta", 30)
    genere = args.get("genere", "M")
    reddito = args.get("reddito", 2000)
    esperienza = args.get("esperienza", 5)
    zona = args.get("zona", "Centro")
    titolo = args.get("titolo", "Diploma")

    # === Carica modello e feature ===
    try:
        model = joblib.load("current_model.pkl")
        with open("current_features.json", "r") as f:
            used_features = json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"Modello non trovato: {e}"}))
        return

    # === Crea un singolo esempio ===
    df = pd.DataFrame([{
        "eta": eta,
        "reddito": reddito,
        "esperienza": esperienza,
        "genere": genere,
        "zona": zona,
        "titolo": titolo
    }])

    # One-hot encoding coerente
    df = pd.get_dummies(df, columns=["genere", "zona", "titolo"], drop_first=False)

    # Assicura che tutte le colonne usate in training esistano
    for col in used_features:
        if col not in df.columns:
            df[col] = 0

    # Allinea colonne all'ordine di addestramento
    X = df[used_features]

    # === Predizione ===
    try:
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
        print(json.dumps({
            "prediction": int(pred),
            "confidence": round(proba * 100, 1) if proba else None,
            "features_used": used_features
        }))
    except Exception as e:
        print(json.dumps({"error": f"Errore predizione: {e}"}))

if __name__ == "__main__":
    main()
