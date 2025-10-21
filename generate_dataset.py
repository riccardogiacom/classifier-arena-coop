import numpy as np
import pandas as pd

# Numero di record
N = 2500
np.random.seed(42)

# --- 1️⃣ Generazione feature di base ---
eta = np.random.randint(18, 65, N)
genere = np.random.choice(["M", "F"], N, p=[0.52, 0.48])
esperienza = np.clip((eta - 18) * np.random.beta(2, 5, N), 0, None).astype(int)

zona = np.random.choice(["Centro", "Periferia", "Suburbio"], N, p=[0.45, 0.35, 0.20])
titolo = np.random.choice(["Nessuno", "Diploma", "Laurea", "Master"], N, p=[0.25, 0.35, 0.30, 0.10])

# Reddito base con pattern realistici
reddito = (
    600 +
    esperienza * np.random.uniform(80, 150, N) +
    np.array([200 if z == "Centro" else (-100 if z == "Suburbio" else 0) for z in zona]) +
    np.array([300 if t == "Diploma" else (700 if t == "Laurea" else (1200 if t == "Master" else -200)) for t in titolo])
)
reddito = np.clip(reddito + np.random.normal(0, 200, N), 400, 8000)

# --- 2️⃣ Funzione di probabilità per l’assunzione ---
prob = (
    0.05 * np.clip(eta - 18, 0, 40)              # età media favorita
    + 0.08 * esperienza
    + 0.0004 * reddito
)

# Titolo aumenta probabilità in modo non lineare
prob += np.array([
    -5 if t == "Nessuno" else (3 if t == "Diploma" else (7 if t == "Laurea" else 9))
    for t in titolo
])

# Zona influisce moderatamente
prob += np.array([3 if z == "Centro" else (-2 if z == "Suburbio" else 0) for z in zona])

# Genere con piccolo bias simulato (F leggermente penalizzate)
prob += np.where(genere == "F", -1.5, 0)

# Rumore casuale
prob += np.random.normal(0, 4, N)

# --- 3️⃣ Conversione in probabilità [0, 1] ---
prob = 1 / (1 + np.exp(-0.1 * (prob - 25)))  # sigmoide

# --- 4️⃣ Assegnazione target "assunto" ---
assunto = np.random.binomial(1, prob)

# --- 5️⃣ Bilanciamento (circa 50/50) ---
# Se il dataset è sbilanciato, sovracampioniamo la classe minoritaria
df = pd.DataFrame({
    "eta": eta,
    "genere": genere,
    "reddito": reddito.astype(int),
    "esperienza": esperienza,
    "zona": zona,
    "titolo": titolo,
    "assunto": assunto
})

# Controllo bilanciamento
balance = df["assunto"].mean()
if balance < 0.45 or balance > 0.55:
    print(f"⚖️ Bilanciamento iniziale: {balance:.2f} → ribilanciamento in corso...")
    yes = df[df["assunto"] == 1]
    no = df[df["assunto"] == 0]
    min_size = min(len(yes), len(no))
    df = pd.concat([yes.sample(min_size, replace=True), no.sample(min_size, replace=True)])
    df = df.sample(frac=1).reset_index(drop=True)

# --- 6️⃣ Salvataggio ---
df.to_csv("dataset.csv", index=False)
print("✅ Dataset v3 generato!")
print(df["assunto"].value_counts(normalize=True).round(3) * 100)
print("📊 Esempi:\n", df.head(5))
