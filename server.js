import express from "express";
import http from "http";
import { Server } from "socket.io";
import { spawn } from "child_process";
import fs from "fs";
import csv from "csv-parser";

const app = express();
const server = http.createServer(app);
const io = new Server(server);
const PORT = process.env.PORT || 10000;

app.use(express.static("./public")); // serve index.html e frontend
let dataset = [];

// === FUNZIONE: Carica dataset dal CSV ===
function loadDataset() {
  return new Promise((resolve, reject) => {
    dataset = [];
    fs.createReadStream("dataset.csv")
      .pipe(csv())
      .on("data", (row) => {
        dataset.push({
          eta: parseInt(row.eta),
          genere: row.genere,
          reddito: parseFloat(row.reddito),
          esperienza: parseInt(row.esperienza),
          zona: row.zona,
          titolo: row.titolo,
          assunto: parseInt(row.assunto),
        });
      })
      .on("end", () => {
        console.log(`📦 Dataset caricato (${dataset.length} righe)`);
        resolve();
      })
      .on("error", (err) => reject(err));
  });
}

// === FUNZIONE: esegue script Python ===
function runPython(script, args, callback) {
  const jsonArgs = JSON.stringify(args);
  const py = spawn("python", [script, jsonArgs]);
  let output = "";
  let error = "";

  py.stdout.on("data", (data) => (output += data.toString()));
  py.stderr.on("data", (data) => (error += data.toString()));

  py.on("close", () => {
    if (error) console.error("🐍 Errore Python:", error);
    if (!output.trim()) {
      console.warn("⚠️ Nessun output Python ricevuto");
      callback(null);
      return;
    }
    try {
      const parsed = JSON.parse(output);
      callback(parsed);
    } catch (err) {
      console.error("❌ Errore parsing JSON:", err.message, output);
      callback(null);
    }
  });
}

// === Scegli un caso casuale dal dataset ===
function pickCase() {
  if (dataset.length === 0) return null;
  return dataset[Math.floor(Math.random() * dataset.length)];
}

// === Mappa utenti con i loro punteggi ===
let users = {};

// === SOCKET.IO ===
io.on("connection", async (socket) => {
  console.log(`🟢 ${socket.id} connesso`);

  socket.on("joinGame", (name) => {
    users[socket.id] = { name, score: 0, acc: 0, fair: 0 };
    io.emit("leaderboard", Object.values(users));
    console.log(`👤 ${name} si è unito al gioco`);
  });

  // Addestramento del modello
  socket.on("trainModel", (data) => {
    console.log("▶️ Addestramento modello con:", data);
    runPython("model_eval.py", data, (res) => {
      if (!res) {
        socket.emit("modelError", { message: "Errore nell'addestramento del modello." });
        return;
      }
      console.log("✅ Modello addestrato:", res);
      users[socket.id].model_info = res;
      socket.emit("modelReady", res);
      io.emit("leaderboard", Object.values(users));
    });
  });

  // Richiedi un caso
  socket.on("requestCase", () => {
    const c = pickCase();
    if (!c) {
      socket.emit("modelError", { message: "Dataset non disponibile" });
      return;
    }
    socket.emit("newCase", c);
  });

  // Predizione su un caso
  socket.on("predictCase", (data) => {
    const caso = {
      eta: parseInt(data.eta),
      genere: data.genere,
      reddito: parseFloat(data.reddito),
      esperienza: parseInt(data.esperienza),
      zona: data.zona,
      titolo: data.titolo,
    };

    console.log(`🤔 Predizione per ${users[socket.id].name}:`, caso);

    runPython("model_predict.py", caso, (res) => {
      if (!res) {
        socket.emit("modelError", { message: "Errore durante la predizione." });
        return;
      }

      const model_pred = parseInt(res.prediction);
      const true_label = parseInt(data.assunto);
      const user_pred = parseInt(data.user_pred);

      const model_correct = model_pred === true_label;
      const user_correct = user_pred === true_label;

      // Aggiorna punteggi
      let delta = 0;
      if (user_correct) delta += 5;
      if (model_correct) delta += 5;
      users[socket.id].score += delta;

      // Feedback personalizzato
      let feedback = "";
      if (user_pred === model_pred && model_correct)
        feedback = "🎯 Tu e il modello avete azzeccato insieme!";
      else if (user_pred === model_pred && !model_correct)
        feedback = "😬 Tu e il modello avete sbagliato insieme!";
      else if (model_correct)
        feedback = "🧠 Il modello ha ragione, tu no!";
      else if (user_correct)
        feedback = "✨ Tu hai ragione, il modello no!";
      else
        feedback = "🤷 Nessuno dei due ha indovinato!";

      socket.emit("predictionFeedback", {
        feedback,
        model_pred,
        correct: user_correct,
      });

      console.log(`🧩 ${users[socket.id].name} → User:${user_pred} | Model:${model_pred} | Real:${true_label}`);
      io.emit("leaderboard", Object.values(users));
    });
  });

  // Disconnessione
  socket.on("disconnect", () => {
    console.log(`🔴 ${socket.id} disconnesso`);
    delete users[socket.id];
    io.emit("leaderboard", Object.values(users));
  });
});

// === AVVIO SERVER ===
await loadDataset();
server.listen(PORT, () =>
  console.log(`✅ Server attivo su porta ${PORT}`)
);
