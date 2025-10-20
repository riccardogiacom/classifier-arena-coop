// ---- IMPORTS ----
import express from "express";
import http from "http";
import { Server } from "socket.io";
import { spawn } from "child_process";
import fs from "fs";

// ---- EXPRESS + SOCKET.IO SETUP ----
const app = express();
const server = http.createServer(app);
const io = new Server(server);
const PORT = process.env.PORT || 10000;

app.use(express.static("public"));
const users = {};
let currentCase = null;

// ---- FUNZIONE DI ESECUZIONE PYTHON ----
function runPython(args, callback) {
  const py = spawn("python3", ["model_eval.py", JSON.stringify(args)]);
  let output = "";
  let errorOutput = "";

  py.stdout.on("data", (d) => (output += d.toString()));
  py.stderr.on("data", (d) => (errorOutput += d.toString()));

  py.on("close", (code) => {
    if (errorOutput) console.error("🐍 Errore Python:", errorOutput);
    if (!output.trim()) {
      console.error("⚠️ Nessun output Python ricevuto.");
      callback(null);
      return;
    }
    try {
      const parsed = JSON.parse(output);
      callback(parsed);
    } catch (err) {
      console.error("⚠️ Errore nel parsing JSON:", err.message, output);
      callback(null);
    }
  });
}

// ---- SELEZIONA UN CASO DAL DATASET ----
function pickCase() {
  const rows = fs.readFileSync("dataset.csv", "utf8").split("\n").slice(1, -1);
  const rand = rows[Math.floor(Math.random() * rows.length)].split(",");
  return {
    eta: rand[0],
    genere: rand[1],
    esperienza: rand[2],
    zona: rand[3],
    titolo: rand[4],
    assunto: rand[5],
  };
}

// ---- SOCKET.IO LOGICA DI GIOCO ----
io.on("connection", (socket) => {
  console.log(`🟢 ${socket.id} connesso`);

  socket.on("joinGame", (name) => {
    users[socket.id] = { name, score: 0, acc: 0, fair: 0 };
    io.emit("leaderboard", Object.values(users));
  });

  socket.on("trainModel", (data) => {
    runPython(data, (res) => {
      if (!res) {
        socket.emit("modelError", { message: "Errore nell'elaborazione del modello." });
        return;
      }
      socket.emit("modelReady", res);
    });
  });

  socket.on("requestCase", () => {
    currentCase = pickCase();
    io.emit("newCase", currentCase);
  });

  socket.on("predictCase", (data) => {
    const correct = parseInt(currentCase.assunto) === parseInt(data.pred);
    if (correct) users[socket.id].score += 10;
    else users[socket.id].score -= 5;
    io.emit("leaderboard", Object.values(users));
  });

  socket.on("disconnect", () => {
    delete users[socket.id];
    io.emit("leaderboard", Object.values(users));
  });
});

// ---- AVVIO SERVER ----
server.listen(PORT, () => console.log(`✅ Server attivo su porta ${PORT}`));
