import express from "express";
import http from "http";
import { Server } from "socket.io";
import { spawn } from "child_process";
import fs from "fs";

const app = express();
const server = http.createServer(app);
const io = new Server(server);
app.use(express.static("public"));

const PORT = process.env.PORT || 3000;
const users = {};
let currentCase = null;

// scegli un caso casuale dal dataset
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

io.on("connection", (socket) => {
  console.log(`🟢 ${socket.id} connesso`);

  socket.on("joinGame", (name) => {
    users[socket.id] = { name, score: 0, acc: 0, fair: 0 };
    io.emit("leaderboard", Object.values(users));
  });

  socket.on("trainModel", (data) => {
    const args = JSON.stringify(data);
    const py = spawn("python", ["model_eval.py", args]);
    let output = "";
    py.stdout.on("data", (d) => (output += d.toString()));
    py.on("close", () => {
      try {
        const res = JSON.parse(output);
        users[socket.id] = { ...users[socket.id], ...res };
        socket.emit("modelReady", res);
        io.emit("leaderboard", Object.values(users));
      } catch (err) {
        console.error("Errore Python:", err, output);
      }
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

server.listen(PORT, () => console.log(`✅ Server attivo su porta ${PORT}`));
