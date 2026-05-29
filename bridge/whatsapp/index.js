/**
 * Daedalus WhatsApp bridge (unofficial).
 *
 * A thin companion process the Python side (daedalus/surfaces/whatsapp.py) spawns and
 * talks to over localhost HTTP. It owns the whatsapp-web.js client: it logs in by
 * printing a QR code you scan with your phone, queues inbound messages for Python to
 * drain, and sends replies Python hands back.
 *
 * HTTP API (all JSON):
 *   GET  /status            -> { ready: bool, qr: string|null }
 *   GET  /messages          -> { messages: [ { from, body, id } ] }   (drains the queue)
 *   POST /send  {to, body}  -> { ok: true }
 *
 * This is intentionally tiny and stateless beyond an in-memory inbound queue. The
 * WhatsApp session itself is persisted by LocalAuth under ./.wwebjs_auth so you only
 * scan the QR once.
 *
 * CAVEAT: whatsapp-web.js is an unofficial automation of WhatsApp Web and can break or
 * get an account flagged. Use a throwaway/secondary number and treat this as the least
 * stable surface. Prefer Telegram for anything important.
 */

const express = require("express");
const qrcode = require("qrcode-terminal");
const { Client, LocalAuth } = require("whatsapp-web.js");

const PORT = parseInt(process.env.PORT || "8765", 10);

// --- in-memory state --------------------------------------------------------
let ready = false;
let lastQr = null;
const inbound = []; // queue of { from, body, id }, drained by GET /messages

// --- WhatsApp client --------------------------------------------------------
const client = new Client({
  authStrategy: new LocalAuth({ dataPath: ".wwebjs_auth" }),
  puppeteer: { args: ["--no-sandbox", "--disable-setuid-sandbox"] },
});

client.on("qr", (qr) => {
  lastQr = qr;
  console.log("[bridge] scan this QR with WhatsApp > Linked Devices > Link a Device:");
  qrcode.generate(qr, { small: true });
});

client.on("ready", () => {
  ready = true;
  lastQr = null;
  console.log("[bridge] WhatsApp client is ready.");
});

client.on("disconnected", (reason) => {
  ready = false;
  console.log("[bridge] disconnected:", reason);
});

client.on("message", (msg) => {
  // Ignore group chats and status broadcasts; answer 1:1 messages only.
  if (msg.from === "status@broadcast" || msg.from.endsWith("@g.us")) return;
  inbound.push({ from: msg.from, body: msg.body || "", id: msg.id ? msg.id._serialized : "" });
});

client.initialize();

// --- HTTP API ---------------------------------------------------------------
const app = express();
app.use(express.json());

app.get("/status", (_req, res) => {
  res.json({ ready, qr: lastQr });
});

app.get("/messages", (_req, res) => {
  const batch = inbound.splice(0, inbound.length); // drain
  res.json({ messages: batch });
});

app.post("/send", async (req, res) => {
  const { to, body } = req.body || {};
  if (!to || !body) {
    return res.status(400).json({ ok: false, error: "missing 'to' or 'body'" });
  }
  try {
    await client.sendMessage(to, body);
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err) });
  }
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`[bridge] HTTP API listening on http://127.0.0.1:${PORT}`);
});
