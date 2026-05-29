# Daedalus WhatsApp bridge (unofficial)

WhatsApp has no official self-hosted bot API, so this folder is a small **Node companion
process** that Daedalus drives to reach WhatsApp. It uses the community
[`whatsapp-web.js`](https://github.com/pedroslopez/whatsapp-web.js) library, which
automates WhatsApp Web behind the scenes.

## ⚠ Read this first

- This is **unofficial** and **reverse-engineered**. WhatsApp can change things and break
  it at any time.
- Automating a personal account can get it **rate-limited or banned**. Use a
  **secondary / throwaway number**.
- This is the **least stable** Daedalus surface. For anything you depend on, use Telegram
  (`dae telegram`).

## Requirements

- **Node.js** (LTS) — includes `npm`. Get it from <https://nodejs.org/>.
- A phone with WhatsApp to scan the login QR (once).

## How it runs

You don't normally start this by hand — `dae whatsapp` spawns it for you, installing
dependencies on first run and waiting for the QR login. If you want to run it standalone:

```bash
cd bridge/whatsapp
npm install
PORT=8765 node index.js   # then scan the QR printed in the terminal
```

The bridge exposes a localhost-only HTTP API the Python side polls:

| Method | Path        | Purpose                                            |
|--------|-------------|----------------------------------------------------|
| GET    | `/status`   | `{ ready, qr }` — login state + current QR string  |
| GET    | `/messages` | drains the inbound 1:1 message queue               |
| POST   | `/send`     | `{ to, body }` — send a reply                      |

The login session is cached under `bridge/whatsapp/.wwebjs_auth/`, so you only scan the
QR once. Delete that folder to force a fresh login.

## Configuration

`dae` reads `WHATSAPP_BRIDGE_URL` from your `.env` (default `http://127.0.0.1:8765`). The
port in that URL is passed to the bridge as `PORT`.
