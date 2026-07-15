# Mobile Field Capture Companion

An offline-first PWA + FastAPI backend for recording audio and tapping tempo
from a phone, syncing captures into the same `data/{dataset}/tracks/` +
`annotations/*.beats` structure `tools/annotator` reads. See
`docs/mobilecompanionfeasibility.md` for the design rationale.

## Local HTTPS / LAN dev setup

Phones require a secure context (HTTPS) for microphone access and for
installing the app as a PWA — plain `http://<lan-ip>` will not work. This
repo uses [mkcert](https://github.com/FiloSottile/mkcert) to generate a
locally-trusted certificate for dev use.

### One-time setup (per machine)

```bash
brew install mkcert
mkcert -install   # installs the local CA into your Mac's trust store — needs your sudo password interactively
```

`mkcert -install` prompts for your macOS password (it calls `sudo security
add-trusted-cert` under the hood) — run it yourself in a real terminal, it
can't be scripted non-interactively.

### Generate the dev certificate

Find your Mac's LAN IP (it can change between networks/DHCP leases):

```bash
ipconfig getifaddr en0
```

Then, from `tools/mobile_companion/certs/` (create it if missing):

```bash
cd tools/mobile_companion/certs
mkcert -cert-file dev-cert.pem -key-file dev-key.pem localhost 127.0.0.1 <your-lan-ip>
```

Re-run this whenever your LAN IP changes. These files are gitignored —
they're machine-local dev artifacts, not something to commit.

### Trusting the cert on your phone

`mkcert -install` only trusts the CA on *this Mac*. Your phone's browser
needs to trust it too:

1. Find the CA root: `mkcert -CAROOT` (prints a directory containing
   `rootCA.pem`).
2. AirDrop / email `rootCA.pem` to your phone.
3. **iPhone**: open the file → Settings will prompt to install the profile →
   then go to *Settings → General → About → Certificate Trust Settings* and
   enable full trust for the mkcert root.
4. **Android**: open the file → *Settings → Security → Encryption &
   credentials → Install a certificate → CA certificate*.

### Running the server over HTTPS

```bash
uv run uvicorn tools.mobile_companion.server:app \
  --host 0.0.0.0 --port 8443 \
  --ssl-keyfile tools/mobile_companion/certs/dev-key.pem \
  --ssl-certfile tools/mobile_companion/certs/dev-cert.pem
```

Then, from your phone (same WiFi network as the laptop):

```
https://<your-lan-ip>:8443/health
```

should return `{"status": "ok"}` with no certificate warning, once the CA
is trusted on the phone.
