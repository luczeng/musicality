# Mobile Field Capture Companion

An offline-first PWA + FastAPI backend for recording audio and tapping tempo
from a phone, syncing captures into the same `data/{dataset}/tracks/` +
`annotations/*.beats` structure `tools/annotator` reads. See
`docs/mobilecompanionfeasibility.md` for the design rationale.

## Requirements

`ffmpeg` must be installed on the machine running the server (`brew install
ffmpeg`). Phones record compressed audio — webm/opus on Android, mp4/aac on
iOS — and decoding those formats falls back to `librosa`/`audioread` shelling
out to `ffmpeg`; without it, uploads from real devices fail with a 400
("could not decode audio") even though everything else works fine.

## Remote connectivity + HTTPS (via Tailscale)

The phone needs to reach the laptop from **any** network — home WiFi, mobile
data, anywhere — not just the same LAN. A plain WiFi address like
`192.168.178.96` only resolves inside the home network, so it can't work for
that. This repo uses [Tailscale](https://tailscale.com), a private mesh VPN,
so the laptop and phone can always find each other regardless of network,
plus it can auto-issue trusted HTTPS certificates (no manual cert-trusting
dance needed).

### One-time setup (per device)

**Laptop**: `brew install --cask tailscale`, then open the Tailscale app and
log in (opens a browser — pick any account, e.g. Google/Microsoft/GitHub).
This needs to be done interactively by hand, it can't be scripted.

**Phone**: install the Tailscale app from the App Store / Play Store, log in
with the **same account**.

Once both are signed in, check the laptop's Tailscale identity:

```bash
tailscale status   # confirms it's connected
tailscale ip        # e.g. 100.103.0.56
```

The laptop also gets a stable hostname of the form
`<device-name>.<tailnet-name>.ts.net` (run `tailscale status` to see it) —
this is what the phone will use to reach it, since the IP is more likely to
be memorized wrong than the name.

### Enabling HTTPS certificates

This is a one-time toggle in the Tailscale admin console (can't be done from
the command line — needs your login in a browser):

1. Go to https://login.tailscale.com/admin/dns and turn on **HTTPS
   Certificates**.
2. Then, on the laptop:
   ```bash
   cd tools/mobile_companion/certs
   tailscale cert <device-name>.<tailnet-name>.ts.net
   ```
   This writes a real, trusted certificate + key (no phone-side trust setup
   needed at all — unlike a self-signed cert, the phone already trusts it).
   Re-run this before the cert's expiry (Tailscale certs auto-renew if you
   set up a cron/launchd job to re-run this periodically; for now, manual
   re-issue is enough).

### Running the server over HTTPS

```bash
uv run uvicorn tools.mobile_companion.server:app \
  --host 0.0.0.0 --port 8443 \
  --ssl-keyfile tools/mobile_companion/certs/<device-name>.<tailnet-name>.ts.net.key \
  --ssl-certfile tools/mobile_companion/certs/<device-name>.<tailnet-name>.ts.net.crt
```

Then, from your phone (on WiFi *or* mobile data — Tailscale must be running
in the phone's Tailscale app):

```
https://<device-name>.<tailnet-name>.ts.net:8443/health
```

should return `{"status": "ok"}` with no certificate warning.

## Local same-WiFi dev fallback (mkcert)

For quick local testing without Tailscale running, a self-signed cert for
the LAN IP still works, though the phone must manually trust it and be on
the same WiFi:

```bash
brew install mkcert
mkcert -install   # needs your sudo password interactively, run it yourself
cd tools/mobile_companion/certs
mkcert -cert-file dev-cert.pem -key-file dev-key.pem localhost 127.0.0.1 <your-lan-ip>
```

Trusting it on the phone: find the CA root with `mkcert -CAROOT`, AirDrop/
email `rootCA.pem` to the phone, then on **iPhone**: open the file → install
the profile → *Settings → General → About → Certificate Trust Settings* →
enable full trust; on **Android**: open the file → *Settings → Security →
Encryption & credentials → Install a certificate → CA certificate*.

This path is a fallback only — Tailscale above is the primary way this app
is meant to be used.
