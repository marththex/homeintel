# Enabling the Microphone on Your Phone (Tailscale HTTPS)

The voice/mic button needs a **secure context** (HTTPS or `localhost`). On desktop
`localhost` it works out of the box. On your phone, the app is typically reached
over plain HTTP on a LAN IP, where mobile browsers (iOS Safari in particular)
block the microphone. This sets up a private HTTPS origin via **Tailscale Serve**
— no public internet exposure, and a publicly-trusted certificate so the phone's
browser trusts it with no profile install.

## Prerequisites
- Tailscale installed and logged in on the machine running the app and on your phone.
- In the Tailscale admin console: **MagicDNS** enabled and **HTTPS Certificates** enabled.
- Your app host's MagicDNS name, e.g. `mypc.tailXXXX.ts.net`.

## Recommended: one HTTPS origin (frontend + API together)

Serving the built UI and the API from one origin avoids mixed-content and CORS.

1. Build the frontend against a **relative** API base so it calls the same origin:

   ```bash
   cd frontend
   # empty base → api.ts uses relative URLs like /chat, /health
   VITE_API_BASE_URL= npm run build
   ```

2. Have FastAPI serve the built `frontend/dist`. In `backend/main.py`, **after** all
   `app.include_router(...)` lines, add:

   ```python
   from fastapi.staticfiles import StaticFiles
   # API routes are registered above and take precedence; this serves the SPA.
   app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="frontend")
   ```

3. Start the API (it now serves the UI too) on port 8000.

4. Expose it over HTTPS within your tailnet:

   ```bash
   tailscale serve --bg --https=443 http://127.0.0.1:8000
   ```

5. On your phone (with Tailscale connected), open
   `https://mypc.tailXXXX.ts.net` → **Share → Add to Home Screen**. The mic now works.

## Alternative: two HTTPS ports (no backend code change)

Keep the dev server and API separate, expose both over HTTPS, and point the
frontend at the HTTPS API:

```bash
tailscale serve --bg --https=443  http://127.0.0.1:5173   # frontend
tailscale serve --bg --https=8443 http://127.0.0.1:8000   # API
```

Then build the frontend with `VITE_API_BASE_URL=https://mypc.tailXXXX.ts.net:8443`
and add that origin to `CORS_ALLOW_ORIGINS` in `.env` (comma-separated), then
restart the backend. (One origin is simpler — prefer the recommended approach
above.)

## Troubleshooting
- **Mic still blocked:** confirm the address bar shows `https://…ts.net` (not an IP)
  and a valid lock. `getUserMedia` is unavailable on insecure origins.
- **Blank page / 404 on refresh:** ensure `frontend/dist` exists (run the build) and
  the `StaticFiles` mount is the **last** thing added in `main.py`.
- **Mixed-content errors in console:** the page is HTTPS but an API call is HTTP —
  use the same-origin (recommended) setup, or the two-HTTPS-ports alternative.
- **`tailscale serve` status:** `tailscale serve status`; reset with `tailscale serve reset`.
