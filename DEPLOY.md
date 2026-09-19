# Deployment

Live: frontend https://aptino-claim-frontend.twistark.cloud, API https://aptino-claim-backend.twistark.cloud/health (nginx + HTTPS in front of the compose stack).

Everything is Dockerized and deployed to a VPS (tested sizing: 4 vCPU /
16 GB RAM / 200 GB disk — far more than needed; ~2 GB RAM is plenty).

## VPS via GitHub Actions + docker compose (primary)

`.github/workflows/deploy.yml` runs on every push to `main`:

1. **build-and-push**: builds `Dockerfile` (backend) and `Dockerfile.frontend`
   (slim Streamlit image) and pushes both to Docker Hub, tagged `:latest`
   and `:<git-sha>`.
2. **deploy**: copies `docker-compose.yml` to `~/aptino-claim` on the VPS
   over SSH, then runs `docker compose pull && docker compose up -d --wait`.
   `--wait` blocks until the healthchecks pass (backend first, the
   frontend `depends_on` it), and the job fails with container logs if they
   don't.

Required repo secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `SSH_HOST`,
`SSH_PORT`, `SSH_USER`, `SSH_PRIVATE_KEY`.

VPS prerequisites: Docker + the compose plugin installed, `SSH_USER` in
the `docker` group, the public half of `SSH_PRIVATE_KEY` in
`~/.ssh/authorized_keys`, and ports **8000** (API) and **8501** (frontend)
open in the firewall.

- API: `http://<SSH_HOST>:8000` (`/health`, `/analyze`)
- Frontend: `http://<SSH_HOST>:8501` (talks to the backend over the compose
  network at `http://backend:7860`, no public URL baked in)
- Models are cached in the `models` volume, so only the first deploy pays
  the ~400 MB download (first boot takes a few minutes).
- Optional LLM rationale: set `LLM_PROVIDER`, `OPENAI_API_KEY`,
  `OPENAI_BASE_URL`, `OPENAI_MODEL` in a `.env` file next to the compose
  file on the VPS (never commit keys).
- HTTPS: put Caddy/nginx in front if you need it.
- Rollback: `IMAGE_TAG=<old-sha> DOCKERHUB_USERNAME=<user> docker compose up -d`
  in `~/aptino-claim` on the VPS.

Run the same stack locally: `DOCKERHUB_USERNAME=local docker compose up --build`.

## Alternative: Render backend + HF Streamlit frontend

1. https://dashboard.render.com → **New +** → **Web Service** → connect
   the `aptino-claim-engine` GitHub repo.
2. Render auto-detects `Dockerfile` at the repo root (env: Docker). Plan:
   **Free**.
3. No environment variables are required for the default
   `LLM_PROVIDER=offline` mode (already set via `render.yaml`). To enable
   LLM-phrased rationales, add `LLM_PROVIDER=openai_compatible`,
   `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` as Render
   environment variables (never commit them).
4. Deploy. Render builds the Dockerfile and exposes the service on the
   `$PORT` it injects (the Dockerfile already reads `$PORT`, defaulting to
   7860 for other hosts).
5. Confirm: `GET https://<your-service>.onrender.com/health` →
   `{"status": "ok", ...}`.

(`render.yaml` at the repo root lets you instead use Render's "Blueprint"
one-click flow — New + → Blueprint → pick this repo.)

Free-tier note: the service spins down after 15 minutes idle and cold-starts
(~30-60s, including first-time model download) on the next request — fine
for review/demo use.

### Frontend — Hugging Face Streamlit Space

```bash
pip install -U huggingface_hub
huggingface-cli login          # paste a Write-scope token — huggingface.co/settings/tokens
```

Create the Space via https://huggingface.co/new-space → SDK = **Streamlit**
(e.g. name it `aptino-claim-frontend`).

```bash
git checkout -b deploy-hf-frontend main
# README.md on this branch already has the Streamlit Space frontmatter
# (title/sdk/app_file) — see the branch if you need to re-create it.
git remote add hf-frontend https://huggingface.co/spaces/<your-username>/aptino-claim-frontend
git push hf-frontend deploy-hf-frontend:main
git checkout main
```

Set the Space secret `API_URL` to the Render backend URL from step 1
(Settings → Variables and secrets), then restart the Space.

### Verify

- `GET https://<backend>.onrender.com/health` → `{"status": "ok", ...}`
- Open the frontend Space, run a supplied case, confirm citations + trace render.

### Updating after a code change

- Backend: push to `main` on GitHub — Render auto-redeploys.
- Frontend: `git checkout deploy-hf-frontend && git merge main && git push hf-frontend deploy-hf-frontend:main && git checkout main`

## Alternative: both on Hugging Face

If your HF account later has Docker Spaces enabled, `deploy-hf-backend`
(also branched from `main`, with its own Space-config `README.md` +
`app_port: 7860`) can be pushed the same way to a Docker Space instead of
using Render — see git history / the `deploy-hf-backend` branch.
