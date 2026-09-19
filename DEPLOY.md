# Deployment

Backend on **Render** (free Web Service, deploys straight from GitHub —
no Docker-tier account restriction like some Hugging Face free accounts
hit). Frontend on a **Hugging Face Streamlit Space** (Streamlit SDK Spaces
have no such restriction; only the Docker SDK does on some accounts).

## 0. Backend on your own VPS via GitHub Actions (preferred)

`.github/workflows/deploy.yml` runs on every push to `main`:
`pytest` → build + push Docker image to Docker Hub → SSH into the VPS,
pull the new image, replace the container, wait for `/health`.

Required repo secrets (Settings → Secrets and variables → Actions):
`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `SSH_HOST`, `SSH_PORT`,
`SSH_USER`, `SSH_PRIVATE_KEY`.

VPS prerequisites: Docker installed, `SSH_USER` in the `docker` group,
the public half of `SSH_PRIVATE_KEY` in `~/.ssh/authorized_keys`, and the
host port (`HOST_PORT`, default 8000, set in the workflow) open in the
firewall. Models are cached in the named volume `aptino-claim-models`, so
only the first deploy pays the download. The API is then at
`http://<SSH_HOST>:8000` (`/health`, `/analyze`); put nginx/Caddy in front
for HTTPS if the frontend is served over HTTPS.

The same workflow also deploys the **Streamlit frontend**
(`Dockerfile.frontend`, slim image: only `streamlit` + `requests`) as a
second container on the shared `aptino-net` Docker network. It reaches the
backend by container name (`API_URL=http://aptino-claim-backend:7860`), so
no public URL is baked in. Frontend: `http://<SSH_HOST>:8501` (open port
8501 in the firewall). The backend is deployed and health-checked first;
the frontend only rolls out if it is healthy.

Rollback: on the VPS, `docker run` the previous `:<git-sha>` tag.

## 1. Backend — Render (alternative)

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

## 2. Frontend — Hugging Face Streamlit Space

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

## 3. Verify

- `GET https://<backend>.onrender.com/health` → `{"status": "ok", ...}`
- Open the frontend Space, run a supplied case, confirm citations + trace render.

## Updating after a code change

- Backend: push to `main` on GitHub — Render auto-redeploys.
- Frontend: `git checkout deploy-hf-frontend && git merge main && git push hf-frontend deploy-hf-frontend:main && git checkout main`

## Alternative: both on Hugging Face

If your HF account later has Docker Spaces enabled, `deploy-hf-backend`
(also branched from `main`, with its own Space-config `README.md` +
`app_port: 7860`) can be pushed the same way to a Docker Space instead of
using Render — see git history / the `deploy-hf-backend` branch.
