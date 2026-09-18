# Deploying to Hugging Face Spaces

Two Spaces are needed under your HF account: one Docker Space for the
FastAPI backend, one Streamlit Space for the frontend. The `main` branch
stays a normal GitHub repo; each Space gets its own local branch with a
Space-specific `README.md` (HF reads Space config from that file's YAML
frontmatter), then that branch is pushed to the Space's git remote.

## 0. One-time setup

```bash
pip install -U huggingface_hub
huggingface-cli login          # paste an HF access token (Write scope) — huggingface.co/settings/tokens
```

Create the two Spaces (web UI is simplest: huggingface.co/new-space):
- **Backend**: SDK = Docker, name e.g. `aptino-claim-backend`
- **Frontend**: SDK = Streamlit, name e.g. `aptino-claim-frontend`

## 1. Backend (Docker Space)

```bash
git checkout -b deploy-hf-backend main
cat > README.md <<'EOF'
---
title: Aptino Claim Backend
emoji: 🩺
sdk: docker
app_port: 7860
---
EOF
git add README.md
git commit -m "HF Space config: docker backend"
git remote add hf-backend https://huggingface.co/spaces/<your-username>/aptino-claim-backend
git push hf-backend deploy-hf-backend:main
git checkout main
```

No secrets are required for the default `LLM_PROVIDER=offline` mode. To
enable LLM-phrased rationales, set `LLM_PROVIDER`, `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `OPENAI_MODEL` as **Space secrets** (Settings → Variables
and secrets) rather than committing them.

Once built, the API is at `https://<your-username>-aptino-claim-backend.hf.space`.

## 2. Frontend (Streamlit Space)

```bash
git checkout -b deploy-hf-frontend main
cat > README.md <<'EOF'
---
title: Aptino Claim Frontend
emoji: 🩺
sdk: streamlit
app_file: frontend/streamlit_app.py
---
EOF
git add README.md
git commit -m "HF Space config: streamlit frontend"
git remote add hf-frontend https://huggingface.co/spaces/<your-username>/aptino-claim-frontend
git push hf-frontend deploy-hf-frontend:main
git checkout main
```

Set the Space secret `API_URL` to the backend URL from step 1
(`https://<your-username>-aptino-claim-backend.hf.space`) under Settings →
Variables and secrets, then restart the Space.

## 3. Verify

- `GET https://<backend-space>/health` → `{"status": "ok", ...}`
- Open the frontend Space, run a supplied case, confirm citations + trace render.

## Updating after a code change

```bash
git checkout deploy-hf-backend && git merge main && git push hf-backend deploy-hf-backend:main && git checkout main
git checkout deploy-hf-frontend && git merge main && git push hf-frontend deploy-hf-frontend:main && git checkout main
```
