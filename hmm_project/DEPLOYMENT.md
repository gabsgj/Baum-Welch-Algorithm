# Deploying to Zeabur

## Project Structure

The deployment is configured with these files:

| File | Purpose |
|---|---|
| `app.py` | Root entry point — exports the Flask `app` object |
| `zbpack.json` | Zeabur build config — Python version, entry, start command |
| `requirements.txt` | Python dependencies (pip) |
| `pyproject.toml` | Project metadata & dependency spec |

## How It Works

- **Zeabur auto-detects** `app.py` at the project root and recognises it as a Flask application.
- **Gunicorn** is launched with the `eventlet` worker class to support Flask-SocketIO WebSockets.
- The custom `start_command` in `zbpack.json` overrides the default Gunicorn invocation:
  ```
  gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app
  ```
- `$PORT` is injected by Zeabur at runtime.

## Deploy Steps

### 1. Push to GitHub

```bash
git add .
git commit -m "Ready for Zeabur deployment"
git push origin main
```

### 2. Create a Zeabur Project

1. Log in at https://zeabur.com
2. Click **New Project** → **Deploy from Git**
3. Authorise GitHub and select your repository
4. **Set Root Directory** → `hmm_project` (Settings → Root Directory) since the repo has a parent folder

### 3. Verify

Once the build finishes you'll receive a public URL (e.g. `https://your-app.zeabur.app`).
Open it to access the HMM Baum–Welch Training Dashboard.

## Environment Variables (Optional)

| Variable | Default | Description |
|---|---|---|
| `PORT` | Set by Zeabur | HTTP port — do **not** override |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |

## Local Testing

```bash
# Install dependencies
pip install -e .

# Run with Gunicorn (production-like)
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 app:app

# Or run directly
python app.py
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails | Check Zeabur build logs; ensure `requirements.txt` is complete |
| WebSocket not connecting | Confirm `--worker-class eventlet` is in the start command |
| 404 on `/` | Verify root directory is set to `hmm_project` |
| Missing packages | Run `pip install -e .` locally to verify deps resolve |
