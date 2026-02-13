# Deploying to Zeabur

This guide walks you through deploying the HMM Baum-Welch application to **Zeabur**, a modern deployment platform.

## Prerequisites

1. **GitHub account** with your code pushed to a repository
2. **Zeabur account** (sign up at https://zeabur.com)
3. The deployment files have been added to your project:
   - `wsgi.py` — WSGI application entry point
   - `zeabur.json` — Zeabur configuration
   - `Procfile` — Process file for production
   - `requirements.txt` — Dependencies file
   - `runtime.txt` — Python version

## Step 1: Push Your Code to GitHub

```bash
cd d:\Baum-Welch-Algorithm\hmm_project
git init
git add .
git commit -m "Add deployment configuration for Zeabur"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub username and repository name.

## Step 2: Connect and Deploy on Zeabur

1. **Log in to Zeabur**: https://zeabur.com
2. **Create a new project**:
   - Click "New Project" or "Create"
   - Select "Deploy from Git"
3. **Authorize GitHub**:
   - Click "Install GitHub App"
   - Authorize Zeabur to access your repositories
4. **Select your repository**:
   - Find and select your HMM Baum-Welch repository
   - Choose your branch (typically `main`)
5. **Configure the deployment**:
   - Zeabur will automatically detect it's a Python project
   - It will read `zeabur.json` and `Procfile` for configuration
   - The build should start automatically

## Step 3: Environment Configuration (Optional)

If your app needs environment variables, add them in Zeabur:

1. Go to your project dashboard
2. Navigate to **Settings** → **Environment Variables**
3. Add any needed variables (e.g., `FLASK_ENV=production`)

## Step 4: Verify Deployment

Once the build completes:

1. You'll get a **public URL** (e.g., `https://your-app.zeabur.app`)
2. Click the URL to open your HMM dashboard
3. Test the training functionality with sample data

## What Was Added

### `wsgi.py`
- Entry point for production WSGI servers (Gunicorn)
- Handles Socket.IO integration for WebSocket support
- Listens on environment variable `PORT` (or 5000 locally)

### `zeabur.json`
- Tells Zeabur to build with Python
- Configures it as a WSGI application
- Points to the `wsgi:app` entry point

### `Procfile`
- Specifies how to run the app in production
- Uses Gunicorn with Eventlet worker for WebSocket support
- Binds to `0.0.0.0` on the PORT environment variable

### `pyproject.toml` (Updated)
- Added `gunicorn>=21.0` to dependencies
- Required for production deployments

## Troubleshooting

### Build Fails
- **Check logs**: View build logs in Zeabur dashboard
- **Python version**: Ensure Python 3.11+ is available
- **Dependencies**: Verify all dependencies install correctly
  ```bash
  pip install -e .
  ```

### App Runs But Dashboard Doesn't Load
- **CORS issues**: The Flask app allows all origins (`cors_allowed_origins="*"`)
- **WebSocket**: Zeabur supports WebSocket for Socket.IO

### WebSocket Connection Fails
- **Eventlet worker**: The Procfile uses `--worker-class eventlet` which supports WebSocket
- **Check console**: Browser DevTools → Network/Console for WebSocket errors

## Local Testing Before Deploy

Test the WSGI setup locally:

```bash
pip install gunicorn
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 wsgi:app
```

Visit `http://localhost:5000` in your browser.

## Performance Notes

- **Single worker**: The Procfile uses `-w 1` (one worker) since Eventlet handles concurrency
- **Memory**: Python apps typically need 512 MB+ RAM
- **Timeouts**: Long training runs might timeout (configure in Zeabur settings if needed)

## Next Steps

- Monitor your app in the Zeabur dashboard
- Set up custom domain (if desired)
- Configure auto-deployments on GitHub push
- Add monitoring/logging (Zeabur has built-in logs)

---

For more info: https://docs.zeabur.com/en/guide/deploy
