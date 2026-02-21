FROM python:3.11-slim

LABEL "language"="python"
LABEL "framework"="flask"

WORKDIR /app

COPY . .

# Install the package and all dependencies declared in pyproject.toml
RUN pip install --no-cache-dir -e .

EXPOSE 8080

# socketio.run() reads $PORT (set by Zeabur) and falls back to 5000 locally
CMD ["python", "app.py"]
