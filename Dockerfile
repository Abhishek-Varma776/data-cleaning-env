FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Environment defaults (optional, but fine)
ENV TASK_NAME=easy

EXPOSE 7860

# Correct entrypoint (IMPORTANT)
CMD ["uvicorn", "server.app:main", "--host", "0.0.0.0", "--port", "7860"]