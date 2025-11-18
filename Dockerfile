# Use lightweight official Python image
FROM python:3.11-slim

# -----------------------
# Environment variables
# -----------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# -----------------------
# Install system dependencies
# -----------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Set working directory
# -----------------------
WORKDIR /app

# -----------------------
# Copy all application files
# -----------------------
COPY . .

# -----------------------
# Install Python dependencies
# -----------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------
# Expose Cloud Run port
# -----------------------
EXPOSE 8080

# -----------------------
# Run the Flask app
# -----------------------
CMD ["python", "main.py"]
