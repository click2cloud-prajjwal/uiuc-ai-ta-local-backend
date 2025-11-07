FROM python:3.10-slim

# Install only required runtime packages (Tesseract + SSL certs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        ca-certificates \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Install Python dependencies (no local build tools needed)
# RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001

CMD ["python", "worker.py"]
