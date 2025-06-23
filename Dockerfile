
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt .

# Instalar PyTorch com CUDA + outras dependências
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

