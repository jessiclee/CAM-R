FROM python:3.11-slim
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
COPY src .
CMD ["python", "./app.py"]
