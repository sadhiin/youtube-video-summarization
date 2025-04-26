FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

#ECHO "export PYTHONPATH=/app" >> ~/.bashrc
#RUN echo "export PYTHONPATH=/app" >> ~/.bashrc

#Expose the ports
EXPOSE 8000 8501


CMD ["bash"]