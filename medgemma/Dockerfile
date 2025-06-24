FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

WORKDIR /medgemma

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3-venv \
    libpq-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .
COPY images ./images/

EXPOSE 7860

CMD [ "python3", "app.py" ]