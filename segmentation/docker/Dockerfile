FROM ros:humble-ros-base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    nano \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    python3-venv \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /tmp/rgbt-requirements.txt

RUN python -m pip install --no-cache-dir \
        -r /tmp/rgbt-requirements.txt \
    && rm -f /tmp/rgbt-requirements.txt

RUN python -m pip install --no-cache-dir \
    onnx==1.17.0 \
    tensorrt-cu12==10.13.3.9

RUN python -m ipykernel install \
    --sys-prefix \
    --name rgbt \
    --display-name "Python (RGB-T Transformer)"

WORKDIR /workspace

EXPOSE 8898

CMD ["/bin/bash"]
