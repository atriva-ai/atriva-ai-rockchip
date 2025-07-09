# Use Ubuntu 24.04 as base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.10 (default for Ubuntu 22.04)
RUN apt update && apt install -y \
    python3 python3-pip \
    libopencv-dev \
    libdrm-dev libjpeg-dev \
    libv4l-dev libtinfo5 \
    curl wget unzip git net-tools build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the correct RKNN Toolkit wheel
# Ensure the wheel matches cp310 (Python 3.10)
COPY rknn_toolkit2-1.6.0-cp310-cp310-linux_aarch64.whl /tmp/

# Install RKNN toolkit
RUN pip3 install /tmp/rknn_toolkit2-1.6.0-cp310-cp310-linux_aarch64.whl

# Create a working directory
WORKDIR /app

# Create an empty models directory
RUN mkdir -p /app/models

# Copy application files (excluding empty directories)
COPY . /app

# Ensure models directory exists (for volume mounting)
VOLUME ["/app/models"]

# Create a virtual environment
RUN python3 -m venv /app/venv

# Activate virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose FastAPI port
EXPOSE 8001

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
