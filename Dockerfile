# Base OS
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install required system packages
RUN apt update && apt install -y \
    python3 python3-pip python3-venv \
    libopencv-dev libdrm-dev libjpeg-dev \
    libv4l-dev libtinfo5 \
    curl wget unzip git net-tools build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy RKNN Toolkit wheel
COPY rknn_toolkit2-2.3.2-*.whl /tmp/

# Create and activate virtual environment
RUN python3 -m venv /app/venv

# Install RKNN and requirements into venv
COPY requirements.txt .
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install /tmp/rknn_toolkit2-2.3.2-*.whl && \
    /app/venv/bin/pip install -r requirements.txt

# Add venv to PATH so CMD can access installed packages
ENV PATH="/app/venv/bin:$PATH"

# Copy rest of the application
COPY . .

# Ensure models dir exists and can be mounted
RUN mkdir -p /app/models
VOLUME ["/app/models"]

# Expose FastAPI port
EXPOSE 8001

# Start FastAPI app using uvicorn (from venv)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
