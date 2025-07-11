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
    curl wget unzip git net-tools build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy RKNN toolkit from local rknpu directory
COPY rknpu/rknn_toolkit2-2.3.2-*.whl /tmp/

# Copy RKNN runtime library to system library folder
COPY rknpu/librknnrt.so /usr/lib64/
RUN ldconfig

# Copy RKNN server binary to system path
COPY rknpu/rknn_server /usr/local/bin/
RUN chmod +x /usr/local/bin/rknn_server

# Copy RKNN startup scripts
COPY rknpu/start_rknn.sh /usr/local/bin/
COPY rknpu/restart_rknn.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start_rknn.sh /usr/local/bin/restart_rknn.sh

# Create and activate virtual environment
RUN python3 -m venv /app/venv

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install RKNN and requirements into venv (this layer will be cached unless requirements.txt changes)
RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install /tmp/rknn_toolkit2-2.3.2-*.whl && \
    /app/venv/bin/pip install -r requirements.txt

# Add venv to PATH so CMD can access installed packages
ENV PATH="/app/venv/bin:$PATH"

# Copy rest of the application (this layer changes frequently but doesn't invalidate pip install)
COPY . .

# Ensure models dir exists and can be mounted
RUN mkdir -p /app/models
RUN mkdir -p /tmp/models && chmod 755 /tmp/models
VOLUME ["/app/models"]

# Expose FastAPI port
EXPOSE 8001

# Create startup script that starts both RKNN server and FastAPI
RUN echo '#!/bin/bash\n\
# Start RKNN server in background\n\
start_rknn.sh &\n\
RKNN_PID=$!\n\
\n\
# Wait a moment for RKNN server to start\n\
sleep 2\n\
\n\
# Start FastAPI app\n\
exec uvicorn main:app --host 0.0.0.0 --port 8001\n\
' > /app/start.sh && chmod +x /app/start.sh

# Start both RKNN server and FastAPI app
CMD ["/app/start.sh"]
