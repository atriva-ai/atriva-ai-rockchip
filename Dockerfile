# Base OS
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install required system packages
# Some mirrors have Release files that are "not valid yet" if the container clock differs.
# Disable APT date validation during update to avoid build failures.
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get install -y \
    python3 python3-pip python3-venv \
    libopencv-dev libdrm-dev libjpeg-dev \
    libv4l-dev libtinfo5 \
    curl wget unzip git net-tools build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Download RKNN toolkit and runtime components
COPY scripts/download_rknn.sh /tmp/
RUN chmod +x /tmp/download_rknn.sh

# Copy RKNN runtime components (if available locally)
# These need to be obtained from Rockchip and placed in rknpu/ directory
COPY rknpu/librknnrt.so /usr/lib64/ 2>/dev/null || echo "⚠️ librknnrt.so not found - please obtain from Rockchip"
COPY rknpu/rknn_server /usr/local/bin/ 2>/dev/null || echo "⚠️ rknn_server not found - please obtain from Rockchip"
RUN ldconfig 2>/dev/null || true
RUN chmod +x /usr/local/bin/rknn_server 2>/dev/null || true

# Copy RKNN startup scripts
COPY rknpu/start_rknn.sh /usr/local/bin/ 2>/dev/null || echo "⚠️ start_rknn.sh not found"
COPY rknpu/restart_rknn.sh /usr/local/bin/ 2>/dev/null || echo "⚠️ restart_rknn.sh not found"
RUN chmod +x /usr/local/bin/start_rknn.sh /usr/local/bin/restart_rknn.sh 2>/dev/null || true

# Create and activate virtual environment
RUN python3 -m venv /app/venv

# Copy requirements first (for better caching)
COPY requirements.txt .

# Download RKNN toolkit and install requirements
RUN /tmp/download_rknn.sh && \
    /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install /app/rknpu/rknn_toolkit2-*.whl && \
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
