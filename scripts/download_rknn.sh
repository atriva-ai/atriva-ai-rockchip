#!/bin/bash
# Download RKNN Toolkit2 and runtime components

set -e

RKNN_VERSION="2.3.2"
ARCH="aarch64"
PYTHON_VERSION="cp310"
RKNPU_DIR="/app/rknpu"

echo "🔧 Downloading RKNN Toolkit2 v${RKNN_VERSION}..."

# Create rknpu directory
mkdir -p ${RKNPU_DIR}

# Download RKNN Toolkit2 wheel
WHEEL_URL="https://github.com/rockchip-linux/rknn-toolkit2/releases/download/v${RKNN_VERSION}/rknn_toolkit2-${RKNN_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-manylinux_2_17_${ARCH}.manylinux2014_${ARCH}.whl"
echo "📦 Downloading wheel from: ${WHEEL_URL}"
wget -O ${RKNPU_DIR}/rknn_toolkit2-${RKNN_VERSION}-${PYTHON_VERSION}-${PYTHON_VERSION}-manylinux_2_17_${ARCH}.manylinux2014_${ARCH}.whl "${WHEEL_URL}"

# Download runtime library (if available)
# Note: This might need to be obtained from Rockchip directly
echo "📚 Note: librknnrt.so and rknn_server need to be obtained from Rockchip"
echo "   Place them in the rknpu/ directory before building"

echo "✅ RKNN Toolkit2 download completed"
