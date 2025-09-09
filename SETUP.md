# RKNN Setup Guide

## Prerequisites

Before building the AI inference service, you need to obtain the following RKNN components from Rockchip:

### Required Files

1. **librknnrt.so** - RKNN runtime library
2. **rknn_server** - RKNN server binary
3. **start_rknn.sh** - RKNN server startup script
4. **restart_rknn.sh** - RKNN server restart script

### Where to Get These Files

These files are typically found in:
- Rockchip RKNN Toolkit2 installation directory
- Rockchip development board SDK
- Official Rockchip documentation/releases

### Setup Steps

1. **Create rknpu directory** (if not exists):
   ```bash
   mkdir -p services/atriva-ai-rockchip/rknpu/
   ```

2. **Copy required files** to `rknpu/` directory:
   ```bash
   # Copy from your RKNN installation
   cp /path/to/your/rknn/librknnrt.so services/atriva-ai-rockchip/rknpu/
   cp /path/to/your/rknn/rknn_server services/atriva-ai-rockchip/rknpu/
   cp /path/to/your/rknn/start_rknn.sh services/atriva-ai-rockchip/rknpu/
   cp /path/to/your/rknn/restart_rknn.sh services/atriva-ai-rockchip/rknpu/
   ```

3. **Make scripts executable**:
   ```bash
   chmod +x services/atriva-ai-rockchip/rknpu/*.sh
   ```

4. **Build the service**:
   ```bash
   docker compose build ai_inference
   ```

## What Gets Downloaded Automatically

- **rknn_toolkit2 wheel**: Downloaded during Docker build from GitHub releases
- **Python dependencies**: Installed from requirements.txt

## Troubleshooting

### Missing Runtime Components
If you see warnings about missing `librknnrt.so` or `rknn_server`:
1. Ensure you've copied the files to the `rknpu/` directory
2. Check file permissions (should be executable for scripts)
3. Verify the files are from the correct architecture (aarch64)

### Build Failures
- Check internet connection (needed for wheel download)
- Verify Python version compatibility (currently using Python 3.10)
- Ensure all required system packages are installed

## File Structure

```
services/atriva-ai-rockchip/
├── rknpu/                    # RKNN runtime components (not in git)
│   ├── librknnrt.so         # Runtime library
│   ├── rknn_server          # Server binary
│   ├── start_rknn.sh        # Startup script
│   └── restart_rknn.sh      # Restart script
├── scripts/
│   └── download_rknn.sh     # Downloads toolkit wheel
└── Dockerfile               # Updated to use dynamic download
```
