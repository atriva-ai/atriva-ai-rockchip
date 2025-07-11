#!/bin/bash

PID_FILE="/tmp/rknn_server.pid"
LOG_FILE="/tmp/rknn_server.log"

echo "$(date): Restarting RKNN server..." >> "$LOG_FILE"

# Kill existing RKNN server processes
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date): Stopping existing RKNN server (PID: $PID)" >> "$LOG_FILE"
        kill "$PID"
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Kill any remaining rknn_server processes
pkill -f rknn_server 2>/dev/null

# Kill start_rknn.sh process
pkill -f start_rknn.sh 2>/dev/null

# Wait a moment
sleep 1

# Start RKNN server
start_rknn.sh &
echo "$(date): RKNN server restart initiated" >> "$LOG_FILE"