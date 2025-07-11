#!/bin/bash

# Set up logging
LOG_FILE="/tmp/rknn_server.log"
PID_FILE="/tmp/rknn_server.pid"

# Function to start RKNN server
start_rknn_server() {
    echo "$(date): Starting RKNN server..." >> "$LOG_FILE"
    rknn_server >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "$(date): RKNN server started with PID $(cat $PID_FILE)" >> "$LOG_FILE"
}

# Function to stop RKNN server
stop_rknn_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "$(date): Stopping RKNN server (PID: $PID)" >> "$LOG_FILE"
            kill "$PID"
            rm -f "$PID_FILE"
        fi
    fi
}

# Clean up on exit
trap 'stop_rknn_server; exit' TERM INT

# Main loop
while true; do
    # Check if RKNN server is running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "$(date): RKNN server crashed, restarting..." >> "$LOG_FILE"
            rm -f "$PID_FILE"
            start_rknn_server
        fi
    else
        echo "$(date): RKNN server not running, starting..." >> "$LOG_FILE"
        start_rknn_server
    fi
    
    sleep 5
done
