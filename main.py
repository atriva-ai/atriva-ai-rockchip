from fastapi import FastAPI
from app.routes import router  # Import API routes
from app.shared_data import list_available_cameras, get_shared_frames_path, get_shared_temp_path
import os
import logging

# import debugpy

# debugpy.listen(("0.0.0.0", 5678))  # Allow debugger to connect
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

app = FastAPI(title="AI Vision API")

# Configure logging to reduce repetitive logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

# Include Routes
app.include_router(router)

@app.get("/")
def root():
    return {"message": "AI Vision API is running!"}

@app.get("/health")
def health_check():
    """Health check endpoint with shared volume status."""
    # Simple health check without logging
    return {"status": "healthy"}
