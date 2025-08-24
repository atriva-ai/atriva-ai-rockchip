#!/usr/bin/env python3
"""
Test script for AI service vehicle tracking functionality
"""

import requests
import json
import time
import sys
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8001"
CAMERA_ID = "1"

def test_vehicle_tracking():
    """Test the complete vehicle tracking workflow in AI service"""
    
    print("🚗 Testing AI Service Vehicle Tracking Functionality")
    print("=" * 60)
    
    # Test 1: Start vehicle tracking
    print("\n1. Starting vehicle tracking...")
    try:
        config = {
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.8
        }
        response = requests.post(
            f"{BASE_URL}/vehicle-tracking/start/",
            data={"camera_id": CAMERA_ID, "tracking_config": json.dumps(config)}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Tracking started: {result['message']}")
        else:
            print(f"❌ Failed to start tracking: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error starting tracking: {e}")
        return False
    
    # Test 2: Check tracking status
    print("\n2. Checking tracking status...")
    try:
        response = requests.get(f"{BASE_URL}/vehicle-tracking/status/{CAMERA_ID}")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status retrieved: {status}")
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return False
    
    # Test 3: Process frame with tracking
    print("\n3. Processing frame with tracking...")
    try:
        response = requests.post(
            f"{BASE_URL}/vehicle-tracking/process-frame/",
            data={"camera_id": CAMERA_ID, "frame_number": 0}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Frame processed: {result['tracked_vehicles']} vehicles tracked")
            print(f"   Saved path: {result['saved_path']}")
        else:
            print(f"❌ Failed to process frame: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return False
    
    # Test 4: Get annotated frame
    print("\n4. Getting annotated frame...")
    try:
        response = requests.get(f"{BASE_URL}/vehicle-tracking/annotated-frame/{CAMERA_ID}")
        if response.status_code == 200:
            # Save the annotated frame for inspection
            output_file = f"ai_tracked_frame_{CAMERA_ID}.jpg"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Annotated frame saved as: {output_file}")
            
            # Check headers
            tracking_enabled = response.headers.get('X-Vehicle-Tracking', 'disabled')
            tracked_vehicles = response.headers.get('X-Tracked-Vehicles', '0')
            saved_path = response.headers.get('X-Saved-Path', '')
            
            print(f"   Tracking: {tracking_enabled}")
            print(f"   Vehicles: {tracked_vehicles}")
            print(f"   Saved: {saved_path}")
        else:
            print(f"❌ Failed to get annotated frame: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting annotated frame: {e}")
        return False
    
    # Test 5: Update configuration
    print("\n5. Updating tracking configuration...")
    try:
        new_config = {
            "track_thresh": 0.6,
            "track_buffer": 25,
            "match_thresh": 0.7
        }
        response = requests.put(
            f"{BASE_URL}/vehicle-tracking/config/{CAMERA_ID}",
            json=new_config
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Configuration updated: {result['message']}")
        else:
            print(f"❌ Failed to update config: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False
    
    # Test 6: Stop tracking
    print("\n6. Stopping vehicle tracking...")
    try:
        response = requests.post(
            f"{BASE_URL}/vehicle-tracking/stop/",
            data={"camera_id": CAMERA_ID}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Tracking stopped: {result['message']}")
        else:
            print(f"❌ Failed to stop tracking: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error stopping tracking: {e}")
        return False
    
    # Test 7: Cleanup
    print("\n7. Cleaning up resources...")
    try:
        response = requests.delete(f"{BASE_URL}/vehicle-tracking/cleanup/{CAMERA_ID}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Cleanup completed: {result['message']}")
        else:
            print(f"❌ Failed to cleanup: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All AI service tests completed successfully!")
    print("\nGenerated files:")
    print(f"  - ai_tracked_frame_{CAMERA_ID}.jpg (annotated frame from AI service)")
    
    return True

def test_error_handling():
    """Test error handling scenarios"""
    
    print("\n🔍 Testing Error Handling")
    print("=" * 60)
    
    # Test 1: Try to start tracking on non-existent camera
    print("\n1. Testing non-existent camera...")
    try:
        response = requests.post(
            f"{BASE_URL}/vehicle-tracking/start/",
            data={"camera_id": "999"}
        )
        if response.status_code == 404:
            print("✅ Correctly handled non-existent camera")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Try to get status for non-existent camera
    print("\n2. Testing status for non-existent camera...")
    try:
        response = requests.get(f"{BASE_URL}/vehicle-tracking/status/999")
        if response.status_code == 404:
            print("✅ Correctly handled non-existent camera status")
        else:
            print(f"❌ Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Error handling tests completed!")

if __name__ == "__main__":
    print("AI Service Vehicle Tracking Test Suite")
    print("Make sure the AI service is running on localhost:8001")
    print("Make sure you have at least one camera with decoded frames available")
    
    try:
        # Run main tests
        success = test_vehicle_tracking()
        
        if success:
            # Run error handling tests
            test_error_handling()
        
        print("\n🎉 Test suite completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)
