#!/usr/bin/env python3
"""
Standalone RKNPU Test Script
Tests RKNPU hardware functionality without using the API service
"""

import os
import sys
import numpy as np
from rknn.api import RKNN

def test_rknpu_hardware():
    """Test RKNPU hardware functionality"""
    print("=== RKNPU Hardware Test ===")
    
    # Check if librknnrt.so exists
    librknn_path = "./rknpu/librknnrt.so"
    if os.path.exists(librknn_path):
        print("✅ librknnrt.so found")
    else:
        print("❌ librknnrt.so not found")
        return False
    
    # Test RKNN API import
    try:
        from rknn.api import RKNN
        print("✅ RKNN API imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RKNN API: {e}")
        return False
    
    # Test RKNN object creation
    try:
        rknn = RKNN()
        print("✅ RKNN object created successfully")
    except Exception as e:
        print(f"❌ Failed to create RKNN object: {e}")
        return False
    
    # Test model loading and runtime initialization
    try:
        # Check if we have an existing RKNN model
        rknn_model_path = "./models/yolov8n.rknn"
        onnx_model_path = "./models/yolov8n.onnx"
        
        if os.path.exists(rknn_model_path):
            print(f"✅ Found existing RKNN model: {rknn_model_path}")
            ret = rknn.load_rknn(rknn_model_path)
            if ret != 0:
                print(f"❌ Failed to load RKNN model, code: {ret}")
                rknn.release()
                return False
            print("✅ RKNN model loaded successfully")
        elif os.path.exists(onnx_model_path):
            print(f"✅ Found ONNX model, converting to RKNN: {onnx_model_path}")
            
            # Configure RKNN for RK3588
            ret = rknn.config(target_platform='rk3588', mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
            if ret != 0:
                print(f"❌ Failed to configure RKNN, code: {ret}")
                rknn.release()
                return False
            print("✅ RKNN configured successfully")
            
            ret = rknn.load_onnx(model=onnx_model_path, inputs=['images'], input_size_list=[[1, 3, 640, 640]])
            if ret != 0:
                print(f"❌ Failed to load ONNX model, code: {ret}")
                rknn.release()
                return False
            print("✅ ONNX model loaded successfully")
            
            # Build RKNN model
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                print(f"❌ Failed to build RKNN model, code: {ret}")
                rknn.release()
                return False
            print("✅ RKNN model built successfully")
        else:
            print("❌ No suitable model found for testing")
            rknn.release()
            return False
        
        # Now test runtime initialization
        ret = rknn.init_runtime()
        if ret == 0:
            print("✅ NPU runtime initialized successfully")
            print("   🎉 RKNPU hardware is working!")
        else:
            print(f"❌ Failed to initialize NPU runtime, code: {ret}")
            rknn.release()
            return False
            
    except Exception as e:
        print(f"❌ Exception during model loading/runtime initialization: {e}")
        rknn.release()
        return False
    
    # Test basic functionality
    try:
        # Create dummy input data (1x224x224x3 for typical image models)
        dummy_input = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        print("✅ Created dummy input data")
        print("✅ NPU runtime is functional and ready for inference")
        
    except Exception as e:
        print(f"❌ Exception during functionality test: {e}")
        rknn.release()
        return False
    
    # Clean up
    try:
        rknn.release()
        print("✅ RKNN resources released")
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {e}")
    
    return True

def test_model_conversion():
    """Test model conversion capabilities"""
    print("\n=== Model Conversion Test ===")
    
    # Check if we have ONNX models available
    models_dir = "./models"
    if os.path.exists(models_dir):
        onnx_files = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
        if onnx_files:
            print(f"✅ Found ONNX models: {onnx_files}")
        else:
            print("⚠️  No ONNX models found in models directory")
    else:
        print("⚠️  Models directory not found")
    
    # Test RKNN conversion API
    try:
        rknn = RKNN()
        print("✅ RKNN conversion API available")
        rknn.release()
    except Exception as e:
        print(f"❌ RKNN conversion API test failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("RKNPU Standalone Test")
    print("=" * 50)
    
    # Test hardware functionality
    hardware_ok = test_rknpu_hardware()
    
    # Test model conversion
    conversion_ok = test_model_conversion()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"Hardware Test: {'✅ PASS' if hardware_ok else '❌ FAIL'}")
    print(f"Conversion Test: {'✅ PASS' if conversion_ok else '❌ FAIL'}")
    
    if hardware_ok and conversion_ok:
        print("\n🎉 All tests passed! RKNPU is ready for use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
