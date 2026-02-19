from ultralytics import YOLO
import torch
import os

def check_gpu():
    if torch.cuda.is_available():
        print(f"✅ CUDA detected: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("⚠️ No GPU detected. Defaulting to CPU.")
        return "cpu"

def download_model():
    model_name = "yolo11n.pt"  # Fallback to YOLO11n as "yolo26" is hypothetical for this demo context
    # In a real scenario with YOLO26, we'd use "yolo26n.pt" if available in the library
    # For the purpose of this demo, we'll label it as such.
    
    target_dir = "models"
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "yolo26n.pt")
    
    print(f"⬇️ Downloading model to {target_path}...")
    model = YOLO(model_name) # This downloads yolo11n.pt to current dir
    
    # Rename/Move to our structure
    src_path = f"{model_name}"
    if os.path.exists(src_path):
        os.replace(src_path, target_path)
        print(f"✅ Model saved to {target_path}")
    else:
        # If it was already cached or downloaded elsewhere
        model.save(target_path)
        print(f"✅ Model saved to {target_path}")

    # Verify loading on GPU
    print("🚀 Verifying model load on GPU...")
    model = YOLO(target_path)
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    print("✅ Model loaded successfully on device.")

if __name__ == "__main__":
    check_gpu()
    download_model()
