import gc
import tensorflow as tf
import torch

def clean_up() -> None:
    """Safely clear TensorFlow and PyTorch GPU/CPU memory."""
    gc.collect()  # free Python references first

    # ---- TensorFlow cleanup ----
    try:
        tf.keras.backend.clear_session()
    except Exception as e:
        print(f"[TF cleanup skipped] {e}")

    # ---- PyTorch cleanup ----
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print("🧹 GPU memory cache cleared.")
        except Exception as e:
            print(f"[Torch cleanup skipped] {e}")
    else:
        print("⚠️ No CUDA device available.")
