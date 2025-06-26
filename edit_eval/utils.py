import os
import time
import hashlib
from pathlib import Path
import transformers
import torch
import random
import numpy as np 

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_unique_model_path(save_dir: str, 
                             cuda_device: str, 
                             hparams_dir: str, 
                             editing_method: str,
                             start_sample: int,
                             end_sample: int) -> str:
    """
    Generate a unique path for saving the edited model.
    
    Args:
        metrics_save_dir: Base directory for saving metrics
        cuda_device: CUDA device identifier
        hparams_dir: Directory containing hyperparameters
        editing_method: Name of the editing method
        start_sample: Start index of samples
        end_sample: End index of samples
        
    Returns:
        str: Unique path for the model
    """
    # Extract experiment name from hparams
    hparams_name = os.path.basename(hparams_dir).rsplit(".", 1)[0]
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create a unique hash from the parameters
    params_str = f"{hparams_dir}_{editing_method}_{start_sample}_{end_sample}_{timestamp}"
    unique_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    # Create the model directory name
    model_dir_name = f"{hparams_name}_{editing_method}_samples_{start_sample}_{end_sample}_time_{timestamp}_hash_{unique_hash}"
    
    # Construct the full path
    model_save_path = os.path.join(
        save_dir,
        cuda_device,
        model_dir_name
    )
    
    return model_save_path