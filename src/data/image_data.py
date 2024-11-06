"""Module for processing and loading image data for CNN input."""

import cv2
import numpy as np
import os
from typing import List, Tuple
from ...config import CHART_SIZE

def load_images(
    image_dir: str,
    target_size: Tuple[int, int] = CHART_SIZE
) -> np.ndarray:
    """
    Load and preprocess images for CNN input.
    
    Args:
        image_dir: Directory containing chart images
        target_size: Target size for image resizing
        
    Returns:
        Array of preprocessed images
    """
    images = []
    image_files = sorted(os.listdir(image_dir))
    
    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
        except Exception as e:
            print(f"Error loading image {image_file}: {str(e)}")
            continue
            
    return np.array(images)