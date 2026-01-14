"""
Utility functions for the micro-expression recognition system
"""

import os
from sklearn.preprocessing import LabelEncoder
from config import CONFIG


def load_labels():
    """
    Load class labels from the dataset directory
    
    Returns:
        numpy.ndarray: Array of class labels
    """
    global classes  # Declare as global variable
    
    categories = sorted([
        d for d in os.listdir(CONFIG["data_dir"])
        if os.path.isdir(os.path.join(CONFIG["data_dir"], d))
    ])
    label_encoder = LabelEncoder().fit(categories)
    classes = label_encoder.classes_
    
    return classes