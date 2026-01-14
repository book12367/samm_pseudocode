"""
Configuration module for Dual-Branch Network with DTW for Micro-Expression Recognition
"""

import os

# System configuration
CONFIG = {
    "data_dir": "data/videos_casme",         # Dataset path
    "use_refined_landmarks": True,           # Whether to use refined landmarks (False=468, True=478)
    "target_length": 20,                     # Unified sequence length
    "batch_size": 2,                         # Training batch size
    "epochs": 20,                            # Maximum training epochs
    "model_path": "me_model_tf.h5",          # Model save path
    "min_frames": 10,                        # Minimum frames for real-time detection
    "is_training": True,                     # Whether in training mode
    "AugTimes": 5,                           # Data augmentation times, effective in training mode
    "learning_rate": 1e-4,                   # Learning rate for optimizer
    "early_stopping_patience": 5,            # Patience for early stopping
    "lr_reduce_patience": 3,                 # Patience for learning rate reduction
    "lr_reduce_factor": 0.5,                 # Factor for learning rate reduction
    "dropout_rate_cnn": 0.3,                 # Dropout rate for CNN branch
    "dropout_rate_final": 0.5,               # Dropout rate for final layer
    "cnn_filters": 64,                       # Number of filters in CNN
    "lstm_units": 32,                        # Number of units in LSTM
    "dense_units": 64,                       # Number of units in dense layer
    "noise_factor": 0.03,                    # Noise factor for data augmentation
}

# Calculate derived values
LANDMARK_POINTS = 478 if CONFIG["use_refined_landmarks"] else 468
CONFIG["input_shape"] = (CONFIG["target_length"], LANDMARK_POINTS * 3)