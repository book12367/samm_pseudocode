"""
Training module for the micro-expression recognition model
"""

import tensorflow as tf
import numpy as np
from model import build_model, MetricsCallback
from dataset import load_dataset, split_dataset
from config import CONFIG


def train_model():
    """
    Train the micro-expression recognition model
    
    Returns:
        tensorflow.keras.Model: Trained model
    """
    # Load data
    X, y = load_dataset()
    print(f"Dataset loaded: {X.shape} sequences, {len(globals().get('classes', []))} classes")

    # Split dataset
    X_train, X_val, y_train, y_val = split_dataset(X, y)

    # Build model
    num_classes = len(np.unique(y))
    model = build_model(num_classes)
    model.summary()  # Print model architecture

    # Training configuration
    callbacks = [
        MetricsCallback(X_val, y_val),
        tf.keras.callbacks.EarlyStopping(
            patience=CONFIG["early_stopping_patience"], 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=CONFIG["lr_reduce_factor"], 
            patience=CONFIG["lr_reduce_patience"]
        )
    ]

    # Start training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks
    )

    print("Training finished")

    # Save model
    model.save(CONFIG["model_path"], save_format='tf')
    print(f"Model saved to {CONFIG['model_path']}")
    return model