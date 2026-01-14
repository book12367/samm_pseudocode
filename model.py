"""
Neural network model definition for micro-expression recognition
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, f1_score
from config import CONFIG


def build_model(num_classes):
    """
    Build dual-branch neural network model (CNN + BiLSTM)

    Args:
        num_classes (int): Number of output classes

    Returns:
        tensorflow.keras.Model: Compiled model
    """
    inputs = tf.keras.Input(shape=CONFIG["input_shape"])

    # Branch 1: CNN for local feature processing
    cnn = tf.keras.layers.Conv1D(
        CONFIG["cnn_filters"],
        5,
        padding='same'
    )(inputs)
    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.ReLU()(cnn)
    cnn = tf.keras.layers.Dropout(CONFIG["dropout_rate_cnn"])(cnn)

    # Branch 2: BiLSTM for temporal feature processing
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(CONFIG["lstm_units"], return_sequences=True)
    )(inputs)
    lstm = tf.keras.layers.Conv1D(
        CONFIG["cnn_filters"],
        3,
        padding='same'
    )(lstm)  # Unify channel count

    # Feature fusion (concatenate along feature axis)
    merged = tf.keras.layers.concatenate([cnn, lstm], axis=-1)

    # Joint spatiotemporal feature processing
    x = tf.keras.layers.Conv1D(128, 3, padding='same')(merged)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Classification layers
    x = tf.keras.layers.Dense(
        CONFIG["dense_units"],
        activation='relu',
        kernel_regularizer='l2'
    )(x)
    x = tf.keras.layers.Dropout(CONFIG["dropout_rate_final"])(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=CONFIG["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


class MetricsCallback(tf.keras.callbacks.Callback):
    """Callback to compute validation recall and F1 score at the end of each epoch"""

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        # Predict on validation set
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate metrics
        val_recall = recall_score(self.y_val, y_pred_classes, average='macro')
        val_f1 = f1_score(self.y_val, y_pred_classes, average='macro')

        # Log to training logs
        logs = logs or {}
        logs['val_recall'] = val_recall
        logs['val_f1'] = val_f1

        # Print metrics
        print(f" - val_recall: {val_recall:.4f} - val_f1: {val_f1:.4f}")