"""
Single video detection module for micro-expression recognition
"""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from feature_extraction import FeatureExtractor
from sequence_processing import SequenceProcessor
from config import CONFIG, LANDMARK_POINTS


def detect_single_video(video_path, model_path=None, show_processing=True):
    """
    Detect micro-expression in a single video
    
    Args:
        video_path (str): Path to the video file
        model_path (str): Path to the model file (optional)
        show_processing (bool): Whether to show processing progress
        
    Returns:
        dict: Detection results with prediction, confidence, and diagnostics
    """
    # Initialize configuration
    model_path = model_path or CONFIG["model_path"]
    extractor = FeatureExtractor()
    processor = SequenceProcessor()

    # Result dictionary
    result = {
        "status": "success",
        "predicted_class": None,
        "confidence": 0.0,
        "class_probabilities": {},
        "error": None,
        "warning": []
    }

    try:
        # Check file existence
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        # Load model
        if not os.path.exists(model_path):
            raise ValueError(f"Model file does not exist: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Process video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        landmarks_seq = []
        missed_frames = 0

        # Process with progress bar
        progress = tqdm(total=total_frames, desc="Processing video frames", disable=not show_processing)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Facial landmark detection
            landmarks = extractor.process_frame(frame)
            if landmarks is not None:
                landmarks_seq.append(landmarks)
            else:
                missed_frames += 1

            progress.update(1)
        cap.release()
        progress.close()

        # Check valid frame count
        if len(landmarks_seq) < CONFIG["min_frames"]:
            raise ValueError("Insufficient valid frames for analysis")

        # Process sequence
        aligned_seq = processor.align_sequence(np.array(landmarks_seq))
        processed_seq = processor.normalize(aligned_seq)

        # Execute prediction
        predictions = model.predict(processed_seq[np.newaxis, ...], verbose=0)[0]
        predicted_idx = np.argmax(predictions)

        # Get classes from dataset module
        from dataset import classes as dataset_classes
        classes = dataset_classes
        if classes is None:
            # If classes not available, try to load them
            try:
                from dataset import load_dataset
                _, _ = load_dataset()  # This will set the global classes variable
                from dataset import classes as dataset_classes
                classes = dataset_classes
            except:
                # If still not available, try to load from saved model info or use a default
                classes = [f"class_{i}" for i in range(len(predictions))]

        # Build result
        result.update({
            "predicted_class": classes[predicted_idx] if classes is not None else f"class_{predicted_idx}",
            "confidence": float(np.max(predictions)),
            "class_probabilities": {
                cls: float(prob) for cls, prob in zip(
                    classes if classes is not None else [f"class_{i}" for i in range(len(predictions))], 
                    predictions
                )
            },
            "warning": [f"Detected {missed_frames} missed frames"] if missed_frames > 0 else []
        })

    except Exception as e:
        result.update({
            "status": "error",
            "error": str(e)
        })

    # Add diagnostic information
    result["diagnostics"] = {
        "video_path": video_path,
        "total_frames": total_frames if 'total_frames' in locals() else 0,
        "valid_frames": len(landmarks_seq) if 'landmarks_seq' in locals() else 0,
        "processing_time": progress.format_dict["elapsed"] if show_processing and 'progress' in locals() else None
    }

    return result