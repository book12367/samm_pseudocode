"""
Main module for Dual-Branch Network with DTW for Micro-Expression Recognition

This module provides the main entry points for training, real-time detection,
and single video detection.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
from training import train_model
from realtime_detection import realtime_demo, RealTimeDetector
from single_video_detection import detect_single_video
from utils import load_labels
from config import CONFIG


def main():
    """Main function demonstrating the complete workflow"""

    # Load class labels
    classes = load_labels()
    print(f"Loaded {len(classes)} classes: {classes}")

    # Training the model (uncomment to run)
    # trained_model = train_model()

    # Example: Single video detection
    start_time = time.time()
    result = detect_single_video(
        video_path="test.mp4",
        show_processing=True
    )
    print(f"Detection time: {time.time() - start_time:.2f} seconds")

    # Print results
    print("\nDetection Results:")
    if result["status"] == "success":
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nDetailed probability distribution:")
        for cls, prob in result["class_probabilities"].items():
            print(f"  {cls}: {prob:.2%}")
    else:
        print(f"Detection failed: {result['error']}")

    # Display warning messages
    if result["warning"]:
        print("\nWarning messages:")
        for warn in result["warning"]:
            print(f"  - {warn}")

    # Uncomment to run real-time demo
    # realtime_demo()


if __name__ == "__main__":
    main()