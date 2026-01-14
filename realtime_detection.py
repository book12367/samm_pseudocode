"""
Real-time detection module for micro-expression recognition
"""

import time
import cv2
import numpy as np
import tensorflow as tf
from feature_extraction import FeatureExtractor
from sequence_processing import SequenceProcessor
from config import CONFIG


class RealTimeDetector:
    """Real-time micro-expression detector"""
    
    def __init__(self, model_path=None):
        model_path = model_path or CONFIG["model_path"]
        self.model = tf.keras.models.load_model(model_path)
        self.face_mesh_extractor = FeatureExtractor()
        self.processor = SequenceProcessor()
        self.buffer = []

    def detect(self, frame):
        """
        Process a single frame and detect micro-expression
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (processed_frame, prediction) where prediction is the class name or None
        """
        # Extract facial landmarks
        landmarks = self.face_mesh_extractor.process_frame(frame)
        if landmarks is None:
            return frame, None

        # Add to buffer
        self.buffer.append(landmarks)

        # Maintain buffer length
        if len(self.buffer) > CONFIG["target_length"]:
            self.buffer = self.buffer[-CONFIG["target_length"]:]

        # Start prediction when minimum frames reached
        if len(self.buffer) >= CONFIG["min_frames"]:
            try:
                processed_seq = self.processor.process(np.array(self.buffer))
                pred = self.model.predict(processed_seq[np.newaxis, ...], verbose=0)[0]
                label_idx = np.argmax(pred)
                
                # Get classes from dataset module
                from dataset import classes as dataset_classes
                classes = dataset_classes
                if classes is None:
                    # If classes not available, load them
                    from dataset import load_dataset
                    _, _ = load_dataset()  # This will set the global classes variable
                    from dataset import classes as dataset_classes
                    classes = dataset_classes
                
                return frame, classes[label_idx] if classes is not None else None
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                return frame, None

        return frame, None


def realtime_demo():
    """Run real-time demonstration"""
    detector = RealTimeDetector()

    # Open camera
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Micro-expression Detection", cv2.WINDOW_NORMAL)

    start_time = time.time()
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        counter += 1  # Count frames
        # Process frame and display result
        processed_frame, pred = detector.detect(frame)
        if pred:
            cv2.putText(
                processed_frame, 
                f"Prediction: {pred}", 
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

        if (time.time() - start_time) != 0:  # Real-time display of FPS
            cv2.putText(
                frame, 
                "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), 
                (5, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (0, 0, 255), 3
            )
            counter = 0
            start_time = time.time()
        
        cv2.imshow("Micro-expression Detection", processed_frame)

        # Exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()