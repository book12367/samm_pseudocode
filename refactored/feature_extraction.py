"""
Feature extraction module for facial landmark detection
"""

import cv2
import numpy as np
import mediapipe as mp
from config import CONFIG


class FeatureExtractor:
    """Facial landmark extractor using MediaPipe"""
    
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=CONFIG["use_refined_landmarks"],
            min_detection_confidence=0.5
        )

    def process_video(self, video_path):
        """
        Process video file and extract facial landmarks
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            numpy.ndarray: Array of flattened facial landmarks for each frame
        """
        cap = cv2.VideoCapture(video_path)
        landmarks_seq = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in results.multi_face_landmarks[0].landmark
                ])
                landmarks_seq.append(landmarks.flatten())

        cap.release()
        return np.array(landmarks_seq)

    def process_frame(self, frame):
        """
        Process a single frame and extract facial landmarks
        
        Args:
            frame (numpy.ndarray): Input frame image
            
        Returns:
            numpy.ndarray or None: Flattened facial landmarks if detected, None otherwise
        """
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for lm in results.multi_face_landmarks[0].landmark
            ])
            return landmarks.flatten()
        return None