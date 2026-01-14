"""
Dataset loading and preprocessing module
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from feature_extraction import FeatureExtractor
from sequence_processing import SequenceProcessor
from data_augmentation import DataAugmenter
from config import CONFIG


# Global variable to store class labels
classes = None


def load_dataset():
    """
    Load and preprocess the dataset

    Returns:
        tuple: (X, y) where X is features and y is labels
    """
    global classes  # Declare as global variable for other functions to use

    categories = sorted([
        d for d in os.listdir(CONFIG["data_dir"])
        if os.path.isdir(os.path.join(CONFIG["data_dir"], d))
    ])
    label_encoder = LabelEncoder().fit(categories)
    classes = label_encoder.classes_

    extractor = FeatureExtractor()
    processor = SequenceProcessor()

    X, y = [], []
    for label_name in categories:
        label_idx = label_encoder.transform([label_name])[0]
        video_dir = os.path.join(CONFIG["data_dir"], label_name)

        video_files = [
            f for f in os.listdir(video_dir)
            if f.lower().endswith('.mp4')
        ]

        for video_file in tqdm(video_files, desc=f'Processing {label_name}'):
            video_path = os.path.join(video_dir, video_file)
            raw_seq = extractor.process_video(video_path)

            # Data filtering and processing
            if len(raw_seq) < 5:  # Filter out too short sequences
                continue

            try:
                processed = processor.process(raw_seq)
                if CONFIG["is_training"]:  # Only augment during training
                    for _ in range(CONFIG["AugTimes"]):
                        augmented = DataAugmenter().augment(processed)
                        X.append(augmented)
                        y.append(label_idx)
                else:
                    X.append(processed)
                    y.append(label_idx)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
                continue

    X = np.array(X).astype(np.float32)
    y = np.array(y)
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and validation sets
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Labels
        test_size (float): Proportion of validation set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )