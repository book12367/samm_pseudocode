"""
Sequence processing module with DTW alignment
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from config import CONFIG, LANDMARK_POINTS


class SequenceProcessor:
    """Sequence processor with DTW alignment functionality"""
    
    def __init__(self):
        self.target_length = CONFIG["target_length"]
        # Use all landmarks for alignment
        self.selected_indices = list(range(LANDMARK_POINTS))

    def align_sequence(self, sequence):
        """
        Align sequence using Dynamic Time Warping (DTW)
        
        Args:
            sequence (numpy.ndarray): Input sequence of shape (frames, features)
            
        Returns:
            numpy.ndarray: Aligned sequence of shape (target_length, features)
        """
        if len(sequence) < 2:
            return np.zeros((self.target_length, sequence.shape[1]))

        # Extract features for DTW alignment
        selected_features = sequence[:, self.selected_indices].reshape(len(sequence), -1)

        # Generate reference sequence using linear interpolation
        original_length = len(sequence)
        x_orig = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, self.target_length)
        R_selected = np.zeros((self.target_length, selected_features.shape[1]))

        for dim in range(selected_features.shape[1]):
            f = interp1d(x_orig, selected_features[:, dim], kind='linear', fill_value="extrapolate")
            R_selected[:, dim] = f(x_new)

        # Calculate DTW path
        _, path = fastdtw(selected_features, R_selected, dist=euclidean)

        # Build aligned sequence based on path
        aligned = np.zeros((self.target_length, sequence.shape[1]))
        counts = np.zeros(self.target_length)

        for s_idx, r_idx in path:
            aligned[r_idx] += sequence[s_idx]
            counts[r_idx] += 1

        # Handle unaligned positions and normalize
        counts[counts == 0] = 1  # Avoid division by zero
        aligned /= counts[:, np.newaxis]

        return aligned

    def normalize(self, sequence):
        """
        Normalize sequence using mean and standard deviation
        
        Args:
            sequence (numpy.ndarray): Input sequence
            
        Returns:
            numpy.ndarray: Normalized sequence
        """
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        return (sequence - mean) / (std + 1e-8)

    def process(self, sequence):
        """
        Complete processing pipeline: align and normalize
        
        Args:
            sequence (numpy.ndarray): Input sequence
            
        Returns:
            numpy.ndarray: Processed sequence
        """
        aligned = self.align_sequence(sequence)
        return self.normalize(aligned)