"""
Data augmentation module for micro-expression sequences
"""

import numpy as np
from scipy.interpolate import interp1d
from config import CONFIG


class DataAugmenter:
    """Data augmenter for temporal sequences"""
    
    def __init__(self):
        self.noise_factor = CONFIG["noise_factor"]

    def augment(self, sequence):
        """
        Apply augmentation to a sequence
        
        Args:
            sequence (numpy.ndarray): Input sequence to augment
            
        Returns:
            numpy.ndarray: Augmented sequence
        """
        # Temporal augmentation
        if np.random.rand() > 0.5:
            sequence = self.temporal_warping(sequence)

        # Spatial augmentation (add noise)
        sequence += np.random.normal(0, self.noise_factor, sequence.shape)
        return sequence

    def temporal_warping(self, seq):
        """
        Apply temporal warping to sequence
        
        Args:
            seq (numpy.ndarray): Input sequence
            
        Returns:
            numpy.ndarray: Temporally warped sequence
        """
        x = np.linspace(0, 1, len(seq))
        new_x = np.linspace(0, 1, len(seq)) + np.random.normal(0, 0.1, len(seq))
        new_x = np.clip(new_x, 0, 1)
        return interp1d(x, seq, axis=0)(new_x)