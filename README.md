# SAMM Micro-expression Recognition Algorithm Pseudocode

## Project Overview

This project contains complete algorithm pseudocode for the SAMM (Spontaneous Affective Facial Expressions in the Wild) micro-expression recognition system. The system uses facial landmark detection and deep learning techniques to identify and classify micro-expressions.

## Directory Structure

```
pseudocode/
├── chinese/                 # Chinese pseudocode
│   ├── training/            # Training algorithms
│   │   └── training_algorithm.txt
│   ├── inference/           # Inference algorithms
│   │   └── inference_algorithm.txt
│   ├── real_time_inference/ # Real-time inference algorithms
│   │   └── real_time_algorithm.txt
│   └── preprocessing/       # Preprocessing algorithms
│       └── preprocessing_algorithm.txt
└── english/                 # English pseudocode
    ├── training/            # Training algorithms
    │   └── training_algorithm.txt
    ├── inference/           # Inference algorithms
    │   └── inference_algorithm.txt
    ├── real_time_inference/ # Real-time inference algorithms
    │   └── real_time_algorithm.txt
    └── preprocessing/       # Preprocessing algorithms
        └── preprocessing_algorithm.txt
```

## Algorithm Modules

### 1. Training Algorithm

The training algorithm implements a deep learning model for micro-expression recognition with key features:

- **Feature Extraction**: Uses MediaPipe Face Mesh to detect facial landmarks
- **Sequence Alignment**: Uses Dynamic Time Warping (DTW) to align video sequences of different lengths
- **Model Architecture**: Hybrid architecture combining CNN and BiLSTM for spatiotemporal feature extraction
- **Data Augmentation**: Includes temporal warping and noise addition techniques
- **Evaluation Metrics**: Computes recall and F1 scores during training

### 2. Inference Algorithm

The inference algorithm is used for micro-expression classification on single video files:

- **Video Processing**: Extracts facial landmarks frame by frame
- **Sequence Processing**: Aligns and normalizes landmark sequences
- **Model Prediction**: Uses the trained model for classification
- **Result Output**: Returns predicted class, confidence, and probability distribution

### 3. Real-time Inference Algorithm

The real-time inference algorithm enables live micro-expression detection:

- **Buffer Management**: Maintains a fixed-length frame buffer
- **Real-time Processing**: Continuously processes camera input
- **Performance Optimization**: Displays real-time FPS to monitor performance
- **User Interface**: Provides visualization interface showing detection results

### 4. Preprocessing Algorithm

The preprocessing algorithm handles data preparation and validation:

- **Data Loading**: Reads label information from Excel files
- **File Validation**: Checks video file existence and integrity
- **Property Checking**: Validates video frame count, resolution, and other properties
- **Data Cleaning**: Cleans and standardizes data
- **Format Conversion**: Converts image sequences to video format

## Pseudocode Features

- **Non-executable**: Uses custom syntax to prevent direct execution
- **Algorithm Clarity**: Preserves complete algorithm logic and flow
- **Structured Design**: Employs object-oriented and modular design
- **Detailed Comments**: Includes comprehensive comments explaining algorithm details
- **Bilingual Support**: Provides both Chinese and English versions

## Usage Instructions

Each algorithm file contains complete implementation details and can be used directly for:

- Algorithm understanding and learning
- System design reference
- Academic research support
- Code implementation guidance

## Technology Stack

- **Face Detection**: MediaPipe Face Mesh
- **Deep Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas
- **Computer Vision**: OpenCV
- **Sequence Alignment**: FastDTW

## Algorithm Workflow

1. **Data Preprocessing**: Load and validate SAMM dataset
2. **Feature Extraction**: Extract facial landmark sequences
3. **Sequence Alignment**: Align variable-length sequences using DTW
4. **Model Training**: Train CNN-BiLSTM hybrid model
5. **Model Inference**: Classify micro-expressions on new data
6. **Result Evaluation**: Compute classification accuracy and related metrics

## References

- SAMM Dataset: Spontaneous Affective Facial Expressions in the Wild
- MediaPipe: Google's machine learning framework
- DTW: Dynamic Time Warping algorithm
- Deep Learning: CNN and LSTM network architectures

---

[中文版 README](README_CN.md) | [English README](README.md)
