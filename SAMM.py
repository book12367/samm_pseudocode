import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, f1_score
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# 系统配置
CONFIG = {
    "data_dir": r"E:\LW\SAMM_videos",         # 数据集路径
    "use_refined_landmarks": True, # 是否使用细节优化关键点（False=468，True=478）
    "target_length": 20,           # 统一序列长度
    "batch_size": 2,               # 训练批大小
    "epochs": 100,                  # 最大训练轮次
    "model_path": "me_model_tf.h5",   # 模型保存路径
    "min_frames": 10,               # 实时检测最小帧数
    "is_training": True,        # 是否为训练模式
    "AugTimes": 5            # 数据增强次数 训练模式生效
}

# 自动计算输入维度
LANDMARK_POINTS = 478 if CONFIG["use_refined_landmarks"] else 468
CONFIG["input_shape"] = (CONFIG["target_length"], LANDMARK_POINTS * 3)

class FeatureExtractor:
    """面部关键点提取器"""
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=CONFIG["use_refined_landmarks"],
            min_detection_confidence=0.5)

    def process_video(self, video_path):
        """处理视频文件"""
        cap = cv2.VideoCapture(video_path)
        landmarks_seq = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] 
                                    for lm in results.multi_face_landmarks[0].landmark])
                landmarks_seq.append(landmarks.flatten())
        
        cap.release()
        return np.array(landmarks_seq)

class SequenceProcessor:
    """序列处理器"""
    def __init__(self):
        self.target_length = CONFIG["target_length"]
        # 定义用于对齐的关键点索引, 需要调整
        # self.selected_indices = [0, 4, 17, 37, 61, 291, 267, 36, 39, 40, 185, 159]
        self.selected_indices = [x for x in range(LANDMARK_POINTS)]
    
    def align_sequence(self, sequence):
        """使用DTW进行动态序列对齐"""
        if len(sequence) < 2:
            return np.zeros((self.target_length, sequence.shape[1]))
        
        # 提取关键点特征用于DTW对齐
        selected_features = sequence[:, self.selected_indices].reshape(len(sequence), -1)
        
        # 生成参考序列（使用线性插值）
        original_length = len(sequence)
        x_orig = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, self.target_length)
        R_selected = np.zeros((self.target_length, selected_features.shape[1]))
        
        for dim in range(selected_features.shape[1]):
            f = interp1d(x_orig, selected_features[:, dim], kind='linear', fill_value="extrapolate")
            R_selected[:, dim] = f(x_new)
        
        # 计算DTW路径
        _, path = fastdtw(selected_features, R_selected, dist=euclidean)
        
        # 根据路径构建对齐后的序列
        aligned = np.zeros((self.target_length, sequence.shape[1]))
        counts = np.zeros(self.target_length)
        
        for s_idx, r_idx in path:
            aligned[r_idx] += sequence[s_idx]
            counts[r_idx] += 1
        
        # 处理未对齐的位置并归一化
        counts[counts == 0] = 1  # 避免除以零
        aligned /= counts[:, np.newaxis]
        
        return aligned
    
    def normalize(self, sequence):
        """数据标准化"""
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        return (sequence - mean) / (std + 1e-8)
    
    def process(self, sequence):
        """完整处理流程"""
        processed = self.align_sequence(sequence)
        return self.normalize(processed)

class MetricsCallback(tf.keras.callbacks.Callback):
    """在每个epoch结束时计算验证集的召回率和F1分数"""
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        # 预测验证集
        y_pred = self.model.predict(self.X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 计算指标
        val_recall = recall_score(self.y_val, y_pred_classes, average='macro')
        val_f1 = f1_score(self.y_val, y_pred_classes, average='macro')
        
        # 记录到训练日志
        logs['val_recall'] = val_recall
        logs['val_f1'] = val_f1
        
        # 输出指标结果
        print(f" - val_recall: {val_recall:.4f} - val_f1: {val_f1:.4f}")

def build_model():
    inputs = tf.keras.Input(shape=CONFIG["input_shape"])
    
    # 时空特征并行提取
    # 分支1：CNN处理局部特征
    cnn = tf.keras.layers.Conv1D(64, 5, padding='same')(inputs)
    cnn = tf.keras.layers.BatchNormalization()(cnn)
    cnn = tf.keras.layers.ReLU()(cnn)
    cnn = tf.keras.layers.Dropout(0.3)(cnn)
    
    # 分支2：BiLSTM处理时序特征
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True))(inputs)
    lstm = tf.keras.layers.Conv1D(64, 3, padding='same')(lstm)  # 统一通道数
    
    # # 特征融合（沿特征轴拼接）
    merged = tf.keras.layers.concatenate([cnn, lstm], axis=-1)
    
    # 时空特征联合处理
    x = tf.keras.layers.Conv1D(128, 3, padding='same')(merged)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # 分类层
    x = tf.keras.layers.Dense(64, activation='relu', 
                            kernel_regularizer='l2')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(classes), activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 优化器配置
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    return model

class DataAugmenter:
    """数据增强器"""
    def __init__(self):
        self.noise_factor = 0.03
        
    def augment(self, sequence):
        # 时序增强
        if np.random.rand() > 0.5:
            sequence = self.temporal_warping(sequence)
        
        # 空间增强
        sequence += np.random.normal(0, self.noise_factor, sequence.shape)
        return sequence
        
    def temporal_warping(self, seq):
        """时间扭曲增强"""
        x = np.linspace(0, 1, len(seq))
        new_x = np.linspace(0, 1, len(seq)) + np.random.normal(0, 0.1, len(seq))
        new_x = np.clip(new_x, 0, 1)
        return interp1d(x, seq, axis=0)(new_x)

def load_dataset():
    """加载并预处理数据集"""
    global classes  # 声明为全局变量以便其他函数使用
    
    categories = sorted([d for d in os.listdir(CONFIG["data_dir"]) 
                        if os.path.isdir(os.path.join(CONFIG["data_dir"], d))])
    print(categories)
    label_encoder = LabelEncoder().fit(categories)
    classes = label_encoder.classes_

    print(classes)
    
    extractor = FeatureExtractor()
    processor = SequenceProcessor()
    
    X, y = [], []
    for label_name in categories:
        print(label_name)
        label_idx = label_encoder.transform([label_name])[0]
        video_dir = os.path.join(CONFIG["data_dir"], label_name)
        
        video_files = [f for f in os.listdir(video_dir) 
                      if f.lower().endswith('.mp4')]
        
        for video_file in tqdm(video_files, desc=f'Processing {label_name}'):
            video_path = os.path.join(video_dir, video_file)
            raw_seq = extractor.process_video(video_path)
            
            # 数据过滤和处理
            if len(raw_seq) < 5:  # 过滤过短序列
                continue
                
            try:
                processed = processor.process(raw_seq)
                if CONFIG["is_training"]:  # 仅在训练时增强
                    for _ in range(CONFIG["AugTimes"]):
                        processed = DataAugmenter().augment(processed)
                        X.append(processed)
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

def train():
    """训练流程"""
    # 加载数据
    X, y = load_dataset()
    print(f"Dataset loaded: {X.shape} sequences, {len(classes)} classes")
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 构建模型
    model = build_model()
    model.summary()  # 打印模型结构
    
    # 训练配置
    callbacks = [
        MetricsCallback(X_val, y_val),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    # 开始训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks
    )

    print("Training finished")
    
    # 保存模型
    model.save(CONFIG["model_path"], save_format='tf')
    print(f"Model saved to {CONFIG['model_path']}")
    return model

def load_label():
    """加载标签"""
    global classes  # 声明为全局变量以便其他函数使用
    
    categories = sorted([d for d in os.listdir(CONFIG["data_dir"]) 
                        if os.path.isdir(os.path.join(CONFIG["data_dir"], d))])
    label_encoder = LabelEncoder().fit(categories)
    classes = label_encoder.classes_

class RealTimeDetector:
    """实时检测器"""
    def __init__(self):
        self.model = tf.keras.models.load_model(CONFIG["model_path"])
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=CONFIG["use_refined_landmarks"],
            min_detection_confidence=0.5)
        self.processor = SequenceProcessor()
        self.buffer = []
        
    def detect(self, frame):
        """处理单帧图像"""
        # 转换颜色空间
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测面部关键点
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return frame, None
            
        # 提取特征
        landmarks = np.array([[lm.x, lm.y, lm.z] 
                            for lm in results.multi_face_landmarks[0].landmark])
        self.buffer.append(landmarks.flatten())
        
        # 保持缓冲区长度
        if len(self.buffer) > CONFIG["target_length"]:
            self.buffer = self.buffer[-CONFIG["target_length"]:]
            
        # 达到最小帧数开始预测
        if len(self.buffer) >= CONFIG["min_frames"]:
            try:
                processed = self.processor.process(np.array(self.buffer))
                pred = self.model.predict(processed[np.newaxis, ...], verbose=0)[0]
                label_idx = np.argmax(pred)
                return frame, classes[label_idx]
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                load_label()
                return frame, None
            
        return frame, None

def realtime_demo():
    """实时演示"""
    detector = RealTimeDetector()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Micro-expression Detection", cv2.WINDOW_NORMAL)
    
    start_time = time.time()
    counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        counter += 1  # 计算帧数
        # 处理帧并显示结果
        processed_frame, pred = detector.detect(frame)
        if pred:
            cv2.putText(processed_frame, f"Prediction: {pred}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if (time.time() - start_time) != 0:  # 实时显示帧数
            cv2.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (5, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                    3)
            # print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()
        cv2.imshow("Micro-expression Detection", processed_frame)
        
        # 退出键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def detect_single_video(video_path, model_path=None, show_processing=True):
    # 初始化配置
    model_path = model_path or CONFIG["model_path"]
    extractor = FeatureExtractor()
    processor = SequenceProcessor()
    
    # 结果字典
    result = {
        "status": "success",
        "predicted_class": None,
        "confidence": 0.0,
        "class_probabilities": {},
        "error": None,
        "warning": []
    }

    try:
        # 检查文件存在性
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 加载模型
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # 处理视频
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        landmarks_seq = []
        missed_frames = 0

        # 带进度条的处理
        progress = tqdm(total=total_frames, desc="处理视频帧", disable=not show_processing)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 面部关键点检测
            results = extractor.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] 
                                    for lm in results.multi_face_landmarks[0].landmark])
                landmarks_seq.append(landmarks.flatten())
            else:
                missed_frames += 1

            progress.update(1)
        cap.release()
        progress.close()

        # 检查有效帧数
        if len(landmarks_seq) < CONFIG["min_frames"]:
            raise ValueError("有效帧数不足，无法进行分析")

        # 处理序列
        aligned_seq = processor.align_sequence(np.array(landmarks_seq))
        processed_seq = processor.normalize(aligned_seq)

        # 执行预测
        predictions = model.predict(processed_seq[np.newaxis, ...], verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        
        # 构建结果
        result.update({
            "predicted_class": classes[predicted_idx],
            "confidence": float(np.max(predictions)),
            "class_probabilities": {cls: float(prob) for cls, prob in zip(classes, predictions)},
            "warning": [f"检测丢失{missed_frames}帧"] if missed_frames > 0 else []
        })

    except Exception as e:
        result.update({
            "status": "error",
            "error": str(e)
        })
    
    # 添加诊断信息
    result["diagnostics"] = {
        "video_path": video_path,
        "total_frames": total_frames,
        "valid_frames": len(landmarks_seq),
        "processing_time": progress.format_dict["elapsed"] if show_processing else None
    }
    
    return result

if __name__ == "__main__":
    # 训练模型（首次运行时取消注释）
    # train()
    
    # 运行实时检测
    # realtime_demo()

    # 运行预测
    # 加载类别标签
    load_label()
    
    # 执行检测
    start_time = time.time()
    test_result = detect_single_video(
        video_path=r"E:\LW\SAMM_videos\Contempt\014_5_2.mp4",
        # video_path='surprise_22_13-33.mp4',
        show_processing=True
    )
    print("检测时间：", time.time() - start_time)
    
    # 打印结果
    print("\n检测结果：")
    if test_result["status"] == "success":
        print(f"预测类别: {test_result['predicted_class']}")
        print(f"置信度: {test_result['confidence']:.2%}")
        print("\n详细概率分布：")
        for cls, prob in test_result["class_probabilities"].items():
            print(f"  {cls}: {prob:.2%}")
    else:
        print(f"检测失败: {test_result['error']}")
    
    # 显示警告信息
    if test_result["warning"]:
        print("\n警告信息：")
        for warn in test_result["warning"]:
            print(f"  - {warn}")