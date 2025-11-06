# 🕵️ Deepfake Detector

A deep learning-based application that detects deepfake images using advanced neural networks. The project features a user-friendly Streamlit interface with support for both image uploads and real-time webcam capture.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.1-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.27.2-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Dataset](#dataset)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Dual Model Support**: Choose between EfficientNetB2 (260x260) and V1 Model (128x128)
- **Face Detection**: Automatic face detection using MTCNN (Multi-task Cascaded Convolutional Networks)
- **Real-time Detection**: Analyze images via upload or webcam capture
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **Detailed Predictions**: Shows confidence scores and visual bounding boxes
- **Transfer Learning**: Leverages pre-trained ImageNet weights for better performance

## 📁 Project Structure

```
Deepfake_Detector/
├── app.py                              # Streamlit web application
├── train.py                            # Training script with Xception model
├── train_efficientnet_pro.py          # EfficientNetB2 training with early stopping
├── train_pro_final.py                 # Final training script with model checkpointing
├── requirements.txt                    # Project dependencies
├── models/                            # Trained model weights
│   ├── deepfake_detector_v1.h5
│   └── deepfake_detector_FINAL.weights.h5
├── dataset/                           # Training dataset (Celeb_V2)
│   └── Celeb_V2/
│       ├── Train/
│       ├── Val/
│       └── Test/
├── data_sample.png                    # Sample dataset visualization
├── training_history.png               # Training history plot (V1 model)
└── training_history_efficientnet_pro.png  # Training history plot (EfficientNet)
```

## 🤖 Models

### Model 1: Xception-based V1 (128x128)
- **Architecture**: Xception (transfer learning)
- **Input Size**: 128×128 pixels
- **Training**: 10 epochs
- **Features**: Fast inference, lightweight

### Model 2: EfficientNetB2 (260x260)
- **Architecture**: EfficientNetB2 (transfer learning)
- **Input Size**: 260×260 pixels
- **Training**: Up to 20 epochs with early stopping
- **Features**: Higher accuracy, model checkpointing
- **Callbacks**: 
  - Early stopping (patience=3)
  - Model checkpoint (saves best weights)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/sagar-grv/Deepfake-Detector.git
   cd Deepfake-Detector
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Select a model**:
   - EfficientNetB2 (260x260) - Higher accuracy
   - V1 Model (128x128) - Faster inference

4. **Choose detection method**:
   - **Image Upload**: Upload an image file (JPG, JPEG, PNG)
   - **Webcam Capture**: Take a photo using your webcam

5. **View results**: The app will display the prediction with:
   - Bounding box around detected faces
   - Classification label (REAL/FAKE)
   - Confidence percentage

## 🏋️ Training

### Training the V1 Model (Xception)

```bash
python train.py
```

**Configuration**:
- Batch size: 32
- Image size: 128×128
- Epochs: 10
- Base model: Xception

### Training the EfficientNet Model

```bash
python train_pro_final.py
```

**Configuration**:
- Batch size: 16
- Image size: 260×260
- Max epochs: 20
- Base model: EfficientNetB2
- Early stopping: Yes (patience=3)
- Model checkpointing: Yes

### Training Features

- **Data Augmentation**: Built-in via `image_dataset_from_directory`
- **Prefetching**: Optimized data pipeline for faster training
- **Binary Classification**: REAL vs FAKE images
- **Evaluation Metrics**: 
  - Accuracy
  - Loss
  - Classification report
  - Confusion matrix

## 📊 Dataset

The project uses the **Celeb-DF v2** dataset structure:

```
dataset/
└── Celeb_V2/
    ├── Train/
    │   ├── FAKE/
    │   └── REAL/
    ├── Val/
    │   ├── FAKE/
    │   └── REAL/
    └── Test/
        ├── FAKE/
        └── REAL/
```

**Note**: The dataset is not included in this repository due to size constraints. You'll need to organize your own dataset following the structure above.

### Recommended Datasets

- [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC (Deepfake Detection Challenge)](https://ai.facebook.com/datasets/dfdc/)

## 📈 Results

### Training Visualizations

The training scripts generate plots showing:
- Training vs Validation Accuracy
- Training vs Validation Loss

These are saved as `training_history.png` and `training_history_efficientnet_pro.png`.

### Model Performance

After training, the scripts output:
- Test accuracy and loss
- Classification report (precision, recall, F1-score)
- Confusion matrix

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras 2.10.1
- **Web Framework**: Streamlit 1.27.2
- **Face Detection**: MTCNN 0.1.1
- **Image Processing**: OpenCV 4.8.1.78, Pillow 10.0.0
- **Data Analysis**: NumPy 1.26.4, scikit-learn
- **Visualization**: Matplotlib

## 🔧 Key Components

### Face Detection (MTCNN)

The application uses MTCNN (Multi-task Cascaded Convolutional Networks) for accurate face detection before classification.

### Transfer Learning

Both models leverage pre-trained weights from ImageNet, allowing the model to:
- Train faster
- Require less data
- Achieve higher accuracy

### Preprocessing

- **EfficientNet**: Uses `efficientnet.preprocess_input` (scaling to [-1, 1])
- **Xception**: Uses `xception.preprocess_input` (scaling to [-1, 1])

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sagar**

- GitHub: [@sagar-grv](https://github.com/sagar-grv)

## 🙏 Acknowledgments

- TensorFlow and Keras teams for the excellent deep learning framework
- Streamlit for the intuitive web app framework
- The creators of the Celeb-DF dataset
- The open-source community

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!
