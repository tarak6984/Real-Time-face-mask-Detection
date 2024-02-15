# Real-Time Face Mask Detection

A real-time face mask detection system built with Python, OpenCV, TensorFlow, and Keras. This deep learning model detects whether a person is wearing a face mask or not using live camera feed.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Applications](#applications)
- [License](#license)

## 🔍 Overview

This project uses a deep learning model based on MobileNetV2 architecture to detect face masks in real-time video streams. The system can identify whether a person is wearing a mask with high accuracy and displays the result with bounding boxes and confidence scores.

## ✨ Features

- **Real-time Detection**: Processes live video feed from webcam
- **High Accuracy**: Uses MobileNetV2 pre-trained model with custom layers
- **Fast Performance**: Optimized for real-time processing
- **Visual Feedback**: Color-coded bounding boxes (Green for mask, Red for no mask)
- **Confidence Score**: Displays detection probability percentage
- **Multi-face Detection**: Can detect multiple faces simultaneously

## 🛠 Technologies Used

- **Python 3.x**
- **TensorFlow**: Deep learning framework
- **Keras**: High-level neural networks API
- **OpenCV**: Computer vision library
- **MobileNetV2**: Pre-trained CNN architecture
- **NumPy**: Numerical computing
- **imutils**: Image processing utilities

## 📦 Installation

### Prerequisites
- Python 3.6 or higher
- Webcam/Camera

### Step 1: Clone the Repository
```bash
git clone https://github.com/tarak6984/Real-Time-face-mask-Detection.git
cd Real-Time-face-mask-Detection
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with TensorFlow, you may need to install it separately:
```bash
pip install tensorflow==2.4.0
```

### Step 3: Verify Installation
Make sure all the required files are present:
- `main.py` - Main detection script
- `train_mask_detector.py` - Model training script
- `mask_detector.model` - Pre-trained model file
- `face_detector/` - Face detection model files
  - `deploy.prototxt`
  - `res10_300x300_ssd_iter_140000.caffemodel`

## 🚀 Usage

### Running the Face Mask Detector

Simply run the main script to start real-time detection:

```bash
python main.py
```

- The webcam will activate and start detecting faces
- Green bounding box = Person wearing mask
- Red bounding box = Person not wearing mask
- Press **'q'** to quit the application

### Training Your Own Model (Optional)

If you want to train the model with your own dataset:

1. Organize your dataset in the following structure:
```
dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_mask/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

2. Update the `DIRECTORY` path in `train_mask_detector.py` (line 28):
```python
DIRECTORY = r"path/to/your/dataset"
```

3. Run the training script:
```bash
python train_mask_detector.py
```

## 🧠 How It Works

### Detection Pipeline

1. **Face Detection**: Uses OpenCV's DNN module with Caffe model to detect faces in the frame
2. **Preprocessing**: Detected faces are resized to 224x224 and preprocessed for the model
3. **Classification**: MobileNetV2 model classifies each face as "Mask" or "No Mask"
4. **Visualization**: Results are displayed with bounding boxes and confidence scores

### Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**:
  - AveragePooling2D (7x7)
  - Flatten Layer
  - Dense Layer (128 units, ReLU activation)
  - Dropout (0.5)
  - Dense Layer (2 units, Softmax activation)

### Training Parameters
- Learning Rate: 1e-4
- Epochs: 20
- Batch Size: 32
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy

## 📁 Project Structure

```
Real-Time-face-mask-Detection/
│
├── face_detector/
│   ├── deploy.prototxt              # Face detector config
│   └── res10_300x300_ssd_iter_140000.caffemodel  # Face detector weights
│
├── main.py                          # Main detection script
├── train_mask_detector.py           # Model training script
├── mask_detector.model              # Trained mask detection model
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

## 🎯 Applications

This face mask detection system can be deployed in various scenarios:

- **Healthcare Facilities**: Hospitals, clinics, and medical centers
- **Educational Institutions**: Schools, colleges, and universities
- **Public Transportation**: Airports, train stations, bus terminals
- **Retail Spaces**: Shopping malls, supermarkets, stores
- **Corporate Offices**: Building entrances and security checkpoints
- **Restaurants & Entertainment**: Theaters, restaurants, gyms
- **Smart City Surveillance**: Public area monitoring systems

## 📊 Model Performance

The model achieves high accuracy in detecting face masks with:
- Fast inference time suitable for real-time applications
- Robust detection under various lighting conditions
- Support for multiple face detection in a single frame

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MobileNetV2 architecture from TensorFlow
- OpenCV DNN face detector
- Face mask dataset contributors

---

**Created by [tarak6984](https://github.com/tarak6984)**

For questions or feedback, feel free to open an issue on GitHub.



