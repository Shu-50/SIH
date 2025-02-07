# ISL Gesture Recognition

This project is designed to recognize **Indian Sign Language (ISL) hand gestures** using computer vision and machine learning. It involves collecting gesture data, training a model, and performing real-time gesture detection.

## **Features**
- Collect and label hand gesture data
- Train a machine learning model for gesture classification
- Detect ISL gestures in real-time using a webcam

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/Shu-50/SIH/Sign Language.git
cd isl-gesture-recognition
```

### **2. Install Dependencies**
Ensure you have Python installed. Install required packages:
```bash
pip install -r requirements.txt
```

---

## **Steps to Run the Project**

**1. Collect Data**
Run the script to collect hand gesture data:
```bash
python scripts/collect_data.py
```
- Press **'s'** to save the current gesture.
- Input the corresponding ISL label (e.g., `hello`, `thank_you`).
- The gesture data will be saved in `isl_data.csv`.

### **2. Train the Model**
Train the model using the collected dataset:
```bash
python scripts/train_model.py
```
- The trained model will be saved as `isl_gesture_classifier.pkl`.

**3. Real-Time Gesture Detection**
Run the real-time gesture recognition using your webcam:
```bash
python scripts/detect_gesture.py
```

## **Project Structure**
```
├── scripts/
│   ├── collect_data.py    # Collect gesture data
│   ├── train_model.py     # Train the model
│   ├── detect_gesture.py  # Detect gestures in real-time
│
├── models/
│   ├── isl_gesture_classifier.pkl  # Trained model
│
├── datasets/
│   ├── isl_data.csv      # Collected dataset
│
├── requirements.txt      # Required Python packages
├── README.md             # Project Documentation
```

 **Requirements**
- Python 3.x
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- MediaPipe

Install dependencies using:
```bash
pip install -r requirements.txt
```

**Contributing**
Contributions are welcome! Feel free to **fork**, **open issues**, or submit **pull requests**.

**License**
This project is open-source and available under the **MIT License**.

**Contact**
For any issues or suggestions, please contact: **lawhares@example.com**

