# YOLOv11 Object Detection on BDD100K Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)

This repository contains an end-to-end project for training a YOLOv8 model on a subset of the BDD100K dataset for object detection in autonomous driving scenarios. It includes a Jupyter notebook for data preparation, exploratory data analysis (EDA), model training, validation, and export, along with a web-based demo application built with React (frontend) and FastAPI (backend) for real-time inference on uploaded images.

The model detects classes such as cars, trucks, pedestrians, traffic signs, and traffic lights. 

## Features
- **Data Processing**: Cleaning, EDA, and preparation of BDD100K dataset annotations and images.
- **Model Training**: Fine-tuning YOLOv8 on a subset of the dataset for object detection.
- **Validation & Metrics**: Evaluation with mAP, precision, recall, and confusion matrices.
- **Web Demo**: Upload images for inference, displaying bounding boxes and confidence scores.
- **Export**: Trained model exported as `.pt` for deployment.

## Dataset
This project uses a subset of the [BDD100K dataset](https://www.bdd100k.com/), which includes images and annotations for object detection in driving scenarios. The notebook processes the training split, handling missing/corrupted data, duplicates, and outliers.

**Classes**: car, truck, bus, pedestrian, bike, rider, motor, train, traffic light, traffic sign.

## Installation

### Prerequisites
- Python 3.8+
- Node.js (for React frontend)
- Git

### Steps
1. **Clone the Repository**:
   ```
   git clone https://github.com/maged-94/GTC-RealTime-Object-Detection.git
   ```

2. **Backend (FastAPI)**:
   - Install dependencies:
     ```
     pip install -r backend/requirements.txt
     ```
     (Assumed requirements: `fastapi`, `uvicorn`, `ultralytics`, `pillow`, `python-multipart`)
   - Place the trained model (`best.pt`) in `backend/models/best.pt`. (Download from notebook output or train your own.)
   - Update `MODEL_PATH` in `backend/main.py` if necessary.

3. **Frontend (React)**:
   - Navigate to the frontend directory:
     ```
     cd frontend
     npm install
     ```

## Usage

### Training the Model
1.  Run cells sequentially:
   - **Data Loading & Cleaning**: Processes BDD100K images and annotations.
   - **EDA**: Visualizes dataset statistics (e.g., category counts, bounding box sizes).
   - **Dataset Preparation**: Converts annotations to YOLO format.
   - **Training**: Fine-tunes YOLOv8 (adjust epochs, batch size, etc., as needed).
   - **Validation**: Evaluates model performance on a test subset.
2. Export the model: The notebook saves `best.pt` to `/kaggle/working/models/`.

### Running the Web Demo
1. **Start Backend**:
   ```
   cd backend
   uvicorn main:app --reload --port 8000
   ```
   Ensure `MODEL_PATH` points to the correct `best.pt` file.

2. **Start Frontend**:
   ```
   cd frontend
   npm run dev
   ```
   Open the provided Vite dev server URL (e.g., `http://localhost:5173`).

3. **Demo Interaction**:
   - Select an image file using the "Choose Image" button.
   - Click "Upload & Predict" to run inference.
   - View bounding boxes with labels and confidence scores overlaid on the image.
   - Raw JSON results are displayed below the image.

### Example Output
- **Notebook**: Outputs training logs, confusion matrix, precision-recall curves, and sample predictions (e.g., `384x640 5 trucks, 3 pedestrians, 4 signs`).
- **Demo**: For an input image, outputs like:
  ```json
  {
    "predictions": [
      {
        "label": "car",
        "confidence": 0.92,
        "bbox": [100.5, 150.2, 300.7, 350.4],
        "original_width": 1280,
        "original_height": 720
      }
    ]
  }
  ```




## Technologies Used
- **Model**: YOLOv8 (Ultralytics)
- **Backend**: FastAPI, Pillow
- **Frontend**: React, Axios
- **Data Processing**: Pandas, NumPy, OpenCV
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Kaggle/Jupyter Notebook

## Project Structure
```
GTC-RealTime-Object-Detection/
├── backend/                  # FastAPI backend code
│   ├── main.py               # API for YOLO inference
│   ├── models/best.pt        # Trained YOLOv8 model
│   └── requirements.txt      # Backend dependencies
├── frontend/                 # React frontend code
│   ├── src/App.jsx           # Main React component
│   └── package.json          # Frontend dependencies
├── README.md                # This file
└── LICENSE                  # MIT License
```

## Contributing
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-feature`.
3. Commit changes: `git commit -m 'Add new feature'`.
4. Push to branch: `git push origin feature/new-feature`.
5. Open a Pull Request.


## Acknowledgments
- **BDD100K Dataset**: UC Berkeley AI Research.
- **YOLOv8**: Ultralytics team.
- Inspired by open-source object detection tutorials.
