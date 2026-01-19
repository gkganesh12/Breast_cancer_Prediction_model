<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🎗️ Breast Cancer Prediction Model</h1>

<p align="center">
  <strong>An interactive machine learning dashboard for predicting breast cancer diagnosis using Decision Tree Classifier</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-model-performance">Model Performance</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-license">License</a>
</p>

---

## ✨ Features

- 🔮 **Real-time Predictions** - Interactive sliders to input tumor characteristics
- 🔍 **Manual Control** - "Get Diagnosis" button to trigger predictions only when you're ready
- 📜 **Prediction History** - Save and compare different scenarios in a history table
- 📊 **Model Insights** - Comprehensive performance metrics including confusion matrix and ROC curve
- 🔬 **Data Explorer** - Explore the dataset with visualizations and correlation heatmaps
- 💡 **Quick Demo** - Pre-loaded sample cases for malignant and benign tumors
- 📈 **Feature Importance** - Understand which factors contribute most to each decision

## 🎯 Demo

🌐 **Live Demo**: [https://breastcancer-gk.streamlit.app/](https://breastcancer-gk.streamlit.app/)

The dashboard provides four main sections:

| Tab | Description |
|-----|-------------|
| 🔮 **Predict** | Input tumor features and get real-time predictions with confidence scores |
| 📊 **Model Insights** | View accuracy, precision, recall, F1 score, confusion matrix, and ROC curve |
| 🔬 **Data Explorer** | Explore the dataset distribution and feature correlations |
| ℹ️ **About** | Learn about the project, dataset, and model details |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or uv package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/gkganesh12/Breast_cancer_Prediction_model.git
   cd Breast_cancer_Prediction_model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Using Makefile (Alternative)

```bash
# Install uv if not already installed
pip install --upgrade uv

# Create virtual environment
make create_environment

# Install dependencies
make requirements

# Run the app (optional: build predictions)
make breast_cancer_prediction
```

## 💻 Usage

### Interactive Prediction

1. Use the sliders in the sidebar to input tumor characteristics
2. Watch the prediction update in real-time
3. View the confidence gauge and probability distribution
4. Explore which features influenced the prediction most

### Quick Demo Mode

- Click **🔴 Malignant** to load a sample malignant case
- Click **🟢 Benign** to load a sample benign case

## 📈 Model Performance

The Decision Tree Classifier achieves excellent performance on the test set:

| Metric | Score |
|--------|-------|
| **Accuracy** | ~95% |
| **Precision** | ~93% |
| **Recall** | ~95% |
| **F1 Score** | ~94% |

### Model Configuration

```python
DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    min_samples_split=0.01,
    min_samples_leaf=0.01
)
```

## 📊 Dataset

This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository.

| Property | Value |
|----------|-------|
| **Source** | [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| **Total Samples** | 569 |
| **Benign Cases** | 357 (62.7%) |
| **Malignant Cases** | 212 (37.3%) |
| **Features** | 30 numeric features |

### Features Explained

The features are computed from digitized images of fine needle aspirates (FNA) of breast masses:

| Feature | Description |
|---------|-------------|
| Radius | Mean of distances from center to points on the perimeter |
| Texture | Standard deviation of gray-scale values |
| Perimeter | Perimeter of the cell nucleus |
| Area | Area of the cell nucleus |
| Smoothness | Local variation in radius lengths |
| Compactness | Perimeter² / Area - 1.0 |
| Concavity | Severity of concave portions of the contour |
| Concave Points | Number of concave portions of the contour |
| Symmetry | Symmetry of the cell nucleus |
| Fractal Dimension | "Coastline approximation" - 1 |

Each feature is computed as **mean**, **standard error**, and **worst** (mean of the three largest values).

## 📁 Project Structure

```
Breast_cancer_Prediction_model/
├── 📄 app.py                    # Streamlit dashboard application
├── 📄 requirements.txt          # Python dependencies
├── 📄 Makefile                  # Build automation
├── 📄 pyproject.toml            # Project configuration
├── 📁 data/
│   └── 📁 raw/                  # Raw dataset files
├── 📁 models/                   # Trained model files
├── 📁 notebooks/                # Jupyter notebooks for analysis
│   ├── 01-data-overview.ipynb
│   ├── 02-model-training.ipynb
│   ├── 03-pca-feature-engineering.ipynb
│   ├── 04-hyperparameter-tuning.ipynb
│   └── 05-model-explainability.ipynb
├── 📁 figures/                  # Generated plots and visualizations
└── 📁 docs/                     # Documentation
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

## ⚠️ Disclaimer

> **Important**: This tool is for **educational and demonstration purposes only**. It should **NOT** be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024-2026 Ganesh Khetawat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files.
```

## 👨‍💻 Author

**Ganesh Khetawat**

- GitHub: [@gkganesh12](https://github.com/gkganesh12)

---

<p align="center">
  Made with ❤️ for cancer awareness and early detection
</p>

<p align="center">
  <a href="https://github.com/gkganesh12/Breast_cancer_Prediction_model">
    <img src="https://img.shields.io/github/stars/gkganesh12/Breast_cancer_Prediction_model?style=social" alt="Stars">
  </a>
</p>
