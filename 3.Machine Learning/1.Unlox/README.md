# ML-Powered Service Recommendation System

**Client:** Tailyo Technologies  
**Project:** Service Recommendation Engine  
**Status:** Major Project Phase

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)

## 📖 Project Overview

This Machine Learning-powered application is designed for **Tailyo Technologies** to help businesses find the perfect service providers. By analyzing user requirements such as business type, budget, location, and specific needs, the system utilizes advanced NLP and similarity algorithms to recommend the most relevant services.

## ✨ Key Features

-   **Smart Recommendations**: Uses TF-IDF and Cosine Similarity to match user descriptions with service offerings.
-   **Multi-Criteria Filtering**: Precision filtering based on Business Category, Price Range, Location, and Language Support.
-   **Interactive UI**: A polished, Amazon-style card interface built with Streamlit for seamless user interaction.
-   **Explainable AI**: Provides "Why This Match?" insights for every recommendation.
-   **Text-Based Branding**: Custom "Tailyo Technologies" branding for a professional corporate identity.

## 🛠️ Tech Stack

-   **Frontend**: Streamlit (Python)
-   **Backend Logic**: Python
-   **Machine Learning**: scikit-learn (TF-IDF, Cosine Similarity, Nearest Neighbors)
-   **Data Processing**: Pandas, NumPy
-   **Testing**: Pytest

## 🚀 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd ml-service-recommendation
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    streamlit run app/streamlit_app.py
    ```

## 📂 Project Structure

```
ml-service-recommendation/
├── app/
│   ├── static/          # Images and assets
│   └── streamlit_app.py # Main application entry point
├── data/
│   ├── raw/             # Original dataset
│   ├── processed/       # Feature matrices
│   └── cleaned/         # Cleaned CSV data
├── src/
│   ├── models/          # ML Engine and Encoders
│   ├── preprocessing/   # Data cleaning and feature engineering scripts
│   └── utils/           # Helper functions
├── tests/               # Automated test suite
├── docs/                # User and Developer documentation
├── notebooks/           # Jupyter notebooks for experiments
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🧪 Running Tests

To verify the system integrity, run the automated test suite:

```bash
pytest tests/
```

---
© 2025 Tailyo Technologies. All Rights Reserved.
