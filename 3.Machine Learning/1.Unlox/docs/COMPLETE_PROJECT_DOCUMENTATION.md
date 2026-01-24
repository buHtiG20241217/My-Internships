# Tailyo Technologies: Service Recommendation System - Complete Project Documentation

**Date:** December 9, 2025
**Version:** 1.0.0
**Status:** Deployed & Verified

---

## 📑 Table of Contents
1.  [Executive Summary](#1-executive-summary)
2.  [System Architecture](#2-system-architecture)
3.  [Methodology & Algorithm](#3-methodology--algorithm)
4.  [User Guide](#4-user-guide)
5.  [Developer Guide & API](#5-developer-guide--api)
6.  [Evaluation & Results](#6-evaluation--results)
7.  [Visual Gallery](#7-visual-gallery)
8.  [Appendix: Setup & Installation](#8-appendix-setup--installation)

---

## 1. Executive Summary
This document serves as the comprehensive knowledge base for the Service Recommendation System developed for Tailyo Technologies. The system provides intelligent, personalized service recommendations using a hybrid machine learning approach, wrapped in a responsive, brand-aligned Streamlit interface.

**Key Achievements:**
*   **Intelligent Matching:** Hybrid content-based + strict filtering engine.
*   **User-Centric UI:** "Amazon-style" results card carousel.
*   **Robustness:** 100% automated test pass rate.
*   **Observability:** "Why This Match?" explanation engine for transparency.

---

## 2. System Architecture
The application follows a modular architecture separating the presentation layer (Streamlit) from the core logic (ML Engine) and data layer.

```mermaid
graph TD
    User[User Interface (Streamlit)] <--> |Input/Results| App[Main Application]
    App --> |Query| Engine[Recommendation Engine]
    
    subgraph Core Logic
        Engine --> |Encode| Encoder[User Encoder]
        Engine --> |Rank| Ranker[Cosine/KNN Ranker]
        Engine --> |Explain| Explainer[Explanation Generator]
    end
    
    subgraph Data Layer
        Encoder --> |Load| EncodersPKL[Encoders.pkl]
        Ranker --> |Load| Features[Features.npy]
        Ranker --> |Load| CleanedData[Cleaned Data CSV]
    end
```

---

## 3. Methodology & Algorithm

### 3.1 Hybrid Recommendation Strategy
The core engine utilizes a **Content-Based Filtering** approach enhanced with **Strict Filtering** pre-processors.

1.  **Input Processing**: User preferences (Budget, Language, Location, Business Type) are encoded into a vector.
    *   *Categorical Encoding*: One-Hot Encoding for Business Type (`Target_Business_Type`).
    *   *Ordinal Encoding*: Weighted mapping for Price Category (`Low`=0.25 to `Premium`=1.0).
    *   *Text Analysis*: TF-IDF Vectorization for the `Description` field to capture semantic intent (boosted by 10x for relevance).

2.  **Filtering Layer**:
    *   **Strict Mode**: Pre-filters candidates to ensure they exactly match critical constraints (Business Type, Location) before expensive ranking calculations.
    *   **Price Logic**: Ensures recommendations are strictly at or below the user's specified budget.

3.  **Ranking Algorithm**:
    *   **Cosine Similarity**: Measures the cosine of the angle between the multi-dimensional User Vector and Service Vectors.
        $$ \text{Similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} $$
    *   **Output**: Services are ranked by similarity score (0-100%).

4.  **Explanation Generation**:
    *   A rule-based post-processor compares attributes of recommended services against user inputs to generate human-readable "Why this match?" reasons (e.g., "Within your budget", "Perfect match for E-commerce").

---

## 4. User Guide

### 🚀 Getting Started
1.  **Launch the Application**: Ensure the application is running in your browser (default: `http://localhost:8501`).
2.  **View the Header**: You should see the **Tailyo Technologies** branding at the top.

### 🎯 Finding Services
The application uses a 3-step process to understand your requirements:

#### Step 1: Location Preference
-   **Remote**: Choose this for digital services deliverable from anywhere.
-   **Specific City**: Choose this for local providers in **Delhi, Mumbai, Bengaluru, or Chennai**.

#### Step 2: Business & Budget
-   **Business Category**: Select your industry (e.g., *E-commerce, Clinic, Tech Startup*).
-   **Budget Range**:
    -   *Low*: Cost-effective solutions.
    -   *Medium*: Balanced quality and cost.
    -   *High*: Premium providers.
    -   *Premium*: Top-tier, specialized agencies.

#### Step 3: Specific Needs
-   **Preferred Languages**: Select **English**, **Hindi**, or **Both**.
-   **Service Type & Description**: Choose a category and optionally refine the description.

### 📊 Understanding Results
Click **"Find Matching Services"** to generate recommendations. The system displays the **Top 3** matches.

*   **Match Score**: Percentage indicating fit.
    *   🟢 **HIGH MATCH (>80%)**
    *   🔵 **GOOD MATCH (60-79%)**
    *   🟠 **ALTERNATIVE (<60%)**
*   **Why This Match?**: Explains the AI's reasoning (e.g., "Matches your budget").

### ❓ Troubleshooting
-   **"No Services Found"**: Try broadening your search (e.g., switch Location to **Remote**).
-   **Slow Loading**: First search typically takes 2-3 seconds for model loading; subsequent searches are instant.

---

## 5. Developer Guide & API

### 🏗️ Architecture Overview
1.  **Preprocessing Layer**: Text normalization (`src/preprocessing/data_cleaner.py`) and feature engineering (`src/preprocessing/feature_engineering.py`).
2.  **Model Layer**: 
    -   `recommendation_engine.py`: Loads artifacts and orchestrates logic.
    -   `user_encoder.py`: Transforms 1x5 logical input into 1xN feature vector.
3.  **Application Layer**: `streamlit_app.py` handles state and UI rendering.

### 🔄 workflows
**Adding New Data:**
1.  Add new raw data to `data/raw/`.
2.  Run cleaning: `python src/preprocessing/data_cleaner.py`
3.  Regenerate features: `python src/preprocessing/feature_engineering.py`
4.  Restart Streamlit app.

---

## 6. Evaluation & Results

### 6.1 Testing Metrics
The system passed a complete 8-point system integration test suite (100% Pass Rate).

| Test Case | Result | Description |
| :--- | :--- | :--- |
| **Engine Initialization** | ✅ | Validated loading of 1000+ services and feature matrices. |
| **Strict Filtering** | ✅ | Confirmed exclusions of mismatched business types. |
| **Match Scores** | ✅ | Scores verified to be descending and within 0-100% range. |
| **Data Quality** | ✅ | No missing values in critical feature columns. |

### 6.2 Performance
*   **Latency**: Sub-200ms response time for recommendation generation.
*   **Scalability**: Vectorized operations allow efficient scaling.

---

## 7. Visual Gallery

### Application Interface & Results
*Below are screenshots of the deployed application.*

![Results Page](file:///E:/Internship/ml-service-recommendation/screenshots/Screenshot%202025-12-09%20190744.png)

![User Input Form](file:///E:/Internship/ml-service-recommendation/screenshots/Screenshot%202025-12-09%20190807.png)

---

## 8. Appendix: Setup & Installation

### Installation
1.  **Clone & Environment**:
    ```bash
    git clone <repo>
    cd ml-service-recommendation
    python -m venv venv
    .\venv\Scripts\activate
    ```
2.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run**:
    ```bash
    streamlit run app/streamlit_app.py
    ```

---
**Report Generated By:** Kiran Soorya R S
**For:** Tailyo Technologies
