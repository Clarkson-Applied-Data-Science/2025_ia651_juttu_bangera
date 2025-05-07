# Credit Card Fraud Detection Using Unsupervised Learning

## Abstract

Fraud detection in credit card transactions is challenged by extreme class imbalance and evolving fraud tactics. This project applies unsupervised learning techniques to flag anomalies without relying on labeled fraud examples, and benchmarks performance against supervised models to highlight trade‑offs.

## Dataset

* **Source**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Records**: 284,807 transactions over two days
* **Fraud Cases**: 492 (0.172%)
* **Features**: 30 (V1–V28 PCA components, `Time`, `Amount`, `Class`)

## Methodology

1. **Preprocessing**

   * Scaled features with `StandardScaler` & `MinMaxScaler`
   * Selected high‑variance and low‑correlation features
2. **Dimensionality Reduction**

   * Used UMAP to project data into 2D for visualization of anomaly clusters
3. **Exploratory Data Analysis**

   * Time‑of‑day patterns: fraud spikes between 0–6 hrs
   * PCA component distributions differ for fraud vs. normal transactions

## Models Implemented

* **Supervised** (`creditcard_supervised.ipynb`)

  * Logistic Regression, Random Forest, XGBoost
* **Unsupervised** (`creditCard.ipynb`)

  * Isolation Forest, Local Outlier Factor, One‑Class SVM

## Performance Comparison

### Supervised Models

| Model               | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Logistic Regression | 0.92      | 0.88   | 0.90     |
| Random Forest       | 0.95      | 0.93   | 0.94     |
| XGBoost             | 0.96      | 0.94   | 0.95     |

### Unsupervised Models

| Algorithm            | Precision | Recall | F1-Score |
| -------------------- | --------- | ------ | -------- |
| Isolation Forest     | 0.70      | 0.60   | 0.65     |
| Local Outlier Factor | 0.68      | 0.58   | 0.62     |
| One-Class SVM        | 0.65      | 0.55   | 0.60     |

## Why Unsupervised?

* **No Labeled Dependence**: Eliminates costly, time‑consuming fraud labeling
* **Adaptability**: Detects novel fraud tactics without retraining on new labels
* **Real‑Time Scalability**: Flags anomalies on incoming streams immediately
* **Robust to Imbalance**: Designed for skewed classes, avoiding majority bias

## Data Visualization

Visual aids were created in Jupyter notebooks to explore feature behavior:

* **Heatmap**: Correlation matrix of features and target
  ![Untitled-1](https://github.com/user-attachments/assets/787e3675-9643-470a-955a-58290753326f)

* **UMAP 2D Projection**: Cluster separation of outliers
  ![Untitled](https://github.com/user-attachments/assets/bae72717-210a-4f9b-939b-d63cc1265890)

* **Histograms & Boxplots**: Skewness and IQR of `Amount`, `Time`, key V‑features
  ![Untitled-1](https://github.com/user-attachments/assets/6995f0aa-bc68-4469-8a44-dbe0d27c0942)
  ![Untitled](https://github.com/user-attachments/assets/0479f8c8-0407-4c0c-bb48-a123632fc301)

* **Class Distribution**: Highlighted extreme imbalance (\~0.17% fraud)
  ![Untitled-1](https://github.com/user-attachments/assets/1a393cdd-aafc-4fee-9304-05cb22dda53f)

* **DBSCAN Cliustering**: Highlighted clusters of fraud.
![Untitled](https://github.com/user-attachments/assets/eb5af6a4-2318-47f9-b513-a0accb031d50)

* **Fraud Heavy Clusters**:
  ![Untitled-1](https://github.com/user-attachments/assets/079b958c-a8fe-4bec-8765-58581ee0002e)

* **Isolation Forest: Contamination Vs. Precision/Recall/F1-Score**
  ![Untitled](https://github.com/user-attachments/assets/df071dfa-14a9-41dd-ad49-f11bf25ef806)



## Usage

1. Clone this repo:

   ```bash
   git clone https://github.com/Clarkson-Applied-Data-Science/2025_ia651_juttu_bangera.git
   cd 2025_ia651_juttu_bangera
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch notebooks:

   ```bash
   jupyter notebook creditcard_supervised.ipynb
   jupyter notebook creditCard.ipynb
   ```

## Conclusion

While supervised models achieve higher F1‑scores (0.90+), they risk overfitting and depend on labeled fraud examples. Unsupervised methods, with F1‑scores around 0.60–0.65, offer a more flexible, label‑free, and robust solution for real‑time anomaly detection in dynamic fraud environments.

## Acknowledgments

* **Course**: IA651 Applied Machine Learning, Clarkson University
* **Supervisor**: Prof. Michael Gilbert

## Authors

* Ashish Varma Juttu, Clarkson University
* Sagar Bangera, Clarkson University
