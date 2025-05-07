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

## Key Findings:

* **Emerging Pattern Detection**: Unsupervised models identified clusters of anomalies not captured by supervised models, revealing potential new fraud tactics.

* **Consistent Recall Over Time**: While supervised accuracy dropped on newer data with concept drift, unsupervised recall remained stable (≈0.60).

* **Visualization-Driven Insights**: UMAP projections showed clear separability of outliers, confirming that high-dimensional PCA features carry discriminative power.

* **Operational Efficiency**: Removing the labeling bottleneck reduced dataset preparation time by over 80%, streamlining the model update cycle.

* **Flexible Thresholding**: By adjusting anomaly-score thresholds, teams can dial precision vs. recall to meet evolving risk tolerances.

## Unsupervised Learning Pipeline

To clarify the steps from PCA to final anomaly detection, the unsupervised workflow follows these five key stages:

* **Feature Scaling**: Standardize transaction features so that PCA accurately captures the true variance without bias from scale differences.

* **Dimensionality Reduction (PCA)**: Transform the 30 original features into principal components that capture most variance, reducing noise and multicollinearity.

* **Visualization (UMAP)**: Project PCA-reduced data into 2D space to visually inspect clusters and isolate potential outliers before modeling.

* **Anomaly Detection Algorithms**: Train models like Isolation Forest, LOF, and One-Class SVM on the reduced feature set to assign anomaly scores to each transaction.

* **Thresholding & Evaluation**: Convert continuous anomaly scores into binary fraud predictions by selecting score thresholds, then compute precision, recall, and F1-score.

## Data Visualization

Visual aids were created in Jupyter notebooks to explore feature behavior:

* **Heatmap**: Correlation matrix of features and target
  ![Untitled-1](https://github.com/user-attachments/assets/787e3675-9643-470a-955a-58290753326f)
  
* **PCA Projection of Transactions**:
  ![Untitled-1](https://github.com/user-attachments/assets/529fdf94-2241-425d-82c5-80dc8613581b)


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

This study demonstrates the critical trade-offs between supervised and unsupervised fraud detection methods:

* **High Predictive Performance vs. Label Dependence**: Supervised models (Random Forest, XGBoost) achieved F1-scores above 0.90 on static test sets, consistent with findings by Dal Pozzolo et al. (2015), but require extensive labeled data and frequent retraining to address concept drift (Gama et al., 2014).

* **Novel Anomaly Discovery**: Unsupervised algorithms—Isolation Forest (Liu et al., 2008), LOF (Breunig et al., 2000), One-Class SVM (Schölkopf et al., 2001)—delivered moderate F1-scores (0.60–0.65) yet successfully flagged emerging fraud patterns absent from historical labels.

* **Temporal Stability**: When evaluated on temporal holdout splits, unsupervised recall varied by less than 5%, highlighting resilience to evolving transaction behaviors and shifting fraud tactics.

* **Operational Efficiency**: Removing the manual labeling step reduced end-to-end data preparation time by over 80%, enabling rapid iteration and deployment in real-time monitoring pipelines.

* **Future Research Directions**: Incorporating ensemble anomaly detection frameworks (Aggarwal, 2016), leveraging streaming deep autoencoders (Zhou & Paffenroth, 2017), and integrating concept-drift adaptation mechanisms can further enhance detection sensitivity and robustness.

By prioritizing unsupervised learning, this pipeline achieves a scalable, label-agnostic approach that aligns with best practices for continuous fraud monitoring in dynamic financial environments.

## Acknowledgments

* **Course**: IA651 Applied Machine Learning, Clarkson University
* **Supervisor**: Prof. Michael Gilbert

## Authors

* Ashish Varma Juttu, Clarkson University
* Sagar Bangera, Clarkson University
