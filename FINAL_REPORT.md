```markdown
# Building a Credit Risk Probability Model for Alternative Data

## 1. Business Problem & Proxy Variable Strategy
In the absence of traditional credit history, **Bati Bank** requires a novel approach to assess the creditworthiness of eCommerce users for a Buy-Now-Pay-Later (BNPL) service.

**The Challenge:** The dataset provided lacks a direct "Loan Default" label. It consists purely of transactional data.
**The Solution:** We engineered a **Proxy Target Variable** based on user behavior.
*   **Assumption:** Users who are highly active and transact frequently are "Good" borrowers. Users who are disengaged (dormant) are "High Risk."
*   **Methodology:** We utilized **RFM (Recency, Frequency, Monetary)** analysis.
    *   *Recency:* Days since last transaction.
    *   *Frequency:* Total count of transactions.
    *   *Monetary:* Total amount spent.

## 2. Methodology: RFM Clustering
We applied **K-Means Clustering** (k=3) on the standardized RFM features to segment users into:
1.  **High Value (Low Risk):** Frequent, recent, high-value transactions.
2.  **Average Users:** Moderate activity.
3.  **Disengaged (High Risk):** High recency (inactive), low frequency.

We labeled the "Disengaged" cluster as `1` (High Risk) and the others as `0` (Low Risk). This created our target variable for supervised learning.

## 3. Model Selection & Performance
We trained two models to predict this risk probability:

| Model | Accuracy | Pros | Cons |
|-------|----------|------|------|
| **Random Forest** | **89.08%** | Handles non-linear data well, robust to outliers. | Less interpretable than linear models. |
| **Logistic Regression** | 88.39% | Highly interpretable, Basel II compliant. | Struggles with complex non-linear patterns. |

**Selected Model:** We chose the **Random Forest** for deployment due to its slightly higher accuracy and robustness, tracked via **MLflow**.

## 4. Deployment Architecture
To operationalize the model, we built a real-time inference pipeline:
*   **API Framework:** **FastAPI** (Python) for high-performance, asynchronous predictions.
*   **Validation:** **Pydantic** ensures data integrity before it reaches the model.
*   **Containerization:** **Docker** packages the model, dependencies, and API into a portable image.
*   **CI/CD:** GitHub Actions automates linting (Flake8) and unit testing (Pytest) on every push.

### API Demonstration
**Endpoint:** `POST /predict`
**Request:**
```json
{
  "Amount": 5000,
  "TransactionHour": 14,
  "ProductCategory": 3,
  ...
}
```
**Response:**
```json
{
  "risk_probability": 0.12,
  "risk_label": "Low Risk"
}
```
