# Credit Risk Probability Model for Alternative Data

## 1. Credit Scoring Business Understanding

### Basel II Accord & Interpretability
The Basel II Accord emphasizes the "Internal Ratings-Based" (IRB) approach. This requires financial institutions to not only estimate risk but to understand the **drivers of that risk**.
*   **Impact:** We need a model that is **transparent**. A "black box" model (like a deep neural network) might be rejected by regulators because we cannot explain *why* a specific user was denied a loan. We favor Explainable AI (XAI) or interpretable models like Logistic Regression with Weight of Evidence (WoE).

### The Proxy Variable Strategy
Since Xente is a transactional platform and not a bank, we lack a historical "Loan Default" column.
*   **The Strategy:** We create a proxy based on **RFM (Recency, Frequency, Monetary)** analysis.
*   **The Logic:** Users who are "disengaged" (High Recency, Low Frequency, Low Monetary) are statistically less likely to be reliable borrowers compared to active, high-volume users.
*   **Business Risk:** The risk is **Type II Error (False Negative)**—classifying a good potential borrower as "High Risk" just because they haven't used the platform recently, leading to lost revenue.

### Model Trade-offs
1.  **Logistic Regression (w/ WoE):**
    *   *Pros:* Highly interpretable, standard in banking (Scorecards), statistically robust.
    *   *Cons:* Misses complex non-linear relationships.
2.  **Gradient Boosting (XGBoost/LGBM):**
    *   *Pros:* High accuracy, captures complex patterns.
    *   *Cons:* Harder to interpret. Requires SHAP/LIME for explanation to satisfy Basel II.
    *   *Decision:* We will train both, but prioritize interpretability for the final recommendation.
