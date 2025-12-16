import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import os

# Ensure images dir exists
os.makedirs("images", exist_ok=True)

# Load Data
print("Loading data...")
df = pd.read_csv("data/processed/train_labeled.csv")

# --- Plot 1: Transaction Distribution (EDA) ---
plt.figure(figsize=(10, 6))
sns.histplot(df["Amount"], bins=50, kde=True, color="blue")
plt.title("Distribution of Transaction Amounts (Log Scale)")
plt.yscale("log")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/eda_distribution.png")
print("Saved images/eda_distribution.png")

# --- Plot 2: Correlation Matrix (EDA) ---
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("images/correlation.png")
print("Saved images/correlation.png")

# --- Plot 3: ROC Curve (Model Performance) ---
X = df.drop(["risk_label", "CustomerId"], axis=1, errors="ignore")
y = df["risk_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/roc_curve.png")
print("Saved images/roc_curve.png")
