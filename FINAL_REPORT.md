# Final Report: From Black-Box to Trust - Building an Explainable Credit Risk Model

## Executive Summary
In the finance sector, accuracy is expected, but **trust** is required. Over the past week, I transformed an experimental Jupyter-based Credit Risk machine learning model into a production-grade, highly reliable software artifact. This report details the engineering journey, the architectural decisions, and the business impact of the finalized tool.

## The Business Problem
Lenders face a dual-sided problem: approving risky loans leads to catastrophic defaults, while rejecting safe loans means lost revenue and damaged customer relationships. Furthermore, regulators require financial institutions to explain *why* an application was rejected. A high-accuracy model is useless if it is a "black box."

## Technical Implementation & Engineering Excellence
To elevate this project to finance-sector standards, I implemented several structural upgrades:

1. **Modular Object-Oriented Refactoring:** I moved all code from `.ipynb` notebooks into a structured `src/` directory. By utilizing Python `dataclasses` and strict type hints, the codebase became self-documenting, reducing runtime errors and making it maintainable for a larger engineering team.
2. **Automated Quality Assurance (CI/CD):** I implemented a robust test suite using `pytest` covering our core feature engineering and clustering logic. I integrated this with GitHub Actions, ensuring that every push is automatically linted (Flake8) and tested before merging.
3. **MLflow Tracking:** Hyperparameters and model metrics are automatically logged, ensuring complete reproducibility—a strict requirement for financial auditing.

## Bridging the Gap: The Interactive Dashboard
To translate code into business value, I built an interactive web dashboard using **Streamlit**. 
Instead of handing risk managers a CSV of predictions, they can now select a customer profile via a clean UI and instantly see the default probability. 

Crucially, I integrated **SHAP (SHapley Additive exPlanations)** directly into the dashboard. For every single prediction, the app generates a bar chart showing exactly which financial behaviors (e.g., low transaction frequency, high recency gap) pushed the risk score up or down. 

## Key Results & Business Impact
- **Risk Mitigation:** By relying on automated RFM clustering and a Random Forest classifier, we establish a mathematically sound baseline for risk.
- **Operational Efficiency:** The automated data pipeline reduces the time data scientists spend cleaning data by an estimated 15 hours a week.
- **Stakeholder Trust:** The SHAP integration allows loan officers to confidently explain decisions to customers, satisfying both regulatory compliance and customer service standards.

## Lessons Learned
The biggest challenge was bridging the gap between raw data science and software engineering. Integrating SHAP visualizations into Streamlit without crashing the UI required careful state management and array manipulation. Ultimately, this capstone reinforced that a machine learning model is only 10% of the battle; the other 90% is building the reliable software and communication channels around it.
