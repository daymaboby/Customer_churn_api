# Customer Churn Prediction Model

## Objective
Predict whether a customer will churn in the next 60 days using historical behavioral and transactional data.


## Project Structure

data/
churn_model.ipynb
model.pkl
metrics.json
error_analysis.md
model_card.md
requirements.txt
README.md


## Workflow

### 1. Data Preparation
- Loaded customer, order, ticket, web, and campaign data
- Created RFM features:
  - Recency
  - Frequency
  - Monetary value
- Engineered additional features:
  - Ticket count
  - Refund rate
  - Web activity
  - Campaign engagement

### 2. Modeling Approach
- Baseline Model: Logistic Regression
- Final Model: Random Forest Classifier

### 3. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

### 4. Threshold Selection
A custom threshold (0.40) was used to improve recall and capture more potential churners.


## How to Run

```bash
pip install -r requirements.txt
