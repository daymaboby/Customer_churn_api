# Monitoring Plan

## 1. Data Drift
Monitor changes in input features:
- Recency distribution
- Frequency changes
- Monetary value shifts

## 2. Prediction Drift
Track churn probability distribution over time.

## 3. Business Metrics
- Retention campaign conversion rate
- Revenue saved from retained customers

## 4. API Health
- Response time
- Error rate
- Timeout frequency

## 5. Retraining Trigger
Retrain model if:
- ROC-AUC drops significantly
- Data drift detected
- Business performance declines