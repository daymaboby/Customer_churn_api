import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

import pandas as pd

orders = pd.read_csv("data/orders.csv")
customers = pd.read_csv("data/customers (1).csv")
tickets = pd.read_csv("data/support_tickets.csv")
web = pd.read_csv("data/web_events_snapshot.csv")
campaigns = pd.read_csv("data/intervention_history.csv")
churn = pd.read_csv("data/churn_labels.csv")


orders["order_date"] = pd.to_datetime(orders["order_date"])
snapshot_date = orders["order_date"].max()

rfm = orders.groupby("customer_id").agg({
    "order_date": lambda x: (snapshot_date - x.max()).days,
    "order_id": "count",
    "gross_amount": "sum"
})
ticket = tickets.groupby("customer_id").size().rename("ticket_count")
refund = orders.groupby("customer_id")["returned"].mean().rename("refund_rate")
web_feat = web.groupby("customer_id")["product_views_30d"].mean().rename("web_activity")
campaign = campaigns.groupby("customer_id").size().rename("campaign_engagement")
X = rfm.join(ticket, how="left") \
       .join(refund, how="left") \
       .join(web_feat, how="left") \
       .join(campaign, how="left")
X = X.fillna(0)
y = churn.set_index("customer_id")["churn_next_60d"]
X, y = X.align(y, join="inner", axis=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

joblib.dump(rf_model, "model.pkl")