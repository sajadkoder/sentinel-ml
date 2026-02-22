"""
Load testing script using Locust
"""
import random
from locust import HttpUser, task, between

from datetime import datetime


class FraudDetectionUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    merchant_categories = [
        "grocery", "gas_station", "restaurant", "online_shopping",
        "entertainment", "travel", "utilities", "healthcare"
    ]
    
    transaction_types = ["purchase", "withdrawal", "transfer"]
    device_types = ["mobile", "desktop", "tablet"]
    countries = ["US", "UK", "CA", "DE", "FR"]
    
    @task(10)
    def predict_single(self):
        transaction = {
            "user_id": f"user_{random.randint(1, 1000)}",
            "amount": round(random.uniform(10, 5000), 2),
            "merchant_id": f"merchant_{random.randint(1, 500)}",
            "merchant_category": random.choice(self.merchant_categories),
            "transaction_type": random.choice(self.transaction_types),
            "device_type": random.choice(self.device_types),
            "location_country": random.choice(self.countries)
        }
        
        self.client.post(
            "/api/v1/predict",
            json=transaction,
            name="/api/v1/predict"
        )
    
    @task(1)
    def predict_batch(self):
        transactions = []
        for _ in range(random.randint(5, 20)):
            transactions.append({
                "user_id": f"user_{random.randint(1, 1000)}",
                "amount": round(random.uniform(10, 5000), 2),
                "merchant_id": f"merchant_{random.randint(1, 500)}",
                "merchant_category": random.choice(self.merchant_categories),
                "transaction_type": random.choice(self.transaction_types),
                "device_type": random.choice(self.device_types),
                "location_country": random.choice(self.countries)
            })
        
        self.client.post(
            "/api/v1/predict/batch",
            json={"transactions": transactions},
            name="/api/v1/predict/batch"
        )
    
    @task(2)
    def health_check(self):
        self.client.get("/health", name="/health")
    
    @task(1)
    def model_info(self):
        self.client.get("/api/v1/model/info", name="/api/v1/model/info")
