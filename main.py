import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import scipy.stats as stats
import numpy as np

app = FastAPI()

# CONFIGURAZIONE CRUCIALE PER LOVABLE
# Lovable girerà su un dominio diverso, quindi dobbiamo aprire le porte.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione metterai il dominio di Lovable, per ora * va bene
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    cost: float
    price: float
    mean: float
    std_dev: float

@app.get("/")
def read_root():
    return {"status": "active", "message": "Ops Backend is running"}

@app.post("/newsvendor")
def calculate_newsvendor(data: InputData):
    if data.price <= data.cost:
        return {"error": "Il prezzo deve essere maggiore del costo"}
    
    # 1. Calcoli Economici
    underage_cost = data.price - data.cost
    overage_cost = data.cost 
    critical_ratio = underage_cost / (underage_cost + overage_cost)
    
    # 2. Calcolo Quantità Ottima (Distribuzione Normale Inversa)
    optimal_q = stats.norm.ppf(critical_ratio, loc=data.mean, scale=data.std_dev)
    
    # 3. Profitto Atteso (Expected Profit) - Formula standard Newsvendor
    # Z-score standardizzato
    z = (optimal_q - data.mean) / data.std_dev
    # Standard Loss Function L(z)
    loss_z = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
    expected_sales = data.mean - (data.std_dev * loss_z)
    expected_profit = (data.price - data.cost) * expected_sales - (data.cost * (optimal_q - expected_sales)) # Semplificato (recupero 0)

    # 4. Dati per il grafico (Distribuzione)
    x = np.linspace(data.mean - 3*data.std_dev, data.mean + 3*data.std_dev, 50)
    y = stats.norm.pdf(x, loc=data.mean, scale=data.std_dev)
    chart_data = [{"x": round(float(xi), 1), "y": float(yi)} for xi, yi in zip(x, y)]

    return {
        "optimal_quantity": round(optimal_q, 2),
        "critical_ratio": round(critical_ratio, 4),
        "expected_profit": round(expected_profit, 2),
        "chart_data": chart_data
    }
