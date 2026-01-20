import pandas as pd
import numpy as np
import scipy.stats as stats
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from prophet import Prophet
import logging

# Configurazione
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELLI DATI MRP ---
class Item(BaseModel):
    id: str
    name: str
    lt: int       # Lead Time
    ss: int       # Safety Stock
    oh: int       # On Hand
    ls: int       # Lot Size
    type: str     # 'Finished' or 'Component'

class BomLine(BaseModel):
    parent: str
    child: str
    qty: int

class MRPInput(BaseModel):
    items: List[Item]
    bom: List[BomLine]
    mps: Dict[str, Dict[str, int]] # { "ItemA": { "1": 50, "3": 20 } }
    horizon: int = 10

# --- MODELLI DATI IMPIANTI ---
class StationInput(BaseModel):
    id: int
    name: str
    ct: float # Cycle Time
    a: float  # Availability
    p: float  # Performance
    q: float  # Quality

class PlantDesignInput(BaseModel):
    demand_year: float
    days_year: int
    shifts_day: int
    hours_shift: float
    stations: List[StationInput]

# --- 1. MRP ENGINE (Logic porting from JS to Python) ---
@app.post("/api/mrp")
def run_mrp(data: MRPInput):
    # 1. Costruzione Grafo BOM per calcolo livelli (Low-Level Code)
    G = nx.DiGraph()
    G.add_nodes_from([i.id for i in data.items])
    for rel in data.bom:
        G.add_edge(rel.parent, rel.child, qty=rel.qty)
    
    # Verifica cicli
    if not nx.is_directed_acyclic_graph(G):
        raise HTTPException(status_code=400, detail="Errore: La BOM contiene cicli infiniti.")

    # Calcolo Low Level Code (Distanza massima dalla radice)
    levels = {}
    for node in G.nodes():
        # In un DAG, il livello è la lunghezza del percorso più lungo da una "sorgente"
        # Semplificazione: Topological Sort
        levels[node] = 0 # Placeholder

    # Ordine topologico: Prima i prodotti finiti, poi i componenti
    try:
        sorted_nodes = list(nx.topological_sort(G))
    except:
        sorted_nodes = [i.id for i in data.items] # Fallback

    # Struttura risultati
    results = {}
    
    # Mappa rapida Items
    items_map = {i.id: i for i in data.items}

    # Inizializzazione griglia
    for item_id in sorted_nodes:
        item = items_map.get(item_id)
        if not item: continue

        row = []
        current_inventory = item.oh
        
        for t in range(1, data.horizon + 1):
            period_data = {
                "period": t,
                "grossReq": 0,
                "schedReceipt": 0,
                "projAvail": 0,
                "netReq": 0,
                "plannedOrderReceipt": 0,
                "plannedOrderRelease": 0
            }
            
            # 1. Gross Requirements da MPS
            mps_val = data.mps.get(item_id, {}).get(str(t), 0)
            period_data["grossReq"] += mps_val

            # 2. Gross Requirements da Fabbisogni Dipendenti (Padri)
            # Cerco chi è padre di questo item
            parents = list(G.predecessors(item_id))
            for parent in parents:
                # Prendo i rilasci del padre in questo periodo
                # BOM Qty
                edge_data = G.get_edge_data(parent, item_id)
                qty_per = edge_data['qty']
                
                # Il padre rilascia l'ordine nel periodo T. 
                # Quindi il figlio serve nel periodo T (Gross Req).
                if parent in results:
                     parent_release = results[parent][t-1]["plannedOrderRelease"]
                     period_data["grossReq"] += parent_release * qty_per
            
            # 3. Calcolo Proiezione
            prev_proj = current_inventory if t == 1 else results[item_id][t-2]["projAvail"]
            projected = prev_proj + period_data["schedReceipt"] - period_data["grossReq"]

            # 4. Netting & Lotting
            net = 0
            receipt = 0
            
            if projected < item.ss:
                net = item.ss - projected
                # Lot Sizing logic
                if item.ls <= 1: # L4L
                    receipt = net
                else: # Fixed Lot
                    receipt = np.ceil(net / item.ls) * item.ls
                
                projected = prev_proj + period_data["schedReceipt"] + receipt - period_data["grossReq"]
            
            period_data["projAvail"] = projected
            period_data["netReq"] = max(0, net)
            period_data["plannedOrderReceipt"] = receipt

            # 5. Offsetting (Lead Time)
            # Se devo ricevere al periodo T, devo rilasciare a T - LT
            # Nota: Salviamo il release nel periodo "di rilascio", ma qui stiamo iterando T "di fabbisogno".
            # Quindi dobbiamo tornare indietro nella lista dei risultati o posticipare la scrittura.
            # Metodo più semplice: Scriviamo il release nel periodo corrente T, ma logicamente appartiene a T-LT.
            # NO: La logica standard è: se release è a T=1, serve per coprire fabbisogno a T=1+LT.
            # Qui usiamo la logica inversa: Ho receipt a T, quindi ho rilasciato a T-LT.
            
            # Salviamo receipt qui. Il release lo calcoliamo "nel passato" se possibile, 
            # ma dato che iteriamo in avanti, dobbiamo scrivere il release nel bucket T-LT.
            # Per semplicità nel JSON response, calcoliamo il release per QUESTO periodo T
            # (ovvero: cosa devo rilasciare OGGI per avere roba tra LT settimane?)
            # Questo richiederebbe lookahead.
            
            # Approccio corretto per loop sequenziale:
            # Ho un planned receipt a T. Vado a scrivere +receipt nel campo 'release' del periodo T-LT.
            row.append(period_data)

        # Post-processing per i Release (Offsetting)
        for t_idx, r_data in enumerate(row):
            receipt_qty = r_data["plannedOrderReceipt"]
            if receipt_qty > 0:
                release_period_idx = t_idx - item.lt
                if release_period_idx >= 0:
                    row[release_period_idx]["plannedOrderRelease"] += receipt_qty
        
        results[item_id] = row

    return results

# --- 2. PLANT ENGINEERING ENGINE ---
@app.post("/api/plant-design")
def run_plant_design(data: PlantDesignInput):
    # Parametri Globali
    time_avail_year_hours = data.days_year * data.shifts_day * data.hours_shift
    time_avail_year_sec = time_avail_year_hours * 3600
    takt_time = time_avail_year_sec / data.demand_year if data.demand_year > 0 else 0
    
    stations_res = []
    current_input_req = data.demand_year # Iniziamo dall'output richiesto (Logica Pull invertita per calcolo scarti)
    
    # Calcolo a ritroso per definire l'input necessario (causa scarti)
    # Ma per il loop, iteriamo e basta, poi applichiamo fattore scarto
    # ATTENZIONE: Se Station 1 alimenta Station 2, Station 1 deve produrre di più se Station 2 scarta.
    # L'array arriva [Stazione1, Stazione2...]. Dobbiamo calcolare gli scarti dal fondo.
    
    # 1. Calcolo Input richiesto per ogni fase (Reverse Loop)
    inputs_required = [0] * len(data.stations)
    temp_req = data.demand_year
    for i in range(len(data.stations) - 1, -1, -1):
        s = data.stations[i]
        input_needed = temp_req / s.q # Se Q=0.9, devo produrre 100/0.9
        inputs_required[i] = input_needed
        temp_req = input_needed

    # 2. Calcolo Macchine
    for i, s in enumerate(data.stations):
        input_qty = inputs_required[i]
        
        # Ore necessarie = (Pezzi * TempoCiclo) / (OEE_factors)
        # OEE parziale (A * P) perché Q è già considerato nell'aumento dell'input
        # Time operating theoretical = Input * CT
        # Real Time needed = (Input * CT) / (A * P) 
        # Convertiamo CT (min o sec?) assumiamo minuti se non spec. Facciamo sec per coerenza col JS.
        # JS usava minuti probabilmente, ma qui standardizziamo. 
        # Assumiamo CT in MINUTI come nel tuo codice JS originale (2.0 min).
        
        hours_needed = (input_qty * s.ct) / (s.a * s.p * 60)
        machines = np.ceil(hours_needed / time_avail_year_hours)
        saturation = (hours_needed / (machines * time_avail_year_hours)) * 100 if machines > 0 else 0
        
        oee = s.a * s.p * s.q * 100
        
        stations_res.append({
            "id": s.id,
            "name": s.name,
            "input_qty": round(input_qty, 0),
            "machines": int(machines),
            "saturation": round(saturation, 1),
            "oee": round(oee, 1)
        })

    return {
        "global": {
            "takt_time": round(takt_time, 2),
            "total_hours": time_avail_year_hours
        },
        "stations": stations_res
    }

# --- 3. RELIABILITY ENGINE (RBD) ---
@app.post("/api/rbd")
def run_rbd(mtbf: float, mttr: float, n_series: int = 3, n_parallel: int = 2):
    if mtbf + mttr == 0: return {"error": "Zero division"}
    
    A = mtbf / (mtbf + mttr)
    A_series = A ** n_series
    A_parallel = 1 - (1 - A) ** n_parallel
    
    return {
        "availability_single": round(A * 100, 2),
        "availability_series": round(A_series * 100, 2),
        "availability_parallel": round(A_parallel * 100, 2)
    }

# --- Endpoint Check ---
@app.get("/")
def root():
    return {"status": "Ops Suite v2.0 Running"}
