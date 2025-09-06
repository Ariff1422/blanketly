import math
from typing import List, Union, Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChallengeInput(BaseModel):
    goods: List[str]
    ratios: List[List[Union[float, int]]]

def build_rate_matrix(goods: List[str], ratios: List[List[Union[float, int]]]) -> List[List[float]]:
    """Build the exchange rate matrix from the input ratios."""
    n = len(goods)
    rates = [[0.0] * n for _ in range(n)]
    
    for ratio in ratios:
        frm, to, rate = int(ratio[0]), int(ratio[1]), float(ratio[2])
        rates[frm][to] = rate
    
    return rates

def find_best_arbitrage_cycle(rates: List[List[float]]) -> Optional[List[int]]:
    """Find the arbitrage cycle with the highest gain using exhaustive search."""
    n = len(rates)
    best_gain = 1.0
    best_cycle = None
    
    # Check all cycles of length 3
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and k != i:
                    if rates[i][j] > 0 and rates[j][k] > 0 and rates[k][i] > 0:
                        gain = rates[i][j] * rates[j][k] * rates[k][i]
                        if gain > best_gain:
                            best_gain = gain
                            best_cycle = [i, j, k, i]
    
    # Check all cycles of length 4
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if len(set([i, j, k, l])) == 4:  # All different vertices
                        if (rates[i][j] > 0 and rates[j][k] > 0 and 
                            rates[k][l] > 0 and rates[l][i] > 0):
                            gain = rates[i][j] * rates[j][k] * rates[k][l] * rates[l][i]
                            if gain > best_gain:
                                best_gain = gain
                                best_cycle = [i, j, k, l, i]
    
    # Check cycles of length 5 for larger graphs
    if n >= 5:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        for m in range(n):
                            if len(set([i, j, k, l, m])) == 5:  # All different
                                if (rates[i][j] > 0 and rates[j][k] > 0 and 
                                    rates[k][l] > 0 and rates[l][m] > 0 and rates[m][i] > 0):
                                    gain = rates[i][j] * rates[j][k] * rates[k][l] * rates[l][m] * rates[m][i]
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_cycle = [i, j, k, l, m, i]
    
    return best_cycle

def bellman_ford_arbitrage(rates: List[List[float]]) -> Optional[List[int]]:
    """Alternative implementation using Bellman-Ford for negative cycle detection."""
    n = len(rates)
    
    # Convert to weight matrix with -log(rate)
    weights = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if rates[i][j] > 0:
                weights[i][j] = -math.log(rates[i][j])
    
    # Bellman-Ford with all distances initialized to 0
    dist = [0.0] * n
    pred = [-1] * n
    
    # Relax edges n times
    for iteration in range(n):
        for u in range(n):
            for v in range(n):
                if weights[u][v] != math.inf:
                    if dist[u] + weights[u][v] < dist[v] - 1e-10:
                        dist[v] = dist[u] + weights[u][v]
                        pred[v] = u
    
    # Check for negative cycles
    for u in range(n):
        for v in range(n):
            if weights[u][v] != math.inf:
                if dist[u] + weights[u][v] < dist[v] - 1e-10:
                    # Found negative cycle, reconstruct it
                    # Walk back n steps to ensure we're in the cycle
                    node = v
                    for _ in range(n):
                        if pred[node] != -1:
                            node = pred[node]
                    
                    # Collect cycle
                    cycle = [node]
                    current = pred[node]
                    while current != node and current != -1:
                        cycle.append(current)
                        current = pred[current]
                    
                    if current == node:
                        cycle.append(node)  # Close the cycle
                        cycle.reverse()  # Make it forward order
                        return cycle
    
    return None

def calculate_cycle_gain(cycle: List[int], rates: List[List[float]]) -> float:
    """Calculate the total gain for a given cycle."""
    if not cycle or len(cycle) < 2:
        return 1.0
    
    gain = 1.0
    for i in range(len(cycle) - 1):
        u, v = cycle[i], cycle[i + 1]
        if rates[u][v] > 0:
            gain *= rates[u][v]
        else:
            return 1.0  # Invalid cycle
    
    return gain

@app.post("/The-Ink-Archive")
async def process_trades(data: List[ChallengeInput]):
    """Process arbitrage challenges and return the best trading cycles."""
    results = []
    
    for challenge in data:
        # Build rate matrix
        rates = build_rate_matrix(challenge.goods, challenge.ratios)
        
        # Find the best arbitrage cycle
        cycle = find_best_arbitrage_cycle(rates)
        
        # If exhaustive search fails, try Bellman-Ford
        if cycle is None:
            cycle = bellman_ford_arbitrage(rates)
        
        if cycle is None:
            # No arbitrage opportunity found
            results.append({"path": [], "gain": 0.0})
        else:
            # Convert cycle indices to good names
            path = [challenge.goods[i] for i in cycle]
            
            # Calculate gain
            gain = calculate_cycle_gain(cycle, rates)
            
            # Convert to percentage gain (subtract 1 and multiply by 100)
            percentage_gain = (gain - 1.0) * 100.0
            
            results.append({
                "path": path,
                "gain": percentage_gain
            })
    
    return results

@app.get("/")
async def root():
    return {"message": "Ink Archive Arbitrage Detection Service"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8060)
