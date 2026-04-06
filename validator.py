"""Scientific Validator Module for GSNR-Aware Routing Strategies."""

import csv
import logging
from pathlib import Path

# Base Imports
from spectrum_manager import SpectrumManager
import fiber_model
from optimizer import RSAOptimizer
from traffic_engine import TrafficEngine
from dataset_loader import load_topology

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_experiment(graph_path: str, output_csv: str) -> None:
    # Pre-flight check
    if not Path(graph_path).exists():
        logger.error(f"Cannot find benchmark graph at {graph_path}. Attempting to proceed natively using generated mock graph if necessary, but returning error initially.")
        raise FileNotFoundError(f"Missing {graph_path}")
        
    strategies = ["shortest_path", "static_gsnr", "dynamic_gsnr"]
    traffic_loads = [10.0, 50.0, 100.0, 150.0, 200.0]  # Erlangs (lambda/mu)
    
    results = []
    
    # We simulate exactly 1000 requests per run for mathematically strict reproducibility
    TOTAL_REQUESTS = 1000
    
    logger.info(f"Starting validation suite on {graph_path}")
    logger.info(f"Total combinations matrix: {len(strategies) * len(traffic_loads)}")
    
    for load in traffic_loads:
        for strategy in strategies:
            logger.info(f"--- Processing Load: {load} Erlangs | Strategy: {strategy} ---")
            
            # 1. Re-initialize fresh topology to zero-out spectrum states cleanly
            graph = load_topology(graph_path)
            
            # 2. Re-initialize manager structures natively corresponding to the empty graph bounds
            spec_manager = SpectrumManager(graph, total_slots=320)
            
            opt = RSAOptimizer(
                graph=graph,
                spectrum_manager=spec_manager,
                fiber_model=fiber_model
            )
            
            # 3. Setup Traffic Engine identically handling Load = lambda / mu parameters
            engine = TrafficEngine(
                graph=graph,
                spectrum_manager=spec_manager,
                optimizer=opt,
                arrival_rate_lambda=load,
                service_rate_mu=1.0,
                bit_rate_gbps=100.0,
                random_seed=42, # Lock random seed so requests are identical geometrically avoiding variance biases
                strategy=strategy
            )
            
            # 4. Burn the block explicitly across the timeline
            metrics = engine.run(total_requests=TOTAL_REQUESTS)
            
            # 5. Extract KPI matrix bounds securely
            run_data = {
                "strategy": strategy,
                "traffic_load_erlangs": load,
                "blocking_probability": metrics.get("blocking_probability", 0.0),
                "network_utilization_pct": metrics.get("network_utilization", 0.0) * 100.0,
                "average_latency_ms": metrics.get("average_latency_ms", 0.0),
                "average_snr_db": metrics.get("average_snr_db", 0.0),
                "total_energy_w": metrics.get("total_energy_consumption_w", 0.0),
                "energy_per_bit": metrics.get("energy_per_bit", 0.0)
            }
            results.append(run_data)
            
    # Output to standardized CSV logic safely 
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        "strategy", 
        "traffic_load_erlangs", 
        "blocking_probability", 
        "network_utilization_pct",
        "average_latency_ms",
        "average_snr_db",
        "total_energy_w",
        "energy_per_bit"
    ]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
        
    logger.info(f"Validation successful! Saved exact benchmark results explicitly to {csv_path}")

if __name__ == "__main__":
    TEST_GRAPH = "dataset/graphs/graph-rediris-wfq-0.txt"
    OUTPUT_FILE = "dataset/validation_results.csv"
    run_experiment(TEST_GRAPH, OUTPUT_FILE)
