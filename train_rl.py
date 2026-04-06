"""
Reinforcement Learning Execution Engine over Optical Graphs.
"""
import logging
import torch
from pathlib import Path
from dataset_loader import load_topology
from spectrum_manager import SpectrumManager
import fiber_model
from optimizer import RSAOptimizer
from traffic_engine import TrafficEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_agent(graph_path: str, eps: int = 2000, model_out: str = "ai/trained_dqn.pth"):
    logger.info(f"Bootstrapping DQN Training exactly {eps} interactions on {graph_path}")
    
    graph = load_topology(graph_path)
    spec_manager = SpectrumManager(graph, total_slots=320)
    
    opt = RSAOptimizer(
        graph=graph,
        spectrum_manager=spec_manager,
        fiber_model=fiber_model
    )
    
    engine = TrafficEngine(
        graph=graph,
        spectrum_manager=spec_manager,
        optimizer=opt,
        arrival_rate_lambda=50.0, # Target 50 Erlangs
        service_rate_mu=1.0,
        bit_rate_gbps=100.0,
        strategy="rl"
    )
    
    # Pre-train Engine run
    metrics = engine.run(total_requests=eps)
    
    logger.info(f"Training Complete! Final blocking rate: {metrics['blocking_probability']:.2%}")
    logger.info(f"Network bounded to utilization: {metrics['network_utilization']:.2%}")
    
    if engine.rl_agent is not None:
        torch.save(engine.rl_agent.policy_net.state_dict(), model_out)
        logger.info(f"DQN Trained PyTorch bounds exported seamlessly to: {model_out}")

if __name__ == "__main__":
    TRAIN_GRAPH = "dataset/graphs/graph-rediris-wfq-0.txt"
    OUTPUT_MODEL = "ai/trained_dqn.pth"
    # Create dir if not exist
    Path("ai").mkdir(exist_ok=True)
    train_agent(TRAIN_GRAPH, eps=5000, model_out=OUTPUT_MODEL)
