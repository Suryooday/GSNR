import asyncio
import csv
import logging
import os
import psutil
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api_service import SimulationAPI, SimulatorConfig
from traffic_engine import TrafficEngine
from logging_utils import configure_logging

configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GSNR-Aware Optical Network Simulator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = SimulationAPI(SimulatorConfig())
api.build()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()


@app.get("/topology")
def get_topology():
    if not api.topology:
        api.build()
    
    nodes = []
    links = []
    
    for n in api.topology.graph.nodes():
        nodes.append({"id": n, "type": api.topology.graph.nodes[n].get("type", "unknown")})
        
    for u, v in api.topology.graph.edges():
        links.append({"source": u, "target": v, "length_km": api.topology.graph[u][v].get("length_km", 0)})
        
    return {"nodes": nodes, "links": links}


class OptimizeRequest(BaseModel):
    source: str
    destination: str
    bit_rate: float = 100.0

@app.post("/optimize")
def optimize_path(req: OptimizeRequest):
    result = api.compute_path(req.source, req.destination)
    if result:
        return {
            "success": True,
            "path": result["path"],
            "modulation": result["modulation"],
            "gsnr_linear": result["gsnr_linear"],
            "latency_ms": result["latency_ms"],
            "power_w": result["power_w"],
            "score": result["score"]
        }
    return {"success": False, "message": "No feasible path found"}


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if data == "start":
                    logger.info("Real-time live simulation sweep engaged over websockets.")
                    asyncio.create_task(run_live_simulation())
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket execution cycle encountered internal disruption: {str(e)}")
    except WebSocketDisconnect:
        logger.info("WebSocket natively disconnected from stream.")
        manager.disconnect(websocket)
        
@app.get("/system/health")
def api_system_health():
    """Returns absolute production statistics parsing runtime metrics directly natively."""
    process = psutil.Process(os.getpid())
    return {
        "status": "healthy",
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "active_threads": process.num_threads(),
        "graph_nodes": len(api.topology.graph.nodes()),
        "graph_edges": len(api.topology.graph.edges()),
        "websocket_active_clients": len(manager.active_connections)
    }

async def run_live_simulation():
    import random
    import heapq
    from traffic_engine import ActiveConnection
    
    # Load dataset if available, otherwise just run a few random
    dataset = []
    try:
        import csv
        with open('sample_trace.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset.append(row)
    except FileNotFoundError:
        pass

    engine = TrafficEngine(
        graph=api.topology.graph,
        spectrum_manager=api.spectrum_manager,
        optimizer=api.optimizer,
        dataset=dataset if dataset else None
    )
    
    # Enable continuous metrics calculation
    def get_network_metrics():
        utilization = engine._current_utilization()
        
        # Calculate Absolute Wattage Draw
        total_energy_w = 0.0
        # 1. Amplifiers
        for u, v in api.topology.graph.edges():
            edge_len = float(api.topology.graph[u][v].get("length_km", 80.0))
            num_amps = max(0, int(edge_len / 80.0))
            is_active = any(api.topology.graph[u][v].get("occupied_slots", []))
            
            amp_draw = 30.0 if is_active else 10.0
            total_energy_w += (num_amps * amp_draw)
            
        # 2. Transponders
        total_energy_w += (len(active_connections) * 200.0) 
        
        # Energy Efficiency
        total_gbps = sum(conn.holding_time for conn in active_connections.values()) # approximation using scale
        energy_per_bit = (total_energy_w / total_gbps) if total_gbps > 0 else 0.0
        
        return {
            "utilization": utilization, 
            "total_energy_w": total_energy_w,
            "energy_per_bit": energy_per_bit
        }
        
    total_requests = len(dataset) if dataset else 500
    current_time = 0.0
    
    if dataset and len(dataset) > 0:
        next_arrival_time = float(dataset[0].get("time", 0.0))
    else:
        next_arrival_time = engine._exp_sample(engine.arrival_rate_lambda)
        
    # Chaos Config
    mean_time_between_failures = 10.0 # Virtual time units
    next_failure_time = current_time + engine._exp_sample(1.0 / mean_time_between_failures)
        
    arrivals_seen = 0
    event_queue = []
    event_counter = 0 # break ties in heapq
    active_connections = {} # dict of request_id: Connection
    
    # Helper to push to queue avoiding tie breaks on objects
    def push_event(t, event_type, payload):
        nonlocal event_counter
        event_counter += 1
        heapq.heappush(event_queue, (t, event_counter, event_type, payload))
        
    def path_uses_edge(path, u, v):
        for k in range(len(path)-1):
            if (path[k] == u and path[k+1] == v) or (path[k] == v and path[k+1] == u):
                return True
        return False

    while arrivals_seen < total_requests or event_queue or current_time <= next_arrival_time:
        next_departure_time = event_queue[0][0] if event_queue else float("inf")
        
        # Determine next event
        times = [
            (next_arrival_time, "arrival", None),
            (next_departure_time, "queue_event", None),
            (next_failure_time, "failure", None)
        ]
        times.sort(key=lambda x: x[0])
        current_time, event_type, _ = times[0]
        
        if event_type == "arrival":
            arrivals_seen += 1
            request_id = arrivals_seen
            
            if dataset:
                req_data = dataset[engine.dataset_index]
                engine.dataset_index += 1
                src = str(req_data["src"])
                dst = str(req_data["dst"])
                holding_time = float(req_data.get("holding_time", 1.0))
                req_bit_rate = float(req_data.get("bit_rate", 100.0))
            else:
                src, dst = engine._random_src_dst()
                holding_time = engine._exp_sample(engine.service_rate_mu)
                req_bit_rate = engine.bit_rate_gbps
                
            # Use optimizer without touching TrafficEngine's inner queue since we bypassed it entirely.
            result = engine._run_optimizer(source=src, destination=dst)
            if result:
                path = list(result["path"])
                release_time = current_time + holding_time
                conn = ActiveConnection(
                    request_id=request_id,
                    path=path,
                    slots=list(result["slots"]),
                    release_time=release_time,
                    latency_ms=float(result.get("latency_ms", 0.0)),
                )
                conn.source = src 
                conn.destination = dst
                conn.holding_time = holding_time
                
                active_connections[request_id] = conn
                push_event(release_time, "departure", request_id)
                
                await manager.broadcast({
                    "type": "add_traffic",
                    "request_id": request_id,
                    "path": path,
                    "color": "#00ff00",
                    "latency_ms": float(result.get("latency_ms", 0.0)),
                    "snr_db": float(result.get("snr_db", 0.0)),
                    "slots_used": len(list(result.get("slots", [])))
                })
            else:
                await manager.broadcast({
                    "type": "block",
                    "src": src,
                    "dst": dst
                })

            if dataset:
                if engine.dataset_index < len(dataset):
                    next_arrival_time = float(dataset[engine.dataset_index].get("time", current_time))
                else:
                    next_arrival_time = float('inf')
            else:
                next_arrival_time = current_time + engine._exp_sample(engine.arrival_rate_lambda)
                
        elif event_type == "queue_event":
            _, _, q_type, payload = heapq.heappop(event_queue)
            
            if q_type == "departure":
                request_id = payload
                conn = active_connections.pop(request_id, None)
                if conn is not None:
                    engine.spectrum_manager.release_slots(conn.path, conn.slots)
                    await manager.broadcast({
                        "type": "remove_traffic",
                        "request_id": request_id,
                        "path": conn.path
                    })
                    
            elif q_type == "recovery":
                u, v, saved_data = payload
                # Restore edge
                api.topology.graph.add_edge(u, v, **saved_data)
                await manager.broadcast({
                    "type": "link_recovery",
                    "source": u,
                    "target": v
                })
                
        elif event_type == "failure":
            edges = list(api.topology.graph.edges(data=True))
            if edges:
                u, v, edge_data = random.choice(edges)
                
                # Disconnect logic
                api.topology.graph.remove_edge(u, v)
                await manager.broadcast({
                    "type": "link_failure",
                    "source": u,
                    "target": v
                })
                
                # Schedule recovery (5.0s virtual time)
                push_event(current_time + 5.0, "recovery", (u, v, dict(edge_data)))
                
                # Sever active connections using this edge
                dropped_reqs = []
                for req_id, conn in list(active_connections.items()):
                    if path_uses_edge(conn.path, u, v):
                        # Graceful dismount
                        engine.spectrum_manager.release_slots(conn.path, conn.slots)
                        active_connections.pop(req_id)
                        dropped_reqs.append(conn)
                        await manager.broadcast({
                            "type": "remove_traffic",
                            "request_id": req_id,
                            "path": conn.path
                        })
                        
                # Autonomous Reroute Trial
                for failed_conn in dropped_reqs:
                    r_result = engine._run_optimizer(source=failed_conn.source, destination=failed_conn.destination)
                    if r_result:
                        new_path = list(r_result["path"])
                        # Retain original release time to honor lifecycle
                        new_conn = ActiveConnection(
                            request_id=failed_conn.request_id,
                            path=new_path,
                            slots=list(r_result["slots"]),
                            release_time=failed_conn.release_time,
                            latency_ms=float(r_result.get("latency_ms", 0.0)),
                        )
                        new_conn.source = failed_conn.source
                        new_conn.destination = failed_conn.destination
                        active_connections[new_conn.request_id] = new_conn
                        
                        await manager.broadcast({
                            "type": "add_traffic",
                            "request_id": new_conn.request_id,
                            "path": new_path,
                            "color": "#ffff00", # Yellow for Rerouted
                            "rerouted": True,
                            "latency_ms": float(r_result.get("latency_ms", 0.0)),
                            "snr_db": float(r_result.get("snr_db", 0.0)),
                            "slots_used": len(list(r_result.get("slots", [])))
                        })
                    else:
                        await manager.broadcast({
                            "type": "block",
                            "src": failed_conn.source,
                            "dst": failed_conn.destination,
                            "reason": "reroute_failed"
                        })
            
            # Schedule next global failure
            next_failure_time = current_time + engine._exp_sample(1.0 / mean_time_between_failures)
            
        # Real time sync:
        metrics = get_network_metrics()
        await manager.broadcast({
            "type": "update_metrics",
            "utilization": metrics["utilization"],
            "total_energy_w": metrics["total_energy_w"],
            "energy_per_bit": metrics["energy_per_bit"]
        })
        await asyncio.sleep(0.1) # smooth loop tick

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
