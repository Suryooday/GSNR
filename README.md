# Optical Network Digital Twin 🌐⚡️

An industry-grade, real-time Digital Twin and simulator for modern **Routing and Spectrum Assignment (RSA)** in elastic optical networks. Built with a robust Python/FastAPI backend and a highly interactive 3D React-Three-Fiber frontend.

## 🚀 What We Have Built
1. **Real-Time Physics Engine (`traffic_engine.py`)**: A purely time-series event-driven simulation loop loading standard dataset CSV traces. It dynamically parses arrivals, departures, bandwidth requirements, and physical holding times.
2. **True Multi-Objective Optimizer (`optimizer.py`)**: Integrates dynamic distance, exact hop latency, physical Signal-to-Noise ranges (GSNR), and absolute Watts Power Consumption simultaneously to score optimal paths.
3. **Deep Q-Network (DQN) AI Integration**: Pluggable PyTorch RL models analyzing network states globally rather than relying entirely on isolated algorithmic heuristics.
4. **Autonomous Chaos Mechanisms**: An internal event loop simulating completely unpredictable physical link failures (`link_failure` and `link_recovery`) and deploying internal autonomic reroutes directly saving active connections!
5. **Interactive 3D Dashboard**: A modern Web-GL Next.js frontend rendering dynamic Congestion Matrix Heatmaps, animated traffic flows, live Green Energy constraints (Total Watts / Energy-Per-Bit), and absolute interactive Raycasted tooltips displaying live latencies and GSNRs per Edge!

---

## ⚙️ How It Works
1. **The Backend**: Powered by `FastAPI`, the system creates an event-driven physics simulation operating continuously over an `async` loop. Using `networkx` paired with an internal `LRU_Cache` engine, it geometrically binds physical topologies.
2. **The Optimizer**: When traffic arrives, the internal physics bounds deploy multi-objective mapping mapping K-Shortest simple paths over constraints, checking continuous slot capabilities locally before deploying the payload.
3. **The WebSockets**: Successful routes, blocks, failed edges, and absolute energy thresholds are encoded live into a JSON data stream traversing via WebSocket endpoints (`/ws/stream`) to all connected interfaces.
4. **The Frontend**: The React Application mounts physical spheres representing core nodes mapped exactly to topology configurations. Array intercepts map continuous overlapping pathways rendering dynamic color gradients natively (Green -> Yellow -> Red) mirroring the physical saturation load per fiber link immediately! 
5. **Validation Testing (`validator.py`)**: Can be run globally sweeping load across varying Erlangs to plot absolute analytics output natively into standardized CSV graphs benchmarking distinct behaviors over thousands of iterations.

---

## ⚡️ Novelty: How This Differs From Traditional Systems

### 1. "Green Networking" (Conditional Marginal Energy Modeling) 
**Traditional Approach**: Simulators generally calculate path costs simply via distance (e.g., shortest-path mapping mapping generic fixed metrics). 
**Our System**: Deploys an actual physical hardware mapping architecture! Amplifiers lying on a sleeping (Idle) edge possess heavy 'spin-up' costs compared to active fibers (e.g., jumping from `10W Idle` -> `30W Active`). Our internal objective dynamically adjusts energy scales organically creating a "Highway" paradigm! The router natively learns to heavily cluster data payloads onto already energized links, keeping large segments of the network isolated and *asleep*, drastically dropping absolute global physical footprint computationally!

### 2. Dynamic Continuous GSNR Interference
**Traditional Approach**: Simple linear assumptions bounding channel capabilities generically over a fixed reach. 
**Our System**: Natively surveys contiguous frequencies analyzing Non-Linear Interference bounds factoring adjacent slot interactions directly influencing signal-to-noise margins seamlessly restricting modulation constraints realistically dynamically instead of statistically!

### 3. Digital Twin Observability (Live State) 
**Traditional Approach**: Most simulators run batch CLI operations taking in arrays and spitting out end-state percentages completely blinding researchers during the process.
**Our System**: We built an absolute Live Digital Twin reflecting precise geometric interactions in 3D natively over real-time async websockets gracefully catching failures mimicking a true telecommunications operations dashboard natively. 

## 🛠 Usage
*   **Run Server**: `uvicorn main:app --reload`
*   **Run Frontend**: `npm run dev`
*   **Run Validation Sets**: `python validator.py`
