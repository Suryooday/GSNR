"use client";

import React, { useEffect, useState, useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Sphere, Line, Text, Html } from "@react-three/drei";

// --- Custom Animated Line Component for Network Edges ---
const AnimatedEdge = ({ 
  start, 
  end, 
  sourceId,
  targetId,
  edgeInfo,
  isFailed = false,
  setHoverInfo
}: any) => {
  const lineRef = useRef<any>(null);
  const totalSlotsUsed = edgeInfo.slotsUsed;
  
  useFrame((state, delta) => {
    if (lineRef.current && lineRef.current.material) {
        if (totalSlotsUsed > 0 && !isFailed) {
            lineRef.current.material.dashOffset -= delta * 3.0; // Animate traffic
        }
    }
  });

  // Calculate congestion color: Green -> Yellow -> Red
  let color = "#404040"; // idle
  let lineWidth = 1;
  const utilization = totalSlotsUsed / 320.0;
  
  if (isFailed) {
    color = "#ef4444"; // crimson red for broken links
    lineWidth = 4;
  } else if (totalSlotsUsed > 0) {
    lineWidth = 3;
    if (utilization > 0.8) color = "#ef4444"; // High congestion
    else if (utilization > 0.5) color = "#f59e0b"; // Medium 
    else color = "#10b981"; // Low 
  }

  return (
    <group
      onPointerOver={(e) => {
          e.stopPropagation();
          setHoverInfo({
            x: e.clientX, y: e.clientY,
            title: `Link: ${sourceId} ↔ ${targetId}`,
            lines: [
                `Status: ${isFailed ? 'OFFLINE' : 'ACTIVE'}`,
                `Slots Used: ${totalSlotsUsed} / 320`,
                `Congestion: ${(utilization * 100).toFixed(1)}%`,
                `Active Tunnels: ${edgeInfo.connCount}`,
                `Avg Latency: ${edgeInfo.avgLatency} ms`,
                `Avg GSNR: ${edgeInfo.avgSnr} dB`,
            ]
          });
      }}
      onPointerOut={() => setHoverInfo(null)}
    >
      <Line
        ref={lineRef}
        points={[start, end]}
        color={color}
        lineWidth={lineWidth}
        transparent
        opacity={isFailed ? 1.0 : (totalSlotsUsed > 0 ? 0.9 : 0.3)}
        dashed={totalSlotsUsed > 0 && !isFailed}
        dashScale={5}
        dashSize={1}
        dashOffset={0}
        gapSize={1}
      />
    </group>
  );
};


export default function DigitalTwin() {
  const [topology, setTopology] = useState<any>({ nodes: [], links: [] });
  const [activeTraffic, setActiveTraffic] = useState<any[]>([]);
  const [brokenLinks, setBrokenLinks] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any>({ utilization: 0, total_energy_w: 0, energy_per_bit: 0 });
  
  const [hoverInfo, setHoverInfo] = useState<any>(null);
  
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Fetch base topology
    fetch("http://127.0.0.1:8000/topology")
      .then((res) => res.json())
      .then((data) => setTopology(data))
      .catch((err) => console.error("Could not fetch topology:", err));

    // Connect WebSocket for live traffic
    const ws = new WebSocket("ws://127.0.0.1:8000/ws/stream");
    wsRef.current = ws;

    ws.onopen = () => console.log("Connected to simulation WS");

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "add_traffic") {
        setActiveTraffic((prev) => [...prev, msg]);
      } else if (msg.type === "remove_traffic") {
        setActiveTraffic((prev) => prev.filter((t) => t.request_id !== msg.request_id));
      } else if (msg.type === "link_failure") {
        setBrokenLinks((prev) => [...prev, { source: msg.source, target: msg.target }]);
      } else if (msg.type === "link_recovery") {
        setBrokenLinks((prev) => prev.filter(l => !(l.source === msg.source && l.target === msg.target)));
      } else if (msg.type === "update_metrics") {
        setMetrics({
          utilization: msg.utilization,
          total_energy_w: msg.total_energy_w,
          energy_per_bit: msg.energy_per_bit
        });
      }
    };

    return () => ws.close();
  }, []);

  const startSimulation = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send("start");
    }
  };

  // Node position calculation
  const nodePositions: Record<string, any> = useMemo(() => {
    const positions: any = {};
    const getNodePosition = (index: number, total: number, type: string) => {
      const angle = (index / total) * Math.PI * 2;
      if (type === "spine") return [Math.cos(angle) * 3, 2, Math.sin(angle) * 3];
      return [Math.cos(angle) * 6, -2, Math.sin(angle) * 6];
    };
    
    const spines = topology.nodes.filter((n: any) => n.type === "spine");
    const leafs = topology.nodes.filter((n: any) => n.type === "leaf");

    spines.forEach((node: any, i: number) => { positions[node.id] = getNodePosition(i, spines.length, "spine"); });
    leafs.forEach((node: any, i: number) => { positions[node.id] = getNodePosition(i, leafs.length, "leaf"); });
    topology.nodes.forEach((node: any, i: number) => {
      if (!positions[node.id]) positions[node.id] = getNodePosition(i, topology.nodes.length, "leaf");
    });
    return positions;
  }, [topology]);

  // Pre-calculate Edge Load matrix natively to draw heatmaps correctly!
  const getEdgeInfo = (u: string, v: string) => {
     let slotsUsed = 0;
     let totalLatency = 0;
     let totalSnr = 0;
     let connCount = 0;
     
     activeTraffic.forEach(conn => {
         const p = conn.path;
         for (let j = 0; j < p.length - 1; j++) {
             if ((p[j] === u && p[j+1] === v) || (p[j] === v && p[j+1] === u)) {
                 slotsUsed += (conn.slots_used || 8);
                 totalLatency += (conn.latency_ms || 0);
                 totalSnr += (conn.snr_db || 0);
                 connCount += 1;
             }
         }
     });
     
     return {
         slotsUsed,
         avgLatency: connCount > 0 ? (totalLatency / connCount).toFixed(2) : 0,
         avgSnr: connCount > 0 ? (totalSnr / connCount).toFixed(2) : 0,
         connCount
     };
  };

  const checkFailed = (u: string, v: string) => {
     return brokenLinks.some(bl => (bl.source === u && bl.target === v) || (bl.source === v && bl.target === u));
  };

  return (
    <div className="w-full h-screen bg-neutral-900 flex flex-col font-sans overflow-hidden">
      {/* 2D HTML Overlays */}
      <div className="absolute top-0 left-0 z-10 p-6 flex items-center justify-between w-full pointer-events-none">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Optical Network Simulator</h1>
          <p className="text-emerald-400 font-medium tracking-wide">Industry Grade Dashboard</p>
        </div>
        <button 
          onClick={startSimulation}
          className="pointer-events-auto bg-emerald-600 hover:bg-emerald-500 transition-colors text-white px-6 py-2 rounded-full font-semibold shadow-lg shadow-emerald-900/50"
        >
          Start Live Traffic
        </button>
      </div>

      <div className="absolute bottom-6 left-6 z-10 bg-neutral-800/80 backdrop-blur-md p-5 rounded-xl border border-neutral-700 w-72">
        <h3 className="text-white text-sm font-bold uppercase tracking-wider mb-3">Green Energy Matrix</h3>
        <div className="flex justify-between items-center text-xs text-neutral-400 mb-2 border-b border-neutral-700/50 pb-2">
          <span>Active Connections</span>
          <span className="text-emerald-400 font-mono text-sm">{activeTraffic.length}</span>
        </div>
        <div className="flex justify-between items-center text-xs text-neutral-400 mb-2 border-b border-neutral-700/50 pb-2">
          <span>Total Power (W)</span>
          <span className="text-amber-400 font-mono text-sm">{metrics.total_energy_w?.toFixed(1) || 0} W</span>
        </div>
        <div className="flex justify-between items-center text-xs text-neutral-400">
          <span>Energy per Bit</span>
          <span className="text-cyan-400 font-mono text-sm">{metrics.energy_per_bit?.toFixed(4) || 0} W/Gbps</span>
        </div>
      </div>
      
      {/* Dynamic Hover Tooltip following Mouse Coordinates perfectly! */}
      {hoverInfo && (
        <div 
          className="absolute z-50 bg-black/90 p-3 rounded-md border border-neutral-600 shadow-2xl pointer-events-none text-white whitespace-nowrap transition-opacity duration-100"
          style={{ left: hoverInfo.x + 15, top: hoverInfo.y + 15 }}
        >
          <div className="font-bold text-sm border-b border-neutral-700 pb-1 mb-1 text-emerald-300">{hoverInfo.title}</div>
          {hoverInfo.lines.map((line: string, idx: number) => (
             <div key={idx} className="text-xs text-neutral-300 font-mono my-0.5">{line}</div>
          ))}
        </div>
      )}

      {/* React Three Fiber Scene */}
      <Canvas camera={{ position: [12, 10, 18], fov: 45 }}>
        <color attach="background" args={["#111111"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 20, 10]} intensity={1.5} />
        <pointLight position={[-10, -10, -10]} intensity={0.5} />
        <OrbitControls makeDefault enableDamping dampingFactor={0.05} />

        <group>
          {topology.nodes.map((node: any) => {
            const pos = nodePositions[node.id];
            const isSpine = node.id.includes("S");
            return (
              <group 
                key={node.id} 
                position={pos}
                onPointerOver={(e) => {
                    e.stopPropagation();
                    setHoverInfo({
                      x: e.clientX, y: e.clientY,
                      title: `Optical Node: ${node.id}`,
                      lines: [
                          `Type: ${isSpine ? "Core Spine Router" : "Edge Leaf Matrix"}`,
                          `Coordinates: [${pos.map((p:any) => p.toFixed(1)).join(', ')}]`
                      ]
                    });
                }}
                onPointerOut={() => setHoverInfo(null)}
              >
                <Sphere args={[isSpine ? 0.4 : 0.3, 32, 32]}>
                  <meshStandardMaterial 
                    color={isSpine ? "#3b82f6" : "#6366f1"} 
                    roughness={0.1}
                    metalness={0.9}
                    emissive={isSpine ? "#1e3a8a" : "#312e81"}
                    emissiveIntensity={0.5}
                  />
                </Sphere>
                <Text position={[0, -0.6, 0]} fontSize={0.3} color="#a3a3a3" anchorX="center" anchorY="middle" font="https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfMZhrib2Bg-4.ttf">
                  {node.id}
                </Text>
              </group>
            );
          })}

          {topology.links.map((link: any, i: number) => {
            const start = nodePositions[link.source];
            const end = nodePositions[link.target];
            if (!start || !end) return null;

            const edgeInfo = getEdgeInfo(link.source, link.target);
            const isFailed = checkFailed(link.source, link.target);

            return (
              <AnimatedEdge 
                 key={`link-${i}`}
                 start={start}
                 end={end}
                 sourceId={link.source}
                 targetId={link.target}
                 edgeInfo={edgeInfo}
                 isFailed={isFailed}
                 setHoverInfo={setHoverInfo}
              />
            );
          })}
        </group>
      </Canvas>
    </div>
  );
}
