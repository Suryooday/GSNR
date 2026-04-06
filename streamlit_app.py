"""Streamlit UI for GSNR-aware optical network simulation."""

from __future__ import annotations

import logging
from typing import List

import streamlit as st

from api_service import SimulationAPI, SimulatorConfig
from logging_utils import configure_logging
from visualization import (
    plot_blocking_probability,
    plot_gsnr_vs_distance,
    plot_latency_vs_load,
    plot_topology,
)

logger = logging.getLogger(__name__)

def main() -> None:
    configure_logging()
    st.set_page_config(page_title="Optical RSA Simulator", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        [data-testid="stSidebar"] {
            background-color: #121826;
        }
        .metric-card {
            padding: 0.9rem;
            border-radius: 0.7rem;
            border: 1px solid #2A2F3A;
            background: #151B2B;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("GSNR-Aware Optical Network Simulator")

    with st.sidebar:
        st.header("Simulation Settings")
        number_of_nodes = st.slider("Number of nodes", min_value=4, max_value=40, value=12, step=1)
        input_power_mw = st.slider("Input power (mW)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        noise_figure_db = st.slider("Noise figure (dB)", min_value=3.0, max_value=8.0, value=5.0, step=0.1)
        traffic_load = st.slider("Traffic load (lambda)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
        service_rate_mu = st.slider("Service rate (mu)", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
        sim_requests = st.slider("Traffic requests", min_value=50, max_value=1000, value=250, step=50)

    signal_power_w = input_power_mw * 1e-3
    api = SimulationAPI(
        SimulatorConfig(
            number_of_nodes=number_of_nodes,
            input_power_w=signal_power_w,
            noise_figure_db=noise_figure_db,
            traffic_load_lambda=traffic_load,
            service_rate_mu=service_rate_mu,
        )
    )
    api.build()

    if api.topology is None:
        st.error("Topology initialization failed.")
        return

    nodes = api.get_nodes()
    st.subheader("Route Query")
    col_src, col_dst, col_btn = st.columns([2, 2, 1.2])
    with col_src:
        source = st.selectbox("Source", options=nodes, index=0)
    with col_dst:
        default_dst_index = 1 if len(nodes) > 1 else 0
        destination = st.selectbox("Destination", options=nodes, index=default_dst_index)
    with col_btn:
        st.write("")
        st.write("")
        compute_clicked = st.button("Compute Path", use_container_width=True)

    if source == destination:
        st.warning("Source and destination must be different.")
        compute_clicked = False

    if compute_clicked:
        result = api.compute_path(source=source, destination=destination)

        if result is None:
            st.error("No feasible RSA solution found for the selected node pair.")
        else:
            path = list(result["path"])
            snr_db = float(result["snr_db"])
            ber = float(result["ber"])
            latency_ms = float(result["latency_ms"])

            metric_cols = st.columns(4)
            metric_cols[0].metric("Path", " -> ".join(path))
            metric_cols[1].metric("GSNR (dB)", f"{snr_db:.2f}")
            metric_cols[2].metric("BER", f"{ber:.2e}")
            metric_cols[3].metric("Latency (ms)", f"{latency_ms:.3f}")

            try:
                fig_topology, _ = plot_topology(
                    api.topology.graph,
                    selected_path=path,
                    title="Topology with Selected Path",
                )
                st.pyplot(fig_topology, use_container_width=True)
            except Exception:
                logger.exception("Failed to render topology plot.")
                st.warning("Topology plot could not be rendered.")

            # GSNR vs distance: evaluate shortest routes from source to all reachable nodes.
            eval_slots = list(result["slots"])
            distances, gsnr_db_values = api.gsnr_distance_curve(source=source, slots=eval_slots)

            plot_col1, plot_col2 = st.columns(2)
            with plot_col1:
                if distances:
                    fig_gsnr, _ = plot_gsnr_vs_distance(
                        distances_km=distances,
                        gsnr_db=gsnr_db_values,
                        title="GSNR vs Distance",
                    )
                    st.pyplot(fig_gsnr, use_container_width=True)
                else:
                    st.info("Not enough data points for GSNR-distance plot.")

            # Latency/load and blocking probability from load sweep.
            load_points = [max(0.05, traffic_load * factor) for factor in (0.5, 0.8, 1.0, 1.2, 1.5)]
            avg_latency_points, blocking_points = api.traffic_sweep(
                load_points=load_points,
                requests=sim_requests,
            )

            with plot_col2:
                fig_latency, _ = plot_latency_vs_load(
                    offered_load=load_points,
                    latency_ms=avg_latency_points,
                    title="Latency vs Load",
                )
                st.pyplot(fig_latency, use_container_width=True)

            fig_blocking, _ = plot_blocking_probability(
                x_values=load_points,
                blocking_probabilities=blocking_points,
                x_label="Traffic Load (lambda)",
                title="Blocking Probability",
            )
            st.pyplot(fig_blocking, use_container_width=True)


if __name__ == "__main__":
    main()
