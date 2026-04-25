# ============================================================
# COMPLETE FIXED STREAMLIT UI
# Save this as app.py and run: streamlit run app.py
#
# BEFORE RUNNING:
# 1. Run the complete training notebook in Colab
# 2. Note the INPUT_SIZE it prints
# 3. Note the HEAVY_TRAFFIC_THRESHOLD it prints
# 4. Put all 4 files in models/ folder:
#      models/best_traffic_model.pt
#      models/uncert_gru_model.pt
#      models/scaler.pkl
#      models/default_sequence.npy
# 5. Update INPUT_SIZE and HEAVY_TRAFFIC_THRESHOLD below
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import folium_static

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(layout="wide")
st.title("Smart Traffic Routing Assistant")
st.write(
    "This system predicts upcoming traffic levels in Madrid, "
    "estimates prediction uncertainty, and suggests adaptive "
    "route replanning when congestion is likely."
)

# ============================================================
# SET THESE TWO VALUES FROM YOUR COLAB OUTPUT
# The training notebook prints both values clearly
# ============================================================
INPUT_SIZE               = 12   # confirmed from Colab
HEAVY_TRAFFIC_THRESHOLD  = 520  # confirmed from Colab p75

SEQ_LEN = 12

# ============================================================
# MODEL DEFINITIONS
# These must exactly match your Colab training code
# ============================================================
class TunableLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        if num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUUncertaintyModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.3):
        super().__init__()
        self.gru     = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


# ============================================================
# LOAD MODELS — cached so they only load once
# ============================================================
@st.cache_resource
def load_all():
    scaler = joblib.load("models/scaler.pkl")

    traffic_model = TunableLSTMModel(INPUT_SIZE)
    traffic_model.load_state_dict(
        torch.load("models/best_traffic_model.pt", map_location="cpu")
    )
    traffic_model.eval()

    uncert_model = GRUUncertaintyModel(INPUT_SIZE)
    uncert_model.load_state_dict(
        torch.load("models/uncert_gru_model.pt", map_location="cpu")
    )
    # No .eval() — dropout must stay ready for MC Dropout

    # Real 12-row sequence saved from training data
    default_seq = np.load("models/default_sequence.npy")

    return scaler, traffic_model, uncert_model, default_seq


try:
    scaler, traffic_model, uncert_model, default_seq = load_all()
except Exception as e:
    st.error(
        f"Failed to load models: {e}\n\n"
        "Make sure all 4 files are in the models/ folder:\n"
        "  best_traffic_model.pt\n"
        "  uncert_gru_model.pt\n"
        "  scaler.pkl\n"
        "  default_sequence.npy"
    )
    st.stop()


# ============================================================
# MC DROPOUT UNCERTAINTY FUNCTION
# ============================================================
def mc_dropout_uncertainty(model, X, samples=30):
    """
    Runs model 30 times with dropout active.
    Returns 10th and 90th percentile of predictions.
    The spread = prediction uncertainty.
    """
    model.train()  # CRITICAL — keeps dropout active
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            pred = model(X).squeeze().cpu().numpy()
            # Handle both scalar and array outputs
            if np.isscalar(pred):
                preds.append(float(pred))
            else:
                preds.append(float(pred.mean()))
    preds = np.array(preds)
    lower = float(np.percentile(preds, 10))
    upper = float(np.percentile(preds, 90))
    return lower, upper


# ============================================================
# MADRID NAMED LOCATIONS
# User picks origin and destination from these
# Routes will visibly differ between distant pairs
# ============================================================
LOCATIONS = {
    "Puerta del Sol (City Centre)": (40.4168, -3.7038),
    "Atocha Station":               (40.4065, -3.6907),
    "Bernabeu Stadium":             (40.4531, -3.6883),
    "Retiro Park":                  (40.4153, -3.6844),
    "Barajas Airport":              (40.4719, -3.5626),
    "Casa de Campo":                (40.4200, -3.7474),
    "Salamanca District":           (40.4270, -3.6780),
    "Vallecas":                     (40.3840, -3.6520),
}


# ============================================================
# USER INPUTS — SECTION 1: CONDITIONS
# ============================================================
st.subheader("Step 1 — Enter Current Road and Weather Conditions")

col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("Temperature (°C)",   -5.0, 40.0, 20.0)
    rain        = st.slider("Rainfall (mm)",        0.0, 50.0,  0.0)
    wind        = st.slider("Wind Speed (km/h)",    0.0, 20.0,  5.0)
with col2:
    hour  = st.slider("Hour of Day",            0,  23, 12)
    lanes = st.slider("Number of Lanes",        1,   6,  2)
    speed = st.slider("Speed Limit (km/h)",    30, 120, 60)


# ============================================================
# USER INPUTS — SECTION 2: ROUTE SELECTION
# ============================================================
st.subheader("Step 2 — Select Your Route")
st.write(
    "Choose an origin and destination. "
    "The system will show how the route changes under congestion."
)

col3, col4 = st.columns(2)
with col3:
    origin_name = st.selectbox(
        "Origin",
        list(LOCATIONS.keys()),
        index=0
    )
with col4:
    # Remove origin from destination options
    dest_options = [k for k in LOCATIONS.keys() if k != origin_name]
    dest_name = st.selectbox(
        "Destination",
        dest_options,
        index=1
    )

origin_coords = LOCATIONS[origin_name]
dest_coords   = LOCATIONS[dest_name]


# ============================================================
# PREDICT BUTTON
# ============================================================
if st.button("Predict Traffic and Plan Route", type="primary"):

    # ----------------------------------------------------------
    # BUILD INPUT SEQUENCE
    # Uses real 12-row sequence from training data as base
    # Replaces the last row (most recent timestep) with
    # the user's current conditions
    # This is correct — model sees 11 real historical rows
    # plus the user's current conditions as row 12
    # ----------------------------------------------------------
    sequence = default_seq.copy().astype(np.float32)  # shape: (12, INPUT_SIZE)

    user_row = sequence[-1].copy()

    # Map user inputs to correct column indices
    # These match the column order from your training notebook
    # If your Colab printed a different order, update these numbers
    # The Colab notebook prints: "index 0 = hour_sin" etc
    user_row[0] = float(np.sin(2 * np.pi * hour / 24))  # hour_sin
    user_row[1] = float(np.cos(2 * np.pi * hour / 24))  # hour_cos
    # index 2 = week_day — left as real value from sequence
    # index 3 = latitude — left as real value
    # index 4 = longitude — left as real value
    user_row[5] = float(wind)
    user_row[6] = float(rain)
    user_row[7] = float(temperature)
    user_row[8] = float(lanes)
    user_row[9] = float(speed)
    # remaining indices = highway/oneway one-hot — left as real values

    sequence[-1] = user_row

    # Convert to tensor shape (1, 12, INPUT_SIZE)
    X = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

    # ----------------------------------------------------------
    # TRAFFIC PREDICTION
    # ----------------------------------------------------------
    with torch.no_grad():
        mean = float(traffic_model(X).squeeze().item())
    mean = max(mean, 0.0)

    # ----------------------------------------------------------
    # UNCERTAINTY ESTIMATION
    # ----------------------------------------------------------
    lower, upper = mc_dropout_uncertainty(uncert_model, X)
    lower        = max(lower, 0.0)
    upper        = max(upper, lower)  # ensure upper >= lower
    uncertainty  = (upper - lower) / 2.0

    # ----------------------------------------------------------
    # DISPLAY PREDICTION RESULTS
    # ----------------------------------------------------------
    st.subheader("Traffic Prediction Results")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted Traffic",    f"{mean:.0f} vehicles / 15 min")
    with m2:
        st.metric("Expected Range",       f"{lower:.0f} – {upper:.0f}")
    with m3:
        st.metric("Prediction Uncertainty", f"±{uncertainty:.0f} vehicles")

    st.write("**Uncertainty bar** — fuller bar means prediction is less reliable:")
    progress_value = float(min(1.0, uncertainty / max(mean, 1)))
    st.progress(progress_value)

    # ----------------------------------------------------------
    # DECISION LOGIC
    # Threshold comes from actual data distribution (p75)
    # not an arbitrary hardcoded number
    # ----------------------------------------------------------
    heavy_traffic    = mean > HEAVY_TRAFFIC_THRESHOLD
    high_uncertainty = uncertainty > 0.35 * mean

    if heavy_traffic or high_uncertainty:
        reason = []
        if heavy_traffic:
            reason.append(f"predicted traffic ({mean:.0f}) exceeds threshold ({HEAVY_TRAFFIC_THRESHOLD})")
        if high_uncertainty:
            reason.append(f"uncertainty (±{uncertainty:.0f}) is high relative to prediction")
        st.error(
            f"⚠ Congestion likely — {' and '.join(reason)}. "
            "Alternative route recommended."
        )
        replan = True
    else:
        st.success(
            f"✓ Traffic stable — predicted {mean:.0f} vehicles/15 min "
            f"with uncertainty ±{uncertainty:.0f}. Current route is fine."
        )
        replan = False

    # ----------------------------------------------------------
    # ROUTE VISUALISATION
    # This section is completely rewritten
    # ----------------------------------------------------------
    st.subheader("Route Visualisation")

    # Show what the map will display
    if replan:
        st.info(
            "🔴 Congestion detected. "
            "Green = original fastest route. "
            "Red = suggested reroute avoiding main roads."
        )
    else:
        st.info("🟢 Traffic stable. Showing fastest route in green.")

    # Calculate midpoint for map centre
    mid_lat = (origin_coords[0] + dest_coords[0]) / 2
    mid_lon = (origin_coords[1] + dest_coords[1]) / 2

    # Download road network
    # dist=5000 gives 5km radius from midpoint
    # covers all Madrid location pairs comfortably
    with st.spinner("Loading Madrid road network — this takes 10-20 seconds..."):
        try:
            G = ox.graph_from_point(
                (mid_lat, mid_lon),
                dist=5000,
                network_type="drive"
            )
        except Exception as e:
            st.error(
                f"Could not load road network: {e}. "
                "Check your internet connection and try again."
            )
            st.stop()

    # Find nearest real road graph nodes to chosen locations
    # FIX: replaces broken nodes[100] / nodes[-100]
    try:
        origin_node = ox.distance.nearest_nodes(
            G,
            X=float(origin_coords[1]),  # longitude
            Y=float(origin_coords[0])   # latitude
        )
        dest_node = ox.distance.nearest_nodes(
            G,
            X=float(dest_coords[1]),
            Y=float(dest_coords[0])
        )
    except Exception as e:
        st.error(f"Could not find road nodes near selected locations: {e}")
        st.stop()

    # Safety check
    if origin_node == dest_node:
        st.warning(
            "The selected locations map to the same road junction. "
            "Please choose locations further apart."
        )
        st.stop()

    # Apply traffic-based penalties to road edges
    # FIX: stronger tiered penalties so route actually changes
    # when congestion is predicted
    for u, v, k, data in G.edges(keys=True, data=True):
        base         = float(data.get("length", 1.0))
        highway_type = str(data.get("highway", ""))

        if replan:
            # Heavy penalties on main roads forces algorithm
            # to prefer quieter side streets
            if "motorway" in highway_type or "trunk" in highway_type:
                penalty = 1.0 + (mean / 80.0)    # very heavy
            elif "primary" in highway_type:
                penalty = 1.0 + (mean / 250.0)   # heavy
            elif "secondary" in highway_type:
                penalty = 1.0 + (mean / 600.0)   # moderate
            else:
                penalty = 1.0 + (mean / 2000.0)  # side roads barely penalised
        else:
            penalty = 1.0  # no congestion — pure distance routing

        data["traffic_weight"] = base * penalty
        data["base_weight"]    = base

    # Compute both routes
    try:
        # Normal route — shortest by distance
        normal_route = nx.shortest_path(
            G, origin_node, dest_node, weight="base_weight"
        )
        # Replanned route — shortest avoiding penalised main roads
        replan_route = nx.shortest_path(
            G, origin_node, dest_node, weight="traffic_weight"
        )
    except nx.NetworkXNoPath:
        st.error(
            "No connected route found between these locations. "
            "The road network download may not cover both points. "
            "Try locations closer together."
        )
        st.stop()
    except Exception as e:
        st.error(f"Routing failed: {e}")
        st.stop()

    # Convert node IDs to lat/lon coordinates
    normal_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in normal_route]
    replan_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in replan_route]

    # Build Folium map
    m = folium.Map(location=(mid_lat, mid_lon), zoom_start=13)

    if replan:
        # Show both routes for comparison
        folium.PolyLine(
            normal_coords,
            color="green",
            weight=4,
            opacity=0.65,
            tooltip="Original route (would be congested)"
        ).add_to(m)

        folium.PolyLine(
            replan_coords,
            color="red",
            weight=5,
            opacity=0.9,
            tooltip="Suggested reroute (avoids main roads)"
        ).add_to(m)

        # Add legend to map
        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px;
            left: 30px;
            background-color: white;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 14px;
            z-index: 9999;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        ">
            <b>Route Legend</b><br><br>
            <span style="color: green; font-size: 18px;">&#9472;&#9472;</span>
            &nbsp; Original route<br>
            <span style="color: red; font-size: 18px;">&#9472;&#9472;</span>
            &nbsp; Suggested reroute
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

    else:
        # Just show the normal route
        folium.PolyLine(
            normal_coords,
            color="green",
            weight=5,
            opacity=0.9,
            tooltip="Fastest route — traffic stable"
        ).add_to(m)

    # Add start and end markers
    # FIX: uses actual route coordinates not arbitrary fixed points
    folium.Marker(
        location=normal_coords[0],
        tooltip=f"Start: {origin_name}",
        icon=folium.Icon(color="blue", icon="play", prefix="fa")
    ).add_to(m)

    folium.Marker(
        location=normal_coords[-1],
        tooltip=f"End: {dest_name}",
        icon=folium.Icon(color="red", icon="flag", prefix="fa")
    ).add_to(m)

    # RENDER MAP
    # FIX: folium_static is OUTSIDE all if/else blocks
    # Previously it was inside if replan: which meant
    # the map never showed when traffic was stable
    folium_static(m, width=900, height=550)

    # Summary message below map
    if replan:
        st.warning(
            f"Route updated to avoid congestion. "
            f"Predicted: {mean:.0f} vehicles/15 min "
            f"(uncertainty ±{uncertainty:.0f}). "
            f"Main roads are heavily penalised — side roads preferred."
        )
    else:
        st.success(
            f"Fastest route shown. "
            f"Traffic stable at {mean:.0f} vehicles/15 min "
            f"(uncertainty ±{uncertainty:.0f})."
        )