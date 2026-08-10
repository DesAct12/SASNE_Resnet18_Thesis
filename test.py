# ==============================
# Imports
# ==============================
import os
import numpy as np
from SASNE import SASNE_from_Z, SASNE_biharmonic
import plotly.graph_objects as go

# ==============================
# Checkpoints to embed
# ==============================

SELECTED_CHECKPOINTS = ["input"]
# Example:
# [1]
# [1,2]
# [2,4,"output"]
# [1,2,3,4,"output"]

# ==============================
# Config
# ==============================
checkpoints_config = ["input", 1, 2, 3, 4, "output"]
epochs_config = list(range(50))

# --- Paths
DATA_FILENAME = "SAGD_matrix.npy"

SAVE_DIR = "plots_sasne"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================
# Color helpers
# ==============================
def interpolate_color(c1, c2, t):
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )

def rgb_str(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"

checkpoint_color_pairs = [
    ((0, 102, 255), (0, 255, 255)),
    ((255, 0, 0), (255, 200, 0)),
    ((0, 180, 0), (180, 255, 0)),
    ((150, 0, 200), (255, 0, 150)),
    ((255, 140, 0), (255, 255, 0))
]

checkpoint_colors = {
    "input": ((50,50,50), (180,180,180)),
    1: ((0,102,255), (0,255,255)),
    2: ((255,0,0), (255,200,0)),
    3: ((0,180,0), (180,255,0)),
    4: ((150,0,200), (255,0,150)),
    "output": ((255,140,0), (255,255,0)),
}

# ==============================
# Load matrix
# ==============================
print("[STEP] Loading matrix...")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, DATA_FILENAME)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find '{DATA_FILENAME}'.\n"
        f"Looked here: {DATA_PATH}"
    )

W_np = np.load(DATA_PATH)
N = W_np.shape[0]

if W_np.shape[0] != W_np.shape[1]:
    raise ValueError(f"Distance matrix must be square. Got {W_np.shape}.")

if np.any(W_np < 0):
    raise ValueError("Distance matrix has negative values.")

# ==============================
# Labels
# ==============================
labels = []

for checkpoint in checkpoints_config:

    if checkpoint == "input":
        labels.append(("input", 0))
    else:
        for epoch in epochs_config:
            labels.append((checkpoint, epoch))

labels = labels[:N]

# ==============================
# Filter selected checkpoints
# ==============================

selected_indices = [
    i
    for i, (c, e) in enumerate(labels)
    if c in SELECTED_CHECKPOINTS
]

N = W_np.shape[0]

checkpoints = [
    c
    for c in checkpoints_config
    if c in SELECTED_CHECKPOINTS
]

def get_xyz(indices, emb):
    return emb[indices, 0], emb[indices, 1], emb[indices, 2]

# ==============================
# SASNE
# ==============================

print("\n[STEP] Computing full SASNE geometry...")

Z_full, eigenval = SASNE_biharmonic(W_np)

print("[OK] Full biharmonic geometry computed")

Z_filtered = Z_full[selected_indices]

labels = [
    labels[i]
    for i in selected_indices
]

print("\n[STEP] Running SASNE...")

embedding_3d = SASNE_from_Z(
    Z_filtered,
    eigenval,
    n_components=3
)

print("[OK] SASNE done")

max_epoch = max(epochs_config)

# ==============================
# Dots and Lines
# ==============================
print("\n[STEP] Plotting INTERACTIVE DOTS AND LINES...")

fig = go.Figure()

for i, checkpoint in enumerate(checkpoints):

    idxs = sorted(
        [
            j
            for j, (c, _) in enumerate(labels)
            if c == checkpoint
        ],
        key=lambda j: labels[j][1]
    )

    xs, ys, zs = get_xyz(idxs, embedding_3d)

    if checkpoint == "input":

        dot_colors = ["white"]

    else:
        
        c_start, c_end = checkpoint_colors[checkpoint]
        
        dot_colors = [
            rgb_str(
                interpolate_color(
                    c_start,
                    c_end,
                    labels[j][1] / max_epoch
                )
            )
            
            for j in idxs
        ]
        
        # First epoch is black
        dot_colors[0] = "black"  

    display_name = (
        "Output layer"
        if checkpoint == "output"
        else f"Checkpoint {checkpoint}"
    )

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            name=f"{display_name} - dots",
            legendgroup=f"{display_name}_dots",
            marker=dict(
                size=4,
                color=dot_colors
            )
        )
    )

    if checkpoint != "input":
        
        first = True
        
        for k in range(len(xs) - 1):
            
            t = k / (len(xs) - 1)
            
            color = rgb_str(
                interpolate_color(
                    c_start,
                    c_end,
                    t
                )
            )
            
            fig.add_trace(
                go.Scatter3d(
                    x=[xs[k], xs[k + 1]],
                    y=[ys[k], ys[k + 1]],
                    z=[zs[k], zs[k + 1]],
                    mode="lines",
                    line=dict(
                        width=2,
                        color=color
                    ),
                    legendgroup=f"{display_name}_lines",
                    name=f"{display_name} - lines" if first else None,
                    showlegend=first
                )
            )
            
            first = False

    if checkpoint != "input":
        
        last_idx = idxs[-1]
        
        xL, yL, zL = get_xyz(
            [last_idx],
            embedding_3d
            )
        
        fig.add_trace(
            go.Scatter3d(
                x=xL,
                y=yL,
                z=zL,
                mode="markers",
                name=f"{display_name} - final",
                marker=dict(
                size=5,
                color="gray"
                ),
                showlegend=True
                )
            )

fig.update_layout(
    title="3D SASNE Embedding",
    scene=dict(
        xaxis_title="SASNE1",
        yaxis_title="SASNE2",
        zaxis_title="SASNE3"
    )
)

fig.write_html(
    os.path.join(
        SAVE_DIR,
        f"sasne_ckp{SELECTED_CHECKPOINTS}.html"
    )
)

print("[OK] Dots and Lines saved")

# ==============================
# DIRECT BIHARMONIC PLOT (NO t-SNE)
# ==============================

print("\n[STEP] Plotting direct biharmonic coordinates...")

biharmonic_3d = 1e-4 * Z_filtered[:,[1,2,3]] * np.sqrt(eigenval[1])

fig_bhd = go.Figure()

for checkpoint in checkpoints:

    idxs = sorted(
        [
            j
            for j, (c, _) in enumerate(labels)
            if c == checkpoint
        ],
        key=lambda j: labels[j][1]
    )

    xs = biharmonic_3d[idxs, 0]
    ys = biharmonic_3d[idxs, 1]
    zs = biharmonic_3d[idxs, 2]

    display_name = (
        "Output layer"
        if checkpoint == "output"
        else f"Checkpoint {checkpoint}"
    )

    if checkpoint == "input":

        fig_bhd.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=6,
                    color="white"
                ),
                name="Input"
            )
        )

        continue

    c_start, c_end = checkpoint_colors[checkpoint]

    dot_colors = [
        rgb_str(
            interpolate_color(
                c_start,
                c_end,
                labels[j][1] / max_epoch
            )
        )
        for j in idxs
    ]

    dot_colors[0] = "black"

    # --------------------------
    # POINTS
    # --------------------------
    fig_bhd.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            name=f"{display_name} - dots",
            legendgroup=f"{display_name}_dots",
            marker=dict(
                size=4,
                color=dot_colors
            )
        )
    )

    # --------------------------
    # COLORED LINE SEGMENTS
    # --------------------------
    first = True

    for k in range(len(xs) - 1):

        t = k / (len(xs) - 1)

        color = rgb_str(
            interpolate_color(
                c_start,
                c_end,
                t
            )
        )

        fig_bhd.add_trace(
            go.Scatter3d(
                x=[xs[k], xs[k + 1]],
                y=[ys[k], ys[k + 1]],
                z=[zs[k], zs[k + 1]],
                mode="lines",
                line=dict(
                    width=2,
                    color=color
                ),
                legendgroup=f"{display_name}_lines",
                name=f"{display_name} - lines" if first else None,
                showlegend=first
            )
        )

        first = False

    # --------------------------
    # FINAL POINT
    # --------------------------
    fig_bhd.add_trace(
        go.Scatter3d(
            x=[xs[-1]],
            y=[ys[-1]],
            z=[zs[-1]],
            mode="markers",
            marker=dict(
                size=5,
                color="gray"
            ),
            name=f"{display_name} - final",
            showlegend=True
        )
    )

fig_bhd.update_layout(
    title="Direct Biharmonic Coordinates (SASNE initial points)",
    scene=dict(
        xaxis_title="BHD 1",
        yaxis_title="BHD 2",
        zaxis_title="BHD 3"
    )
)

fig_bhd.write_html(
    os.path.join(
        SAVE_DIR,
        f"biharmonic_direct_ckp{SELECTED_CHECKPOINTS}.html"
    )
)

print("[OK] Direct biharmonic plot saved")

# ==============================
# Done
# ==============================
print("\n[🎉 DONE]")
print(f"[PATH] {SAVE_DIR}")
