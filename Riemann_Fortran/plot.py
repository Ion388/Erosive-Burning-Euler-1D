from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_state_history(file_path: Path):
    """Load tabular state history data with columns: t x rho u p T M A P rb G."""
    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 11:
        raise ValueError(
            f"Expected 11 columns (t x rho u p T M A P rb G), got {data.shape[1]} columns."
        )

    # Infer nx from first snapshot block (rows with initial time stamp).
    t0 = data[0, 0]
    nx = int(np.sum(np.isclose(data[:, 0], t0)))
    if nx <= 0:
        raise ValueError("Could not infer spatial size nx from input data.")

    # If run was interrupted, drop incomplete trailing rows.
    n_full_rows = (data.shape[0] // nx) * nx
    if n_full_rows == 0:
        raise ValueError("Input file does not contain a complete snapshot block.")
    if n_full_rows != data.shape[0]:
        data = data[:n_full_rows, :]

    nt = data.shape[0] // nx

    # Rows are written in time-major order, then x-major order inside each snapshot.
    reshaped = data.reshape(nt, nx, 11)
    t_unique = reshaped[:, 0, 0]
    x_unique = reshaped[0, :, 1]

    fields = {
        "rho": reshaped[:, :, 2],
        "u": reshaped[:, :, 3],
        "p": reshaped[:, :, 4],
        # "T": reshaped[:, :, 5],
        "M": reshaped[:, :, 6],
        # "A": reshaped[:, :, 7],
        # "P": reshaped[:, :, 8],
        # "rb": reshaped[:, :, 9],  # Extract rb_cycle from inner spatial points only. 
        # "G": reshaped[:, :, 10],  # Extract G_cycle from inner spatial points only. 
    }
    return t_unique, x_unique, fields


def plot_spacetime(t, x, field, name, label, basename, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    mesh = ax.pcolormesh(x, t, field, shading="auto", cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("t [s]")
    ax.set_title(f"{name} space-time map")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{basename}_spacetime.png")
    plt.close(fig)

def plot_exit_history(t, field, name, label, basename, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    ax.plot(t, field, linewidth=2, color='orangered')
    ax.set_xlabel("t [s]")
    ax.set_ylabel(label)
    ax.set_title(f"{name} exit history")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{basename}_exit_history.png")
    plt.close(fig)


def plot_profiles(t, x, field, name, label, basename, out_dir: Path, max_profiles: int):
    nt = t.size
    nplot = min(max_profiles, nt)
    idx = np.linspace(0, nt - 1, nplot, dtype=int)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    for i in idx:
        ax.plot(x, field[i, :], linewidth=2, label=f"t={t[i]:.5g} s")

    ax.set_xlabel("x [m]")
    ax.set_ylabel(label)
    ax.set_title(f"{name} profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{basename}_profiles.png")
    plt.close(fig)


def main():

    labels_state = {
        "rho": "Density rho [kg/m^3]",
        "u": "Velocity u [m/s]",
        "p": "Pressure p [Pa]",
        "T": "Temperature T [K]",
        "M": "Mach number [-]",
        "A": "Area A [m^2]",
        "P": "Perimeter P [m]",
        'rb': "Regression burn cycle [-]",
        'G': "Global burn cycle [-]",
    }

    basenames_state = {
        "rho": "rho",
        "u": "u",
        "p": "pressure",
        "T": "temperature",
        "M": "mach",
        "A": "area",
        "P": "perimeter",
        'rb': "regression_burn_cycle",
        'G': "global_burn_cycle",
    }

    max_profiles = 2

    state_files = sorted(Path(".").glob("riemann_state_history_case*.dat"))
    if not state_files:
        raise FileNotFoundError(
            "No files matching 'riemann_state_history_case*.dat' were found in the current directory."
        )

    state_file = state_files[0]
    
    for i in range(len(state_files)):
        file = state_files[i]
        print(f"Using state file: {file}")
        t_state, x_state, fields_state = load_state_history(file)

        case_id = file.stem.split("case")[-1]
        out_dir = Path(f"plots_state_case{case_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

    

    

    # for name, field in fields_state.items():
    #     plot_spacetime(t_state, x_state, field, name, labels_state[name], basenames_state[name], args.out_dir)
    #     plot_profiles(t_state, x_state, field, name, labels_state[name], basenames_state[name], args.out_dir, args.max_profiles)
    # for name, field in fields_exit.items():
    #     plot_exit_history(t_exit, field, name, labels_exit[name], basenames_exit[name], args.out_dir)

        for name, field in fields_state.items():
            plot_spacetime(t_state, x_state, field, name, labels_state[name], basenames_state[name], out_dir)
            plot_profiles(t_state, x_state, field, name, labels_state[name], basenames_state[name], out_dir, max_profiles)

        print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
