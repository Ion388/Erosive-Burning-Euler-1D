from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_state_history(file_path: Path):
    """Load tabular state history data with columns: t x rho u p M A P."""
    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 8:
        raise ValueError(
            f"Expected 8 columns (t x rho u p M A P), got {data.shape[1]} columns."
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
    reshaped = data.reshape(nt, nx, 8)
    t_unique = reshaped[:, 0, 0]
    x_unique = reshaped[0, :, 1]

    fields = {
        "rho": reshaped[:, :, 2],
        "u": reshaped[:, :, 3],
        "p": reshaped[:, :, 4],
        "M": reshaped[:, :, 5],
        "A": reshaped[:, :, 6],
        "P": reshaped[:, :, 7],
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
    parser = argparse.ArgumentParser(
        description="Read rocket state history file and plot rho, u, p, M, A, P."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to rocket_state_history_case*.dat",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots_state_history"),
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=6,
        help="Maximum number of time profiles per variable.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    t, x, fields = load_state_history(args.input)

    labels = {
        "rho": "Density rho [kg/m^3]",
        "u": "Velocity u [m/s]",
        "p": "Pressure p [Pa]",
        "M": "Mach number [-]",
        "A": "Area A [m^2]",
        "P": "Perimeter P [m]",
    }

    basenames = {
        "rho": "rho",
        "u": "u",
        "p": "pressure",
        "M": "mach",
        "A": "area",
        "P": "perimeter",
    }

    for name, field in fields.items():
        plot_spacetime(t, x, field, name, labels[name], basenames[name], args.out_dir)
        plot_profiles(t, x, field, name, labels[name], basenames[name], args.out_dir, args.max_profiles)

    print(f"Saved plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
