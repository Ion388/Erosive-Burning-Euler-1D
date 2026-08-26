from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def load_state_history(file_path: Path):
    """Load legacy 13-column or current 15-column state history data."""
    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncolumns = data.shape[1]
    if ncolumns not in (13, 15):
        raise ValueError(
            f"Expected 13 or 15 state-history columns, got {ncolumns}."
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
    reshaped = data.reshape(nt, nx, ncolumns)
    t_unique = reshaped[:, 0, 0]
    x_unique = reshaped[0, :, 1]

    if ncolumns == 15:
        # t x rho u p T M A P_wet P_burn web rb G Ts ignited
        column_map = {"P": 8, "rb": 11, "G": 12, "Ts": 13, "ignited": 14}
    else:
        # t x rho u p T M A P rb G Ts ignited
        column_map = {"P": 8, "rb": 9, "G": 10, "Ts": 11, "ignited": 12}
    fields = {
        "rho": reshaped[:, :, 2],
        "u": reshaped[:, :, 3],
        "p": reshaped[:, :, 4],
        "T": reshaped[:, :, 5],
        "M": reshaped[:, :, 6],
        "A": reshaped[:, :, 7],
        "P": reshaped[:, :, column_map["P"]],
        "rb": reshaped[:, :, column_map["rb"]],
        "G": reshaped[:, :, column_map["G"]],
        "Ts": reshaped[:, :, column_map["Ts"]],
        "ignited": reshaped[:, :, column_map["ignited"]],
    }
    return t_unique, x_unique, fields

def load_exit_history(file_path: Path):
    """Load tabular exit history data with columns: t M_exit T_exit p_exit rho_exit u_exit E_exit thrust."""
    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 8:
        raise ValueError(
            f"Expected 8 columns (t M_exit T_exit p_exit rho_exit u_exit E_exit thrust), got {data.shape[1]} columns."
        )

    # Infer nx from first snapshot block (rows with initial time stamp).
    nt = data.shape[0]

    # Rows are written in time-major order, then x-major order inside each snapshot.
    reshaped = data.reshape(nt, 8)
    t_unique = reshaped[:, 0]

    fields = {
        "M_exit": reshaped[:, 1],
        "T_exit": reshaped[:, 2],
        "p_exit": reshaped[:, 3],
        "rho_exit": reshaped[:, 4],
        "u_exit": reshaped[:, 5],
        "E_exit": reshaped[:, 6],
        "thrust": reshaped[:, 7],

    }
    return t_unique, fields


def load_geometry_table(file_path: Path):
    """Load grain geometry data written by grain_levelset_preprocessor."""
    with file_path.open(encoding="ascii") as table_file:
        header = table_file.readline().strip()
        if header != "# GRAIN_LEVELSET_TABLE_V1":
            raise ValueError(f"Unsupported grain geometry table header: {header!r}")
        dimensions = table_file.readline().split()
        if len(dimensions) != 3:
            raise ValueError("Invalid grain geometry table dimensions.")
        nstations, nweb = (int(value) for value in dimensions[:2])
        casing_radius = float(dimensions[2])

    data = np.loadtxt(file_path, comments="#", skiprows=3)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    expected_rows = nstations * nweb
    if data.shape != (expected_rows, 6):
        raise ValueError(
            f"Expected {expected_rows} geometry rows with 6 columns, got {data.shape}."
        )

    data = data.reshape(nstations, nweb, 6)
    return {
        "x": data[:, :, 0][:, 0],
        "w": data[:, :, 1],
        "area": data[:, :, 2],
        "p_burn": data[:, :, 3],
        "p_wet": data[:, :, 4],
        "burned": data[:, :, 5].astype(bool),
        "casing_radius": casing_radius,
    }


def load_levelset_fields(file_path: Path):
    """Load selected signed-distance fields exported by the Fortran preprocessor."""
    with file_path.open(encoding="ascii") as field_file:
        if field_file.readline().strip() != "# GRAIN_LEVELSET_FIELDS_V1":
            raise ValueError("Unsupported level-set field file header.")
        dimensions = field_file.readline().split()
        if len(dimensions) != 4:
            raise ValueError("Invalid level-set field dimensions.")
        nstations, ngrid = (int(value) for value in dimensions[:2])
        xmin, h = (float(value) for value in dimensions[2:])
        field_file.readline()
        x_stations = []
        fields = np.empty((nstations, ngrid, ngrid))
        for station in range(nstations):
            x_stations.append(float(field_file.readline()))
            for row in range(ngrid):
                values = field_file.readline().split()
                if len(values) != ngrid:
                    raise ValueError("Invalid row in level-set field file.")
                fields[station, row, :] = values

    coordinates = xmin + h * np.arange(ngrid)
    return {
        "x": np.asarray(x_stations),
        "coordinates": coordinates,
        "phi": fields,
    }


def selected_geometry_stations(geometry):
    """Return first, middle, and last station indices and display names."""
    nstations = geometry["x"].size
    indices = [0, nstations // 2, nstations - 1]
    names = ["first", "middle", "last"]
    return list(zip(names, indices))


def plot_selected_geometry(geometry, stations, out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    metrics = [
        ("area", "Port area A [m^2]"),
        ("p_wet", "Wet perimeter P_wet [m]"),
        ("p_burn", "Burn perimeter P_burn [m]"),
    ]
    colors = {"first": "tab:blue", "middle": "tab:orange", "last": "tab:green"}

    for axis, (metric, label) in zip(axes, metrics):
        for name, station in stations:
            axis.plot(
                geometry["w"][station],
                geometry[metric][station],
                linewidth=2,
                color=colors[name],
                label=f"{name} cell (x={geometry['x'][station]:.5g} m)",
            )
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Web distance w [m]")
    fig.suptitle("Selected grain cells: geometry evolution", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "selected_cells_geometry.png", dpi=150)
    plt.close(fig)


def animate_selected_geometry(geometry, stations, out_dir: Path):
    """Animate area and perimeters as the selected cells burn through the web."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    metrics = [
        ("area", "Port area A [m^2]"),
        ("p_wet", "Wet perimeter P_wet [m]"),
        ("p_burn", "Burn perimeter P_burn [m]"),
    ]
    colors = {"first": "tab:blue", "middle": "tab:orange", "last": "tab:green"}
    markers = []
    for axis, (metric, label) in zip(axes, metrics):
        metric_markers = []
        for name, station in stations:
            axis.plot(
                geometry["w"][station],
                geometry[metric][station],
                linewidth=2,
                color=colors[name],
                label=name,
            )
            marker, = axis.plot([], [], "o", color=colors[name], markersize=7)
            metric_markers.append(marker)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
        markers.append(metric_markers)
    axes[-1].set_xlabel("Web distance w [m]")

    web = geometry["w"][0]
    area = geometry["area"][0]
    perimeter_burn = geometry["p_burn"][0]
    perimeter_wet = geometry["p_wet"][0]

    def update(frame):
        current_web = web[frame]
        for metric_index, (metric, _) in enumerate(metrics):
            for selected_index, (_, station) in enumerate(stations):
                markers[metric_index][selected_index].set_data(
                    [current_web], [geometry[metric][station, frame]]
                )
        burned_count = sum(
            geometry["burned"][station, frame] for _, station in stations
        )
        state = "burnout reached" if burned_count == len(stations) else "burning"
        fig.suptitle(
            f"Selected grain cells at web distance w={current_web:.5g} m ({state})",
            y=0.995,
        )
        return [marker for metric_markers in markers for marker in metric_markers]

    animation = FuncAnimation(
        fig, update, frames=web.size, interval=35, blit=False, repeat=False
    )
    fig.tight_layout()
    animation.save(out_dir / "selected_cells_geometry.gif", writer=PillowWriter(fps=25))
    plt.close(fig)


def animate_levelset_fields(geometry, levelset, stations, out_dir: Path):
    """Animate phi(x,y)=w contours for the selected physical cross sections."""
    coordinates = levelset["coordinates"]
    xx, yy = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(xx, yy)
    casing_radius = geometry["casing_radius"]
    masked_radius = np.ma.masked_where(radius > casing_radius, radius)
    web = geometry["w"][0]
    # station_indices = [stations[0][1], stations[1][1], stations[2][1]]
    # # print(station_indices)
    # area = geometry["area"][station_indices]
    # # print(area.shape)
    # perimeter_burn = geometry["p_burn"][station_indices]
    # perimeter_wet = geometry["p_wet"][station_indices]
    frame_indices = np.unique(np.r_[np.arange(0, web.size, 2), web.size - 1])
    colors = {"first": "tab:blue", "middle": "tab:orange", "last": "tab:green"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    def draw_frame(frame):
        current_web = web[frame]
        # print(area[:, frame].shape)
        # current_area = area[:, frame]
        # current_perimeter_burn = perimeter_burn[:, frame]
        # current_perimeter_wet = perimeter_wet[:, frame]
        for axis, (name, station), phi in zip(
            axes, stations, levelset["phi"]
        ):
            axis.clear()
            axis.set_aspect("equal")
            axis.set_xlim(-casing_radius * 1.08, casing_radius * 1.08)
            axis.set_ylim(-casing_radius * 1.08, casing_radius * 1.08)
            axis.contourf(
                xx,
                yy,
                np.ma.masked_where(masked_radius.mask, phi),
                levels=[-np.inf, current_web],
                colors=["#dceeff"],
            )
            contour = axis.contour(
                xx,
                yy,
                np.ma.masked_where(masked_radius.mask, phi),
                levels=[current_web],
                colors=[colors[name]],
                linewidths=2.5,
            )
            axis.add_patch(plt.Circle((0.0, 0.0), casing_radius, fill=False, color="black"))
            burned = geometry["burned"][station, frame]
            current_area = geometry["area"][station, frame]
            current_perimeter_burn = geometry["p_burn"][station, frame]
            current_perimeter_wet = geometry["p_wet"][station, frame]
            state = "burned out" if burned else "burning"
            axis.set_title(f"{name} cell\nx={geometry['x'][station]:.5g} m")
            axis.text(
                0.03,
                0.04,
                f"w={current_web:.4g} m\n{state} \n area={current_area:.5g} m^2 \n P_burn={current_perimeter_burn:.5g} m \n P_wet={current_perimeter_wet:.5g} m",
                transform=axis.transAxes,
                fontsize=6,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            axis.set_xlabel("y [m]")
            axis.set_ylabel("z [m]")
            axis.grid(True, alpha=0.2)
            del contour

    animation = FuncAnimation(fig, draw_frame, frames=frame_indices, interval=60, repeat=False)
    animation.save(out_dir / "selected_cells_levelset_4fps.gif", writer=PillowWriter(fps=4))
    animation.save(out_dir / "selected_cells_levelset_8fps.gif", writer=PillowWriter(fps=8))
    animation.save(out_dir / "selected_cells_levelset_16fps.gif", writer=PillowWriter(fps=16))
    plt.close(fig)


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
    script_dir = Path(__file__).resolve().parent
    build_dir = script_dir / "build"

    state_files = sorted(build_dir.glob("rocket_state_history_case*.dat"))
    if not state_files:
        raise FileNotFoundError(
            f"No files matching 'rocket_state_history_case*.dat' were found in '{build_dir}'."
        )

    state_file = state_files[0]
    print(f"Using state file: {state_file}")

    exit_files = sorted(build_dir.glob("rocket_exit_history_case*.dat"))
    if not exit_files:
        raise FileNotFoundError(
            f"No files matching 'rocket_exit_history_case*.dat' were found in '{build_dir}'."
        )
    exit_file = exit_files[0]
    print(f"Using exit file: {exit_file}")

    t_state, x_state, fields_state = load_state_history(state_file)
    t_exit, fields_exit = load_exit_history(exit_file)

    case_id = state_file.stem.split("case")[-1]
    out_dir = Path(f"plots_state_case{case_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_file = build_dir / "grain_geometry_table.dat"
    if not geometry_file.exists():
        raise FileNotFoundError(
            f"No grain geometry table was found at '{geometry_file}'. "
            "Run grain_levelset_preprocessor first."
        )
    geometry = load_geometry_table(geometry_file)
    stations = selected_geometry_stations(geometry)
    levelset_file = build_dir / "grain_levelset_fields.dat"
    if not levelset_file.exists():
        raise FileNotFoundError(
            f"No level-set field file was found at '{levelset_file}'. "
            "Rebuild and run grain_levelset_preprocessor first."
        )
    levelset = load_levelset_fields(levelset_file)
    animate_levelset_fields(geometry, levelset, stations, out_dir)
    plot_selected_geometry(geometry, stations, out_dir)
    animate_selected_geometry(geometry, stations, out_dir)

    max_profiles = 10

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
        'Ts': "Temperature at sensor [K]",
        'ignited': "Ignited [boolean]",
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
        'Ts': "temperature_at_sensor",
        'ignited': "ignited",
    }

    labels_exit = {
        "M_exit": "Exit Mach number [-]",
        "T_exit": "Exit temperature T [K]",
        "p_exit": "Exit pressure p [Pa]",
        "rho_exit": "Exit density rho [kg/m^3]",
        "u_exit": "Exit velocity u [m/s]",
        "E_exit": "Exit total energy E [J/kg]",
        "thrust": "Thrust [N]",
    }
    
    basenames_exit = {
        "M_exit": "M_exit",
        "T_exit": "T_exit",
        "p_exit": "p_exit",
        "rho_exit": "rho_exit",
        "u_exit": "u_exit",
        "E_exit": "E_exit",
        "thrust": "thrust",
    }

    # for name, field in fields_state.items():
    #     plot_spacetime(t_state, x_state, field, name, labels_state[name], basenames_state[name], args.out_dir)
    #     plot_profiles(t_state, x_state, field, name, labels_state[name], basenames_state[name], args.out_dir, args.max_profiles)
    # for name, field in fields_exit.items():
    #     plot_exit_history(t_exit, field, name, labels_exit[name], basenames_exit[name], args.out_dir)

    for name, field in fields_state.items():
        plot_spacetime(t_state, x_state, field, name, labels_state[name], basenames_state[name], out_dir)
        plot_profiles(t_state, x_state, field, name, labels_state[name], basenames_state[name], out_dir, max_profiles)

    for name, field in fields_exit.items():
        plot_exit_history(t_exit, field, name, labels_exit[name], basenames_exit[name], out_dir)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
