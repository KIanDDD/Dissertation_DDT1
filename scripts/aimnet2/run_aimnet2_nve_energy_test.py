from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import Trajectory, read, write
from ase.md.verlet import VelocityVerlet
from aimnet.calculators import AIMNet2ASE


EXPECTED_COMPOSITION = {"C": 13, "H": 47, "N": 2, "O": 16}
TACRINE_ATOM_COUNT = 30
HYDRATION_CUTOFF_A = 5.325


def composition(symbols: list[str]) -> dict[str, int]:
    return {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}


def minimum_pair_distance(positions: np.ndarray) -> float:
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def hydration_diagnostics(atoms) -> tuple[int, float]:
    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.positions, dtype=float)

    tacrine_heavy = [
        i
        for i in range(min(TACRINE_ATOM_COUNT, len(atoms)))
        if symbols[i] != "H"
    ]
    water_oxygen = [
        i
        for i in range(TACRINE_ATOM_COUNT, len(atoms))
        if symbols[i] == "O"
    ]

    if len(water_oxygen) != 16:
        raise ValueError(
            f"Expected 16 water oxygen atoms after atom "
            f"{TACRINE_ATOM_COUNT}; found {len(water_oxygen)}."
        )

    distances = []
    for oxygen_index in water_oxygen:
        d = np.linalg.norm(
            positions[tacrine_heavy] - positions[oxygen_index],
            axis=1,
        )
        distances.append(float(np.min(d)))

    return (
        int(sum(d <= HYDRATION_CUTOFF_A for d in distances)),
        float(max(distances)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continue from the final frame of an AIMNet2 NVT trajectory "
            "and run a short non-periodic NVE energy-conservation test."
        )
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="ASE .traj file from the completed NVT diagnostic run",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", default="aimnet2")
    parser.add_argument("--charge", type=int, default=1)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--write-interval", type=int, default=10)
    args = parser.parse_args()

    if not args.trajectory.is_file():
        raise FileNotFoundError(args.trajectory)

    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.outdir}. "
            "Use a new output directory."
        )

    args.outdir.mkdir(parents=True, exist_ok=True)

    atoms = read(args.trajectory, index=-1)
    symbols = atoms.get_chemical_symbols()
    observed = composition(symbols)

    if len(atoms) != 78:
        raise ValueError(f"Expected 78 atoms; found {len(atoms)}")

    if observed != EXPECTED_COMPOSITION:
        raise ValueError(
            f"Expected composition {EXPECTED_COMPOSITION}; "
            f"found {observed}"
        )

    if bool(np.asarray(atoms.pbc).any()):
        raise ValueError(
            "The final NVT frame unexpectedly has periodic boundaries."
        )

    if not atoms.has("momenta"):
        raise ValueError(
            "The final NVT trajectory frame does not contain momenta. "
            "Do not regenerate velocities for this NVE continuation."
        )

    if not np.isfinite(atoms.positions).all():
        raise ValueError("Non-finite input coordinates")

    if not np.isfinite(atoms.get_momenta()).all():
        raise ValueError("Non-finite input momenta")

    atoms.pbc = False
    atoms.calc = AIMNet2ASE(args.model, charge=args.charge)

    dynamics = VelocityVerlet(
        atoms,
        timestep=args.timestep_fs * units.fs,
    )

    trajectory_path = args.outdir / "aimnet2_nve_energy_test.traj"
    trajectory = Trajectory(trajectory_path, "w", atoms)
    dynamics.attach(trajectory.write, interval=args.write_interval)

    diagnostic_path = args.outdir / "aimnet2_nve_timeseries.csv"
    handle = diagnostic_path.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    fieldnames = [
        "step",
        "time_fs",
        "temperature_K",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "maximum_force_eV_per_A",
        "minimum_pair_distance_A",
        "waters_within_5p325_A",
        "farthest_water_O_to_tacrine_heavy_A",
    ]

    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    def record() -> None:
        forces = np.asarray(atoms.get_forces(), dtype=float)
        water_count, farthest_water = hydration_diagnostics(atoms)

        writer.writerow(
            {
                "step": dynamics.get_number_of_steps(),
                "time_fs": (
                    dynamics.get_number_of_steps()
                    * args.timestep_fs
                ),
                "temperature_K": float(atoms.get_temperature()),
                "potential_energy_eV": float(
                    atoms.get_potential_energy()
                ),
                "kinetic_energy_eV": float(
                    atoms.get_kinetic_energy()
                ),
                "total_energy_eV": float(
                    atoms.get_potential_energy()
                    + atoms.get_kinetic_energy()
                ),
                "maximum_force_eV_per_A": float(
                    np.linalg.norm(forces, axis=1).max()
                ),
                "minimum_pair_distance_A": (
                    minimum_pair_distance(
                        np.asarray(atoms.positions)
                    )
                ),
                "waters_within_5p325_A": water_count,
                "farthest_water_O_to_tacrine_heavy_A": (
                    farthest_water
                ),
            }
        )

        handle.flush()

    record()
    dynamics.attach(record, interval=args.write_interval)

    wall_start = time.perf_counter()
    dynamics.run(args.steps)
    wall_time = time.perf_counter() - wall_start

    if args.steps % args.write_interval != 0:
        trajectory.write(atoms)
        record()

    handle.close()
    trajectory.close()

    final_xyz_path = args.outdir / "aimnet2_nve_final.xyz"
    write(final_xyz_path, atoms)

    with diagnostic_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as read_handle:
        rows = list(csv.DictReader(read_handle))

    total_energy = np.asarray(
        [float(row["total_energy_eV"]) for row in rows],
        dtype=float,
    )
    temperature = np.asarray(
        [float(row["temperature_K"]) for row in rows],
        dtype=float,
    )
    minimum_distance = np.asarray(
        [float(row["minimum_pair_distance_A"]) for row in rows],
        dtype=float,
    )
    water_count = np.asarray(
        [int(row["waters_within_5p325_A"]) for row in rows],
        dtype=int,
    )

    simulated_time_ps = (
        args.steps * args.timestep_fs / 1000.0
    )
    energy_drift_eV = float(
        total_energy[-1] - total_energy[0]
    )

    summary = {
        "status": "complete",
        "calculation_type": (
            "Short non-periodic AIMNet2 Velocity-Verlet NVE "
            "energy-conservation test; not production MD."
        ),
        "input_trajectory": str(
            args.trajectory.resolve()
        ),
        "model": args.model,
        "charge": args.charge,
        "atom_count": len(atoms),
        "composition": observed,
        "timestep_fs": args.timestep_fs,
        "steps": args.steps,
        "simulated_time_fs": (
            args.steps * args.timestep_fs
        ),
        "write_interval_steps": args.write_interval,
        "initial_total_energy_eV": float(
            total_energy[0]
        ),
        "final_total_energy_eV": float(
            total_energy[-1]
        ),
        "total_energy_drift_eV": energy_drift_eV,
        "total_energy_drift_eV_per_ps": (
            energy_drift_eV / simulated_time_ps
            if simulated_time_ps > 0
            else None
        ),
        "total_energy_range_eV": float(
            total_energy.max() - total_energy.min()
        ),
        "maximum_absolute_energy_deviation_eV": float(
            np.max(
                np.abs(
                    total_energy - total_energy[0]
                )
            )
        ),
        "mean_temperature_K": float(
            temperature.mean()
        ),
        "minimum_temperature_K": float(
            temperature.min()
        ),
        "maximum_temperature_K": float(
            temperature.max()
        ),
        "minimum_pair_distance_over_trajectory_A": (
            float(minimum_distance.min())
        ),
        "initial_waters_within_5p325_A": int(
            water_count[0]
        ),
        "final_waters_within_5p325_A": int(
            water_count[-1]
        ),
        "minimum_waters_within_5p325_A": int(
            water_count.min()
        ),
        "wall_time_s": wall_time,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "aimnet_version": version("aimnet"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "scientific_caution": (
            "This test evaluates numerical energy conservation "
            "for one finite +1 cluster and one timestep. It does "
            "not establish agreement with VASP or represent bulk "
            "aqueous tacrine."
        ),
    }

    summary_path = (
        args.outdir
        / "aimnet2_nve_energy_test_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(
        f"\nSaved results to: "
        f"{args.outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
