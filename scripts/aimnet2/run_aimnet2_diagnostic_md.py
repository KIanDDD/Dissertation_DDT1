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
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
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
        i for i in range(min(TACRINE_ATOM_COUNT, len(atoms))) if symbols[i] != "H"
    ]
    water_oxygen = [
        i for i in range(TACRINE_ATOM_COUNT, len(atoms)) if symbols[i] == "O"
    ]
    if len(water_oxygen) != 16:
        raise ValueError(f"Expected 16 water oxygens; found {len(water_oxygen)}")
    distances = []
    for oxygen_index in water_oxygen:
        d = np.linalg.norm(positions[tacrine_heavy] - positions[oxygen_index], axis=1)
        distances.append(float(np.min(d)))
    return int(sum(d <= HYDRATION_CUTOFF_A for d in distances)), float(max(distances))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Short, non-periodic AIMNet2 NVT diagnostic MD for one N16 cluster."
    )
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", default="aimnet2")
    parser.add_argument("--charge", type=int, default=1)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--timestep-fs", type=float, default=0.25)
    parser.add_argument("--friction-per-fs", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--write-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=701)
    args = parser.parse_args()

    if not args.xyz.is_file():
        raise FileNotFoundError(args.xyz)
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.outdir}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    atoms = read(args.xyz)
    symbols = atoms.get_chemical_symbols()
    observed = composition(symbols)

    if len(atoms) != 78:
        raise ValueError(f"Expected 78 atoms; found {len(atoms)}")
    if observed != EXPECTED_COMPOSITION:
        raise ValueError(f"Expected {EXPECTED_COMPOSITION}; found {observed}")
    if bool(np.asarray(atoms.pbc).any()):
        raise ValueError("Input unexpectedly has periodic boundary conditions")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Non-finite input coordinates")

    atoms.pbc = False
    atoms.calc = AIMNet2ASE(args.model, charge=args.charge)

    initial_energy = float(atoms.get_potential_energy())
    initial_forces = np.asarray(atoms.get_forces(), dtype=float)
    initial_max_force = float(np.linalg.norm(initial_forces, axis=1).max())
    initial_min_distance = minimum_pair_distance(np.asarray(atoms.positions))
    initial_water_count, initial_farthest_water = hydration_diagnostics(atoms)

    rng = np.random.RandomState(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_k,
        friction=args.friction_per_fs / units.fs,
        fixcm=True,
    )

    trajectory_path = args.outdir / "aimnet2_diagnostic_nvt.traj"
    trajectory = Trajectory(trajectory_path, "w", atoms)
    dynamics.attach(trajectory.write, interval=args.write_interval)

    diagnostic_path = args.outdir / "aimnet2_diagnostic_timeseries.csv"
    handle = diagnostic_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
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
        ],
    )
    writer.writeheader()

    def record() -> None:
        forces = np.asarray(atoms.get_forces(), dtype=float)
        water_count, farthest_water = hydration_diagnostics(atoms)
        writer.writerow(
            {
                "step": dynamics.get_number_of_steps(),
                "time_fs": dynamics.get_number_of_steps() * args.timestep_fs,
                "temperature_K": float(atoms.get_temperature()),
                "potential_energy_eV": float(atoms.get_potential_energy()),
                "kinetic_energy_eV": float(atoms.get_kinetic_energy()),
                "total_energy_eV": float(atoms.get_potential_energy() + atoms.get_kinetic_energy()),
                "maximum_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max()),
                "minimum_pair_distance_A": minimum_pair_distance(np.asarray(atoms.positions)),
                "waters_within_5p325_A": water_count,
                "farthest_water_O_to_tacrine_heavy_A": farthest_water,
            }
        )
        handle.flush()

    record()
    dynamics.attach(record, interval=args.write_interval)

    start = time.perf_counter()
    dynamics.run(args.steps)
    wall_time = time.perf_counter() - start

    if args.steps % args.write_interval != 0:
        trajectory.write(atoms)
        record()

    handle.close()
    trajectory.close()

    final_energy = float(atoms.get_potential_energy())
    final_forces = np.asarray(atoms.get_forces(), dtype=float)
    final_max_force = float(np.linalg.norm(final_forces, axis=1).max())
    final_min_distance = minimum_pair_distance(np.asarray(atoms.positions))
    final_water_count, final_farthest_water = hydration_diagnostics(atoms)

    write(args.outdir / "aimnet2_diagnostic_final.xyz", atoms)

    with diagnostic_path.open("r", newline="", encoding="utf-8") as read_handle:
        rows = list(csv.DictReader(read_handle))

    temperatures = np.asarray([float(row["temperature_K"]) for row in rows])
    total_energies = np.asarray([float(row["total_energy_eV"]) for row in rows])
    minimum_distances = np.asarray([float(row["minimum_pair_distance_A"]) for row in rows])
    water_counts = np.asarray([int(row["waters_within_5p325_A"]) for row in rows])

    summary = {
        "status": "complete",
        "calculation_type": "Short non-periodic AIMNet2 Langevin-NVT diagnostic MD; not production MD.",
        "input_xyz": str(args.xyz.resolve()),
        "model": args.model,
        "charge": args.charge,
        "atom_count": len(atoms),
        "composition": observed,
        "temperature_target_K": args.temperature_k,
        "timestep_fs": args.timestep_fs,
        "friction_per_fs": args.friction_per_fs,
        "steps": args.steps,
        "simulated_time_fs": args.steps * args.timestep_fs,
        "write_interval_steps": args.write_interval,
        "random_seed": args.seed,
        "initial_potential_energy_eV": initial_energy,
        "final_potential_energy_eV": final_energy,
        "initial_maximum_force_eV_per_A": initial_max_force,
        "final_maximum_force_eV_per_A": final_max_force,
        "initial_minimum_pair_distance_A": initial_min_distance,
        "final_minimum_pair_distance_A": final_min_distance,
        "minimum_pair_distance_over_trajectory_A": float(minimum_distances.min()),
        "mean_temperature_K": float(temperatures.mean()),
        "minimum_temperature_K": float(temperatures.min()),
        "maximum_temperature_K": float(temperatures.max()),
        "total_energy_range_eV": float(total_energies.max() - total_energies.min()),
        "initial_waters_within_5p325_A": initial_water_count,
        "final_waters_within_5p325_A": final_water_count,
        "minimum_waters_within_5p325_A": int(water_counts.min()),
        "initial_farthest_water_O_to_tacrine_heavy_A": initial_farthest_water,
        "final_farthest_water_O_to_tacrine_heavy_A": final_farthest_water,
        "wall_time_s": wall_time,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "aimnet_version": version("aimnet"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "scientific_caution": (
            "This finite +1 microhydrated cluster is non-periodic. "
            "The run tests numerical stability only and is not bulk aqueous MD or VASP validation."
        ),
    }

    (args.outdir / "aimnet2_diagnostic_md_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
