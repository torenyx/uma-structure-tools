#!/usr/bin/env python3
"""Minimal reusable batch structure relaxation with UMA and torch-sim.

This script intentionally implements only the reusable geometry-optimization
path:

1. torch-sim's batched LBFGS optimizer;
2. one FairChemModel shared by the whole batch;
3. GPU batch inference (default batch size 72);
4. resumable output handling.

It accepts ASE-readable structure files (CIF, XYZ/EXTXYZ, and other formats
supported by ASE). Existing ``move_mask``, ASE FixAtoms constraints, and
OC20-style tags are preserved. If no constraint information is present, all
atoms are treated as mobile.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _pre_parse_gpu_id() -> str:
    """Set CUDA visibility before importing torch/FairChem."""

    for i, arg in enumerate(sys.argv):
        if arg == "--gpu-id" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--gpu-id="):
            return arg.split("=", 1)[1]
    return "0"


os.environ.setdefault("CUDA_VISIBLE_DEVICES", _pre_parse_gpu_id())

# Loaded only after argument parsing so ``--help`` and lightweight packaging
# checks work on machines that do not have the CUDA/FairChem stack installed.


def load_runtime_dependencies() -> None:
    global np, torch, ts, ASEFixAtoms, read, write
    global FairChemModel, TorchSimFixAtoms, state_to_atoms
    import numpy as np  # noqa: PLW0603
    import torch  # noqa: PLW0603
    import torch_sim as ts  # noqa: PLW0603
    from ase.constraints import FixAtoms as ASEFixAtoms  # noqa: PLW0603
    from ase.io import read, write  # noqa: PLW0603
    from fairchem.core.calculate.torchsim_interface import (  # noqa: PLW0603
        FairChemModel,
    )
    from torch_sim.constraints import FixAtoms as TorchSimFixAtoms  # noqa: PLW0603
    from torch_sim.io import state_to_atoms  # noqa: PLW0603


SUPPORTED_SUFFIXES = {
    ".cif",
    ".extxyz",
    ".pdb",
    ".xyz",
}


@dataclass
class RelaxResult:
    name: str
    input_path: str
    output_xyz: str | None
    converged: bool
    steps: int
    energy_eV: float | None
    fmax_eV_A: float | None
    natoms: int
    nfixed: int
    elapsed_s: float
    error: str | None = None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reusable UMA batch structure optimization via torch-sim LBFGS"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        action="append",
        type=Path,
        help="One structure file; repeat --input for a small explicit batch.",
    )
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing ASE-readable structure files (non-recursive).",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--model-or-ckpt",
        default=os.environ.get("UMA_MODEL"),
        help="UMA checkpoint; defaults to UMA_MODEL.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--task-name", default="oc20")
    parser.add_argument("--batch-size", type=int, default=72)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help=(
            "Use one InFlightAutoBatcher queue for all inputs; converged systems "
            "are replaced immediately from the waiting queue."
        ),
    )
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--maxstep", type=float, default=0.04)
    parser.add_argument("--memory", type=int, default=50)
    parser.add_argument(
        "--max-memory-scaler",
        type=float,
        default=None,
        help=(
            "Optional n_atoms memory budget for dynamic batching. If omitted, "
            "it is set to batch-size times the largest input structure."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("uma_relax_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(out_dir / "relax.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def input_files(args: argparse.Namespace) -> list[Path]:
    if args.input:
        paths = [path.expanduser().resolve() for path in args.input]
    else:
        root = args.input_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Input directory not found: {root}")
        paths = [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        paths.sort(key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError("No input structures found")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))
    return paths


def fixed_mask_from_atoms(atoms) -> np.ndarray:
    """Return a strict per-atom fixed mask while preserving input constraints."""

    move_mask = atoms.arrays.get("move_mask")
    if move_mask is not None:
        movable = np.asarray(move_mask, dtype=bool)
        if movable.ndim == 2:
            fully_movable = movable.all(axis=1)
            fully_fixed = (~movable).all(axis=1)
            if not np.all(fully_movable | fully_fixed):
                raise ValueError("Partial Cartesian move_mask is unsupported")
            movable = fully_movable
        elif movable.ndim != 1:
            raise ValueError(f"Unexpected move_mask shape: {movable.shape}")
        if len(movable) != len(atoms):
            raise ValueError("move_mask length does not match atom count")
        return ~movable

    fixed = np.zeros(len(atoms), dtype=bool)
    saw_constraint = False
    for constraint in atoms.constraints:
        if not isinstance(constraint, ASEFixAtoms):
            raise ValueError(
                "Only FixAtoms-style constraints are supported; "
                f"got {type(constraint).__name__}"
            )
        saw_constraint = True
        fixed[np.asarray(constraint.get_indices(), dtype=int)] = True
    if saw_constraint:
        return fixed

    tags = np.asarray(atoms.get_tags())
    if len(tags) == len(atoms) and np.any(tags != 0):
        # OC20 convention: tag 0 is fixed slab, nonzero is mobile.
        return tags == 0

    # Generic unconstrained structure: optimize every atom.
    return fixed


def load_atoms(path: Path):
    atoms = read(path)
    fixed = fixed_mask_from_atoms(atoms)
    atoms.set_constraint(ASEFixAtoms(mask=fixed.tolist()))
    atoms.info["uma_input_path"] = str(path)
    atoms.info["nfixed"] = int(fixed.sum())
    return atoms, fixed


def make_state(atoms_list: list, model: FairChemModel):
    state = ts.initialize_state(atoms_list, device=model.device, dtype=model.dtype)
    zeros = torch.zeros(state.n_systems, dtype=torch.long, device=state.device)
    state.system_extras["charge"] = zeros
    state.system_extras["spin"] = zeros
    state.system_extras["orig_index"] = torch.arange(
        state.n_systems, dtype=torch.long, device=state.device
    )

    fixed_indices: list[int] = []
    offset = 0
    for atoms in atoms_list:
        fixed = fixed_mask_from_atoms(atoms)
        fixed_indices.extend((np.flatnonzero(fixed) + offset).tolist())
        offset += len(atoms)
    if fixed_indices:
        state.constraints = TorchSimFixAtoms(
            atom_idx=torch.as_tensor(
                fixed_indices, dtype=torch.long, device=state.device
            )
        )
    return state


def original_order(state) -> np.ndarray:
    n_systems = state.n_systems
    orig = state.system_extras.get("orig_index")
    if orig is None:
        return np.arange(n_systems)
    values = orig.detach().cpu().view(-1).numpy()
    expected = np.arange(n_systems)
    if not np.array_equal(np.sort(values), expected):
        raise ValueError(f"Invalid orig_index: {values.tolist()}")
    return np.argsort(values)


def optimize_batch(atoms_list: list, model: FairChemModel, args: argparse.Namespace):
    state = make_state(atoms_list, model)
    return ts.optimize(
        state,
        model,
        optimizer=(ts.lbfgs_init, ts.lbfgs_step),
        convergence_fn=ts.generate_force_convergence_fn(args.fmax),
        max_steps=args.steps,
        steps_between_swaps=1,
        autobatcher=False,
        pbar=False,
        init_kwargs={"step_size": 1.0, "alpha": 70.0},
        max_history=args.memory,
        max_step=args.maxstep,
    )


def optimize_dynamic(
    atoms_list: list,
    model: FairChemModel,
    args: argparse.Namespace,
):
    """Optimize all inputs through one convergence-aware in-flight queue.

    Each input is initialized as a one-system LBFGS state before entering the
    queue. This avoids materializing all structures as one GPU batch while still
    allowing ``InFlightAutoBatcher`` to refill slots as soon as systems converge.
    """

    if not atoms_list:
        raise ValueError("Dynamic optimization requires at least one structure")

    initialized = []
    natoms = []
    for index, atoms in enumerate(atoms_list):
        state = make_state([atoms], model)
        state.system_extras["orig_index"][0] = index
        optimized = ts.lbfgs_init(
            state=state,
            model=model,
            step_size=1.0,
            alpha=70.0,
        )
        optimized.system_extras["orig_index"][0] = index
        initialized.append(optimized)
        natoms.append(len(atoms))

    memory_scaler = args.max_memory_scaler
    if memory_scaler is None:
        # The previous production setting was batch-size structures. Using the
        # largest structure as the per-slot reference is conservative and keeps
        # the new queue within the established GPU-memory envelope.
        memory_scaler = float(args.batch_size * max(natoms))

    batcher = ts.InFlightAutoBatcher(
        model=model,
        memory_scales_with="n_atoms",
        max_memory_scaler=memory_scaler,
        max_iterations=args.steps,
    )
    batcher.load_states(initialized)

    step_fn = ts.lbfgs_step
    convergence_fn = ts.generate_force_convergence_fn(args.fmax)
    all_completed = []
    state = None
    convergence = None

    while True:
        state, completed = batcher.next_batch(state, convergence)
        all_completed.extend(completed)
        if state is None:
            break

        active_count = state.n_systems
        if active_count <= 0:
            raise RuntimeError("Dynamic batcher returned an empty active state")
        for _ in range(1):
            last_energy = getattr(state, "energy", None)
            state = step_fn(
                state=state,
                model=model,
                max_history=args.memory,
                max_step=args.maxstep,
            )
        convergence = convergence_fn(state, last_energy)

    ordered = batcher.restore_original_order(all_completed)
    return ts.concatenate_states(ordered)


def save_batch(
    final_state,
    originals: list,
    fixed_masks: list[np.ndarray],
    names: list[str],
    paths: list[Path],
    success_dir: Path,
    failed_dir: Path,
    started: float,
    fmax_threshold: float,
) -> list[RelaxResult]:
    relaxed = state_to_atoms(final_state)
    order = original_order(final_state)
    relaxed = [relaxed[i] for i in order]

    energies = final_state.energy.detach().cpu().view(-1).numpy()[order]
    fmax_values = ts.system_wise_max_force(final_state).detach().cpu().numpy()[order]
    n_iter = getattr(final_state, "n_iter", None)
    if n_iter is None:
        n_iter_values = np.full(len(relaxed), -1, dtype=int)
    else:
        n_iter_values = n_iter.detach().cpu().view(-1).numpy()[order]

    forces = final_state.forces.detach().cpu().numpy()
    state_atoms = state_to_atoms(final_state)
    split_points = np.cumsum([len(atoms) for atoms in state_atoms])[:-1]
    force_splits = np.split(forces, split_points)
    force_splits = [force_splits[i] for i in order]

    results: list[RelaxResult] = []
    for i, (name, path, atoms, original, fixed, force) in enumerate(
        zip(names, paths, relaxed, originals, fixed_masks, force_splits, strict=True)
    ):
        converged = bool(fmax_values[i] < fmax_threshold)
        target = success_dir if converged else failed_dir
        other = failed_dir if converged else success_dir
        xyz_path = target / f"{name}.extxyz"
        stale = other / f"{name}.extxyz"
        stale.unlink(missing_ok=True)

        atoms.info.update(original.info)
        atoms.info.update(
            {
                "uma_energy_eV": float(energies[i]),
                "uma_converged": converged,
                "uma_fmax_eV_A": float(fmax_values[i]),
                "uma_steps": int(n_iter_values[i]),
                "nfixed": int(fixed.sum()),
            }
        )
        atoms.set_constraint(ASEFixAtoms(mask=fixed.tolist()))
        atoms.arrays["forces"] = force
        atoms.calc = None
        write(xyz_path, atoms, format="extxyz")
        results.append(
            RelaxResult(
                name=name,
                input_path=str(path),
                output_xyz=str(xyz_path),
                converged=converged,
                steps=int(n_iter_values[i]),
                energy_eV=float(energies[i]),
                fmax_eV_A=float(fmax_values[i]),
                natoms=len(atoms),
                nfixed=int(fixed.sum()),
                elapsed_s=time.time() - started,
            )
        )
    return results


def save_failed(
    paths: list[Path],
    names: list[str],
    failed_dir: Path,
    started: float,
    error: Exception,
) -> list[RelaxResult]:
    results: list[RelaxResult] = []
    for path, name in zip(paths, names, strict=True):
        xyz_path = failed_dir / f"{name}.extxyz"
        try:
            atoms, fixed = load_atoms(path)
            atoms.info["uma_converged"] = False
            atoms.info["uma_error"] = repr(error)
            atoms.calc = None
            write(xyz_path, atoms, format="extxyz")
            natoms = len(atoms)
            nfixed = int(fixed.sum())
            output = str(xyz_path)
        except Exception:
            natoms = 0
            nfixed = 0
            output = None
        results.append(
            RelaxResult(
                name=name,
                input_path=str(path),
                output_xyz=output,
                converged=False,
                steps=0,
                energy_eV=None,
                fmax_eV_A=None,
                natoms=natoms,
                nfixed=nfixed,
                elapsed_s=time.time() - started,
                error=repr(error),
            )
        )
    return results


def main() -> int:
    args = parse_args()
    if not args.model_or_ckpt:
        raise SystemExit("--model-or-ckpt is required (or set UMA_MODEL)")
    if args.batch_size < 1 or args.steps < 1:
        raise SystemExit("--batch-size and --steps must be positive")

    try:
        load_runtime_dependencies()
    except ImportError as exc:
        raise SystemExit(
            "UMA runtime dependencies are unavailable. Run inside catflow2 "
            "or install numpy, torch, torch-sim, ASE, and fairchem-core."
        ) from exc

    out_dir = args.out_dir.expanduser().resolve()
    success_dir = out_dir / "relaxed" / "success"
    failed_dir = out_dir / "relaxed" / "failed"
    success_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(out_dir)
    paths = input_files(args)

    if args.overwrite:
        for directory in (success_dir, failed_dir):
            for pattern in ("*.extxyz",):
                for old in directory.glob(pattern):
                    old.unlink()

    pending: list[Path] = []
    for path in paths:
        name = path.stem
        if not args.overwrite and (
            (success_dir / f"{name}.extxyz").exists()
            or (failed_dir / f"{name}.extxyz").exists()
        ):
            continue
        pending.append(path)

    summary_path = out_dir / "results.jsonl"
    if args.overwrite:
        summary_path.write_text("", encoding="utf-8")
    else:
        summary_path.touch(exist_ok=True)

    logger.info("Structures: total=%d pending=%d", len(paths), len(pending))
    logger.info(
        "Device=%s GPU=%s batch=%d dynamic=%s fmax=%.4f steps=%d maxstep=%.4f",
        args.device,
        args.gpu_id,
        args.batch_size,
        args.dynamic_batch,
        args.fmax,
        args.steps,
        args.maxstep,
    )
    if not pending:
        logger.info("Nothing to do; all outputs already exist.")
        return 0

    model = FairChemModel(
        model=args.model_or_ckpt,
        device=torch.device(args.device),
        task_name=args.task_name,
        compute_stress=False,
    )

    started_all = time.time()
    processed = 0
    converged = 0
    failed = 0
    with summary_path.open("a", encoding="utf-8", buffering=1) as summary:
        chunk_starts = (
            [0]
            if args.dynamic_batch
            else range(0, len(pending), args.batch_size)
        )
        for chunk_start in chunk_starts:
            chunk_paths = pending[chunk_start : chunk_start + args.batch_size]
            if args.dynamic_batch:
                chunk_paths = pending
            names = [
                path.stem
                for path in chunk_paths
            ]
            chunk_started = time.time()
            try:
                loaded = [load_atoms(path) for path in chunk_paths]
                originals = [item[0] for item in loaded]
                fixed_masks = [item[1] for item in loaded]
                if args.dynamic_batch:
                    final_state = optimize_dynamic(originals, model, args)
                else:
                    final_state = optimize_batch(originals, model, args)
                results = save_batch(
                    final_state,
                    originals,
                    fixed_masks,
                    names,
                    chunk_paths,
                    success_dir,
                    failed_dir,
                    chunk_started,
                    args.fmax,
                )
            except Exception as exc:
                logger.exception("Batch %d failed: %s", chunk_start, exc)
                results = save_failed(chunk_paths, names, failed_dir, chunk_started, exc)
                if args.device == "cuda":
                    torch.cuda.empty_cache()

            for result in results:
                summary.write(json.dumps(asdict(result), cls=NumpyEncoder) + "\n")
                processed += 1
                if result.error or not result.converged:
                    failed += 1
                else:
                    converged += 1
            elapsed = max(time.time() - started_all, 1e-9)
            speed = processed / elapsed
            remaining = (len(pending) - processed) / max(speed, 1e-9) / 60
            logger.info(
                "Progress %d/%d | converged=%d failed=%d | %.3f struct/s | ETA %.1f min",
                processed,
                len(pending),
                converged,
                failed,
                speed,
                remaining,
            )

    logger.info("Done: processed=%d converged=%d failed=%d", processed, converged, failed)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
