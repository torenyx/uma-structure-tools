#!/usr/bin/env python3
"""Convert UMA XYZ/EXTXYZ outputs into VASP POSCAR files.

The converter preserves the unit cell, atom order, and fixed-atom information.
It reads ``move_mask`` when present (True means movable), then ASE FixAtoms
constraints, and finally the OC20 convention that ``tag == 0`` is fixed.
Each input becomes ``<out-dir>/<input-stem>/POSCAR``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_runtime_dependencies():
    import numpy as np
    from ase.constraints import FixAtoms
    from ase.io import read, write

    return np, FixAtoms, read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", action="append", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def input_files(args: argparse.Namespace) -> list[Path]:
    if args.input:
        files = [path.expanduser().resolve() for path in args.input]
    else:
        root = args.input_dir.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Input directory not found: {root}")
        files = sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in {".xyz", ".extxyz"}
        )
    if not files:
        raise FileNotFoundError("No XYZ/EXTXYZ input files found")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input file(s) not found: " + ", ".join(missing))
    return files


def fixed_mask(atoms, np, FixAtoms):
    move_mask = atoms.arrays.get("move_mask")
    if move_mask is not None:
        movable = np.asarray(move_mask, dtype=bool)
        if movable.ndim == 2:
            fully_movable = movable.all(axis=1)
            fully_fixed = (~movable).all(axis=1)
            if not np.all(fully_movable | fully_fixed):
                raise ValueError("Partial Cartesian move_mask is unsupported")
            movable = fully_movable
        if movable.ndim != 1 or len(movable) != len(atoms):
            raise ValueError("move_mask must contain one value per atom")
        return ~movable

    fixed = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise ValueError(
                "Only FixAtoms constraints can be converted to Selective Dynamics"
            )
        fixed[np.asarray(constraint.get_indices(), dtype=int)] = True
    if fixed.any():
        return fixed

    tags = np.asarray(atoms.get_tags())
    if len(tags) == len(atoms) and np.any(tags != 0):
        return tags == 0
    return fixed


def convert(path: Path, out_dir: Path, overwrite: bool, np, FixAtoms, read, write) -> Path:
    atoms = read(path)
    if abs(float(atoms.cell.volume)) < 1e-12:
        raise ValueError(f"{path}: a 3D cell is required for POSCAR output")

    fixed = fixed_mask(atoms, np, FixAtoms)
    atoms.set_constraint(FixAtoms(mask=fixed.tolist()))
    target_dir = out_dir / path.stem
    target = target_dir / "POSCAR"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {target}")
    target_dir.mkdir(parents=True, exist_ok=True)
    write(target, atoms, format="vasp", vasp5=True, direct=False, sort=False)
    return target


def main() -> int:
    args = parse_args()
    np, FixAtoms, read, write = load_runtime_dependencies()
    out_dir = args.out_dir.expanduser().resolve()
    files = input_files(args)
    for path in files:
        target = convert(path, out_dir, args.overwrite, np, FixAtoms, read, write)
        print(f"{path} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
