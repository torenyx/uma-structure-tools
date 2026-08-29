#!/usr/bin/env python3
"""Extract UMA atom embeddings, energies, and forces from tagged XYZ files.

The script stores one ``chunk_*.pt`` file per group of structures.  It captures
the UMA backbone ``node_embedding`` after the final backbone norm and keeps the
first scalar channel (``emb_l0``).  ``o_emb_l0`` is the subset with
``tag == 2`` and atomic number 8, and ``slab_emb_l0`` is the mean over
``tag != 2`` atoms.  These definitions match the OC20 adsorbate convention
used by the analysis scripts in this project.

Example::

    python extract_uma_o_embeddings.py \
        --ckpt /path/to/uma-s-1p2.pt \
        --xyz-dirs /path/to/xyz_a /path/to/xyz_b \
        --out-dir /path/to/embeddings
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

def load_runtime_dependencies() -> None:
    global torch, read, pretrained_mlip, data_list_collater, AtomicData, DataLoader
    import torch
    from ase.io import read
    from fairchem.core import pretrained_mlip
    from fairchem.core.datasets import data_list_collater
    from fairchem.core.datasets.atomic_data import AtomicData
    from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="UMA checkpoint.")
    parser.add_argument(
        "--xyz-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Directories containing tagged .xyz files.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-name", default="oc20")
    parser.add_argument(
        "--inference-settings",
        default="default",
        choices=("default", "turbo", "traineval"),
        help="Inference preset passed to fairchem.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many files after the global sorted file list is built.",
    )
    return parser.parse_args()


def logger_for(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("extract_uma_o_embeddings")


def numeric_name_key(path: Path) -> tuple[str, int, str]:
    try:
        number = int(path.stem)
    except ValueError:
        number = 10**18
    return path.parent.name, number, path.name


def collect_xyz_files(xyz_dirs: list[Path]) -> list[Path]:
    files = [path for directory in xyz_dirs for path in directory.glob("*.xyz")]
    return sorted(files, key=numeric_name_key)


def next_chunk_index(out_dir: Path) -> int:
    indices = []
    for path in out_dir.glob("chunk_*.pt"):
        try:
            indices.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return max(indices, default=-1) + 1


def tagged_atomic_data(path: Path, task_name: str) -> AtomicData:
    atoms = read(path)
    if "tags" not in atoms.arrays and "tag" in atoms.arrays:
        atoms.set_tags(atoms.arrays["tag"])
    if "tags" not in atoms.arrays:
        raise ValueError(f"Missing tags array in {path}")

    tags = torch.as_tensor(atoms.get_tags(), dtype=torch.long)
    if tags.numel() != len(atoms):
        raise ValueError(f"Invalid tags length in {path}")

    data = AtomicData.from_ase(atoms, task_name=task_name, r_edges=False)
    # OC20 convention: tag 0 is fixed; positive tags are movable.
    data.tags = tags
    data.fixed = (tags == 0).long()
    return data


class TaggedXyzDataset:
    def __init__(self, files: list[Path], task_name: str) -> None:
        self.files = files
        self.task_name = task_name

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[AtomicData, str]:
        path = self.files[index]
        return tagged_atomic_data(path, self.task_name), f"{path.parent.name}/{path.name}"


def collate(samples: list[tuple[AtomicData, str]]) -> tuple[AtomicData, list[str]]:
    data, names = zip(*samples)
    return data_list_collater(list(data), otf_graph=True), list(names)


def empty_chunk() -> dict[str, list[Any]]:
    return {
        "filenames": [],
        "emb_l0": [],
        "o_emb_l0": [],
        "slab_emb_l0": [],
        "o_indices": [],
        "o_forces": [],
        "energies": [],
        "forces": [],
        "forces_all": [],
        "tags": [],
        "fixed": [],
        "natoms": [],
    }


def capture_node_embedding(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], Any]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: dict[str, torch.Tensor]) -> None:
        captured["node_embedding"] = output["node_embedding"].detach().cpu()

    return captured, model.backbone.register_forward_hook(hook)


def batch_records(
    batch: AtomicData,
    names: list[str],
    captured: dict[str, torch.Tensor],
    prediction: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    embedding = captured.get("node_embedding")
    if embedding is None:
        raise RuntimeError("UMA backbone hook did not capture node_embedding")

    batch_index = batch.batch.detach().cpu()
    atomic_numbers = batch.atomic_numbers.detach().cpu()
    tags_all = batch.tags.detach().cpu().long()
    fixed_all = batch.fixed.detach().cpu().long()
    energies = prediction["energy"].detach().cpu().view(-1)
    forces_all = prediction["forces"].detach().cpu().float()
    if forces_all.shape[0] != atomic_numbers.shape[0]:
        raise RuntimeError(
            f"Expected one force vector per atom; got {forces_all.shape[0]} forces "
            f"for {atomic_numbers.shape[0]} atoms"
        )

    records = []
    for sample_index, name in enumerate(names):
        atom_mask = batch_index == sample_index
        tags = tags_all[atom_mask]
        fixed = fixed_all[atom_mask]
        numbers = atomic_numbers[atom_mask]
        adsorbate_o = (tags == 2) & (numbers == 8)
        if not torch.any(adsorbate_o):
            raise ValueError(f"No adsorbate O atom (tag == 2, Z == 8) found in {name}")
        slab_atoms = tags != 2
        if not torch.any(slab_atoms):
            raise ValueError(f"No slab atoms (tag != 2) found in {name}")

        forces = forces_all[atom_mask]
        emb_l0 = embedding[atom_mask, 0, :].float()
        records.append(
            {
                "filename": name,
                "emb_l0": emb_l0.clone(),
                "o_emb_l0": emb_l0[adsorbate_o].clone(),
                "slab_emb_l0": emb_l0[slab_atoms].mean(dim=0).clone(),
                "o_indices": torch.nonzero(adsorbate_o, as_tuple=False).flatten().long(),
                "o_forces": forces[adsorbate_o].clone(),
                "energy": float(energies[sample_index]),
                "forces": forces[fixed == 0].clone(),
                "forces_all": forces.clone(),
                "tags": tags.clone(),
                "fixed": fixed.clone(),
                "natoms": int(atom_mask.sum()),
            }
        )
    return records


def add_records(chunk: dict[str, list[Any]], records: list[dict[str, Any]]) -> None:
    for record in records:
        chunk["filenames"].append(record["filename"])
        chunk["emb_l0"].append(record["emb_l0"])
        chunk["o_emb_l0"].append(record["o_emb_l0"])
        chunk["slab_emb_l0"].append(record["slab_emb_l0"])
        chunk["o_indices"].append(record["o_indices"])
        chunk["o_forces"].append(record["o_forces"])
        chunk["energies"].append(record["energy"])
        chunk["forces"].append(record["forces"])
        chunk["forces_all"].append(record["forces_all"])
        chunk["tags"].append(record["tags"])
        chunk["fixed"].append(record["fixed"])
        chunk["natoms"].append(record["natoms"])


def save_chunk(
    out_dir: Path,
    index: int,
    chunk: dict[str, list[Any]],
    checkpoint: Path,
    task_name: str,
    inference_settings: str,
    logger: logging.Logger,
) -> None:
    if not chunk["filenames"]:
        return

    path = out_dir / f"chunk_{index:04d}.pt"
    torch.save(
        {
            "filenames": chunk["filenames"],
            "emb_l0": chunk["emb_l0"],
            "o_emb_l0": chunk["o_emb_l0"],
            "slab_emb_l0": chunk["slab_emb_l0"],
            "o_indices": chunk["o_indices"],
            "o_forces": chunk["o_forces"],
            "energies": torch.tensor(chunk["energies"], dtype=torch.float32),
            "forces": chunk["forces"],
            "forces_all": chunk["forces_all"],
            "tags": chunk["tags"],
            "fixed": chunk["fixed"],
            "natoms": torch.tensor(chunk["natoms"], dtype=torch.long),
            "model": checkpoint.stem,
            "task_name": task_name,
            "inference_settings": inference_settings,
            "embedding_policy": (
                "emb_l0 is output['node_embedding'][:, 0, :] from the UMA backbone "
                "after final norm and before the energy/force head. o_emb_l0 is "
                "the subset with tag == 2 and atomic number == 8. slab_emb_l0 is "
                "the mean of emb_l0 over atoms with tag != 2."
            ),
            "force_policy": (
                "forces_all stores raw model forces on all atoms; forces stores "
                "forces_all[fixed == 0], with fixed mapped from tag == 0."
            ),
        },
        path,
    )
    logger.info(
        "Saved %s | structures=%d | %.1f MB",
        path,
        len(chunk["filenames"]),
        path.stat().st_size / 1024**2,
    )


def load_model(checkpoint: Path, device: str, inference_settings: str) -> Any:
    unit = pretrained_mlip.load_predict_unit(
        checkpoint,
        device=device,
        inference_settings=inference_settings,
    )
    model = unit.model.module if hasattr(unit.model, "module") else unit.model
    return unit, model


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.chunk_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size and chunk-size must be positive; num-workers cannot be negative")
    if args.start_index < 0:
        raise ValueError("start-index cannot be negative")

    load_runtime_dependencies()
    logger = logger_for(args.out_dir)
    files = collect_xyz_files(args.xyz_dirs)
    files = files[args.start_index :]
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError("No .xyz files matched --xyz-dirs")

    logger.info("Processing %d structures with %s", len(files), args.device)
    unit, model = load_model(args.ckpt, args.device, args.inference_settings)
    captured, hook_handle = capture_node_embedding(model)

    loader_options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": args.device.startswith("cuda"),
    }
    if args.num_workers:
        loader_options.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(TaggedXyzDataset(files, args.task_name), **loader_options)

    chunk_index = next_chunk_index(args.out_dir)
    chunk = empty_chunk()
    processed = failed = 0
    started = time.perf_counter()
    try:
        for batch_index, (batch, names) in enumerate(loader, start=1):
            try:
                captured.clear()
                batch = batch.to(args.device)
                prediction = unit.predict(batch)
                records = batch_records(batch, names, captured, prediction)
                add_records(chunk, records)
                processed += len(records)
                if len(chunk["filenames"]) >= args.chunk_size:
                    save_chunk(
                        args.out_dir,
                        chunk_index,
                        chunk,
                        args.ckpt,
                        args.task_name,
                        args.inference_settings,
                        logger,
                    )
                    chunk_index += 1
                    chunk = empty_chunk()
            except Exception:
                failed += len(names)
                logger.exception("Batch %d failed (%d structures)", batch_index, len(names))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
    finally:
        save_chunk(
            args.out_dir,
            chunk_index,
            chunk,
            args.ckpt,
            args.task_name,
            args.inference_settings,
            logger,
        )
        hook_handle.remove()

    elapsed = time.perf_counter() - started
    logger.info(
        "Finished | processed=%d | failed=%d | elapsed=%.1f min",
        processed,
        failed,
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
