# UMA structure tools

three small command-line tools for catalytic-structure workflows with UMA:

- `uma_relax_batch.py`: batched torch-sim L-BFGS geometry optimization.
- `uma_to_vasp.py`: conversion of relaxed XYZ/EXTXYZ structures to POSCAR.
- `extract_uma_o_embeddings.py`: UMA inference and extraction of atom, O-site,
  and slab-level embeddings.

## Environment

Run in an environment containing Python 3.10+, PyTorch, ASE, `torch-sim`, and
`fairchem-core`. The UMA checkpoint is downloaded separately and supplied on
the command line.

## Optimize structures

```bash
python uma_relax_batch.py \
  --input-dir ./structures \
  --out-dir ./relaxed \
  --model-or-ckpt /path/to/uma-s-1p2.pt \
  --device cuda \
  --gpu-id 0 \
  --batch-size 72
```

Inputs may also be supplied with repeated `--input` arguments. ASE constraints,
`move_mask`, and OC20-style tags are preserved. Relaxed structures and
`results.jsonl` are written below `--out-dir`.

## Convert UMA outputs to POSCAR

```bash
python uma_to_vasp.py \
  --input-dir ./relaxed/ \
  --out-dir ./vasp_inputs
```

The converter writes one `POSCAR` per input under `--out-dir`, preserving the
cell, atom order, and Selective Dynamics flags. It does not create `POTCAR`,
`INCAR`, or scheduler files.

## Extract embeddings

```bash
python extract_uma_o_embeddings.py \
  --ckpt /path/to/uma-s-1p2.pt \
  --xyz-dirs ./xyz_a ./xyz_b \
  --out-dir ./embeddings \
  --device cuda \
  --batch-size 32
```

Each `chunk_*.pt` contains:

- `emb_l0`: the first scalar channel of the UMA backbone embedding for every atom;
- `o_emb_l0`: entries with `tag == 2` and atomic number 8;
- `slab_emb_l0`: the mean of `emb_l0` over atoms with `tag != 2`;
- model energies, all-atom forces, free-atom forces, tags, and fixed masks.

The extraction order is deterministic: files are sorted by input-directory name
and numeric filename. Existing chunks are continued by choosing the next chunk
index; use a new output directory for a clean run.
