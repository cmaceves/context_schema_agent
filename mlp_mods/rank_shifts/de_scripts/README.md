# de_scripts — per-build paired-donor DE (one-off, params preserved)

One script per `(cell type, disease)` build, named `<build>.py`. Each is a **literal,
self-contained one-off** — the build-specific parameters (dataset id(s), cell type,
tissue, disease arms, DESeq design) are kept verbatim in the script rather than
abstracted into a shared config, so no build's exact recipe is ever lost.

Each script pulls both disease states from CellxGene, builds per-donor pseudobulk over
the matched dataset(s), runs PyDESeq2, and writes:

```
mlp_mods/rank_shifts/<build>_paired/
  pseudobulk_de.tsv       # gene, baseMean, log2FoldChange, padj, healthy_rank, <disease>_rank, rank_shift
  pulled_<cells>.h5ad     # cached raw cells (reruns skip the census pull)
```

That `pseudobulk_de.tsv` is the `de_table` a `de_ppi` build consumes (via
`02_build_ppi/builds_manifest.json` + `de_ppi/config.py`).

## Scripts

| script | dataset(s) | DESeq design | notes |
|---|---|---|---|
| `macrophage_crohn.py` | single IBD atlas `a37f857c` (both arms) | `~disease` | the original; small-intestine macrophages |
| `fibroblast_crohn.py` | 3 small-intestine datasets w/ both arms | `~dataset + disease` | batch covariate across datasets; drops single-arm datasets |

## Adding a build

Copy the closest existing script to `<build>.py`, then set: the dataset id(s) that carry
**both** a normal and a disease arm of your cell type/tissue, the `cell_type` /
`tissue_general` / `disease` filter, and the DESeq `design` (`~disease` for one dataset;
`~dataset + disease` when pooling several for batch control). Output dir + rank-column
names follow `<build>_paired/` and `<disease-arm>_rank` so `de_ppi/config.py` picks them up.

Run with the `.venv` (needs `cellxgene_census` + `pydeseq2`):

```bash
.venv/bin/python mlp_mods/rank_shifts/de_scripts/<build>.py
```
