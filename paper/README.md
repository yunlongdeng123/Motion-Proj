# HARP-3D paper draft

The manuscript uses the official `cvpr-org/author-kit` style pinned in `TEMPLATE_PROVENANCE.md`.

Build from this directory with a TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

All experiment values enter through `results/results_macros.tex`. Do not paste development metrics directly into section text or tables.

The frozen 30-panel/30-video qualitative package is resolved by `PROJECT_PAGE_ASSET_INDEX.md`; the binary bundle remains in
the referenced canonical AutoDL run rather than being duplicated in Git.
