# HARP-3D paper draft

The manuscript uses the official `cvpr-org/author-kit` style pinned in `TEMPLATE_PROVENANCE.md`.

Build from this directory with a TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

All experiment values enter through `results/results_macros.tex`. Do not paste development metrics directly into section text or tables.
