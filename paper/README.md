# HARP-3D paper draft

The manuscript uses the official `cvpr-org/author-kit` style pinned in `TEMPLATE_PROVENANCE.md`.
The exact external inputs and mode switches for anonymous conference and non-anonymous arXiv packages are tracked in
`SUBMISSION_CHECKLIST.md`.

`main.tex` and `arxiv.tex` share `manuscript_metadata.tex`, `manuscript_body.tex`, every section, table, figure, bibliography,
and result macro. The only identity-bearing file is `arxiv_author_metadata.tex`; it is intentionally ignored by Git and must
contain the real `\author{...}` block before building the arXiv version.

Build from this directory with a TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
# after supplying arxiv_author_metadata.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error arxiv.tex
```

All experiment values enter through `results/results_macros.tex`. Do not paste development metrics directly into section text or tables.

The frozen 30-panel/30-video qualitative package is resolved by `PROJECT_PAGE_ASSET_INDEX.md`; the binary bundle remains in
the referenced canonical AutoDL run rather than being duplicated in Git.
