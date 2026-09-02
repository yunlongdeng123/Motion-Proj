# CVPR / arXiv submission handoff

## Current frozen package

- Branch: `research/worldsim-v7-harp3d-cvpr`, based on the terminal V6.7 research branch.
- Anonymous review source: `main.tex` with `\usepackage[review]{cvpr}` and `Anonymous Authors`.
- Non-anonymous source entry: `arxiv.tex`; it shares `manuscript_metadata.tex` and `manuscript_body.tex` with `main.tex` and
  requires the intentionally untracked `arxiv_author_metadata.tex`.
- Main PDF: eight content pages plus one references-only page.
- Supplement PDF: ten pages.
- Results are macro-driven from `results/results_macros.tex`; scientific values must not be copied or edited by hand.
- The latest official public template is CVPR 2026. `TEMPLATE_PROVENANCE.md` records why this is a provisional CVPR 2027 draft.
- V7 method and evaluation expansion is closed. Only submission metadata, an official-kit migration, formatting repair, or a
  legitimate source bug may change this package.

## External inputs still required

These fields cannot be inferred from the repository and must be supplied from real submission metadata:

- final author order, affiliations, and contact information for the non-anonymous arXiv version;
- arXiv primary/secondary categories, license choice, and any acknowledgement text;
- conference paper ID after registration;
- the official CVPR 2027 author kit and final 2027 policy when released.

Do not invent placeholders beyond the existing anonymous author and paper-ID fields, and do not expose private author metadata
in the anonymous review source.

## arXiv release procedure

1. Create the ignored `arxiv_author_metadata.tex` with the supplied real `\author{...}` block; do not edit the anonymous entry.
2. Build `arxiv.tex`, which already uses the official final-style path with page numbers (`\usepackage[pagenumbers]{cvpr}` for
   the currently pinned kit) and has no review-only paper-ID contract.
3. Keep the title, abstract, method, main results, limitations, and supplement scientifically identical to the anonymous version.
4. Compile the main paper and supplement; confirm that all cited figures/tables and bibliography files are included in the source
   package. Do not add project-page or video links until their public release and anonymity status are explicitly decided.
5. Record the submitted arXiv identifier and version here only after the upload succeeds.

## Conference submission procedure

1. When the official CVPR 2027 kit appears, replace only the official style/template files and migrate the documented entry-point
   differences; do not silently use a third-party template.
2. Rebuild in anonymous review mode, insert the assigned paper ID, and recheck the then-current page limit and anonymity policy.
3. Keep pages beyond the content limit references-only. Do not shrink fonts, bibliography spacing, or margins to manufacture fit.
4. Keep the submission and supplement self-contained. In particular, do not add external project, image, or video links that
   extend the reviewed content.
5. Freeze scientific content at upload. Do not use later test-set reads to alter the selected method or headline claims.

## Ready-to-release evidence map

- Claim ownership and prohibited upgrades: `CONTRIBUTION_MAP.md`.
- Qualitative/video identities without duplicated binaries: `PROJECT_PAGE_ASSET_INDEX.md`.
- Template source and official-policy status: `TEMPLATE_PROVENANCE.md`.
- Canonical experiment history and negative evidence: `../docs/EXPERIMENTS.md` and `../docs/RESEARCH_FAILURES.md`.

Current terminal state: the anonymous provisional manuscript is ready; arXiv and conference packaging are waiting only on the
external metadata and official-kit items listed above, not on additional research runs.
