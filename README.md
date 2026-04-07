# DiscreteDiffusion

This repository is a tutorial workspace for discrete diffusion models.

## Purpose

The project builds a pedagogical bridge between:
- forward/reverse Markov chains on finite state spaces,
- practical simulation in PyTorch,
- and learning reverse kernels in the style of score/denoising methods.

## Repository content

- `codes/discrete_diffusion.ipynb`
  - Main notebook tutorial.
  - Two forward kernels on `n=30` states:
    - geometric local random walk on a cycle,
    - masking/absorbing kernel with a trash-bin state.
  - Forward simulation (particles + histogram recursion).
  - Backward simulation using Bayes inverse kernels.
  - Learning section: 2-layer neural network approximating time-dependent reverse kernels `Q^t`.

- `papers/tutorial/tutorial.tex`
  - Main tutorial paper in LaTeX.
  - Unified notation with the notebook (`h^{t+1}=P^\top h^t`, Bayes kernel `Q^t`).
  - Forward process, backward process, admissible kernels and transport polytope viewpoint, and learning objective.

- `papers/internship/internship.tex`
  - Clean LaTeX version of the internship project description.

- `papers/internship/references.bib`
  - BibTeX database for internship references.

- `biblio/summaries/*/summary.tex`
  - Condensed tutorial summaries of the papers stored in `biblio/sources/`.

## Notes

- `biblio/sources/` is considered local reference material and is ignored by git for future additions.
- The notebook and paper are designed to use consistent symbols and indexing.
