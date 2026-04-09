# DiscreteDiffusion

This repository contains companion material for a mathematical tutorial on discrete diffusion models, with a focus on discrete space and discrete time.

## Main Links

- Tutorial paper (PDF): [tutorial/tutorial.pdf](tutorial/tutorial.pdf)
- Tutorial source (LaTeX): [tutorial/tutorial.tex](tutorial/tutorial.tex)
- Discrete Diffusion notebook on `ot4ml`: [python/8-discrete_diffusion.ipynb](https://github.com/gpeyre/ot4ml/blob/main/python/8-discrete_diffusion.ipynb)
- Open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeyre/ot4ml/blob/main/python/8-discrete_diffusion.ipynb)

## Repository Structure

- `tutorial/`
  - Main LaTeX notes in [tutorial/tutorial.tex](tutorial/tutorial.tex)
  - Compiled PDF in [tutorial/tutorial.pdf](tutorial/tutorial.pdf)
  - Bibliography in `tutorial/tutorial.bib`
  - Figure assets in `tutorial/figures/`
  - Auxiliary LaTeX material in `tutorial/aux/`

- `codes/`
  - Local working copy of the notebook in [codes/discrete_diffusion.ipynb](codes/discrete_diffusion.ipynb)

## Scope

The notes and notebook develop a discrete diffusion tutorial around:

- forward noising by Markov kernels,
- Bayes reverse kernels and their coupling interpretation,
- learning reverse transitions from samples,
- and the connection with continuous Gaussian diffusion and denoising score matching.

The public notebook to read or run is the one maintained in the `ot4ml` repository; the notebook stored here is a local companion working version for this project.
