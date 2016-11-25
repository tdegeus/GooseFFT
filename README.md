
<!-- MarkdownTOC -->

- [Introduction](#introduction)
- [Available examples](#available-examples)

<!-- /MarkdownTOC -->

# Introduction

This repository is primarily meant for those researchers who want to dive into the FFT-based micro-mechanical solver. It contains several small, single-file, Python scripts which should lower the learning curve. These examples depend only on standard Python and its scientific libraries [NumPy](http://www.numpy.org) and [SciPy](https://www.scipy.org) (which are widely used, and freely available for all operating systems), No custom software or libraries are used.

The examples are closely related to the following publication:

[1] T.W.J. de Geus, J. Vondřejc, J. Zeman, R.H.J. Peerlings, M.G.D. Geers. Finite strain FFT-based non-linear solvers made simple. Submitted, 2016. [arXiv: 1603.08893](http://arxiv.org/abs/1603.08893)

Since different approaches exists for this type of numerical method, and since there are many styles and programming languages, **anyone is invited to contribute** by:

1. Uploading similar small, preferably single-file, examples for different materials, in different programming languages, or featuring different approaches.

2. Updating this document with the state-of-the-art: an overview of which literature and what approaches are 'on the market'.

# Available examples

1. `finite-strain/hyper-elasticity.py` and `finite-strain/hyper-elasticity_even.py`: the main example of [1] featuring a simple hyper-elastic model in finite strain. As described in [1] the projection operator is slightly different for even or odd grids, here in included in two files. *Great to get started*.

2. `finite-strain/elasto-plasticity.py` and `finite-strain/elasto-plasticity_even.py`: an extension of example 1 with the Simo elasto-plastic model for finite strain. Furthermore, the definition of the projection operator is vectorized.

3. `finite-strain/elasto-plasticity_2D-micrograph`: a realistic example to simulate the mechanical response of the 2-D micrograph of dual-phase steel. Compared to example 2 this example deals with assuming 2-D plane strain.