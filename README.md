
> **Disclaimer**
> 
> This repository is distributed under an MIT license. In layman's term, you are free to do what you want with it. We just kindly request you to cite our papers that form the basis for this repository, and encourage you to be equally open with your code.
> 
> (c) Tom de Geus (tom@geus.me), Jaroslav Vondřejc, 2016

# Contents

- [Introduction](#introduction)
- [Examples](#examples)
- [References](#references)

# Introduction

This repository is primarily meant for those researchers who want to dive into the FFT-based micro-mechanical solver. It contains several small, single file, Python scripts which should lower the learning curve (see Ref. [[1]](http://arxiv.org/abs/1603.08893), which was written with this particular aim). These examples are formulated both in small strain (with linear and non-linear elastic, elasto-plastic, elasto-visco-plastic constitutive models; accompanying [[2]](http://arxiv.org/abs/1601.05970)), and finite strain (with hyper-elastic and elasto-plastic constitutive models; accompanying [[1]](http://arxiv.org/abs/1603.08893)). The examples only depend on standard [Python](https://www.python.org) and its scientific libraries [NumPy](http://www.numpy.org) and [SciPy](https://www.scipy.org), which are freely available for all operating systems and are widely used. No custom software or libraries are used, i.e. there are **no hidden layers of complexity**.

Since different approaches exists for this type of numerical method, and since there are many styles and programming languages, anyone is invited to contribute by uploading similar small, preferably single file, examples for different materials, in different programming languages, or featuring different approaches.

# Examples

The examples are divided in three categories:

- [Basic methodology](#basic-methodology). These examples focus on accessibility while retaining some computational efficiency. **Great to get started.**

- [Different materials models](#different-materials-models). These examples focus on different constitutive relations (*which are completely uncoupled from the FFT-solver*). At the same time, they are implemented more general and more efficiently than the examples that present the basic methodology.

- [Applications](#applications). These examples show just how powerful the FFT-solver can be, even with the very simple Python code. Also, they introduce the concept of 2-D plane strain.

## Basic methodology

*   [`finite-strain/hyper-elasticity.py`](./finite-strain/hyper-elasticity.py) and [`finite-strain/hyper-elasticity-even.py`](./finite-strain/hyper-elasticity-even.py): the main example of Ref. [[1]](http://arxiv.org/abs/1603.08893) featuring a simple hyper-elastic model in finite strain. As described in Ref. [[1]](http://arxiv.org/abs/1603.08893) the projection operator is slightly different for even or odd grids, here included in two files.

*   [`small-strain/linear-elasticity.py`](./small-strain/linear-elasticity.py) and [`small-strain/linear-elasticity-even.py`](./small-strain/linear-elasticity-even.py). The small strain equivalent (linear) of the previous example. Notice that the projection operator is more involved, as the symmetry of the strain tensor has to be enforced.

## Different materials models

*   [`finite-strain/elasto-plasticity.py`](./finite-strain/elasto-plasticity.py) and [`finite-strain/elasto-plasticity_even.py`](./finite-strain/elasto-plasticity_even.py): the Simo elasto-plastic constitutive model for finite strain [[1]](http://arxiv.org/abs/1603.08893). Furthermore, the definition of the projection operator is vectorized.

*   [`small-strain/laminate/non-linear-elasticity.py`](./small-strain/laminate/non-linear-elasticity.py): a laminate of a non-linear elastic and a linear elastic material. For this configuration an analytical solution is available [[2]](http://arxiv.org/abs/1601.05970).

*   [`small-strain/laminate/elasto-plasticity.py`](./small-strain/laminate/elasto-plasticity.py): a laminate of an elasto-plastic and a linear elastic material. For this configuration an analytical solution is available [[2]](http://arxiv.org/abs/1601.05970).

*   [`small-strain/laminate/visco-plasticity.py`](./small-strain/laminate/visco-plasticity.py): a laminate of an elasto-visco-plastic and a linear elastic material. For this configuration an analytical solution is available [[2]](http://arxiv.org/abs/1601.05970).

## Applications

*   [`finite-strain/elasto-plasticity_2D-micrograph`](./finite-strain/elasto-plasticity_2D-micrograph): a realistic example to simulate the mechanical response of the 2-D micrograph of dual-phase steel [[1]](http://arxiv.org/abs/1603.08893).

*   [`small-strain/micrograph`](./small-strain/micrograph): the same example as the previous one, but in small strain [[2]](http://arxiv.org/abs/1601.05970).

# References

[1]: T.W.J. de Geus, J. Vondřejc, J. Zeman, R.H.J. Peerlings, M.G.D. Geers. Finite strain FFT-based non-linear solvers made simple. Computer Methods in Applied Mechanics and Engineering, 2017, 318:412–430. [doi: 10.1016/j.cma.2016.12.032](https://doi.org/10.1016/j.cma.2016.12.032), [arXiv: 1603.08893](http://arxiv.org/abs/1603.08893)

[2]: J. Zeman, T.W.J. de Geus, J. Vondřejc, R.H.J. Peerlings, M.G.D. Geers. A finite element perspective on nonlinear FFT-based micromechanical simulations. International Journal for Numerical Methods in Engineering, 2017, 111(10):903–926. [doi: 10.1002/nme.5481](https://doi.org/10.1002/nme.5481), [arXiv: 1601.05970](http://arxiv.org/abs/1601.05970)
