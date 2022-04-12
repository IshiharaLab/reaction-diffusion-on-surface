# ReactionDiffusiononSurface

Solve reaction-diffusion equations on axisymmetric and deformed spherical surfaces.

## Description

Brusselator model is solved on an axisymmetric cylinder and a deformed sphere.
See Ref. [1] for details.

For Laplacian on triangular mesh, we used discretization used in Ref. [2].

## Requirement

* NumPy
* SciPy
* matplotlib


## Usage

Run scripts Brusselator_cylinder.py and Brusselator_sphere.py on IDE or IPython.
Model parameters can be changed by editing these scripts.

$a333\sin$

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## References

1. R. Nishide and S. Ishihara <br>
"Pattern Propagation Driven by Surface Curvature" (2022) <br>

2. G. Xu <br>
"Discrete Laplace-Beltrami operator on sphere and optimal spherical triangulations"<br>
Int. J. Comput. Geom. Appl. 16, p.75--93 (2006) <br>
