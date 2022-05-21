# ReactionDiffusiononSurface

Solve reaction-diffusion equations on axisymmetric and deformed spherical surfaces.

## Description

Brusselator model is solved on an axisymmetric cylinder and a deformed sphere.
See Ref. [1] for details.

For Laplacian on a cylindrical surface, we used the following discretization where r(x) determine the surface shape. <br>　　　　

<img src="https://latex.codecogs.com/png.image?\inline&space;\dpi{120}\bg{white}~~x&space;\to&space;i\Delta_x,~~\theta&space;\to&space;j\Delta_\theta,~~~u(x,\theta)&space;\to&space;u_{ij}\\~~~~~~~\Delta&space;u&space;=&space;\frac{1}{r_{i}\sqrt{1&plus;{{r'}_{\!i}}^2}\Delta_x^2}\Biggl(\frac{r_{\!i&plus;\frac{1}{2}}}{\sqrt{1&plus;{{r'}_{\!\!i&plus;\frac{1}{2}}}^2}}\left(&space;u_{i&plus;1,j}-u_{i,j}\right)-\frac{r_{i-\frac{1}{2}}}{\sqrt{1&plus;{{r'}_{\!\!i-\frac{1}{2}}}^2}}\left(u_{i,j}-u_{i-1,j}\right)\Biggr)\\~~~~~~~~~~~~~~&plus;~~\frac{1}{r_{\!i}^2&space;&space;\Delta^2_\theta&space;}&space;\biggl(&space;u_{i,j&plus;1}&plus;u_{i,j-1}-2u_{ij}&space;\biggr)&space;\\&space;&space;&space;\\&space;"/>


<!-- $$
\begin{align}
\Delta u_{i} &= \frac{1}{r_{i}\sqrt{1+(\frac{dr}{dx})_{i}^2\Delta_x^2} 
               \Biggl( r_{i+\frac{1}{2}}\left( u_{i+1,j}-u_{i,j}\right)}{\sqrt{1+{r_{i+\frac{1}{2}}}^2}}
\end{align}
$$

$$
\begin{align}
\Delta u &= \frac{1}{r_{\!i}\sqrt{1+{{r'}_{\!i}}^2}\Delta_x^2} 
              \Biggl(  \frac{r_{i+\frac{1}{2}}\left( u_{i+1,j}-u_{i,j}\right)}{\sqrt{1+{{r'}_{\!\!i+\frac{1}{2}}}^2}}  
	                    -\frac{r_{i-\frac{1}{2}}\left( u_{i,j} - u_{i-1,j}\right)}{  \sqrt{1+{{r'}_{\!\!i-\frac{1}{2}}}^2}} 
              \Biggr)\\
	&~+~~ \frac{1}{r_{\!i}^2  \Delta^2_\theta } \biggl( u_{i,j+1}+u_{i,j-1}-2u_{ij} \biggr)
\end{align}
$$  -->

For Laplacian on triangular mesh, we used discretization  in Ref. [2].

## Requirement

* NumPy
* SciPy
* matplotlib


## Usage

Run scripts Brusselator_cylinder.py and Brusselator_sphere.py on IDE or IPython.
Model parameters can be changed by editing these scripts.


## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## References

1. R. Nishide and S. Ishihara <br>
"Pattern Propagation Driven by Surface Curvature" (2022) <br>

2. G. Xu <br>
"Discrete Laplace-Beltrami operator on sphere and optimal spherical triangulations"<br>
Int. J. Comput. Geom. Appl. 16, p.75--93 (2006) <br>
