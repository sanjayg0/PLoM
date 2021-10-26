## Calucation workflow

The algorithm can be divided into four major steps in sequence: **Data Normalization**, 
**Principle Component Analysis**, **Kernerl Density Estimation**, and **ISDE Generation of the Random Variable**.

### Data Normalization
The initial sample **X<sub>r</sub>** typically has wide ranges of values given different quantities of interest. 
Hence, the standard scaling is conducted so that the resulting sample data **X** are in the range of [0,1]:

<img src="https://render.githubusercontent.com/render/math?math=[X]_{ij}=\frac{[X_r]_{ij}-min_k[X_r]_{ik}}{max_k[X_r]_{ik}-min_k[X_r]_{ik}}">

### Principle Component Analysis
Then, a principal component analysis (PCA) of X of dimension <img src="https://render.githubusercontent.com/render/math?math=\nu \leq n"> for 
the initial sample <img src="https://render.githubusercontent.com/render/math?math=x^j, j = 1,2,...,N">, which leads to the random vector **H**
with values in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^\nu"> centered:

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{H}=[\mu]^{-\frac{1}{2}}[\Phi]^T(\mathbf{X}-\hat{\mathbf{x}})">

where <img src="https://render.githubusercontent.com/render/math?math=[\mu]"> is the diagonal matrix of eigenvalues;
<img src="https://render.githubusercontent.com/render/math?math=[\Phi]=[\phi^1, \phi^2,...,\phi^\nu]"> is the orthogonal eigenvectors;
<img src="https://render.githubusercontent.com/render/math?math=\hat{\mathbf{x}}"> is the mean vector of 
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{X}">.

### Kernerl Density Estimation
Following the PCA, a non-parametric representation of probability density function <img src="https://render.githubusercontent.com/render/math?math=p_H">,
using  Gaussian kernel density estimation, is developed:

<img src="https://render.githubusercontent.com/render/math?math=p_{\mathbf{H}}(\eta) = \frac{1}{(\sqrt{2\pi}\hat{s}_\nu)^\nu}\rho(\eta)">
<img src="https://render.githubusercontent.com/render/math?math=\rho(\mathbf{\eta}) = \frac{1}{N}\sum_{j=1}^{N} exp\left\{-\frac{1}{2\hat{s}^2_\nu} \left\lVert \frac{\hat{s}_\nu}{s_\nu}(\mathbf{\eta}^j - \mathbf{\eta}) \right\rVert^2 \right\}">
<img src="https://render.githubusercontent.com/render/math?math=s_\nu = \left\{ \frac{4}{N(2 %2B \nu)} \right\}^{\frac{1}{\nu %2B 4}}">
<img src="https://render.githubusercontent.com/render/math?math=\hat{s}_\nu = \frac{s_\nu}{\sqrt{s_{\nu}^2 %2B \frac{N-1}{N}}}">

### ISDE Generation of the Random Variable
The construction of a nonlinear Ito Stochastic Differential Equation (ISDE) to generate realizations of 
random variable <img src="https://render.githubusercontent.com/render/math?math=\mathbf{H}"> 
allows us to compute the empirical gradient and covariance.

<img src="https://render.githubusercontent.com/render/math?math=d\left[\mathbf{U}(t)\right]=\left[\mathbf{V}(t)\right] d t">
<img src="https://render.githubusercontent.com/render/math?math=d\left[\mathbf{V}(t)\right]=\left[L\left(\left[\mathbf{U}(t)\right]\right)\right] dt-\frac{1}{2} f_{0}\left[\mathbf{V}(t)\right] dt %2B \sqrt{f_{0}} d\left[\mathbf{W}^{\text {wien}}(t)\right]">

where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{U}"> and 
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{V} \in \mathbb{M}_{\nu,N}"> and 
with the initial condition at <img src="https://render.githubusercontent.com/render/math?math=t = 0">.

<img src="https://render.githubusercontent.com/render/math?math=\left[\mathbf{U}(0)\right]=\left[\eta^{\text {init }}\right], \quad\left[\mathbf{V}(0)\right]=\left[\nu^{\text {init }}\right]">

The presented ISDE admits a unique invariant measure, proved in the work of Soize (),

<img src="https://render.githubusercontent.com/render/math?math=\otimes_{\ell=1}^{N}\left\{p_{\mathbf{H}}\left(\mathbf{u}^{\prime}\right) p_{\mathbf{G}}\left(\mathbf{v}^{\prime}\right) d \mathbf{u}^{\prime} d \mathbf{v}^{\prime} \right\}">

The Hamiltonian of the associated dynamical system 
<img src="https://render.githubusercontent.com/render/math?math=$\left\{\left(\mathbf{U}^{\ell}(t), \mathbf{V}^{\ell}(t)\right), t \in \mathbb{R}^ %2B \right\}"> 
related to stochastic process is, <img src="https://render.githubusercontent.com/render/math?math=\mathbb{H}\left(\mathbf{u}^{\ell}, \mathbf{v}^{\ell}\right)=\frac{1}{2}\left\|\mathbf{v}^{\ell}\right\|^{2} %2B \mathcal{V}\left(\mathbf{u}^{\ell}\right)">, 
where 

<img src="https://render.githubusercontent.com/render/math?math=p_{\mathbf{H}}(\boldsymbol{\eta})=c_{0} \rho(\boldsymbol{\eta})">
<img src="https://render.githubusercontent.com/render/math?math=\rho(\boldsymbol{\eta})=\exp \left\{-\mathcal{V}(\boldsymbol{\eta})\right\}, \quad \mathcal{V}(\boldsymbol{\eta})=\psi(\boldsymbol{\eta})">

In this way, the realizations of <img src="https://render.githubusercontent.com/render/math?math=\left[\mathbf{H}\right]"> are obtained by:

<img src="https://render.githubusercontent.com/render/math?math=\left[\mathbf{H}\right]=\left[\mathbf{U}^{\mathrm{st}}\left(t_{\mathrm{st}}\right)\right]=\lim _{t \rightarrow %2B \infty}\left[\mathbf{U}(t)\right]">


