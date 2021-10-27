## Calucation workflow

The algorithm can be divided into four major steps in sequence: **Data Normalization**, 
**Principle Component Analysis**, **Kernerl Density Estimation**, and **ISDE Generation of the Random Variable**.
For details, see 
([Soize and Ghanem, 2016](https://doi.org/10.1016/j.jcp.2016.05.044) and [Soize and Ghanem, 2019](https://doi.org/10.1002/nme.6202)).
However, in brief, the following summarizes the steps:

### Data Normalization
The initial sample **X<sub>r</sub>** typically has wide ranges of values given different quantities of interest. 
As a first step, the data is scaled and shifted so that the utilized sample data **X** are in the range of [0,1]:

<img src="https://render.githubusercontent.com/render/math?math=[X]_{ij}=\frac{[X_r]_{ij}-min_k[X_r]_{ik}}{max_k[X_r]_{ik}-min_k[X_r]_{ik}}">

### Principle Component Analysis
Next, a principal component analysis (PCA) of X is performed to find a lower dimensional space in which the variance of the data can
be explained. Thus a suitable <img src="https://render.githubusercontent.com/render/math?math=\nu \leq n"> 
is sought whereby the first <img src="https://render.githubusercontent.com/render/math?math=\nu"> principal components can explain
the data to a given accuracy.
This leads to the random vector **H** in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^\nu">
such that

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{H}=[\mu]^{-\frac{1}{2}}[\Phi]^T(\mathbf{X}-\hat{\mathbf{x}})">

where <img src="https://render.githubusercontent.com/render/math?math=[\mu]"> is the diagonal matrix of the 
<img src="https://render.githubusercontent.com/render/math?math=\nu"> top eigenvalues and
<img src="https://render.githubusercontent.com/render/math?math=[\Phi]=[\phi^1, \phi^2,...,\phi^\nu]"> is the matrix of the
corresponding orthogonal eigenvectors of the data's convariance matrix.
<img src="https://render.githubusercontent.com/render/math?math=\hat{\mathbf{x}}"> is the mean vector of 
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{X}">.

### Kernerl Density Estimation
Following the PCA, a non-parametric representation of the probability density function <img src="https://render.githubusercontent.com/render/math?math=p_H">,
using Gaussian kernel density estimation, is developed:

<img src="https://render.githubusercontent.com/render/math?math=p_{\mathbf{H}}(\eta) = \frac{1}{(\sqrt{2\pi}\hat{s}_\nu)^\nu}\rho(\eta)">
<img src="https://render.githubusercontent.com/render/math?math=\rho(\mathbf{\eta}) = \frac{1}{N}\sum_{j=1}^{N} exp\left\{-\frac{1}{2\hat{s}^2_\nu} \left\lVert \frac{\hat{s}_\nu}{s_\nu}(\mathbf{\eta}^j - \mathbf{\eta}) \right\rVert^2 \right\}">
<img src="https://render.githubusercontent.com/render/math?math=s_\nu = \left\{ \frac{4}{N(2 %2B \nu)} \right\}^{\frac{1}{\nu %2B 4}}">
<img src="https://render.githubusercontent.com/render/math?math=\hat{s}_\nu = \frac{s_\nu}{\sqrt{s_{\nu}^2 %2B \frac{N-1}{N}}}.">

The <img src="https://render.githubusercontent.com/render/math?math=\mathbf{\eta}^j"> are the realizations of <img src="https://render.githubusercontent.com/render/math?math=\mathbf{H}"> based on the data samples contained in <img src="https://render.githubusercontent.com/render/math?math=\mathbf{X}">.  

### ISDE Generation of the Random Variable
To construct samples from the proposed density, a nonlinear Ito Stochastic Differential Equation (ISDE) is used to generate realizations of 
the random variable <img src="https://render.githubusercontent.com/render/math?math=\mathbf{H}">.  The ISDE is defined via:

<img src="https://render.githubusercontent.com/render/math?math=d\left[\mathbf{U}(t)\right]=\left[\mathbf{V}(t)\right] d t">
<img src="https://render.githubusercontent.com/render/math?math=d\left[\mathbf{V}(t)\right]=\left[L\left(\left[\mathbf{U}(t)\right]\right)\right] dt-\frac{1}{2} f_{0}\left[\mathbf{V}(t)\right] dt %2B \sqrt{f_{0}} d\left[\mathbf{W}^{\mathrm{wien}}(t)\right]">

where <img src="https://render.githubusercontent.com/render/math?math=\mathbf{U},\, \mathbf{V} \in \mathbb{M}_{\nu,N}">, along with
the initial condition at <img src="https://render.githubusercontent.com/render/math?math=t = 0"> of

<img src="https://render.githubusercontent.com/render/math?math=\left[\mathbf{U}(0)\right]=\left[\eta^{\text {init }}\right], \quad\left[\mathbf{V}(0)\right]=\left[\nu^{\text {init }}\right].">

The specification of the terms in this ISDE is designed to represent a Hamiltonion dynamical system which admits the kernel density
estimate as an invariant measure.

### Diffusion map basis ###
The last step in the algorithm is to employ a (non linear) diffusion map basis to identify the potentially nonlinear manifold upon
which the data line.  This allows for a further dimensional reduction and the ISDE is projected into this space before solving it.
The samples generated from its solution are then transformed back to the original dimension of the data.

