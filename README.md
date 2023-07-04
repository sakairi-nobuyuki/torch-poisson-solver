# torch-poisson-solver

## What it is.

This solver try to solve elliptic type PDE with `PyTorch`.

Elliptic type PDE in a 2-dimensional des Cartes coordinate is generally written in the form of 
$\left(\partial_x^2 + \partial_y^2\right)\phi=\rho$, where $x, y \in \bold{R}$, 
$\phi : \bold{R}^2 \rightarrow \bold{R}; (x, y) \mapsto \phi \in \bold{R}$, and 
$\rho : \bold{R}^2 \rightarrow \bold{R}; (x, y) \mapsto \rho \in \bold{R}$, respectively.

In the regime of the numerical simulation, the differencing operators $\left(\partial_x^2 + \partial_y^2\right)$ is approximately represented as an orthogonal matrix with two flinges, say,

$$
L = \begin{pmatrix}
    4 & -1 & 0 & \cdots & 0\\
    -1 & 4 & -1 & \cdots & 0 \\
    \vdots & & & \ddots & 0 \\
    0 & \cdots & \cdots & -1 & 4
\end{pmatrix},
$$

assuming that $\partial_x \phi \simeq \left(\phi_{i+1} - \phi_i\right)/\Delta$, and,
 $\partial_x^2 \phi \simeq \left(\phi_{i+1} - 4\phi_i + \phi_{i-1}\right)/\Delta^2$, where $\Delta$ is a small real number for derivertive and $i$ denotes the $i$ th component of $\phi$ in a discritized space, respectively.

 Let $\Phi$ is a discritized matrix component of $\phi$ in a scalar field of $(x, y) \in \bold{R}^2$, where $x = i\Delta$ and $y = j\Delta$, and say the components of $\Phi$ is denoted as $\hat{\phi}_i^j$, and $\rho$ for <i>visa verca.</i>, the PDE can be rewritten as,

 $$
 L\Phi = \Rho
 $$

 in an approximated matrix form.

 In the past vast amout of theoretical researches for numerical solution of PDE, this equation is solved with an iterative operation, assuming that the residue of the equation, $Res$, against an approximated solution of the equation to be $\tilde{\Phi}$, 

 $$
L\tilde{\Phi} - \Rho = Res.
 $$

In this approximated equation with a residual term, if $Res$ is sufficiently small, the approximated solution $\tilde{\Phi}$ can be regarded to be closer to actual solution $\Phi$. To achieve this calculation, there have been so many practices, the most important one is FEM in the field of material mechanics, and other so-called Poisson solvers.

Let looking this matrix form equation, on the other hand, the formulation of the problem is quite similar to a kind of deep neural network technology, the state-of-the-art technology to artificial intelligence field, if we regard to evaluate $Res$ as a loss function, while we can interpret the term $L\tilde{\Phi}$ as a FC layer. In this time, $\tilde{\Phi}$ is regarded as FC layer and $L$ is a repeatedly input random tensor.