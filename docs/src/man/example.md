# Short example

## Biexponential decay

Let us consider the following model
```math
y(t_j)\;=\;c_1\,e^{-\,R_1 t_j} + c_2\,e^{-\,R_2 t_j}\;=:\;
A_{jk}\left(\bm{x}\right)c_k
```
to describe data ``y_j``, acquired at time points ``t_j``. 
We consider a complex model with ``R_k, c_k \in \mathbb{C}``.

Following the notation in [Variable Projection](@ref), we then just
get 
```math
\bm{x}\;=\;[R_1^\prime,R_1^{\prime\prime},R_2^\prime,R_2^{\prime\prime}]^T
\qquad\text{and}\qquad
A_{jk}\left(\bm{x}\right) = e^{-\,R_k t_j}
```
where ``R_k =: R_k^\prime + i\,R_k^{\prime\prime}``.

The implementation of this model, as expected by [VP4Optim](https://github.com/cganter/VP4Optim.jl), 
can be found in [BiExpDecay.jl](https://github.com/cganter/VP4Optim.jl/blob/main/test/BiExpDecay.jl). 
The corresponding file 
[test_BiExpDecay.jl](https://github.com/cganter/VP4Optim.jl/blob/main/test/test_BiExpDecay.jl)
showcases, how user defined models can be tested for correctness.

## Optimization

One can, for example, use such a model for optimization with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl):
```julia
using Optim
import VP4Optim as VP

# load the model definition
include("BiExpDecay.jl")

# time points, where the data have been collected
ts = collect(range(0, 5, 10))
# specification of the (nonlinear) model parameter names
sym = [:reR1, :imR1, :reR2, :imR2]

# create VP4Optim model instance
bi = BiExpDecay(ts, sym)

# read measured data ...
y = fetch_data_from_somewhere()
# ... and supply them to the model 
VP.set_data!(bi, y)  # unlike y!, set_data! should work for all models

# define some starting value for optimization in the [reR1, imR1, reR2, imR2] space
x0 = [0.1, 0.1, 0.2, -0.2]
# if we want to restrict the optimization space, we can define lower and upper bounds
lo = [0, -π, 0, -π]
up = [1, π, 1, π]

# now we can start the optimization
res = optimize(Optim.only_fg!(fg!(bi)), lo, up, x0, Fminbox(LBFGS()))

# and extract the found results
reR1, imR1, reR2, imR2 = x = res.minimizer

# if we assume these values for our model ...
VP.x!(bi, x)

# ... the missing estimator of the linear coefficients can be simply obtained by
c1, c2 = c = VP.c(bi)
```
