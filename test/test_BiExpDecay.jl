using Random
import VP4Optim as VP

include("BiExpDecay.jl")

# visual confirmation of derivatives
visual = false

# random number seed
rng = MersenneTwister(1)

# include Hessian in derivative test
Hessian = true

# minimal slope
min_slope = 0.8

# generate ModPar structure for given time points
pars = VP.modpar(BEDPar; ts = collect(range(0, 5, 10)))

# true values 
x = [0.05, 0.3, 0.1, 0.7] # actual relaxation rate values (cf. variable sym above)
c = rand(rng, ComplexF64, 2) # linear prefactors of the two expoentials
bi = BiExpDecay(pars) # create model instance
VP.x!(bi, x) # set relaxation rates
y = VP.A(bi) * c # calculate model data at time points ts (required by test function below)

# check that the default implementation of set_data! works
VP.set_data!(bi, y)
@test VP.y(bi) == y

# starting values for optimization
x0 = 0.9x  # relaxation rates
lx = [0, -π, 0, -π] # lower bounds
ux = [1, π, 1, π] # upper bounds

# relative scale of parameters
x_scale = ux - lx # to make different parameters more comparable

# what to test
what = (:consistency, :derivatives, :optimization)

# do the tests
res = VP.check_model(BiExpDecay, pars, x, c, y, what = what, x0 = x0, lx = lx, ux = ux, x_scale = x_scale, visual = visual, rng = rng, Hessian = Hessian, min_slope = min_slope)

