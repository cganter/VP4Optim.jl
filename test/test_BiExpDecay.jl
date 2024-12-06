using Random
import VP4Optim as VP

include("BiExpDecay.jl")

# check out all nonempty combinations x_sym ⊆ sym
ts = collect(range(0, 5, 10)) # time points
sym = [:reR1, :imR1, :reR2, :imR2] # nonlinear variables: two complex relaxation rates
args = (ts, sym) # arguments required by constructor 

# true values 
x = [0.05, 0.3, 0.1, 0.7] # actual relaxation rate values (cf. variable sym above)
c = rand(ComplexF64, 2) # linear prefactors of the two expoentials
bi = BiExpDecay(args...) # create model instance
VP.x!(bi, x) # set relaxation rates
y = VP.A(bi) * c # calculate model data at time points ts (required by test function below)

# starting values for optimization
x0 = 0.9x  # relaxation rates
lx = zeros(4) # lower bounds
ux = ones(4) # upper bounds

# what to test
what = (:consistency,) #, :derivatives, :optimization)

# do the tests
res = VP.check_model(BiExpDecay, args, x, c, y, what = what, x0 = x0, lx = lx, ux = ux)
