using LinearAlgebra, Combinatorics, Test, Optim, Random, Compat
@compat public check_model

"""
    check_model(modcon, args, vals, c_, y_;
    what=(:consistency, :derivatives, :optimization),
    small=sqrt(eps()),
    x0=[], lx=[], ux=[], x_scale=[],
    precon=true,
    visual=false,
    rng=MersenneTwister(),
    Hessian=true,
    log10_rng=range(-6, -3, 10),
    min_slope=0.9)

Tests, which any specific model should pass.

# Arguments
- `modcon::Function`: Constructor to the model to be tested.
- `args::Tuple`: Arguments, as expected by constructor, like `modcon(args...)` or `modcon(args; x_sym=x_sym)`.
- `vals::Vector{Float64}`: *All* nonlinear parameters, the model depends on. As defined in the `Model` field `val`.
- `c_::Vector{Nc, T}`: Linear cofficients, the model depends on.
- `y_::Vector{T}`: Data, corresponding to the *true* parameters `vals` and `c_`. 
- `what::Tuple{Symbol}`: Tests to be performed (see below).
- `small::Float64`: Required as accuracy criterion.
- `x0::Vector{Float64}`: Starting point for optimization and location, where derivatives are tested.
- `lx::Vector{Float64}`: Lower bound of optimization
- `ux::Vector{Float64}`: Upper bound of optimization
- `x_scale::Vector{Float64}`: Scaling vector, such that `δx = randn(size(x)) .* x_scale` becomes reasonable
- `precon::Bool`: Test optimization with and without preconditioner.
- `visual::Bool`: If `true` also generate double-logarithmic plots for the derivative tests.
- `rng::MersenneTwister`: Allows to pass a unique seed (e.g. `MersenneTwister(42)`) for reproducible testing.
- `Hessian::Bool`: Should be set to `false`, if the model does not implement second order derivatives.
- `log10_rng::AbstractVector`: logarithmic range for derivative testing
- `min_slope::Float64`: minimal derivative slope on log-log plot

## Remark

- Tests are performed for every `x_sym ⊆ sym`.
- Returns a dictionary with detailed information about the test results.
- `:consistency ∈ what`: Several basic tests (parameters, names, correct model values)
- `:derivatives ∈ what`: Check first and second order partial derivatives at `x0`.
- `:optimization ∈ what`: Minimize model with `x0` as starting point and bounds `lx` and `ux`.
- An example application can be found in [`test_BiExpDecay.jl`](https://github.com/cganter/VP4Optim.jl/blob/main/test/test_BiExpDecay.jl). This should also work as a template, how to perform tests on own models.
"""
function check_model(modcon, args, vals, c_, y_;
    what=(:consistency, :derivatives, :optimization),
    small=sqrt(eps()),
    x0=[], lx=[], ux=[], x_scale=[],
    precon=true,
    visual=false,
    rng=MersenneTwister(),
    Hessian=true,
    log10_rng=range(-6, -3, 10),
    min_slope=0.9)
    mod = modcon(args...)
    syms = sym(mod)
    d = Dict()

    isempty(x_scale) && (x_scale = ones(length(syms)))
    @assert length(x_scale) == length(syms) && all(x_scale .> 0)

    for xsy in Combinatorics.powerset(syms, 1)
        mod = modcon(args...; x_sym=xsy)
        d[xsy] = Dict()
        d[xsy][:check_subset_args] = (mod, xsy, vals, c_, y_, what, small, x0, lx, ux, precon, d[xsy])
        check_subset(mod, xsy, vals, c_, y_, what, small, x0, lx, ux, x_scale, precon, d[xsy], visual, rng, Hessian, log10_rng, min_slope)
    end

    return d
end

"""
    check_subset(mod::Model{Ny,Nx,Nc,T}, xsy, vals, c_, y_, what, small, x0, lx, ux, x_scale, precon, d, visual, rng, Hessian, log10_rng, min_slope) where {Ny,Nx,Nc,T}

Helper function of [`check_model`](@ref check_model), which performs the tests for a given subset `x_sym ⊆ sym`.

## Remark

- Should not be called directly.
"""
function check_subset(mod::Model{Ny,Nx,Nc,T}, xsy, vals, c_, y_, what, small, x0, lx, ux, x_scale, precon, d, visual, rng, Hessian, log10_rng, min_slope) where {Ny,Nx,Nc,T}
    @assert all(w -> w ∈ (:consistency, :derivatives, :optimization), what)

    y!(mod, y_)
    x_, par_, x0_, x_scale_ = vals[mod.x_ind], vals[mod.par_ind], x0[mod.x_ind], x_scale[mod.x_ind]
    x!(mod, x_)
    par!(mod, par_)

    # consistency checks at the minimum, i.e. for y_ = A(x_) * c_
    if :consistency ∈ what
        # mandatory field values are set and returned correctly
        @test sym(mod) == mod.sym
        @test x_sym(mod) == mod.x_sym == xsy
        psy = filter(x -> x ∉ xsy, sym(mod))
        @test par_sym(mod) == mod.par_sym == psy
        @test val(mod) == mod.val
        @test x(mod) == val(mod)[mod.x_ind]
        @test par(mod) == val(mod)[mod.par_ind]
        @test y(mod) == mod.y
        @test real(y(mod)' * y(mod)) ≈ mod.y2
        @test N_data(mod) == Ny
        @test N_var(mod) == Nx
        @test N_coeff(mod) == Nc
        @test data_type(mod) == T
        # tests for x! and par!
        for is in powerset(1:length(mod.x_ind))
            is = is[randperm(rng, length(is))]
            x!(mod, x_sym(mod)[is], x_[is])
            @test x_ == x(mod) && par_ == par(mod)
        end
        for is in powerset(1:length(mod.par_ind))
            is = is[randperm(rng, length(is))]
            par!(mod, par_sym(mod)[is], par_[is])
            @test x_ == x(mod) && par_ == par(mod)
        end
        for sys in powerset(sym(mod), 1)
            any(sy -> sy ∉ x_sym(mod), sys) &&
                (@test_throws Exception x!(mod, sy, rand(rng, length(sy))))

            any(sy -> sy ∉ par_sym(mod), sys) &&
                (@test_throws Exception par!(mod, sy, rand(rng, length(sy))))
        end
        # be sure to undo any unwanted changes
        x!(mod, x_)
        par!(mod, par_)
        # do we cover all parameters
        @test length(par(mod)) + length(x(mod)) == length(sym(mod))
        # are the variables set correctly
        @test x(mod) == x_
        # are the fixed parameters set correctly
        @test par(mod) == par_
        # do we obtain a local minimum (since x_ should correspond to data y_)
        @test abs(f(mod)(x(mod))) < small
        # calculated data match supplied ones
        @test y_model(mod) ≈ A(mod) * c(mod) ≈ y(mod) == y_
        # the correct linear coefficients are obtained
        @test c(mod) ≈ B(mod) \ b(mod) ≈ c_
        # test correctness of A, B, b
        @test B(mod) ≈ A(mod)' * A(mod)
        @test b(mod) ≈ A(mod)' * y(mod)
        # test dimensions
        @test size(y(mod)) == (Ny,)
        @test size(A(mod)) == (Ny, Nc)
        @test size(B(mod)) == (Nc, Nc)
        @test size(b(mod)) == (Nc,)
        # test that f returns χ²
        x!(mod, x0_)
        @test χ2(mod) ≈ f(mod)(x0_)
        x!(mod, x_)
    end

    # derivatives are tested at x0_ ≠ x_
    if :derivatives ∈ what
        δx = randn(rng, length(x_)) .* x_scale_
        title = ""
        for sy in x_sym(mod)
            title *= " " * string(sy)
        end
        check_fg!(fg!(mod), x0_, δx, title=title, visual=visual, log10_rng=log10_rng, min_slope=min_slope)
        check_fgh!(fgh!(mod), x0_, δx, title=title, visual=visual, Hessian=false, log10_rng=log10_rng, min_slope=min_slope)
        Hessian && check_fgh!(fgh!(mod), x0_, δx, title=title, visual=visual, Hessian=true, log10_rng=log10_rng, min_slope=min_slope)
    end

    if :optimization ∈ what
        x0_, lx_, ux_ = collect(x0[mod.x_ind]), collect(lx[mod.x_ind]), collect(ux[mod.x_ind])

        if precon
            res_precon = optimize(Optim.only_fg!(fg!(mod)), lx_, ux_, x0_, Fminbox(LBFGS(P=P(mod, x0_))))
            @test norm(x_ - res_precon.minimizer) / norm(x_) < 1e-4
            x!(mod, res_precon.minimizer)
            @test norm(c_ - c(mod)) / norm(c_) < 1e-4
            d[:optim_precon] = res_precon
        end

        res_no_precon = optimize(Optim.only_fg!(fg!(mod)), lx_, ux_, x0_, Fminbox(LBFGS()))
        @test norm(x_ - res_no_precon.minimizer) / norm(x_) < 1e-4
        x!(mod, res_no_precon.minimizer)
        @test norm(c_ - c(mod)) / norm(c_) < 1e-4
        d[:optim_no_precon] = res_no_precon
    end
end

"""
    check_fg!(fg!, x0, δx;
    log10_rng=range(-6, -3, 10),
    min_slope=0.9,
    title="Titel",
    plot_size=(1000, 500),
    visual=false)

Helper function of [`check_model`](@ref check_model), which checks the derivatives up to second order for a given subset `x_sym ⊆ sym`.

## Remark

- Should not be called directly.
"""
function check_fg!(fg!, x0, δx;
    log10_rng=range(-6, -3, 10),
    min_slope=0.9,
    title="Titel",
    plot_size=(1000, 500),
    visual=false)

    log_rng = 10 .^ log10_rng

    lx = length(x0)
    F = zero(eltype(x0))
    G = zeros(eltype(x0), lx)

    # set up values, where the function shall be evaluated
    hs = [lr * δx for lr in log_rng]
    nhs = [norm(h) for h in hs]   # norm of h

    # initialize plot vector
    plts = []

    # evaluate function value, gradient and Hessian at x0
    f0 = fg!(F, G, x0)
    G0 = deepcopy(G) # the Hessian is not changed from here on and does not need to be saved

    # evaluate function values at different x0 + h
    fs = [fg!(F, nothing, x0 + h) for h in hs]

    # linear approximation of function (to test gradient)
    flas = [f0 + G0' * h for h in hs]

    # deviation of linear approximations
    # (divided by norm(h), such that it should vanish linearly for small h)
    δfla = [abs(fh - fla) / nh for (fh, fla, nh) in zip(fs, flas, nhs)]

    # approximate slope of linear deviations
    sfla = δfla[1] / nhs[1]

    # calculate linear regression slope of the logarithms
    log_nhs = log.(nhs)
    log_δfla = log.(δfla)
    slope = (sum(log_nhs) * sum(log_δfla) - length(nhs) * (log_nhs' * log_δfla)) /
            (sum(log_nhs)^2 - length(nhs) * (log_nhs' * log_nhs))

    @test slope > min_slope

    if visual
        # initialize plot vector
        plts = []

        # plot result for gradient
        push!(plts, plot(nhs, sfla * nhs, xaxis=:log, yaxis=:log, title=string("gradient: ", title), label="approx"))
        scatter!(nhs, δfla, xaxis=:log, yaxis=:log, label="exact", legend=:topleft)

        # show plots
        display(plot(plts..., size=plot_size))
    end
end

"""
    check_fgh!(fgh!, x0, δx;
    log10_rng=range(-6, -3, 10),
    min_slope=0.9,
    title="Titel",
    plot_size=(1000, 500),
    visual=false,
    Hessian=true)

Helper function of [`check_model`](@ref check_model), which checks the derivatives up to second order for a given subset `x_sym ⊆ sym`.

## Remark

- Should not be called directly.
"""
function check_fgh!(fgh!, x0, δx;
    log10_rng=range(-6, -3, 10),
    min_slope=0.9,
    title="Titel",
    plot_size=(1000, 500),
    visual=false,
    Hessian=true)

    log_rng = 10 .^ log10_rng

    lx = length(x0)
    F = zero(eltype(x0))
    G = zeros(eltype(x0), lx)
    if Hessian
        H0 = zeros(eltype(x0), lx, lx)
    else
        H0 = nothing
    end

    # set up values, where the function shall be evaluated
    hs = [lr * δx for lr in log_rng]
    nhs = [norm(h) for h in hs]   # norm of h

    # initialize plot vector
    plts = []

    # evaluate function value, gradient and Hessian at x0
    f0 = fgh!(F, G, H0, x0)
    G0 = deepcopy(G) # the Hessian is not changed from here on and does not need to be saved

    # evaluate function values at different x0 + h
    fs = [fgh!(F, nothing, nothing, x0 + h) for h in hs]

    # linear approximation of function (to test gradient)
    flas = [f0 + G0' * h for h in hs]

    # deviation of linear approximations
    # (divided by norm(h), such that it should vanish linearly for small h)
    δfla = [abs(fh - fla) / nh for (fh, fla, nh) in zip(fs, flas, nhs)]

    # approximate slope of linear deviations
    sfla = δfla[1] / nhs[1]

    # calculate linear regression slope of the logarithms
    log_nhs = log.(nhs)
    log_δfla = log.(δfla)
    slope = (sum(log_nhs) * sum(log_δfla) - length(nhs) * (log_nhs' * log_δfla)) /
            (sum(log_nhs)^2 - length(nhs) * (log_nhs' * log_nhs))

    @test slope > min_slope 

    if Hessian
        # evaluate gradient at different x0 + h
        Gs = [(fgh!(F, G, nothing, x0 + h); deepcopy(G)) for h in hs]

        # linear approximation of gradient
        Glas = [G0 + H0 * h for h in hs]

        # deviation of linear approximations
        # (divided by norm(h), such that it should vanish linearly for small h)
        δGla = [norm(Gh - Gla) / nh for (Gh, Gla, nh) in zip(Gs, Glas, nhs)]

        # approximate slope of linear deviations
        sGla = δGla[1] / nhs[1]

        # calculate linear regression slope of the logarithms
        log_nhs = log.(nhs)
        log_δGla = log.(δGla)
        slope = (sum(log_nhs) * sum(log_δGla) - length(nhs) * (log_nhs' * log_δGla)) /
                (sum(log_nhs)^2 - length(nhs) * (log_nhs' * log_nhs))

        @test slope > min_slope
    end

    if visual
        # initialize plot vector
        plts = []

        # plot result for gradient
        push!(plts, plot(nhs, sfla * nhs, xaxis=:log, yaxis=:log, title=string("gradient: ", title), label="approx"))
        scatter!(nhs, δfla, xaxis=:log, yaxis=:log, label="exact", legend=:topleft)

        if Hessian
            # plot result for Hessian
            push!(plts, plot(nhs, sGla * nhs, xaxis=:log, yaxis=:log, title=string("Hessian: ", title), label="approx"))
            scatter!(nhs, δGla, xaxis=:log, yaxis=:log, label="exact", legend=:topleft)
        end

        # show plots
        display(plot(plts..., size=plot_size))
    end
end