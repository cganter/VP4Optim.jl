using LinearAlgebra, Combinatorics, Test, Optim

function check_fgh!(fgh!, x0, δx;
    log_rng=10 .^ range(-5, -1, 10), title = "Titel", plot_size=(1000,500))

    lx = length(x0)
    F = zero(eltype(x0))
    G = zeros(eltype(x0), lx)
    H0 = zeros(eltype(x0), lx, lx)

    # set up values, where the function shall be evaluated
    δx *= norm(x0) / norm(δx)    # make the norms of x0 and δx equal
    hs = [lr * δx for lr in log_rng]
    nhs = [norm(h) for h in hs]   # norm of h

    # initialize plot vector
    plts = []

    # evaluate function value, gradient and Hessian at x0
    f0 = fgh!(F, G, H0, x0)
    G0 = deepcopy(G) # the Hessian is not changed from here on and does not need to be saved

    # evaluate function values at different x0 + h
    fs = [fgh!(F, nothing, nothing, x0 + h) for h in hs]

    # evaluate gradient at different x0 + h
    Gs = [(fgh!(F, G, nothing, x0 + h); deepcopy(G)) for h in hs]

    # linear approximation of function (to test gradient)
    flas = [f0 + G0' * h for h in hs]

    # linear approximation of gradient (to test Hessian)
    Glas = [G0 + H0 * h for h in hs]

    # deviation of linear approximations
    # (divided by norm(h), such that it should vanish linearly for small h)
    δfla = [abs(fh - fla) / nh for (fh, fla, nh) in zip(fs, flas, nhs)]
    δGla = [norm(Gh - Gla) / nh for (Gh, Gla, nh) in zip(Gs, Glas, nhs)]

    # approximate slope of linear deviations
    sfla = δfla[1] / nhs[1]
    sGla = δGla[1] / nhs[1]

    # initialize plot vector
    plts = []

    # plot result for gradient
    push!(plts, plot(nhs, sfla * nhs, xaxis=:log, yaxis=:log, title=string("gradient: ", title), label="approx"))
    scatter!(nhs, δfla, xaxis=:log, yaxis=:log, label="exact", legend=:topleft)

    # plot result for Hessian
    push!(plts, plot(nhs, sGla * nhs, xaxis=:log, yaxis=:log, title=string("Hessian: ", title), label="approx"))
    scatter!(nhs, δGla, xaxis=:log, yaxis=:log, label="exact", legend=:topleft)

    # show plots
    display(plot(plts..., size = plot_size))
end

function check_model(constructor, args, mod_par, c_, y_; 
        what=(:consistency, :derivatives, :optimization),
        small = sqrt(eps()),
        x0 = [], lx = [], ux = [],
        precon = true)    
    mod = constructor(args...)
    syms = sym(mod)
    d = Dict()

    for xsy in Combinatorics.powerset(syms, 1)
        mod = constructor(args...; x_sym = xsy)
        d[xsy] = Dict()
        d[xsy][:check_subset_args] = (mod, xsy, mod_par, c_, y_, what, small, x0, lx, ux, precon, d[xsy])
        check_subset(mod, xsy, mod_par, c_, y_, what, small, x0, lx, ux, precon, d[xsy])
    end
    
    return d
end

function check_subset(mod::Model{Ny,Nx,Nc,T}, xsy, mod_val, c_, y_, what, small, x0, lx, ux, precon, d) where {Ny,Nx,Nc,T}
    @assert all(w -> w ∈ (:consistency, :derivatives, :optimization), what)

    y!(mod, y_)
    x_, par_ = mod_val[mod.x_ind], mod_val[mod.par_ind]
    x!(mod, x_)
    par!(mod, par_)

    if :consistency ∈ what
        @test x_sym(mod) == xsy
        psy = filter(x -> x ∉ xsy, sym(mod))
        @test par_sym(mod) == psy
        @test length(par(mod)) + length(x(mod)) == length(sym(mod))
        @test x(mod) == x_
        @test par(mod) == par_
        @test abs(f(mod)(x(mod))) < small
        @test y_model(mod) ≈ y(mod) == y_
        @test c(mod) ≈ c_
    end

    if :derivatives ∈ what
        δx = randn(length(x_))
        title = ""
        for sy in x_sym(mod)
            title *= " " * string(sy)
        end
        check_fgh!(fgh!(mod), x_, δx, title = title)
    end
    
    if :optimization ∈ what
        x0_, lx_, ux_ = collect(x0[mod.x_ind]), collect(lx[mod.x_ind]), collect(ux[mod.x_ind])
        
        if precon
            res_precon = optimize(Optim.only_fg!(fg!(mod)), lx_, ux_, x0_, Fminbox(LBFGS(P = P(mod, x0_))))
            @test norm(x_ - res_precon.minimizer) / norm(x_) < 1e-5
            x!(mod, res_precon.minimizer)
            @test norm(c_ - c(mod)) / norm(c_) < 1e-5
            d[:optim_precon] = res_precon
        end
            
        res_no_precon = optimize(Optim.only_fg!(fg!(mod)), lx_, ux_, x0_, Fminbox(LBFGS()))
        @test norm(x_ - res_no_precon.minimizer) / norm(x_) < 1e-5
        x!(mod, res_no_precon.minimizer)
        @test norm(c_ - c(mod)) / norm(c_) < 1e-5
        d[:optim_no_precon] = res_no_precon
    end
end
