import VP4Optim as VP
using LinearAlgebra, StaticArrays

mutable struct BiExpDecay{Ny,Nx} <: VP.Model{Ny,Nx,2,ComplexF64}
    "Mandatory field: Names of all parameters, variable and fixed."
    sym::Vector{Symbol}
    "Mandatory field: Names of variable parameters."
    x_sym::Vector{Symbol}
    "Mandatory field: Names of fixed parameter."
    par_sym::Vector{Symbol}
    "Mandatory field: Values of all parameters, variable and fixed."
    val::Vector{Float64}
    "Mandatory: Indices of variable parameters in `val` according to the order specified in `x_sym`."
    x_ind::SVector{Nx,Int}
    "Mandatory: Indices of fixed parameters in `val` according to the order specified in `par_sym`."
    par_ind::Vector{Int}
    "Mandatory field: Actual data vector."
    y::SVector{Ny,ComplexF64}
    "Mandatory field: Actual value of `y' * y`"
    y2::Float64
    "Recommended field for matrix A"
    A::SMatrix{Ny,2,ComplexF64}

    # model specific information
    ts::SVector{Ny,Float64}
    cR::SVector{2,ComplexF64}
    ∂B_weights::SVector{Nx,SMatrix{2,2,ComplexF64}}
    ∂b_weights::SVector{Nx,SVector{2,ComplexF64}}
    ∂∂B_weights::SMatrix{Nx,Nx,SMatrix{2,2,ComplexF64}}
    ∂∂b_weights::SMatrix{Nx,Nx,SVector{2,ComplexF64}}
end

struct BEDPar <: VP.ModPar
    ts::Vector{Float64}
    sym::Vector{Symbol}
    x_sym::Vector{Symbol}
end

"""
    BiExpDecay(ts, sym; x_sym=nothing)

Constructor to be called
"""
function BiExpDecay(pars::BEDPar)
    BiExpDecay(Val(length(pars.ts)), Val(length(pars.x_sym)), pars.ts, pars.sym, pars.x_sym)
end

function BEDPar()
    ts = Float64[]
    sym = [:reR1, :imR1, :reR2, :imR2]
    x_sym = deepcopy(sym)

    BEDPar(ts, sym, x_sym)
end

function check(pars::BEDPar)
    @assert length(pars.sym) == 4
    @assert all(sy -> sy ∈ pars.sym, pars.x_sym)
    pars
end

function BiExpDecay(::Val{Ny}, ::Val{Nx}, ts, sym, x_sym) where {Ny,Nx}
    val = zeros(4)
    x_ind = SVector{Nx,Int}(findfirst(x -> x == x_sym[i], sym) for i in 1:Nx)
    par_ind = filter(x -> x ∉ x_ind, 1:4)
    par_sym = sym[par_ind]
    y = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))
    A = SMatrix{Ny,2}(zeros(ComplexF64, Ny, 2))
    ts = SVector{Ny,Float64}(ts)
    cR = SVector{2,ComplexF64}(zeros(ComplexF64, 2))

    ∂B_weights = SVector{4,SMatrix{2,2,ComplexF64}}([
        SMatrix{2,2,ComplexF64}([-2 -1; -1 0]),
        SMatrix{2,2,ComplexF64}([0 1im; -1im 0]),
        SMatrix{2,2,ComplexF64}([0 -1; -1 -2]),
        SMatrix{2,2,ComplexF64}([0 -1im; 1im 0])
    ])

    ∂b_weights = SVector{4,SVector{2,ComplexF64}}([
        SVector{2,ComplexF64}([-1, 0]),
        SVector{2,ComplexF64}([1im, 0]),
        SVector{2,ComplexF64}([0, -1]),
        SVector{2,ComplexF64}([0, 1im]),
    ])

    ∂B_weights = ∂B_weights[x_ind]
    ∂b_weights = ∂b_weights[x_ind]

    ∂∂B_weights = SMatrix{Nx,Nx,SMatrix{2,2,ComplexF64}}(
        ∂B_weights[i] .* ∂B_weights[j] for i in 1:Nx, j in 1:Nx)

    ∂∂b_weights = SMatrix{Nx,Nx,SVector{2,ComplexF64}}(
        ∂b_weights[i] .* ∂b_weights[j] for i in 1:Nx, j in 1:Nx)

    BiExpDecay{Ny,Nx}(sym, x_sym, par_sym, val, x_ind, par_ind, y, 0.0, A, ts, cR, 
        ∂B_weights, ∂b_weights, ∂∂B_weights, ∂∂b_weights)
end

function VP.x_changed!(bi::BiExpDecay)
    cR!(bi)
    A!(bi)
end

function VP.par_changed!(bi::BiExpDecay)
    cR!(bi)
    A!(bi)
end

function cR!(bi::BiExpDecay)
    bi.cR = SVector{2,ComplexF64}(bi.val[i] + im * bi.val[i+1] for i in (1, 3))
end

function A!(bi::BiExpDecay)
    bi.A = exp.(-transpose(bi.cR) .* bi.ts)
end

function VP.Bb!(bi::BiExpDecay)
    A = VP.A(bi)

    B = A' * A
    b = A' * bi.y

    return (B, b)
end

function VP.∂Bb!(bi::BiExpDecay{Ny,Nx}) where {Ny,Nx}
    A = VP.A(bi)

    B = A' * A
    b = A' * bi.y

    tA = bi.ts .* A
    AtA = tA' * A
    Aty = tA' * bi.y

    ∂B = SVector{Nx,SMatrix{2,2,ComplexF64}}(bi.∂B_weights[i] .* AtA for i in 1:Nx)
    ∂b = SVector{Nx,SVector{2,ComplexF64}}(bi.∂b_weights[i] .* Aty for i in 1:Nx)

    return (B, b, ∂B, ∂b)
end

function VP.∂∂Bb!(bi::BiExpDecay{Ny,Nx}) where {Ny,Nx}
    A = VP.A(bi)

    B = A' * A
    b = A' * bi.y

    tA = bi.ts .* A
    AtA = tA' * A
    Aty = tA' * bi.y

    ∂B = SVector{Nx,SMatrix{2,2,ComplexF64}}(bi.∂B_weights[i] .* AtA for i in 1:Nx)
    ∂b = SVector{Nx,SVector{2,ComplexF64}}(bi.∂b_weights[i] .* Aty for i in 1:Nx)

    t2A = bi.ts .* tA
    At2A = t2A' * A
    At2y = t2A' * bi.y

    ∂∂B = SMatrix{Nx,Nx,SMatrix{2,2,ComplexF64}}(bi.∂∂B_weights[i, j] .* At2A for i in 1:Nx, j in 1:Nx)
    ∂∂b = SMatrix{Nx,Nx,SVector{2,ComplexF64}}(bi.∂∂b_weights[i, j] .* At2y for i in 1:Nx, j in 1:Nx)

    return (B, b, ∂B, ∂b, ∂∂B, ∂∂b)
end