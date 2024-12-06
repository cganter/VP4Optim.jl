using LinearAlgebra
using Plots: gr, plot, scatter!
using StaticArrays
gr()

"""
    Model{Ny,Nx,Nc,T}

Abstract supertype of any model specification.

# Type parameters
- `Ny`::Int: length of data vector `y`.
- `Nx`::Int: number of *variable* parameters.
- `Nc::Int`: number of linear coefficients.
- `T <: Union{Float64, ComplexF64}`: `typeof(y)`

# Recommended use

```julia
import VP4Optim as VP

mutable struct SpecificModel{Ny,Nx,Nc,T} <: VP.Model{Ny,Nx,Nc,T}
    # mandatory fields of any Model instance
    sym::Vector{Symbol}         # names of all parameters, variable and fixed 
    x_sym::Vector{Symbol}       # names of variable parameters
    par_sym::Vector{Symbol}     # names of fixed parameters
    val::Vector{Float64}        # values of all parameters, variable and fixed
    x_ind::SVector{Nx,Int}      # indices of variable parameters in field val (order defined by x_sym)
    par_ind::Vector{Int}        # indices of fixed parameters in field val (order defined by par_sym)
    y::SVector{Ny,T}            # actual data vector
    y2::Float64                 # == real(y' * y) (automatically calculated in generic method VP.y!)

    # model-specific information
    # ....
end
```
"""
abstract type Model{Ny,Nx,Nc,T} end

"""
    sym(mod::Model)

Return iterable of model parameter names (each with `type == Symbol`).

**All** model parameters are returned, variable and fixed ones.
These are assumed to be stored in the field `sym` of `mod`.
"""
function sym(mod::Model)
    mod.sym
end

"""
    x_sym(mod::Model)

Return iterable of **variable** model parameters.
"""
function x_sym(mod::Model)
    mod.x_sym
end

"""
    par_sym(mod::Model)

Returns iterable of **fixed** model parameters.
"""
function par_sym(mod::Model)
    mod.par_sym
end

"""
    val(mod::Model)

Returns vector (`::Vector{Float64}`) of **all** model parameters.
"""
function val(mod::Model)
    mod.val
end

"""
    x(mod::Model)

TBW
"""
function x(mod::Model)
    mod.val[mod.x_ind]
end

"""
    par(mod::Model)

TBW
"""
function par(mod::Model)
    mod.val[mod.par_ind]
end

"""
    x!(mod::Model, new_x::AbstractArray)

TBW
"""
function x!(mod::Model, x_syms::AbstractArray, x_vals::AbstractArray)
    for (sy, v) in zip(x_syms, x_vals)
        mod.val[findfirst(s -> s == sy, sym(mod))] = v
    end
    x_changed!(mod)
end

"""
    x!(mod::Model, new_x::AbstractArray)

TBW
"""
function x!(mod::Model, new_x::AbstractArray)
    mod.val[mod.x_ind] = new_x
    x_changed!(mod)
end

"""
    par!(mod::Model, p_syms::AbstractArray, p_vals::AbstractArray)

TBW
"""
function par!(mod::Model, p_syms::AbstractArray, p_vals::AbstractArray)
    for (p, v) in zip(p_syms, p_vals)
        mod.val[findfirst(s -> s == p, sym(mod))] = v
    end
    par_changed!(mod)
end

"""
    par!(mod::Model, new_par::AbstractArray)

Reset fixed parameter values.
"""
function par!(mod::Model, new_par::AbstractArray)
    mod.val[mod.par_ind] = new_par
    par_changed!(mod)
end

"""
    x_changed!(::Model)

TBW
"""
function x_changed!(::Model) end

"""
    par_changed!(::Model)

Notify model that (some of the) fixed parameters `par` have changed.

Function is automatically called by 
Can be used to implement 
"""
function par_changed!(::Model) end

"""
    y!(mod::Model{Ny,Nx,Nc,T}, new_y::AbstractArray) where {Ny,Nx,Nc,T}

Set the data vector.
"""
function y!(mod::Model{Ny,Nx,Nc,T}, new_y::AbstractArray) where {Ny,Nx,Nc,T}
    mod.y = SVector{Ny,T}(new_y)
    mod.y2 = real(mod.y' * mod.y)
end

"""
    y(mod::Model)

Returns the data vector
"""
function y(mod::Model)
    mod.y
end

"""
    A(::Model)

Return VARPRO matrix `A`.

Must be implemented by each model.
"""
function A(mod::Model)
    mod.A
end

"""
    c(mod::Model)

Return VARPRO vector `c`.

Calculates generic soluion `c = B \\ b`.
Can be replaced by model-specific implementation, if desired (e.g. for performance improvements).
"""
function c(mod::Model)
    (B, b) = Bb!(mod)
    B \ b
end

"""
    Bb!(mod::Model)

TBW
"""
function Bb!(mod::Model)
    A_ = A(mod)
    B = A_' * A_
    b = A_' * y(mod)

    return (B, b)
end

"""
    ∂Bb!(::Model)

TBW
"""
function ∂Bb!(::Model)
    error("Missing implementation of ∂Bb!")
end

"""
    ∂∂Bb!(::Model)

TBW
"""
function ∂∂Bb!(::Model)
    error("Missing implementation of ∂∂Bb!")
end

#Requires a prior call of [y!(mod::Model{Ny,Nx,Nc,T}, new_y::AbstractArray) where {Ny,Nx,Nc,T}](@ref) to make any sense.
"""
    y_model(mod::Model)

Compute model prediction `A(mod) * c(mod)` at actual `x`.
"""
function y_model(mod::Model)
    A(mod) * c(mod)
end

"""
    χ2(mod::Model)

Return `χ²` of actual model.
"""
function χ2(mod::Model)
    (B, b) = Bb!(mod)
    f(mod.y2, B, b)
end    

"""
    f(mod::Model)

Return function `f` of argument `x` to be minimized, as expected by Optim.jl
"""
function f(mod::Model)
    x -> begin
        x!(mod, x)
        (B, b) = Bb!(mod)
        f(mod.y2, B, b)
    end
end

"""
    fg!(mod::Model)

Return function `fg!` of three arguments `(F, G, x)` as expected by Optim.jl.
"""
function fg!(mod::Model)
    (F, G, x) -> begin
        x!(mod, x)
        if G !== nothing
            (B, b, ∂B, ∂b) = ∂Bb!(mod)
            fg!(mod, F, G, mod.y2, B, b, ∂B, ∂b)
        elseif F !== nothing
            (B, b) = Bb!(mod)
            f(mod.y2, B, b)
        else
            nothing
        end
    end
end

"""
    fgh!(mod::Model)

Return function `fgh!` of four arguments `(F, G, H, x)` as expected by Optim.jl.
"""
function fgh!(mod::Model)
    (F, G, H, x) -> begin
        x!(mod, x)
        if H !== nothing
            (B, b, ∂B, ∂b, ∂∂B, ∂∂b) = ∂∂Bb!(mod)
            fgh!(mod, F, G, H, mod.y2, B, b, ∂B, ∂b, ∂∂B, ∂∂b)
        elseif G !== nothing
            (B, b, ∂B, ∂b) = ∂Bb!(mod)
            fg!(mod, F, G, mod.y2, B, b, ∂B, ∂b)
        elseif F !== nothing
            (B, b) = Bb!(mod)
            f(mod.y2, B, b)
        else
            nothing
        end
    end
end

"""
    P(mod::Model{Ny,Nx,Nc,T}, x) where {Ny,Nx,Nc,T}

Returns Hessian of model `mod` at `x`.

Can be used as preconditioner.
"""
function P(mod::Model{Ny,Nx,Nc,T}, x) where {Ny,Nx,Nc,T}
    H = Matrix{T}(undef, Nx, Nx)
    fgh!(mod)(nothing, nothing, H, x)
    H
end

#Helper function for [`f(mod::Model)`](@ref).
"""
    f(y2, B, b)

Should not be called directly.
"""
function f(y2, B, b)
    y2 - real(b' * (B \ b))
end

#Helper function for [`fg!(mod::Model)`](@ref).
"""
    fg!(::Model{Ny,Nx,Nc,T}, F, G, y2, B, b, ∂B, ∂b) where {Ny,Nx,Nc,T}

Should not be called directly.
"""
function fg!(::Model{Ny,Nx,Nc,T}, F, G, y2, B, b, ∂B, ∂b) where {Ny,Nx,Nc,T}
    c = B \ b

    for i in 1:Nx
        G[i] = real(c' * (∂B[i] * c - 2∂b[i]))
    end

    if F !== nothing
        return y2 - real(b' * c)
    end
end

#Helper function for [`fgh!(mod::Model)`](@ref).
"""
    fgh!(::Model{Ny,Nx,Nc,T}, F, G, H, y2, B, b, ∂B, ∂b, ∂∂B, ∂∂b) where {Ny,Nx,Nc,T}

Should not be called directly.
"""
function fgh!(::Model{Ny,Nx,Nc,T}, F, G, H, y2, B, b, ∂B, ∂b, ∂∂B, ∂∂b) where {Ny,Nx,Nc,T}
    c = B \ b

    if G !== nothing
        for i in 1:Nx
            G[i] = real(c' * (∂B[i] * c - 2∂b[i]))
        end
    end

    ∂c = SVector{Nx}(B \ (∂b[i] - ∂B[i] * c) for i in 1:Nx)

    for i in 1:Nx
        for j in 1:i
            H[i, j] = real(c' * (∂∂B[i, j] * c - 2∂∂b[i, j])) - 2real(∂c[i]' * B * ∂c[j])

            i != j && (H[j, i] = H[i, j])
        end
    end

    if F !== nothing
        return y2 - real(b' * c)
    end
end

