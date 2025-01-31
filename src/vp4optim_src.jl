using LinearAlgebra, StaticArrays, Compat
using Plots: gr, plot, scatter!
gr()
@compat public N_data, N_var, N_coeff, data_type, Model, sym, x_sym, par_sym, val, x, par, x!, par!, x_changed!, par_changed!,
y, y!, A, B, b, c, Bb!, ∂Bb!, ∂∂Bb!, y_model, χ2, f, fg!, fgh!, P

"""
    Model{Ny,Nx,Nc,T}

Abstract supertype of any model specification.

# Type parameters
- `Ny::Int`: length of data vector `y`.
- `Nx::Int`: number of *variable* parameters.
- `Nc::Int`: number of linear coefficients.
- `T <: Union{Float64, ComplexF64}` is equal to `eltype(y)`
"""
abstract type Model{Ny,Nx,Nc,T} end

"""
    N_data(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}

Returns `Ny` (number of data)
"""
function N_data(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}
    Ny
end

"""
    N_var(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}

Returns `Nx` (number of variable parameters)
"""
function N_var(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}
    Nx
end

"""
    N_coeff(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}

Returns `Nc` (number of linear coefficients)
"""
function N_coeff(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}
    Nc
end

"""
    data_type(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}

Returns `T` (data type of model)
"""
function data_type(::Model{Ny,Nx,Nc,T}) where {Ny,Nx,Nc,T}
    T
end

"""
    sym(mod::Model)

Return *all* model parameter names (type: `Symbol`), variable and fixed.

## Default

- Returns `mod.sym::Vector{Symbol}`.
"""
function sym(mod::Model)
    mod.sym
end

"""
    x_sym(mod::Model)

Return *variable* model parameter names.

## Default

- Returns `mod.x_sym::Vector{Symbol}`.
"""
function x_sym(mod::Model)
    mod.x_sym
end

"""
    par_sym(mod::Model)

Returns *fixed* model parameter names.

## Default

- Returns `mod.par_sym::Vector{Symbol}`.
"""
function par_sym(mod::Model)
    mod.par_sym
end

"""
    val(mod::Model)

Returns *all* model parameters.

## Default

- Returns `mod.val::Vector{Float64}`.
"""
function val(mod::Model)
    mod.val
end

"""
    x(mod::Model)

Returns *variable* model parameters, according to the order specified in `mod.x_ind`.

## Default

- Returns `mod.val[mod.x_ind]::Vector{Float64}`.
"""
function x(mod::Model)
    mod.val[mod.x_ind]
end

"""
    par(mod::Model)

Returns *fixed* model parameters, according to the order specified in `mod.par_ind`.

## Default

- Returns `mod.val[mod.par_ind]::Vector{Float64}`.
"""
function par(mod::Model)
    mod.val[mod.par_ind]
end

"""
    x!(mod::Model, x_syms::AbstractArray, x_vals::AbstractArray)

Resets those variable parameters, which are specified in `x_syms`,
by the values in `x_vals`.

## Default

- Copies the values from `x_vals` into the associated locations.
- Subsequently calls [x_changed!](@ref x_changed!) to trigger optional secondary actions.
- Returns `nothing`.
"""
function x!(mod::Model, x_syms::AbstractArray, x_vals::AbstractArray)
    @assert all(sy -> sy ∈ x_sym(mod), x_syms)
    for (sy, v) in zip(x_syms, x_vals)
        mod.val[findfirst(s -> s == sy, sym(mod))] = v
    end
    x_changed!(mod)
    nothing
end

"""
    x!(mod::Model, new_x::AbstractArray)

Resets *all* variable parameters by the values in `new_x`.

## Default 

- Copies the values from `new_x` into `val[x_ind]` (in this order).
- Subsequently calls [x_changed!](@ref x_changed!) to trigger optional secondary actions.
- Returns `nothing`.
"""
function x!(mod::Model, new_x::AbstractArray)
    mod.val[mod.x_ind] = new_x
    x_changed!(mod)
    nothing
end

"""
    par!(mod::Model, p_syms::AbstractArray, p_vals::AbstractArray)

Resets those fixed parameters, which are specified in `par_syms`,
by the values in `p_vals`.

## Default 

- Copies the values from `p_vals` into the associated locations.
- Subsequently calls [par_changed!](@ref par_changed!) to trigger optional secondary actions.
- Returns `nothing`.
"""
function par!(mod::Model, p_syms::AbstractArray, p_vals::AbstractArray)
    @assert all(sy -> sy ∈ par_sym(mod), p_syms)
    for (sy, v) in zip(p_syms, p_vals)
        mod.val[findfirst(s -> s == sy, sym(mod))] = v
    end
    par_changed!(mod)
    nothing
end

"""
    par!(mod::Model, new_par::AbstractArray)

Resets *all* fixed parameters by the values in `new_par`.

## Default 

- Copies the values from `new_par` into `val[par_ind]` (in this order).
- Subsequently calls [par_changed!](@ref par_changed!) to trigger optional secondary actions.
- Returns `nothing`.
"""
function par!(mod::Model, new_par::AbstractArray)
    mod.val[mod.par_ind] = new_par
    par_changed!(mod)
    nothing
end

"""
    x_changed!(::Model)

Informs user-defined model that `x` has changed.

## Default

- Does nothing.

## Remarks

- Can be used to recalculate any auxiliary model variable (such as `A`), which depends on `x`.
"""
function x_changed!(::Model) end

"""
    par_changed!(::Model)

Informs user-defined model that `par` has changed.

## Default 

- Does nothing.

## Remarks

- Can be used to recalculate any auxiliary model variable (such as `A`), which depends on `par`.
"""
function par_changed!(::Model) end

"""
    y(mod::Model)

Returns the actual data vector.

## Default

- Returns `mod.y::SVector{Ny, T}`.
"""
function y(mod::Model)
    mod.y
end

"""
    y!(mod::Model{Ny,Nx,Nc,T}, new_y::AbstractArray) where {Ny,Nx,Nc,T}

Sets new data values.

## Default

- Resets `mod.y::SVector{Ny, T}` with the content of `new_y`.
- Calculates the squared magnitude of `mod.y` and stores the result in `mod.y2::Float64`.
- Returns nothing.
"""
function y!(mod::Model{Ny,Nx,Nc,T}, new_y::AbstractArray) where {Ny,Nx,Nc,T}
    mod.y = SVector{Ny,T}(new_y)
    mod.y2 = real(mod.y' * mod.y)
    nothing
end

"""
    set_data!(mod::Model, new_data::AbstractArray)

Sets new data values safely.

## Default

- Simply calls [y!](@ref y!).

## Remarks

- Can be used to perform preprocessing of data, if required by specific models.
- In case of doubt, always call [set_data!](@ref set_data!) instead of [y!](@ref y!).
"""
function set_data!(mod::Model, new_data::AbstractArray)
    y!(mod, new_data)
end

"""
    A(::Model)

Return VARPRO matrix `A`.

## Default

- Returns `mod.A`, if the field exists and throws an error otherwise.

## Remarks

- Proper functioning of the method is mandatory. If the field `mod.A` does not exist, the method has to be implemented for the specific model.
- Can be replaced by model-specific implementation, if the field `mod.A` does not exist.
- In that scenario, the methods [x_changed!](@ref x_changed!) and [par_changed!](@ref par_changed!) trigger updates of `mod.A`.
- if the models exhibit redundancy , the return type can be
"""
function A(mod::Model)
    if hasfield(typeof(mod), :A)
        mod.A
    else
        error("method A() not implemented.")
    end
end

"""
    B(mod::Model)

Returns matrix `B::SMatrix{Nc,Nc}`

## Remarks

- Returns `A(mod)' * A(mod)` by default.
"""
function B(mod::Model)
    A(mod)' * A(mod)
end

"""
    b(mod::Model)

Returns vector `b::SVector{Nc}`

## Remarks

- Returns `A(mod)' * y(mod)` by default.
"""
function b(mod::Model)
    A(mod)' * y(mod)
end

"""
    c(mod::Model)

Return VARPRO vector `c`.

## Default

- Gets `B` and `b` from [Bb!](@ref Bb!) and calculates generic solution `c = B \\ b`.

## Remarks

- Can be replaced by model-specific implementation, if desired (e.g. for performance improvements).
"""
function c(mod::Model)
    (B, b) = Bb!(mod)
    B \ b
end

"""
    Bb!(mod::Model)

Return matrix `B = A' * A` and vector `b = A' * y`.

## Default

- Direct calculation, based upon methods [A](@ref A) and [y](@ref y).

## Remarks

- Mandatory routine for the calculation of `χ²` (and its derivatives)
- Can be replaced by model-specific implementation, to improve the performance.
- Expected to return `(B, b)::Tuple`.
- For general models, the return types could be `B::SMatrix{Nc,Nc,T}` and `b::SVector{Nc,T}`
- In most cases, model-specific implementations will be more efficient though.
"""
function Bb!(mod::Model)
    A_ = A(mod)
    B = A_' * A_
    b = A_' * y(mod)
    (B, b)
end

"""
    ∂Bb!(::Model)

Returns up to first order partial derivatives with respect to `x`.

## Default

- None, must be supplied by the user.

## Remarks

- Required for first and second order optimization techniques.
- Returns `(B, b, ∂B, ∂b)::Tuple`
- `(B, b)` as returned from [Bb!](@ref Bb!)
- Types: `∂B::SVector{Nx, SMatrix{Nc,Nc,T}}` and `∂b::SVector{Nx, SVector{Nc,T}}`
- (or anything, which works with the implementation of [fg!](@ref) and [fgh!](@ref fgh!))
"""
function ∂Bb!(::Model)
    error("Missing implementation of ∂Bb!")
end

"""
    ∂∂Bb!(::Model)

Returns up to second order partial derivatives with respect to `x`

## Default

- None, must be supplied by the user.

## Remarks

- Required for second order optimization techniques.
- Returns `(B, b, ∂B, ∂b, ∂∂B, ∂∂b)::Tuple`
- `(B, b, ∂B, ∂b)` as returned from [∂Bb!](@ref ∂Bb!)
- Types: `∂∂B::SMatrix{Nx, Nx, SMatrix{Nc,Nc,T}}` and `∂∂b::SMatrix{Nx, Nx, SVector{Nc,T}}`
- (or anything, which works with the implementation of [fgh!](@ref fgh!))
"""
function ∂∂Bb!(::Model)
    error("Missing implementation of ∂∂Bb!")
end

"""
    y_model(mod::Model)

Compute model prediction `A(x) * c`.

## Default

- Calculates the product of the methods [A](@ref A) and [c](@ref c)

## Remarks

- Return type `== SVector{Ny,T}`
- Can be used to check the model or generate synthetic data.
- If necessary, a model-specific implementation may be needed. (e.g., if the method [A](@ref A) has not be implemented.)
"""
function y_model(mod::Model)
    A(mod) * c(mod)
end

"""
    χ2(mod::Model)

Return `χ² = y² - b' * (B \\ b)` of actual model.

## Default

- Uses `mod.y2` and `(B, b)` from [Bb!](@ref Bb!) to directly calculate the expression.
"""
function χ2(mod::Model)
    (B, b) = Bb!(mod)
    f(mod.y2, B, b)
end

"""
    f(mod::Model)

Return function `f` of argument `x` to be minimized, as expected by Optim.jl

## Remark

- Returns anonymous function `x -> ...` (cf. [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl))
- Depends on [Bb!](@ref Bb!).
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

## Remark

- Returns anonymous function `(F, G, x) -> ...` (cf. [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl))
- Depends on [Bb!](@ref Bb!) and [∂Bb!](@ref ∂Bb!).
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

## Remark

- Returns anonymous function `(F, G, H, x) -> ...` (cf. [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl))
- Depends on [Bb!](@ref Bb!), [∂Bb!](@ref ∂Bb!) and [∂∂Bb!](@ref ∂∂Bb!).
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

Returns Hessian of ``χ²(x)``.

## Remark

- Can be used as preconditioner, as expected by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).
"""
function P(mod::Model{Ny,Nx,Nc,T}, x) where {Ny,Nx,Nc,T}
    H = Matrix{T}(undef, Nx, Nx)
    fgh!(mod)(nothing, nothing, H, x)
    H
end

"""
    f(y2, B, b)

Helper function for [`f(::Model)`](@ref f(::Model)).

## Remark

- Should not be called directly.
"""
function f(y2::Float64, B, b)
    y2 - real(b' * (B \ b))
end

"""
    fg!(::Model{Ny,Nx,Nc,T}, F, G, y2, B, b, ∂B, ∂b) where {Ny,Nx,Nc,T}

Helper function for [`fg!(mod::Model)`](@ref fg!(::Model)).

## Remark

- Should not be called directly.
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

"""
    fgh!(::Model{Ny,Nx,Nc,T}, F, G, H, y2, B, b, ∂B, ∂b, ∂∂B, ∂∂b) where {Ny,Nx,Nc,T}

Helper function for [`fgh!(mod::Model)`](@ref fgh!(::Model)).

## Remark

- Should not be called directly.
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

