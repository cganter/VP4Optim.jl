```@meta
CurrentModule = VP4Optim
```

# Guide

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run

```
pkg> add VP4Optim
```

## Model definition

VARPRO can be applied to any data model that can be written in the form 
``\bm{y}\approx \bm{A}\,\bm{c}``. Apparently, the nontrivial part of the model definition depends on how 
the matrix ``\bm{A}\left(\bm{p}\right)`` is actually defined as a function of tunable parameters ``\bm{p}``. 
With respect to optimization though, we sometimes may only want some subset ``\bm{x}\subseteq\bm{p}`` 
to be variable while keeping the rest fixed[^1]. In addition, we want to remain flexible with respect to 
selecting different subsets of ``\bm{p}`` as variable. 

To address these requirements, any specific VARPRO model in 
[VP4Optim](https://github.com/cganter/VP4Optim.jl) shall be defined as
some subtype of the abstract type `VP4Optim.Model{Ny,Nx,Nc,T}`
```@autodocs
Modules=[VP4Optim]
Order=[:type]
```
Some fields are mandatory in any model definition, as shown in the following example of a complex
valued model[^2]:
```julia
import VP4Optim as VP
using StaticArrays

mutable struct SpecificModel{Ny,Nx,Nc} <: VP.Model{Ny,Nx,Nc,ComplexF64}
    # mandatory fields of any Model instance

    # names of all parameters, variable and fixed 
    # Example: sym == [:a, :b, :c, :d, :e]
    
    sym::Vector{Symbol}     

    # subset of variable parameter names
    # Example: x_sym == [:d, :a, :c] (order defined by x_ind below)
    # Note that this also implies Nx == 3
    
    x_sym::Vector{Symbol}   
    
    # complementary subset of fixed parameter names
    # Example: par_sym == [:e, :b] (order defined by par_ind below)
    
    par_sym::Vector{Symbol} 
    
    # values of all parameters in the order specified by sym
    # Example: val == [a, b, c, d, e]
    
    val::Vector{Float64}
    
    # indices of variable parameters in field val
    # Example: x_ind == SVector{Nx}([4, 1, 3]) (according to the definition of x_sym above)
    # such that
    # x_sym[1] == sym[x_ind[1]] == sym[4] == :d
    # val[x_ind[1]] == val[4] == d
    # x_sym[2] == sym[x_ind[2]] == sym[1] == :a
    # val[x_ind[2]] == val[1] == a
    # x_sym[3] == sym[x_ind[3]] == sym[3] == :c
    # val[x_ind[3]] == val[3] == c
    
    x_ind::SVector{Nx,Int}

    # indices of fixed parameters in field val
    # Example: par_ind == [5, 2] (according to the definition of par_sym above)
    # such that
    # par_sym[1] == sym[par_ind[1]] == sym[5] == :e
    # val[par_ind[1]] == val[5] == e
    # par_sym[2] == sym[par_ind[2]] == sym[2] == :b
    # val[par_ind[2]] == val[2] == b
    
    par_ind::Vector{Int}
    
    # actual data vector
    
    y::SVector{Ny,ComplexF64}            
    
    # == real(y' * y) (by default automatically calculated in generic method VP.y!())
    
    y2::Float64                 

    # optional model-specific information
    # (typically, some allocated workspace to avoid costly redundant calculations)
    # ...
end
```

!!! note
    For general models, one can include the matrix `A::SMatrix{Ny, Nc}` as an additional
    field in any model implementation, since this prevents unnecessary recomputations of `A`.
    With this approach, one would let the methods [x_changed!](@ref x_changed!)
    and [par_changed!](@ref par_changed!) trigger updates of the field `A`.

    For proper functioning of the package, neither field `A` or method [A](@ref A) are mandatory
    though, as long as the methods [Bb!](@ref Bb!), [∂Bb!](@ref ∂Bb!) and [∂∂Bb!](@ref ∂∂Bb!) are
    implemented properly (cf. the method descriptions for more details). 
    Depending on the model, this route can often be preferrable to improve numerical performance.

## Constructor

To benefit from the increased performance of 
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl), the parameters `Ny`, `Nx`, `Nc` need to
be known at *compile time*. This can be accomplished with the following exemplary setup:

A constructor to be called by the user/application, e.g.
```julia
function SpecificModel(args, sym=[:a, :b, :c, :d, :e]; x_sym=nothing)
    # some code, which will be similar for different models

    # name the parameters as desired
    sym = collect(sym)
    
    # specify the variable subset of the parameters
    # (defaults to sym)
    x_sym === nothing && (x_sym = deepcopy(sym))
    Nx = length(x_sym)

    # check that x_sym is a subset of sym
    @assert all(sy -> sy ∈ sym, x_sym)
   
    # Ny will usually be deduced from args
    # (for example by the number of time points, when the data were sampled)
    Ny = some_function_of(args)

    # If not already fixed by SpecificModel, Nc must also be inferred from args:
    Nc = some_other_function_of(args)

    # now we can call a constructor, where Ny, Nx, Nc are converted into types
    SpecificModel(Val(Ny), Val(Nx), Val(Nc), args, sym, x_sym)
end
```
which calls
```julia
function SpecificModel(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, args, sym, x_sym) where {Ny, Nx, Nc}
    # the remaining mandatory fields can be calculated as follows
    val = zeros(length(sym))
    x_ind = SVector{Nx,Int}(findfirst(x -> x == x_sym[i], sym) for i in 1:Nx)
    par_ind = filter(x -> x ∉ x_ind, 1:length(sym))
    par_sym = sym[par_ind]
    y = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))
    y2 = 0.0
    
    # optionally initialize further fields, which appear in SpecificModel
    # ...

    # finally, generate an instance of SpecificModel with the default constructor
    SpecificModel{Ny, Nx, Nc}(sym, x_sym, par_sym, val, x_ind, par_ind, y, y2, ...)
end
```
!!! note
    Often, the number of linear coefficients `Nc` is already determined by the `SpecificModel` type. 
    In such cases, `Nc` will not appear as a type parameter. 
    
    An example is the biexponential (`Nc == 2`) decay model in
    [BiExpDecay.jl](https://github.com/cganter/VP4Optim.jl/blob/main/test/BiExpDecay.jl)
    ```julia
    mutable struct BiExpDecay{Ny, Nx} <: VP.Model{Ny, Nx, 2, ComplexF64}
    ...
    end
    ```
    Also the constructors will then need to be adapted accordingly, as 
    shown there.

## Model parameters

Based upon the mandatory fields in any `Model` instance, the following routines allow access
and (partly) modify the fields (after initialization in the constructor).

```@docs
N_data
N_var
N_coeff
data_type
sym
val
x_sym
x
x!(::Model, ::AbstractArray, ::AbstractArray)
x!(::Model, ::AbstractArray)
par_sym
par
par!(::Model, ::AbstractArray, ::AbstractArray)
par!(::Model, ::AbstractArray)
y
y!
```

!!! note
    Changing the data does *not* require to generate a new model instance!

## VARPRO routines

These methods provide an interface to VARPRO, as described in [Variable Projection](@ref).

```@docs
A
B
b
Bb!
```

!!! note
    As long as `y^2 - b' * (B \ b)` gives the correct result (`χ2`), everything should be fine.\
    The same is true for the partial derivatives.

    In case of doubt, look at how  [f](@ref f), [fg!](@ref fg!) and [fgh!](@ref fgh!) are 
    implemented in the source code.

```@docs
c
y_model
χ2
```

## Interface for [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)

Numerical minimization can be facilitated by the use of powerful optimization libraries, such as
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl). To this end, 
[VP4Optim](https://github.com/cganter/VP4Optim.jl) provides some convenience functions, which 
provide interfaces to ``χ²`` and its partial derivatives (up to second order) in a form as 
typically expected by the optimization libraries. Also a Hessian-based preconditioner is available.

```@docs
f(::Model)
fg!(::Model)
fgh!(::Model)
P
```

## Model specific 

```@docs
x_changed!
par_changed!
∂Bb!
∂∂Bb!
```

## Model testing

```@docs
check_model
```

[^1]:
    In [Variable Projection](@ref), ``\bm{x}`` referred to
    this *variable* part only, while any fixed parameters were absorbed in the actual definition of the
    matrix ``\bm{A}``.
[^2]:
    Here, we assume a fixed data type for the specific model.