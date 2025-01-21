var documenterSearchIndex = {"docs":
[{"location":"man/example/#Short-example","page":"Example","title":"Short example","text":"","category":"section"},{"location":"man/example/#Biexponential-decay","page":"Example","title":"Biexponential decay","text":"","category":"section"},{"location":"man/example/","page":"Example","title":"Example","text":"Let us consider the following model","category":"page"},{"location":"man/example/","page":"Example","title":"Example","text":"y(t_j)=c_1e^-R_1 t_j + c_2e^-R_2 t_j=\nA_jkleft(bmxright)c_k","category":"page"},{"location":"man/example/","page":"Example","title":"Example","text":"to describe data y_j, acquired at time points t_j.  We consider a complex model with R_k c_k in mathbbC.","category":"page"},{"location":"man/example/","page":"Example","title":"Example","text":"Following the notation in Variable Projection, we then just get ","category":"page"},{"location":"man/example/","page":"Example","title":"Example","text":"bmx=R_1^primeR_1^primeprimeR_2^primeR_2^primeprime^T\nqquadtextandqquad\nA_jkleft(bmxright) = e^-R_k t_j","category":"page"},{"location":"man/example/","page":"Example","title":"Example","text":"where R_k = R_k^prime + iR_k^primeprime.","category":"page"},{"location":"man/example/","page":"Example","title":"Example","text":"The implementation of this model, as expected by VP4Optim,  can be found in BiExpDecay.jl.  The corresponding file  test_BiExpDecay.jl showcases, how user defined models can be tested for correctness.","category":"page"},{"location":"man/example/#Optimization","page":"Example","title":"Optimization","text":"","category":"section"},{"location":"man/example/","page":"Example","title":"Example","text":"One can, for example, use such a model for optimization with Optim.jl:","category":"page"},{"location":"man/example/","page":"Example","title":"Example","text":"using Optim\nimport VP4Optim as VP\n\n# load the model definition\ninclude(\"BiExpDecay.jl\")\n\n# time points, where the data have been collected\nts = collect(range(0, 5, 10))\n# specification of the (nonlinear) model parameter names\nsym = [:reR1, :imR1, :reR2, :imR2]\n\n# create VP4Optim model instance\nbi = BiExpDecay(ts, sym)\n\n# read measured data ...\ny = fetch_data_from_somewhere()\n# ... and supply them to the model\nVP.y!(bi, y) \n\n# define some starting value for optimization in the [reR1, imR1, reR2, imR2] space\nx0 = [0.1, 0.1, 0.2, -0.2]\n# if we want to restrict the optimization space, we can define lower and upper bounds\nlo = [0, -π, 0, -π]\nup = [1, π, 1, π]\n\n# now we can start the optimization\nres = optimize(Optim.only_fg!(fg!(bi)), lo, up, x0, Fminbox(LBFGS()))\n\n# and extract the found results\nreR1, imR1, reR2, imR2 = x = res.minimizer\n\n# if we assume these values for our model ...\nVP.x!(bi, x)\n\n# ... the missing estimator of the linear coefficients can be simply obtained by\nc1, c2 = c = VP.c(bi)","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"CurrentModule = VP4Optim","category":"page"},{"location":"man/guide/#Guide","page":"Guide","title":"Guide","text":"","category":"section"},{"location":"man/guide/#Installation","page":"Guide","title":"Installation","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"The package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"pkg> add VP4Optim","category":"page"},{"location":"man/guide/#Model-definition","page":"Guide","title":"Model definition","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"VARPRO can be applied to any data model that can be written in the form  bmyapprox bmAbmc. Apparently, the nontrivial part of the model definition depends on how  the matrix bmAleft(bmpright) is actually defined as a function of tunable parameters bmp.  With respect to optimization though, we sometimes may only want some subset bmxsubseteqbmp  to be variable while keeping the rest fixed[1]. In addition, we want to remain flexible with respect to  selecting different subsets of bmp as variable. ","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"To address these requirements, any specific VARPRO model in  VP4Optim shall be defined as some subtype of the abstract type VP4Optim.Model{Ny,Nx,Nc,T}","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Modules=[VP4Optim]\nOrder=[:type]","category":"page"},{"location":"man/guide/#VP4Optim.Model","page":"Guide","title":"VP4Optim.Model","text":"Model{Ny,Nx,Nc,T}\n\nAbstract supertype of any model specification.\n\nType parameters\n\nNy::Int: length of data vector y.\nNx::Int: number of variable parameters.\nNc::Int: number of linear coefficients.\nT <: Union{Float64, ComplexF64} is equal to eltype(y)\n\n\n\n\n\n","category":"type"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Some fields are mandatory in any model definition, as shown in the following example of a complex valued model[2]:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"import VP4Optim as VP\nusing StaticArrays\n\nmutable struct SpecificModel{Ny,Nx,Nc} <: VP.Model{Ny,Nx,Nc,ComplexF64}\n    # mandatory fields of any Model instance\n\n    # names of all parameters, variable and fixed \n    # Example: sym == [:a, :b, :c, :d, :e]\n    \n    sym::Vector{Symbol}     \n\n    # subset of variable parameter names\n    # Example: x_sym == [:d, :a, :c] (order defined by x_ind below)\n    # Note that this also implies Nx == 3\n    \n    x_sym::Vector{Symbol}   \n    \n    # complementary subset of fixed parameter names\n    # Example: par_sym == [:e, :b] (order defined by par_ind below)\n    \n    par_sym::Vector{Symbol} \n    \n    # values of all parameters in the order specified by sym\n    # Example: val == [a, b, c, d, e]\n    \n    val::Vector{Float64}\n    \n    # indices of variable parameters in field val\n    # Example: x_ind == SVector{Nx}([4, 1, 3]) (according to the definition of x_sym above)\n    # such that\n    # x_sym[1] == sym[x_ind[1]] == sym[4] == :d\n    # val[x_ind[1]] == val[4] == d\n    # x_sym[2] == sym[x_ind[2]] == sym[1] == :a\n    # val[x_ind[2]] == val[1] == a\n    # x_sym[3] == sym[x_ind[3]] == sym[3] == :c\n    # val[x_ind[3]] == val[3] == c\n    \n    x_ind::SVector{Nx,Int}\n\n    # indices of fixed parameters in field val\n    # Example: par_ind == [5, 2] (according to the definition of par_sym above)\n    # such that\n    # par_sym[1] == sym[par_ind[1]] == sym[5] == :e\n    # val[par_ind[1]] == val[5] == e\n    # par_sym[2] == sym[par_ind[2]] == sym[2] == :b\n    # val[par_ind[2]] == val[2] == b\n    \n    par_ind::Vector{Int}\n    \n    # actual data vector\n    \n    y::SVector{Ny,ComplexF64}            \n    \n    # == real(y' * y) (by default automatically calculated in generic method VP.y!())\n    \n    y2::Float64                 \n\n    # optional model-specific information\n    # (typically, some allocated workspace to avoid costly redundant calculations)\n    # ...\nend","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"note: Note\nFor general models, one can include the matrix A::SMatrix{Ny, Nc} as an additional field in any model implementation, since this prevents unnecessary recomputations of A. With this approach, one would let the methods x_changed! and par_changed! trigger updates of the field A.For proper functioning of the package, neither field A or method A are mandatory though, as long as the methods Bb!, ∂Bb! and ∂∂Bb! are implemented properly (cf. the method descriptions for more details).  Depending on the model, this route can often be preferrable to improve numerical performance.","category":"page"},{"location":"man/guide/#Constructor","page":"Guide","title":"Constructor","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"To benefit from the increased performance of  StaticArrays.jl, the parameters Ny, Nx, Nc need to be known at compile time. This can be accomplished with the following exemplary setup:","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"A constructor to be called by the user/application, e.g.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"function SpecificModel(args, sym=[:a, :b, :c, :d, :e]; x_sym=nothing)\n    # some code, which will be similar for different models\n\n    # name the parameters as desired\n    sym = collect(sym)\n    \n    # specify the variable subset of the parameters\n    # (defaults to sym)\n    x_sym === nothing && (x_sym = deepcopy(sym))\n    Nx = length(x_sym)\n\n    # check that x_sym is a subset of sym\n    @assert all(sy -> sy ∈ sym, x_sym)\n   \n    # Ny will usually be deduced from args\n    # (for example by the number of time points, when the data were sampled)\n    Ny = some_function_of(args)\n\n    # If not already fixed by SpecificModel, Nc must also be inferred from args:\n    Nc = some_other_function_of(args)\n\n    # now we can call a constructor, where Ny, Nx, Nc are converted into types\n    SpecificModel(Val(Ny), Val(Nx), Val(Nc), args, sym, x_sym)\nend","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"which calls","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"function SpecificModel(::Val{Ny}, ::Val{Nx}, ::Val{Nc}, args, sym, x_sym) where {Ny, Nx, Nc}\n    # the remaining mandatory fields can be calculated as follows\n    val = zeros(length(sym))\n    x_ind = SVector{Nx,Int}(findfirst(x -> x == x_sym[i], sym) for i in 1:Nx)\n    par_ind = filter(x -> x ∉ x_ind, 1:length(sym))\n    par_sym = sym[par_ind]\n    y = SVector{Ny,ComplexF64}(zeros(ComplexF64, Ny))\n    y2 = 0.0\n    \n    # optionally initialize further fields, which appear in SpecificModel\n    # ...\n\n    # finally, generate an instance of SpecificModel with the default constructor\n    SpecificModel{Ny, Nx, Nc}(sym, x_sym, par_sym, val, x_ind, par_ind, y, y2, ...)\nend","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"note: Note\nOften, the number of linear coefficients Nc is already determined by the SpecificModel type.  In such cases, Nc will not appear as a type parameter. An example is the biexponential (Nc == 2) decay model in BiExpDecay.jlmutable struct BiExpDecay{Ny, Nx} <: VP.Model{Ny, Nx, 2, ComplexF64}\n...\nendAlso the constructors will then need to be adapted accordingly, as  shown there.","category":"page"},{"location":"man/guide/#Model-parameters","page":"Guide","title":"Model parameters","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Based upon the mandatory fields in any Model instance, the following routines allow access and (partly) modify the fields (after initialization in the constructor).","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"sym\nval\nx_sym\nx\nx!(::Model, ::AbstractArray, ::AbstractArray)\nx!(::Model, ::AbstractArray)\npar_sym\npar\npar!(::Model, ::AbstractArray, ::AbstractArray)\npar!(::Model, ::AbstractArray)\ny\ny!","category":"page"},{"location":"man/guide/#VP4Optim.sym","page":"Guide","title":"VP4Optim.sym","text":"sym(mod::Model)\n\nReturn all model parameter names (type: Symbol), variable and fixed.\n\nDefault\n\nReturns mod.sym::Vector{Symbol}.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.val","page":"Guide","title":"VP4Optim.val","text":"val(mod::Model)\n\nReturns all model parameters.\n\nDefault\n\nReturns mod.val::Vector{Float64}.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.x_sym","page":"Guide","title":"VP4Optim.x_sym","text":"x_sym(mod::Model)\n\nReturn variable model parameter names.\n\nDefault\n\nReturns mod.x_sym::Vector{Symbol}.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.x","page":"Guide","title":"VP4Optim.x","text":"x(mod::Model)\n\nReturns variable model parameters, according to the order specified in mod.x_ind.\n\nDefault\n\nReturns mod.val[mod.x_ind]::Vector{Float64}.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.x!-Tuple{VP4Optim.Model, AbstractArray, AbstractArray}","page":"Guide","title":"VP4Optim.x!","text":"x!(mod::Model, x_syms::AbstractArray, x_vals::AbstractArray)\n\nResets those variable parameters, which are specified in x_syms, by the values in x_vals.\n\nDefault\n\nCopies the values from x_vals into the associated locations.\nSubsequently calls x_changed! to trigger optional secondary actions.\nReturns nothing.\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#VP4Optim.x!-Tuple{VP4Optim.Model, AbstractArray}","page":"Guide","title":"VP4Optim.x!","text":"x!(mod::Model, new_x::AbstractArray)\n\nResets all variable parameters by the values in new_x.\n\nDefault\n\nCopies the values from new_x into val[x_ind] (in this order).\nSubsequently calls x_changed! to trigger optional secondary actions.\nReturns nothing.\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#VP4Optim.par_sym","page":"Guide","title":"VP4Optim.par_sym","text":"par_sym(mod::Model)\n\nReturns fixed model parameter names.\n\nDefault\n\nReturns mod.par_sym::Vector{Symbol}.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.par","page":"Guide","title":"VP4Optim.par","text":"par(mod::Model)\n\nReturns fixed model parameters, according to the order specified in mod.par_ind.\n\nDefault\n\nReturns mod.val[mod.par_ind]::Vector{Float64}.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.par!-Tuple{VP4Optim.Model, AbstractArray, AbstractArray}","page":"Guide","title":"VP4Optim.par!","text":"par!(mod::Model, p_syms::AbstractArray, p_vals::AbstractArray)\n\nResets those fixed parameters, which are specified in par_syms, by the values in p_vals.\n\nDefault\n\nCopies the values from p_vals into the associated locations.\nSubsequently calls par_changed! to trigger optional secondary actions.\nReturns nothing.\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#VP4Optim.par!-Tuple{VP4Optim.Model, AbstractArray}","page":"Guide","title":"VP4Optim.par!","text":"par!(mod::Model, new_par::AbstractArray)\n\nResets all fixed parameters by the values in new_par.\n\nDefault\n\nCopies the values from new_par into val[par_ind] (in this order).\nSubsequently calls par_changed! to trigger optional secondary actions.\nReturns nothing.\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#VP4Optim.y","page":"Guide","title":"VP4Optim.y","text":"y(mod::Model)\n\nReturns the actual data vector.\n\nDefault\n\nReturns mod.y::SVector{Ny, T}.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.y!","page":"Guide","title":"VP4Optim.y!","text":"y!(mod::Model{Ny,Nx,Nc,T}, new_y::AbstractArray) where {Ny,Nx,Nc,T}\n\nSets new data values.\n\nDefault\n\nResets mod.y::SVector{Ny, T} with the content of new_y.\nCalculates the squared magnitude of mod.y and stores the result in mod.y2::Float64.\nReturns nothing.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"note: Note\nChanging the data does not require to generate a new model instance!","category":"page"},{"location":"man/guide/#VARPRO-routines","page":"Guide","title":"VARPRO routines","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"These methods provide an interface to VARPRO, as described in Variable Projection.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"A\nBb!","category":"page"},{"location":"man/guide/#VP4Optim.A","page":"Guide","title":"VP4Optim.A","text":"A(::Model)\n\nReturn VARPRO matrix A.\n\nDefault\n\nReturns mod.A, if the field exists and 'nothing' otherwise\n\nRemarks\n\nImplementation is not mandatory,\nCan be replaced by model-specific implementation, if the field mod.A does not exist.\nIn that scenario, the methods x_changed! and par_changed! trigger updates of mod.A.\nif the models exhibit redundancy , the return type can be\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.Bb!","page":"Guide","title":"VP4Optim.Bb!","text":"Bb!(mod::Model)\n\nReturn matrix B = A' * A and vector b = A' * y.\n\nDefault\n\nDirect calculation, based upon methods A and y.\n\nRemarks\n\nMandatory routine for the calculation of χ² (and its derivatives)\nCan be replaced by model-specific implementation, to improve the performance.\nExpected to return (B, b)::Tuple.\nFor general models, the return types could be B::SMatrix{Nc,Nc,T} and b::SVector{Nc,T}\nIn most cases, model-specific implementations will be more efficient though.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"note: Note\nAs long as y^2 - b' * (B \\ b) gives the correct result (χ2), everything should be fine.\nThe same is true for the partial derivatives.In case of doubt, look at how  f, fg! and fgh! are  implemented in the source code.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"c\ny_model\nχ2","category":"page"},{"location":"man/guide/#VP4Optim.c","page":"Guide","title":"VP4Optim.c","text":"c(mod::Model)\n\nReturn VARPRO vector c.\n\nDefault\n\nGets B and b from Bb! and calculates generic solution c = B \\ b.\n\nRemarks\n\nCan be replaced by model-specific implementation, if desired (e.g. for performance improvements).\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.y_model","page":"Guide","title":"VP4Optim.y_model","text":"y_model(mod::Model)\n\nCompute model prediction A(x) * c.\n\nDefault\n\nCalculates the product of the methods A and c\n\nRemarks\n\nReturn type == SVector{Ny,T}\nCan be used to check the model or generate synthetic data.\nIf necessary, a model-specific implementation may be needed. (e.g., if the method A has not be implemented.)\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.χ2","page":"Guide","title":"VP4Optim.χ2","text":"χ2(mod::Model)\n\nReturn χ² = y² - b' * (B \\ b) of actual model.\n\nDefault\n\nUses mod.y2 and (B, b) from Bb! to directly calculate the expression.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#Interface-for-[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)","page":"Guide","title":"Interface for Optim.jl","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"Numerical minimization can be facilitated by the use of powerful optimization libraries, such as Optim.jl. To this end,  VP4Optim provides some convenience functions, which  provide interfaces to χ² and its partial derivatives (up to second order) in a form as  typically expected by the optimization libraries. Also a Hessian-based preconditioner is available.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"f(::Model)\nfg!(::Model)\nfgh!(::Model)\nP","category":"page"},{"location":"man/guide/#VP4Optim.f-Tuple{VP4Optim.Model}","page":"Guide","title":"VP4Optim.f","text":"f(mod::Model)\n\nReturn function f of argument x to be minimized, as expected by Optim.jl\n\nRemark\n\nReturns anonymous function x -> ... (cf. Optim.jl)\nDepends on Bb!.\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#VP4Optim.fg!-Tuple{VP4Optim.Model}","page":"Guide","title":"VP4Optim.fg!","text":"fg!(mod::Model)\n\nReturn function fg! of three arguments (F, G, x) as expected by Optim.jl.\n\nRemark\n\nReturns anonymous function (F, G, x) -> ... (cf. Optim.jl)\nDepends on Bb! and ∂Bb!.\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#VP4Optim.fgh!-Tuple{VP4Optim.Model}","page":"Guide","title":"VP4Optim.fgh!","text":"fgh!(mod::Model)\n\nReturn function fgh! of four arguments (F, G, H, x) as expected by Optim.jl.\n\nRemark\n\nReturns anonymous function (F, G, H, x) -> ... (cf. Optim.jl)\nDepends on Bb!, ∂Bb! and ∂∂Bb!.\n\n\n\n\n\n","category":"method"},{"location":"man/guide/#VP4Optim.P","page":"Guide","title":"VP4Optim.P","text":"P(mod::Model{Ny,Nx,Nc,T}, x) where {Ny,Nx,Nc,T}\n\nReturns Hessian of χ²(x).\n\nRemark\n\nCan be used as preconditioner, as expected by Optim.jl.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#Model-specific","page":"Guide","title":"Model specific","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"x_changed!\npar_changed!\n∂Bb!\n∂∂Bb!","category":"page"},{"location":"man/guide/#VP4Optim.x_changed!","page":"Guide","title":"VP4Optim.x_changed!","text":"x_changed!(::Model)\n\nInforms user-defined model that x has changed.\n\nDefault\n\nDoes nothing.\n\nRemarks\n\nCan be used to recalculate any auxiliary model variable (such as A), which depends on x.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.par_changed!","page":"Guide","title":"VP4Optim.par_changed!","text":"par_changed!(::Model)\n\nInforms user-defined model that par has changed.\n\nDefault\n\nDoes nothing.\n\nRemarks\n\nCan be used to recalculate any auxiliary model variable (such as A), which depends on par.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.∂Bb!","page":"Guide","title":"VP4Optim.∂Bb!","text":"∂Bb!(::Model)\n\nReturns up to first order partial derivatives with respect to x.\n\nDefault\n\nNone, must be supplied by the user.\n\nRemarks\n\nRequired for first and second order optimization techniques.\nReturns (B, b, ∂B, ∂b)::Tuple\n(B, b) as returned from Bb!\nTypes: ∂B::SVector{Nx, SMatrix{Nc,Nc,T}} and ∂b::SVector{Nx, SVector{Nc,T}}\n(or anything, which works with the implementation of fg! and fgh!)\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#VP4Optim.∂∂Bb!","page":"Guide","title":"VP4Optim.∂∂Bb!","text":"∂∂Bb!(::Model)\n\nReturns up to second order partial derivatives with respect to x\n\nDefault\n\nNone, must be supplied by the user.\n\nRemarks\n\nRequired for second order optimization techniques.\nReturns (B, b, ∂B, ∂b, ∂∂B, ∂∂b)::Tuple\n(B, b, ∂B, ∂b) as returned from ∂Bb!\nTypes: ∂∂B::SMatrix{Nx, Nx, SMatrix{Nc,Nc,T}} and ∂∂b::SMatrix{Nx, Nx, SVector{Nc,T}}\n(or anything, which works with the implementation of fgh!)\n\n\n\n\n\n","category":"function"},{"location":"man/guide/#Model-testing","page":"Guide","title":"Model testing","text":"","category":"section"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"check_model","category":"page"},{"location":"man/guide/#VP4Optim.check_model","page":"Guide","title":"VP4Optim.check_model","text":"check_model(modcon, args, vals, c_, y_;\nwhat=(:consistency, :derivatives, :optimization),\nsmall=sqrt(eps()),\nx0=[], lx=[], ux=[], x_scale=[],\nprecon=true,\nvisual=false,\nrng=MersenneTwister(),\nHessian=true,\nlog10_rng=range(-6, -3, 10),\nmin_slope=0.9)\n\nTests, which any specific model should pass.\n\nArguments\n\nmodcon::Function: Constructor to the model to be tested.\nargs::Tuple: Arguments, as expected by constructor, like modcon(args...) or modcon(args; x_sym=x_sym).\nvals::Vector{Float64}: All nonlinear parameters, the model depends on. As defined in the Model field val.\nc_::Vector{Nc, T}: Linear cofficients, the model depends on.\ny_::Vector{T}: Data, corresponding to the true parameters vals and c_. \nwhat::Tuple{Symbol}: Tests to be performed (see below).\nsmall::Float64: Required as accuracy criterion.\nx0::Vector{Float64}: Starting point for optimization and location, where derivatives are tested.\nlx::Vector{Float64}: Lower bound of optimization\nux::Vector{Float64}: Upper bound of optimization\nx_scale::Vector{Float64}: Scaling vector, such that δx = randn(size(x)) .* x_scale becomes reasonable\nprecon::Bool: Test optimization with and without preconditioner.\nvisual::Bool: If true also generate double-logarithmic plots for the derivative tests.\nrng::MersenneTwister: Allows to pass a unique seed (e.g. MersenneTwister(42)) for reproducible testing.\nHessian::Bool: Should be set to false, if the model does not implement second order derivatives.\nlog10_rng::AbstractVector: logarithmic range for derivative testing\nmin_slope::Float64: minimal derivative slope on log-log plot\n\nRemark\n\nTests are performed for every x_sym ⊆ sym.\nReturns a dictionary with detailed information about the test results.\n:consistency ∈ what: Several basic tests (parameters, names, correct model values)\n:derivatives ∈ what: Check first and second order partial derivatives at x0.\n:optimization ∈ what: Minimize model with x0 as starting point and bounds lx and ux.\nAn example application can be found in test_BiExpDecay.jl. This should also work as a template, how to perform tests on own models.\n\n\n\n\n\n","category":"function"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"[1]: In Variable Projection, bmx referred to this variable part only, while any fixed parameters were absorbed in the actual definition of the matrix bmA.","category":"page"},{"location":"man/guide/","page":"Guide","title":"Guide","text":"[2]: Here, we assume a fixed data type for the specific model.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = VP4Optim","category":"page"},{"location":"#VP4Optim.jl","page":"Home","title":"VP4Optim.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Some tools to simplify the implementation of variable projection in Julia.","category":"page"},{"location":"#Variable-Projection","page":"Home","title":"Variable Projection","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Consider a least squares (LS) cost function of the form[1] ","category":"page"},{"location":"","page":"Home","title":"Home","text":"chi^2left(bmx bmcright) =\nleftbmy - bmA(bmx) cdot bmcright^2_2","category":"page"},{"location":"","page":"Home","title":"Home","text":"to be minimized with respect to bmx and bmc","category":"page"},{"location":"","page":"Home","title":"Home","text":"hatbmx hatbmc =\nundersetbmx bmcoperatornameargmin\nchi^2left(bmx bmcright)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Depending on the problem of interest, data vector bmy, matrix bmA and linear  coefficient vector bmc, can be real or complex. Only for the parameter vector bmx, we restrict ourselves to real values[2].","category":"page"},{"location":"","page":"Home","title":"Home","text":"Variable projection (VARPRO) is an established method to reduce the dimensionality of this optimization problem by eliminating the linear parameters bmc. To this end, we exploit that for any given bmx (not necessarily at the minimum), the minimum of  chi^2left(bmx bmcright) with respect to the linear coefficients bmc must satisfy","category":"page"},{"location":"","page":"Home","title":"Home","text":"bmcleft(bmxright) =bmB^-1bmb","category":"page"},{"location":"","page":"Home","title":"Home","text":"where we defined[3]","category":"page"},{"location":"","page":"Home","title":"Home","text":"bmBleft(bmxright)=bmA^astbmA\nqquadqquad\nbmbleft(bmxright)=bmA^astbmy","category":"page"},{"location":"","page":"Home","title":"Home","text":"Assuming this value for bmc, the cost function ","category":"page"},{"location":"","page":"Home","title":"Home","text":"chi^2left(bmxright) =chi^2left(bmxbmcleft(bmxright)right)","category":"page"},{"location":"","page":"Home","title":"Home","text":"depends only on bmx and can be written the form[4]:","category":"page"},{"location":"","page":"Home","title":"Home","text":"chi^2left(bmxright) = y^2 - bmb^astbmB^-1bmb","category":"page"},{"location":"","page":"Home","title":"Home","text":"This allows us to first determine the estimator of the internal parameter","category":"page"},{"location":"","page":"Home","title":"Home","text":"hatbmx =\nundersetbmxoperatornameargmin\nchi^2left(bmxright)","category":"page"},{"location":"","page":"Home","title":"Home","text":"and subsequently simply calculate the optimal linear coefficient","category":"page"},{"location":"","page":"Home","title":"Home","text":"hatbmc = bmcleft(hatbmxright)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that besides reducing the number of independent parameters, the dimensions of bmB and bmb equal the number of elements of the linear coefficient vector bmc, which is often a small number.","category":"page"},{"location":"#Package-Features","page":"Home","title":"Package Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The VP4Optim package provides some tools to simplify the implementation of VARPRO.","category":"page"},{"location":"","page":"Home","title":"Home","text":"template for models (real or complex) to be implemented by the user\nbased upon StaticArrays.jl to improve performance\nwrapper for optimization libraries, such as Optim.jl\nsupport for partial derivatives up to second order\ntest tools to check correctness of user supplied models","category":"page"},{"location":"","page":"Home","title":"Home","text":"[1]: chi^2 can result from a maximum likelihood (ML) minimization problem, except for a missing noise term.  In the most general cases, when the variance of the latter is not constant, such that y_j have  variable standard deviations sigma_j, we can recover the  given expression for chi^2 by simple rescaling: y_jsigma_j to y_j and  A_jksigma_j to A_jk","category":"page"},{"location":"","page":"Home","title":"Home","text":"[2]: Most optimization libraries actually rely on this assumption, but this does not constitute a restriction: Any complex variable z = z^prime + iz^primeprime corresponds to two real ones (e.g. z^prime and z^primeprime).","category":"page"},{"location":"","page":"Home","title":"Home","text":"[3]: bmA^ast denotes the conjugate tranpose of bmA and we suppressed the dependence on bmx to improve readability.    ","category":"page"},{"location":"","page":"Home","title":"Home","text":"[4]: y^2 = bmy^astbmy","category":"page"},{"location":"#Manual","page":"Home","title":"Manual","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"    Pages=[\n        \"man/example.md\",\n        \"man/guide.md\",\n        \"man/api.md\",\n        ]\n    Depth=1","category":"page"},{"location":"man/api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"man/api/","page":"API","title":"API","text":"","category":"page"}]
}
