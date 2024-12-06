using VP4Optim
using Documenter

DocMeta.setdocmeta!(VP4Optim, :DocTestSetup, :(using VP4Optim); recursive=true)

makedocs(;
    modules=[VP4Optim],
    authors="Carl Ganter <cganter@tum.de>",
    sitename="VP4Optim.jl",
    format=Documenter.HTML(;
        canonical="https://cganter.github.io/VP4Optim.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cganter/VP4Optim.jl",
    devbranch="main",
)
