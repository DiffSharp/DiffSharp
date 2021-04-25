(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Reference.dll"
#r "DiffSharp.Backends.Torch.dll"
(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"

Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

# DiffSharp: Differentiable Tensor Programming Made Simple

DiffSharp is a tensor library with support for [differentiable programming](https://en.wikipedia.org/wiki/Automatic_differentiation).
It is designed for use in machine learning, probabilistic programming, optimization and other domains.

🗹 Nested and mixed-mode differentiation

🗹 Common optimizers, model elements, differentiable probability distributions

🗹 F# for robust functional programming 

🗹 PyTorch familiar naming and idioms, efficient LibTorch C++ tensors

🗹 Linux, Windows and CUDA supported

🗹 Use notebooks in Jupyter and Visual Studio Code

🗹 100% open source 


## Differentiable Programming

DiffSharp provides world-leading automatic differentiation capabilities for tensor code, including composable gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector products over arbitrary user code. This goes beyond conventional tensor libraries such as PyTorch and TensorFlow, allowing the use of nested forward and reverse differentiation up to any level. 

With DiffSharp, you can compute higher-order derivatives efficiently and differentiate functions that are internally making use of differentiation and gradient-based optimization. 

<img src="img/anim-intro-2.gif" width="75%" />

## Practical, Familiar and Efficient

DiffSharp comes with a [LibTorch](https://pytorch.org/cppdocs/) backend, using the same C++ and CUDA implementations for tensor computations that power [PyTorch](https://pytorch.org/). On top of these raw tensors (LibTorch's ATen, excluding autograd), DiffSharp implements its own computation graph and differentiation capabilities. It is tested on Linux and Windows and includes support for CUDA 11.

The DiffSharp API is designed to be similar to [the PyTorch Python API](https://pytorch.org/docs/stable/index.html) through very similar
naming and idioms, and where elements have similar names the PyTorch documentation can generally be used as a guide. There are some improvements and DiffSharp supports a richer gradient/differentiation API.

DiffSharp uses [the incredible F# programming language](https://fsharp.org) for tensor programming. F# code is generally faster and more robust than equivalent Python code, while still being succinct and compact like Python, making it an ideal modern AI and machine learning implementation language. This allows fluent and productive code while focusing on the tensor programming domain.

<iframe width="75%" src="https://www.youtube.com/embed/_QnbV6CAWXc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Quick Usage Examples

You can execute this page as an interactive notebook running in your browser, or download it as a script or .NET Interactive Jupyter notebook, using the buttons [![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath={{fsdocs-source-basename}}.ipynb) 
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx) 
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb) on the top of the page. This applies to all documentation pages.

If using Visual Studio Code you can download, edit and execute these notebooks using [the .NET Interactive Notebooks for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode).

First reference the package:

    #r "nuget: DiffSharp-lite, {{fsdocs-package-version}}"

or for LibTorch support:

    #r "nuget: DiffSharp-cpu, {{fsdocs-package-version}}"
*)

open DiffSharp

(** 
Configure:
*)
dsharp.config(dtype=Dtype.Float32, device=Device.CPU, backend=Backend.Reference)

(** 
Defining and adding two tensors:
*)
let t1 = dsharp.tensor [ 0.0 .. 0.2 .. 1.0 ]
let t2 = dsharp.tensor [ 0, 1, 2, 4, 7, 2 ]

t1 + t2

(** 
Computing a convolution:
*)
let t3 = dsharp.tensor [[[[0.0 .. 10.0]]]]
let t4 = dsharp.tensor [[[[0.0 ..0.1 .. 1.0]]]]

t3.conv2d(t4)

(** 
Take the gradient of a vector-to-scalar function:
*)

let f (x: Tensor) = x.exp().sum()

dsharp.grad f (dsharp.tensor([1.8, 2.5]))

(**
Define a model and optimize it:
*)
(*** do-not-eval-file ***)
open DiffSharp.Data
open DiffSharp.Model
open DiffSharp.Util
open DiffSharp.Optim

let epochs = 2
let batchSize = 32
let numBatches = 5

let trainSet = MNIST("../data", train=true, transform=id)
let trainLoader = trainSet.loader(batchSize=batchSize, shuffle=true)

let validSet = MNIST("../data", train=false, transform=id)
let validLoader = validSet.loader(batchSize=batchSize, shuffle=false)

let model = VAE(28*28, 20, [400])

let lr = dsharp.tensor(0.001)
let optimizer = Adam(model, lr=lr)

for epoch = 1 to epochs do
    let batches = trainLoader.epoch(numBatches)
    for i, x, _ in batches do
        model.reverseDiff()
        let l = model.loss(x)
        l.reverse()
        optimizer.step()
        print $"Epoch: {epoch} minibatch: {i} loss: {l}" 

let validLoss = 
    validLoader.epoch() 
    |> Seq.sumBy (fun (_, x, _) -> model.loss(x, normalize=false))

print $"Validation loss: {validLoss/validSet.length}"


(**

Numerous other model definition and gradient/training patterns are supported, see [examples](https://github.com/DiffSharp/DiffSharp/tree/dev/examples).

## More Information

DiffSharp is developed by [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/)
and other contributors, having started as a project supervised by the automatic differentiation wizards [Barak Pearlmutter](https://scholar.google.com/citations?user=AxFrw0sAAAAJ&hl=en) and [Jeffrey Siskind](https://scholar.google.com/citations?user=CgSBtPYAAAAJ&hl=en). 

Please join us [on GitHub](https://github.com/DiffSharp/DiffSharp)!

*)
