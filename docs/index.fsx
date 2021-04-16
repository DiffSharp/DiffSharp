(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net5.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Reference.dll"

(**
DiffSharp: Differentiable Tensor Programming Made Simple
============================================

DiffSharp is a tensor library with support for [differentiable programming](https://en.wikipedia.org/wiki/Automatic_differentiation).
It is designed for use in machine learning, probabilistic programming, optimization and other domains.


✅ Nested and Mixed-Mode Differentiation

✅ PyTorch Familiar Naming and Idioms

✅ F# for Robust Functional AI Programming 

✅ LibTorch Efficient PyTorch C++ Tensors

✅ Linux, Windows and CUDA supported

✅ Use Notebooks in Jupyter and Visual Studio Code

✅ 100% Open Source 


Differentiable Programming
----------------------------

DiffSharp provides world-leading automatic differentiation capabilities for tensor code,
including gradients, Hessians, Jacobians, directional derivatives, and matrix-free Hessian- and Jacobian-vector
products over arbitrary user code.  This goes far beyond traditional tensor libraries such as TensorFlow and PyTorch, allowing the use of nested
forward and reverse differentiation up to any level. With DiffSharp, you can compute higher-order derivatives efficiently and differentiate functions
that are internally making use of differentiation and optimization. 

Differentiation can be applied to any functions accepting and producing DiffSharp Tensor values.
Please see [API Overview](api-overview.html) for a list of available operations.


Practical, Familiar and Efficient
----------------------------

DiffSharp uses PyTorch C++ tensors (minus the gradient computation) as the default
raw-tensor backend. It is tested on Linux and Windows and includes support for CUDA 11.1.

DiffSharp uses [the incredible F# programming language](https://fsharp.org) for tensor programming.
F# code is generally faster and more robust than equivalent Python code, while
still being succinct and compact like Python, making it an ideal modern AI and machine
learning implementation language. This allows fluent and productive tensor programming while
focusing on the tensor programming domain.
To learn more about "F# as a Better Python" see this video:

<iframe width="280" height="157" src="https://www.youtube.com/embed/_QnbV6CAWXc" title="F# as a Better Python" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

The DiffSharp API is designed to be similar to [the PyTorch Python API](https://pytorch.org/docs/stable/index.html) through very similar
naming and idioms, and where elements have similar names the PyTorch documentation can generally be used as a guide.
There are some improvements and DiffSharp supports a richer gradient/differentiation API.


Quick usage examples
-------------------

Below is a  series of simple samples using DiffSharp. You can access this sample as a [script](index.fsx) or a [.NET Interactive Jupyter Notebook](index.ipynb)
(open in [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=index.ipynb)).
If using Visual Studio Code you can download, edit and execute these notebooks
using [the .NET Interactive Notebooks for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.dotnet-interactive-vscode).

First reference the package:

    #r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"

or for Torch support:

    #r "nuget: DiffSharp-cpu,{{fsdocs-package-version}}"
*)

(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"

Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun x writer -> fprintfn writer "%120A" x )
#endif // IPYNB

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

let f (x: Tensor) = sin x.[0] + cos x.[1]

dsharp.grad f (dsharp.tensor(1.83))

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
    let batches = trainLoader.epoch() |> Seq.truncate numBatches
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

Numerous other model definition and gradient/training patterns are supported, see
the [examples](https://github.com/DiffSharp/DiffSharp/tree/dev/examples) directory.

Current features and roadmap
----------------------------

**The library and documentation are undergoing development.**

The primary features of DiffSharp 1.0 are:

- A tensor programming model for F#.

- A reference backend for correctness testing.

- [PyTorch](https://pytorch.org/) backend for CUDA support and highly optimized native tensor operations.

- Nested differentiation for tensors, supporting forward and reverse AD, or any combination thereof, up to any level.

- Matrix-free Jacobian- and Hessian-vector products.

- Common optimizers and model elements including convolutions.

- Probability distributions.

More information
-------------------------

DiffSharp is developed by [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/)
and other contributors, having started as a project supervised by [Barak Pearlmutter](https://scholar.google.com/citations?user=AxFrw0sAAAAJ&hl=en) and [Jeffrey Siskind](https://scholar.google.com/citations?user=CgSBtPYAAAAJ&hl=en). Please join us!

If you are using DiffSharp, please raise any issues you might have [on GitHub](https://github.com/DiffSharp/DiffSharp).
We also have a [Gitter chat room](https://gitter.im/DiffSharp/DiffSharp).
If you would like to cite this library, please use the following information:

_Baydin, A.G., Pearlmutter, B.A., Radul, A.A. and Siskind, J.M., 2017. Automatic differentiation in machine learning: a survey. The Journal of Machine Learning Research, 18(1), pp.5595-5637._ ([link](https://arxiv.org/abs/1502.05767))

*)
