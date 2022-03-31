(*** condition: prepare ***)
#I "../tests/DiffSharp.Tests/bin/Debug/net6.0"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Data.dll"
#r "DiffSharp.Backends.Reference.dll"
#r "DiffSharp.Backends.Torch.dll"
#r "nuget: SixLabors.ImageSharp,1.0.1" 
// These are needed to make fsdocs --eval work. If we don't select a backend like this in the beginning, we get erratic behavior.
DiffSharp.dsharp.config(backend=DiffSharp.Backend.Reference)
DiffSharp.dsharp.seed(123)
open DiffSharp.Util

(*** condition: fsx ***)
#if FSX
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"
#r "nuget: SixLabors.ImageSharp,1.0.1"
#endif // FSX
(*** condition: ipynb ***)
#if IPYNB
// Google Colab only: uncomment and run the following to install dotnet and the F# kernel
// !bash <(curl -Ls https://raw.githubusercontent.com/gbaydin/scripts/main/colab_dotnet6.sh)
#endif // IPYNB
(*** condition: ipynb ***)
#if IPYNB
// Import DiffSharp package
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"
#r "nuget: SixLabors.ImageSharp,1.0.1"

// Set dotnet interactive formatter to plaintext
Formatter.SetPreferredMimeTypesFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiffSharp/diffsharp.github.io/blob/master/{{fsdocs-source-basename}}.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

# Quickstart

Here we cover some key tasks involved in a typical machine learning pipeline and how these can be implemented with DiffSharp. Note that a significant part of DiffSharp's design has been influenced by [PyTorch](https://pytorch.org/) and you would feel mostly at home if you have familiarity with PyTorch.

## Datasets and Data Loaders

DiffSharp provides the `cref:T:DiffSharp.Data.Dataset` type that represents a data source and the `cref:T:DiffSharp.Data.DataLoader` type that handles the loading of data from datasets and iterating over [minibatches](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method) of data.

See the [DiffSharp.Data](/reference/diffsharp-data.html) namespace for the full API reference.

### Datasets

DiffSharp has ready-to-use types that cover main datasets typically used in machine learning, such as `cref:T:DiffSharp.Data.MNIST`, `cref:T:DiffSharp.Data.CIFAR10`, `cref:T:DiffSharp.Data.CIFAR100`, and also more generic dataset types such as `cref:T:DiffSharp.Data.TensorDataset` or `cref:T:DiffSharp.Data.ImageDataset`.

The following loads the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and shows one image entry and the corresponding label.
*)

open DiffSharp
open DiffSharp.Data

// First ten images in MNIST training set
let dataset = MNIST("../data", train=true, transform=id, n=10)

// Inspect a single image and label
let data, label = dataset[7]

// Save image to file
data.saveImage("test.png")

(** *)

(*** hide ***)
pngToHtml "test.png" 64
(*** include-it-raw ***)

// Inspect data as ASCII and show label
printfn "Data: %A\nLabel: %A" (data.toImageString()) label
(*** include-output ***)

(**
 
### Data Loaders

A data loader handles tasks such as constructing minibatches from an underlying dataset on-the-fly, shuffling the data, and moving the data tensors between devices. In the example below we show a single batch of six MNIST images and their corresponding classification labels.

*)

let loader = DataLoader(dataset, shuffle=true, batchSize=6)
let batch, labels = loader.batch()

printfn "%A\nLabels: %A" (batch.toImageString()) labels
(*** include-output ***)

(**

In practice a data loader is typically used to iterate over all minibatches in a given dataset in order to feed each minibatch through a machine learning model. One full iteration over the dataset would be called an "epoch". Typically you would perform multiple such epochs of iterations during the training of a model.

*)

for epoch = 1 to 10 do
    for i, data, labels in loader.epoch() do
        printfn "Epoch %A, minibatch %A" epoch (i+1)
        // Process the minibatch
        // ...
(**

## Models

Many machine learning models are differentiable functions whose parameters can be tuned via [gradient-based optimization](https://en.wikipedia.org/wiki/Gradient_descent), finding an optimum for an objective function that quantifies the fit of the model to a given set of data. These models are typically built as compositions non-linear functions and ready-to-use building blocks such as linear, recurrent, and convolutional layers.

DiffSharp provides the most commonly used model building blocks including convolutions, transposed convolutions, batch normalization, dropout, recurrent and other architectures.

See the [DiffSharp.Model](/reference/diffsharp-model.html) namespace for the full API reference.

### Constructing models, PyTorch style

If you have experience with [PyTorch](https://pytorch.org/), you would find the following way of model definition familiar. Let's look at an example of a [generative adversarial network (GAN)](https://arxiv.org/abs/1406.2661) architecture.
*)
open DiffSharp.Model
open DiffSharp.Compose

// PyTorch style

// Define a model class inheriting the base
type Generator(nz: int) =
    inherit Model()
    let fc1 = Linear(nz, 256)
    let fc2 = Linear(256, 512)
    let fc3 = Linear(512, 1024)
    let fc4 = Linear(1024, 28*28)
    do base.addModel(fc1, fc2, fc3, fc4)
    override self.forward(x) =
        x
        |> dsharp.view([-1;nz])
        |> fc1.forward
        |> dsharp.leakyRelu(0.2)
        |> fc2.forward
        |> dsharp.leakyRelu(0.2)
        |> fc3.forward
        |> dsharp.leakyRelu(0.2)
        |> fc4.forward
        |> dsharp.tanh

// Define a model class inheriting the base
type Discriminator(nz:int) =
    inherit Model()
    let fc1 = Linear(28*28, 1024)
    let fc2 = Linear(1024, 512)
    let fc3 = Linear(512, 256)
    let fc4 = Linear(256, 1)
    do base.addModel(fc1, fc2, fc3, fc4)
    override self.forward(x) =
        x
        |> dsharp.view([-1;28*28])
        |> fc1.forward
        |> dsharp.leakyRelu(0.2)
        |> dsharp.dropout(0.3)
        |> fc2.forward
        |> dsharp.leakyRelu(0.2)
        |> dsharp.dropout(0.3)
        |> fc3.forward
        |> dsharp.leakyRelu(0.2)
        |> dsharp.dropout(0.3)
        |> fc4.forward
        |> dsharp.sigmoid

// Instantiate the defined classes
let nz = 128
let gen = Generator(nz)
let dis = Discriminator(nz)

print gen
print dis
(*** include-output ***)

(**
### Constructing models, DiffSharp style

A key advantage of DiffSharp lies in the [functional programming](https://en.wikipedia.org/wiki/Functional_programming) paradigm enabled by the F# language, where functions are first-class citizens, many algorithms can be constructed by applying and composing functions, and differentiation operations can be expressed as composable [higher-order functions](https://en.wikipedia.org/wiki/Higher-order_function). This allows very succinct (and beautiful) machine learning code to be expressed as a powerful combination of [lambda calculus](https://en.wikipedia.org/wiki/Lambda_calculus) and [differential calculus](https://en.wikipedia.org/wiki/Differential_calculus).

For example, the following constructs the same GAN architecture (that we constructed in PyTorch style in the previous section) using DiffSharp's `-->` composition operator, which allows you to seamlessly compose `Model` instances and differentiable `Tensor->Tensor` functions. 
*)

// DiffSharp style

// Model as a composition of models and Tensor->Tensor functions
let generator =
    dsharp.view([-1;nz])
    --> Linear(nz, 256)
    --> dsharp.leakyRelu(0.2)
    --> Linear(256, 512)
    --> dsharp.leakyRelu(0.2)
    --> Linear(512, 1024)
    --> dsharp.leakyRelu(0.2)
    --> Linear(1024, 28*28)
    --> dsharp.tanh

// Model as a composition of models and Tensor->Tensor functions
let discriminator =
    dsharp.view([-1; 28*28])
    --> Linear(28*28, 1024)
    --> dsharp.leakyRelu(0.2)
    --> dsharp.dropout(0.3)
    --> Linear(1024, 512)
    --> dsharp.leakyRelu(0.2)
    --> dsharp.dropout(0.3)
    --> Linear(512, 256)
    --> dsharp.leakyRelu(0.2)
    --> dsharp.dropout(0.3)
    --> Linear(256, 1)
    --> dsharp.sigmoid

print generator
print discriminator
(*** include-output ***)
