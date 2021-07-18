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
// Google Colab only: uncomment and run the following to install dotnet and the F# kernel
// !bash <(curl -Ls https://raw.githubusercontent.com/gbaydin/scripts/main/colab_dotnet5.sh)
#endif // IPYNB
(*** condition: ipynb ***)
#if IPYNB
// Import DiffSharp package
#r "nuget: DiffSharp-lite,{{fsdocs-package-version}}"

// Set dotnet interactive formatter to plaintext
Formatter.SetPreferredMimeTypeFor(typeof<obj>, "text/plain")
Formatter.Register(fun (x:obj) (writer: TextWriter) -> fprintfn writer "%120A" x )
#endif // IPYNB

(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiffSharp/diffsharp.github.io/blob/master/{{fsdocs-source-basename}}.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath={{fsdocs-source-basename}}.ipynb)&emsp;
[![Script](img/badge-script.svg)]({{fsdocs-source-basename}}.fsx)&emsp;
[![Script](img/badge-notebook.svg)]({{fsdocs-source-basename}}.ipynb)

# Quickstart

Here we cover how some key tasks involved in a machine learning application can be implemented with DiffSharp. Note that a significant part of DiffSharp's design has been influenced by [PyTorch](https://pytorch.org/) and you would feel mostly at home if you have familiarity with PyTorch.

## Datasets and Data Loaders

DiffSharp provides `cref:T:DiffSharp.Data.Dataset` type that represents data sources typically used in machine learning pipelines and the `cref:T:DiffSharp.Data.DataLoader` type that handles the loading of data from datasets and iterating over minibatches of data. 

See the [DiffSharp.Data](/reference/diffsharp-data.html) namespace for the full API reference.

### Datasets

DiffSharp provides ready to use dataset types covering typical datasets in machine learning, such as `cref:T:DiffSharp.Data.MNIST`, `cref:T:DiffSharp.Data.CIFAR10`, `cref:T:DiffSharp.Data.CIFAR100`, and also more generic dataset types such as `cref:T:DiffSharp.Data.TensorDataset` or `cref:T:DiffSharp.Data.ImageDataset`.

The following loads the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and shows one image entry and the corresponding label.
*)

open DiffSharp.Data

let dataset = MNIST("../data", train=true, transform=id)

let data, label = dataset.[7]
printfn "%A\nLabel: %A" (data.toImageString()) label
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

In practice a data loader would be typically used to iterate over all minibatches in a given dataset in order to feed each minibatch through a machine learning model. One full iteration over the dataset would be called an "epoch". Typically you would perform multiple such epochs of iterations during the training of a model.

*)
let epochs = 20

for epoch = 1 to epochs do
    for i, data, labels in loader.epoch() do
        printfn "Epoch %A, minibatch %A" epoch (i+1)
        // Process the minibatch
        // ...
(**

## Models

Machine learning models are typically differentiable functions whose parameters can be tuned via gradient-based optimization, optimizing an objective function quantifying the fit of the model with a given set of data. 

DiffSharp provides the most commonly used model building blocks including convolutions, transposed convolutions, batch normalization, dropout, recurrent and other architectures.

See the [DiffSharp.Model](/reference/diffsharp-model.html) namespace for the full API reference.

### Constructing models, PyTorch style

If you are experience with [PyTorch](https://pytorch.org/), you would find the following way of model definition familiar. Let's look at an example of a [generative adversarial network (GAN)](https://arxiv.org/abs/1406.2661) architecture.
*)

// PyTorch style

(**
### Constructing models, DiffSharp style

If you are fond of functional programming and you would like to benefit from F#'s real strengths in functional programming, you have other model definition paradigms at your disposal. For example, the following constructs the same GAN architecture using DiffSharp's `-->` differentiable composition operator, which allows you to seamlessly compose `Model` instances and differentiable `Tensor->Tensor` functions.
*)

// DiffSharp style


(**
## Optimizers

See the [DiffSharp.Optim](/reference/diffsharp-optim.html) namespace for the full API reference.

## A Complete Typical Training Loop

The following example puts together ...

*)
