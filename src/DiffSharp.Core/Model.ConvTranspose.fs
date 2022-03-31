// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp


/// <summary>A model that applies a 1D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose1d(inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool) =
    inherit Model()
    let biasv = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize))
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSize|], k)
    let b = Parameter <| if biasv then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.addParameter((w, "ConvTranspose1d-weight"), (b, "ConvTranspose1d-bias"))

    /// <summary>Get or set the weight parameter of the model</summary>
    member _.weight
        with get() = w.value
        and set v = w.value <- v

    /// <summary>Get or set the bias parameter of the model</summary>
    member _.bias
        with get() = b.value
        and set v = b.value <- v

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "ConvTranspose1d(%A, %A, %A)" inChannels outChannels kernelSize

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose1d(value, w.value, ?stride=stride, ?padding=padding, ?dilation=dilation)
        if biasv then f + b.value.expand([value.shape[0]; outChannels]).view([value.shape[0]; outChannels; 1]) else f


/// <summary>A model that applies a 2D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose2d(inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve2dKernelSizes kernelSize kernelSizes
    let biasv = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes[0]*kernelSizes[1]))
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSizes[0]; kernelSizes[1]|], k)
    let b = Parameter <| if biasv then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.addParameter((w, "ConvTranspose2d-weight"), (b, "ConvTranspose2d-bias"))

    /// <summary>Get or set the weight parameter of the model</summary>
    member _.weight
        with get() = w.value
        and set v = w.value <- v

    /// <summary>Get or set the bias parameter of the model</summary>
    member _.bias
        with get() = b.value
        and set v = b.value <- v

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "ConvTranspose2d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if biasv then f + b.value.expand([value.shape[0]; outChannels]).view([value.shape[0]; outChannels; 1; 1]) else f

/// <summary>A model that applies a 3D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose3d(inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve3dKernelSizes kernelSize kernelSizes
    let biasv = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes[0]*kernelSizes[1]*kernelSizes[2]))
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSizes[0]; kernelSizes[1]; kernelSizes[2]|], k)
    let b = Parameter <| if biasv then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.addParameter((w, "ConvTranspose3d-weight"), (b, "ConvTranspose3d-bias"))

    /// <summary>Get or set the weight parameter of the model</summary>
    member _.weight
        with get() = w.value
        and set v = w.value <- v

    /// <summary>Get or set the bias parameter of the model</summary>
    member _.bias
        with get() = b.value
        and set v = b.value <- v

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "ConvTranspose3d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose3d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if biasv then f + b.value.expand([value.shape[0]; outChannels]).view([value.shape[0]; outChannels; 1; 1; 1]) else f