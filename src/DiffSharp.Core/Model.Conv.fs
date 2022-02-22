// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp

/// <summary>A model that applies a 1D convolution over an input signal composed of several input planes</summary>
type Conv1d(inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool) =
    inherit Model()
    let biasv = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize))
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSize|], k)
    let b = Parameter <| if biasv then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.addParameter([w;b],["Conv1d-weight";"Conv1d-bias"])

    /// <summary>Get the weight parameter of the model</summary>
    member _.weight = w.value

    /// <summary>Get the bias parameter of the model</summary>
    member _.bias = b.value

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "Conv1d(%A, %A, %A)" inChannels outChannels kernelSize

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv1d(value, w.value, ?stride=stride, ?padding=padding, ?dilation=dilation)
        if biasv then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1]) else f


/// <summary>A model that applies a 2D convolution over an input signal composed of several input planes</summary>
type Conv2d(inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve2dKernelSizes kernelSize kernelSizes
    let biasv = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]))
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]|], k)
    let b = Parameter <| if biasv then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.addParameter([w;b],["Conv2d-weight";"Conv2d-bias"])

    /// <summary>Get the weight parameter of the model</summary>
    member _.weight = w.value

    /// <summary>Get the bias parameter of the model</summary>
    member _.bias = b.value

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "Conv2d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if biasv then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1; 1]) else f


/// <summary>A model that applies a 3D convolution over an input signal composed of several input planes</summary>
type Conv3d(inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve3dKernelSizes kernelSize kernelSizes
    let biasv = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]*kernelSizes.[2]))
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]; kernelSizes.[2]|], k)
    let b = Parameter <| if biasv then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.addParameter([w;b],["Conv3d-weight";"Conv3d-bias"])

    /// <summary>Get the weight parameter of the model</summary>
    member _.weight = w.value

    /// <summary>Get the bias parameter of the model</summary>
    member _.bias = b.value

    /// <summary>TBD</summary>
    override _.ToString() = sprintf "Conv3d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv3d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if biasv then f + b.value.expand([value.shape.[0]; outChannels]).view([value.shape.[0]; outChannels; 1; 1; 1]) else f
