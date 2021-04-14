// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp
open DiffSharp.ShapeChecking

/// <summary>A model that applies a 1D convolution over an input signal composed of several input planes</summary>
type Conv1d(inChannels:Int, outChannels:Int, kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?bias:bool) =
    inherit Model()
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize).ValueOrOne)
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSize|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["Conv1d-weight";"Conv1d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Conv1d(%A, %A, %A)" inChannels outChannels kernelSize

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv1d(value, w.value, ?stride=stride, ?padding=padding, ?dilation=dilation)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool) =
        Conv1d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?bias=bias)

/// <summary>A model that applies a 2D convolution over an input signal composed of several input planes</summary>
type Conv2d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve2dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["Conv2d-weight";"Conv2d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Conv2d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
        Conv2d(Int inChannels, Int outChannels, ?kernelSize=optInt kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?bias=bias)

/// <summary>A model that applies a 3D convolution over an input signal composed of several input planes</summary>
type Conv3d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve3dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels.ValueOrOne*kernelSizes.[0]*kernelSizes.[1]*kernelSizes.[2]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|outChannels; inChannels; kernelSizes.[0]; kernelSizes.[1]; kernelSizes.[2]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["Conv3d-weight";"Conv3d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Conv3d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.conv3d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, ?kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?bias:bool) =
        Conv3d(Int inChannels, Int outChannels, ?kernelSize=optInt kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?bias=bias)

