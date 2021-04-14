// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp
open DiffSharp.ShapeChecking


/// <summary>A model that applies a 1D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose1d(inChannels:Int, outChannels:Int, kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?bias:bool, ?outputPadding: Int) =
    inherit Model()
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSize).ValueOrOne)
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSize|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["ConvTranspose1d-weight";"ConvTranspose1d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "ConvTranspose1d(%A, %A, %A)" inChannels outChannels kernelSize

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose1d(value, w.value, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?bias:bool, ?outputPadding: int) =
        ConvTranspose1d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?bias=bias, ?outputPadding=optInt outputPadding)

/// <summary>A model that applies a 2D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose2d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?outputPadding:Int, ?dilation:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool, ?outputPaddings:seq<Int>) =
    inherit Model()
    let kernelSizes = Shape.resolve2dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSizes.[0]; kernelSizes.[1]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["ConvTranspose2d-weight";"ConvTranspose2d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "ConvTranspose2d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose2d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?outputPadding=outputPadding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations, ?outputPaddings=outputPaddings)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding: int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputPaddings: seq<int>, ?dilations:seq<int>, ?bias:bool) =
        ConvTranspose2d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?outputPadding=optInt outputPadding, ?dilation=optInt dilation, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?bias=bias, ?outputPaddings=optInts outputPaddings)

/// <summary>A model that applies a 3D transposed convolution operator over an input image composed of several input planes.</summary>
type ConvTranspose3d(inChannels:Int, outChannels:Int, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>, ?outputPaddings:seq<Int>, ?dilations:seq<Int>, ?bias:bool) =
    inherit Model()
    let kernelSizes = Shape.resolve3dKernelSizes kernelSize kernelSizes
    let bias = defaultArg bias true
    let k = 1./ sqrt (float (inChannels*kernelSizes.[0]*kernelSizes.[1]*kernelSizes.[2]).ValueOrOne)
    let w = Parameter <| Weight.uniform([|inChannels; outChannels; kernelSizes.[0]; kernelSizes.[1]; kernelSizes.[2]|], k)
    let b = Parameter <| if bias then Weight.uniform([|outChannels|], k) else dsharp.tensor([])
    do base.add([w;b],["ConvTranspose3d-weight";"ConvTranspose3d-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "ConvTranspose3d(%A, %A, %A)" inChannels outChannels kernelSizes

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.convTranspose3d(value, w.value, ?stride=stride, ?strides=strides, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)
        if bias then f + b.value.expand([value.shapex.[0]; outChannels]).view([value.shapex.[0]; outChannels; 1I; 1I; 1I]) else f

    /// <summary>TBD</summary>
    new (inChannels:int, outChannels:int, kernelSize:int, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding: int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings: seq<int>, ?bias:bool) =
        ConvTranspose3d(Int inChannels, Int outChannels, Int kernelSize, ?stride=optInt stride, ?padding=optInt padding, ?dilation=optInt dilation, ?outputPadding=optInt outputPadding, ?kernelSizes=optInts kernelSizes, ?strides=optInts strides, ?paddings=optInts paddings, ?dilations=optInts dilations, ?outputPaddings=optInts outputPaddings, ?bias=bias)

