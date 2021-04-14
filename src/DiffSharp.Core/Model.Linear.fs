// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp

/// <summary>A model that applies a linear transformation to the incoming data: \(y = xA^T + b\)</summary>
type Linear(inFeatures:Int, outFeatures:Int, ?bias:bool) =
    inherit Model()
    let hasBias = defaultArg bias true
    let w = Parameter(Weight.kaiming(inFeatures, outFeatures))
    let k = 1./sqrt (float outFeatures.ValueOrOne)
    let b = Parameter(if hasBias then Weight.uniform([|outFeatures|], k) else dsharp.tensor([]))
    do base.add([w;b],["Linear-weight";"Linear-bias"])
    
    /// <summary>TBD</summary>
    member _.weight = w

    /// <summary>TBD</summary>
    member _.bias = b

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Linear(%A, %A)" inFeatures outFeatures

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.matmul(value, w.value)
        if hasBias then f + b.value else f
        
    /// <summary>TBD</summary>
    new (inFeatures: int, outFeatures: int, ?bias:bool) =
       Linear(Int inFeatures, Int outFeatures, ?bias=bias)
