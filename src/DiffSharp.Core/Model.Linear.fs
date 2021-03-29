// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Model

open DiffSharp

/// <summary>A model that applies a linear transformation to the incoming data: \(y = xA^T + b\)</summary>
type Linear(inFeatures, outFeatures, ?bias:bool) =
    inherit Model()
    let bias = defaultArg bias true
    let w = Parameter(Weight.kaiming(inFeatures, outFeatures))
    let k = 1./sqrt (float outFeatures)
    let b = Parameter(if bias then Weight.uniform([|outFeatures|], k) else dsharp.tensor([]))
    do base.add([w;b],["Linear-weight";"Linear-bias"])

    /// <summary>TBD</summary>
    override _.getString() = sprintf "Linear(%A, %A)" inFeatures outFeatures

    /// <summary>TBD</summary>
    override _.forward(value) =
        let f = dsharp.matmul(value, w.value)
        if bias then f + b.value else f