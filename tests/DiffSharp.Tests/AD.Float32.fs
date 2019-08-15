// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

module DiffSharp.Tests.AD.Float32

open FsCheck.NUnit
open DiffSharp.Util
open DiffSharp.Tests
open DiffSharp.AD.Float32


[<Property>]
let ``AD.32.F.D.FixedPoint``() =
    let g (a:D) (b:D) = (a + b / a) / (D 2.f)
    let p, t = jacobianv' (D.FixedPoint g (D 1.2f)) (D 25.f) (D 1.f)
    Util.(=~)(p, D 5.f) && Util.(=~)(t, D 0.1f)

[<Property>]
let ``AD.32.R.D.FixedPoint``() =
    let g (a:D) (b:D) = (a + b / a) / (D 2.f)
    let p, t = jacobianTv' (D.FixedPoint g (D 1.2f)) (D 25.f) (D 1.f)
    Util.(=~)(p, D 5.f) && Util.(=~)(t, D 0.1f)

[<Property>]
let ``Compute Adjoint``() =
    let tag = DiffSharp.Util.GlobalTagger.Next        
    
    let Wt = toDM [[0.0f; 1.0f]]
    let Wt' = Wt |> makeReverse tag   
    let loss (weights:DM) : D = cos (weights.Item(0,0))
    
    let L = loss Wt'
    let A = computeAdjoints L //Smoke test computeAdjoints, was an issue with single precision

    ()

[<Property>]
let ``Gradient descent``() =

    let minimize (f:DV->D) (x0:DV) = 
        let eta = 1e-2f
        let mutable W = x0
        for _ in [0..10] do
            let L,g = grad' f W
            W <- W - eta*g

    let lossFunction (w:DV) =
        let x = toDM [[1.0; 0.0]]
        let Wg = w.[0..3] |> DM.ofDV 2
        let g = (x*Wg)
        cos g.[0,0]

    minimize lossFunction (DV.create 5 1.0f) //Smoke test


[<Property>]
let ``Gradient descent (with arrays)``() =

    let minimize (f:DV->D) (x0:DV) = 
        let eta = 1e-2f
        let mutable W = x0
        for _ in [0..10] do
            let L,g = grad' f W
            W <- W - eta*g

    let n = 5
    let lossFunction (w:DV) =
        let x = DM.init n n (fun i j -> w.[n*i+j])
        let x' = x.GetSlice(None, None, None, None)
        cos x'.[0,0]

    minimize lossFunction (DV.create (n*n) 1.0f) //Smoke test