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
