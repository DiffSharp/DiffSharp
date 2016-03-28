//
// This file is part of
// DiffSharp: Differentiable Functional Programming
//
// Copyright (c) 2014--2016, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under the LGPL license.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU Lesser General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

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