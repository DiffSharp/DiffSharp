//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under LGPL license.
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

module DiffSharp.Tests

open Xunit
open FsCheck
open FsCheck.Xunit


let ``is nice?`` x =
    not (System.Double.IsInfinity(x) || System.Double.IsNegativeInfinity(x) || System.Double.IsPositiveInfinity(x) || System.Double.IsNaN(x) || (x = System.Double.MinValue) || (x = System.Double.MaxValue))

let (=~) x y =
    abs ((x - y) / x) < 1e-3

[<ReflectedDefinition>]
let inline SS x = (sin x) * (cos (exp x))

let inline VS (x:_[]) = x.[0] * ((sqrt (x.[1] + x.[2])) * (log x.[2])) ** x.[1]

[<ReflectedDefinition>]
let inline VS_S x0 x1 x2 = x0 * ((sqrt (x1 + x2)) * (log x2)) ** x1


// diff
let diff_AD = DiffSharp.AD.DiffOps.diff SS
let diff_ADF = DiffSharp.AD.Forward.DiffOps.diff SS
let diff_ADR = DiffSharp.AD.Reverse.DiffOps.diff SS
let diff_SADF1 = DiffSharp.AD.Specialized.Forward1.DiffOps.diff SS
let diff_SADF2 = DiffSharp.AD.Specialized.Forward2.DiffOps.diff SS
let diff_SADFG = DiffSharp.AD.Specialized.ForwardG.DiffOps.diff SS
let diff_SADFGH = DiffSharp.AD.Specialized.ForwardGH.DiffOps.diff SS
let diff_SADFN = DiffSharp.AD.Specialized.ForwardN.DiffOps.diff SS
let diff_SADR1 = DiffSharp.AD.Specialized.Reverse1.DiffOps.diff SS
let diff_N = DiffSharp.Numerical.DiffOps.diff SS
let diff_S = DiffSharp.Symbolic.DiffOps.diff <@ SS @>

// diff2
let diff2_AD = DiffSharp.AD.DiffOps.diff2 SS
let diff2_ADF = DiffSharp.AD.Forward.DiffOps.diff2 SS
let diff2_ADR = DiffSharp.AD.Reverse.DiffOps.diff2 SS
let diff2_SADF2 = DiffSharp.AD.Specialized.Forward2.DiffOps.diff2 SS
let diff2_SADFN = DiffSharp.AD.Specialized.ForwardN.DiffOps.diff2 SS
let diff2_N = DiffSharp.Numerical.DiffOps.diff2 SS
let diff2_S = DiffSharp.Symbolic.DiffOps.diff2 <@ SS @>

// diffn
let diffn_AD n = DiffSharp.AD.DiffOps.diffn n SS
let diffn_ADF n = DiffSharp.AD.Forward.DiffOps.diffn n SS
let diffn_ADR n = DiffSharp.AD.Reverse.DiffOps.diffn n SS
let diffn_SADFN n = DiffSharp.AD.Specialized.ForwardN.DiffOps.diffn n SS
let diffn_S n = DiffSharp.Symbolic.DiffOps.diffn n <@ SS @>

// diff
[<Property(Verbose = true)>]
let ``diff AD = S`` (x:float) = (``is nice?`` x) ==> (diff_AD (DiffSharp.AD.D x) |> float =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff ADF = S`` (x:float) = (``is nice?`` x) ==> (diff_ADF (DiffSharp.AD.Forward.D x) |> float =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff ADR = S`` (x:float) = (``is nice?`` x) ==> (diff_ADR (DiffSharp.AD.Reverse.D x) |> float =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff SADF1 = S`` (x:float) = (``is nice?`` x) ==> (diff_SADF1 x =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff SADF2 = S`` (x:float) = (``is nice?`` x) ==> (diff_SADF2 x =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff SADFG = S`` (x:float) = (``is nice?`` x) ==> (diff_SADFG x =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff SADFGH = S`` (x:float) = (``is nice?`` x) ==> (diff_SADFGH x =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff SADFN = S`` (x:float) = (``is nice?`` x) ==> (diff_SADFN x =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff SADR1 = S`` (x:float) = (``is nice?`` x) ==> (diff_SADR1 x =~ diff_S x)

[<Property(Verbose = true)>]
let ``diff N = S`` (x:float) = (``is nice?`` x) ==> (diff_N x =~ diff_S x)

// diff2
[<Property(Verbose = true)>]
let ``diff2 AD = S`` (x:float) = (``is nice?`` x) ==> (diff2_AD (DiffSharp.AD.D x) |> float =~ diff2_S x)

[<Property(Verbose = true)>]
let ``diff2 ADF = S`` (x:float) = (``is nice?`` x) ==> (diff2_ADF (DiffSharp.AD.Forward.D x) |> float =~ diff2_S x)

[<Property(Verbose = true)>]
let ``diff2 ADR = S`` (x:float) = (``is nice?`` x) ==> (diff2_ADR (DiffSharp.AD.Reverse.D x) |> float =~ diff2_S x)

[<Property(Verbose = true)>]
let ``diff2 SADF2 = S`` (x:float) = (``is nice?`` x) ==> (diff2_SADF2 x =~ diff2_S x)

[<Property(Verbose = true)>]
let ``diff2 SADFN = S`` (x:float) = (``is nice?`` x) ==> (diff2_SADFN x =~ diff2_S x)

[<Property(Verbose = true)>]
let ``diff2 N = S`` (x:float) = (``is nice?`` x) ==> (diff2_N x =~ diff2_S x)

// diffn
[<Property(Verbose = true)>]
let ``diffn AD = S`` (x:float) = (``is nice?`` x) ==> (diffn_AD 2 (DiffSharp.AD.D x) |> float =~ diffn_S 2 x)

[<Property(Verbose = true)>]
let ``diffn ADF = S`` (x:float) = (``is nice?`` x) ==> (diffn_ADF 2 (DiffSharp.AD.Forward.D x) |> float =~ diffn_S 2 x)

[<Property(Verbose = true)>]
let ``diffn ADR = S`` (x:float) = (``is nice?`` x) ==> (diffn_ADR 2 (DiffSharp.AD.Reverse.D x) |> float =~ diffn_S 2 x)

[<Property(Verbose = true)>]
let ``diffn SADFN = S`` (x:float) = (``is nice?`` x) ==> (diffn_SADFN 2 x =~ diffn_S 2 x)
