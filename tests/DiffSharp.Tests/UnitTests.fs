//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, 2015, National University of Ireland Maynooth.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
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
let inline VS_Symbolic x0 x1 x2 = x0 * ((sqrt (x1 + x2)) * (log x2)) ** x1


// diff
let diff_AD_Forward = DiffSharp.AD.Forward.ForwardOps.diff SS
let diff_AD_Forward2 = DiffSharp.AD.Forward2.Forward2Ops.diff SS
let diff_AD_ForwardG = DiffSharp.AD.ForwardG.ForwardGOps.diff SS
let diff_AD_ForwardGH = DiffSharp.AD.ForwardGH.ForwardGHOps.diff SS
let diff_AD_ForwardN = DiffSharp.AD.ForwardN.ForwardNOps.diff SS
let diff_AD_Reverse = DiffSharp.AD.Reverse.ReverseOps.diff SS
let diff_AD_ForwardReverse = DiffSharp.AD.ForwardReverse.ForwardReverseOps.diff SS
let diff_Numerical = DiffSharp.Numerical.NumericalOps.diff SS
let diff_Symbolic = DiffSharp.Symbolic.SymbolicOps.diff <@ SS @>

// diff2
let diff2_AD_Forward2 = DiffSharp.AD.Forward2.Forward2Ops.diff2 SS
let diff2_AD_ForwardN = DiffSharp.AD.ForwardN.ForwardNOps.diff2 SS
let diff2_AD_ForwardReverse = DiffSharp.AD.ForwardReverse.ForwardReverseOps.diff2 SS
let diff2_Numerical = DiffSharp.Numerical.NumericalOps.diff2 SS
let diff2_Symbolic = DiffSharp.Symbolic.SymbolicOps.diff2 <@ SS @>

// diffn
let diffn_AD_ForwardN n = DiffSharp.AD.ForwardN.ForwardNOps.diffn n SS
let diffn_Symbolic n = DiffSharp.Symbolic.SymbolicOps.diffn n <@ SS @>


// diff
[<Property(Verbose = true)>]
let ``diff AD.Forward = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_AD_Forward x =~ diff_Symbolic x)

[<Property(Verbose = true)>]
let ``diff AD.Forward2 = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_AD_Forward2 x =~ diff_Symbolic x)

[<Property(Verbose = true)>]
let ``diff AD.ForwardG = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_AD_ForwardG x =~ diff_Symbolic x)

[<Property(Verbose = true)>]
let ``diff AD.ForwardGH = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_AD_ForwardGH x =~ diff_Symbolic x)

[<Property(Verbose = true)>]
let ``diff AD.ForwardN = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_AD_ForwardN x =~ diff_Symbolic x)

[<Property(Verbose = true)>]
let ``diff AD.Reverse = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_AD_Reverse x =~ diff_Symbolic x)

[<Property(Verbose = true)>]
let ``diff AD.ForwardReverse = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_AD_ForwardReverse x =~ diff_Symbolic x)

[<Property(Verbose = true)>]
let ``diff Numerical = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff_Numerical x =~ diff_Symbolic x)

// diff2
[<Property(Verbose = true)>]
let ``diff2 AD.Forward2 = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff2_AD_Forward2 x =~ diff2_Symbolic x)

[<Property(Verbose = true)>]
let ``diff2 AD.ForwardN = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff2_AD_ForwardN x =~ diff2_Symbolic x)

[<Property(Verbose = true)>]
let ``diff2 AD.ForwardReverse = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff2_AD_ForwardReverse x =~ diff2_Symbolic x)

[<Property(Verbose = true)>]
let ``diff2 Numerical = Symbolic`` (x:float) = (``is nice?`` x) ==> (diff2_Numerical x =~ diff2_Symbolic x)

// diffn
[<Property(Verbose = true)>]
let ``diffn AD.ForwardN = Symbolic`` (x:float) = (``is nice?`` x) ==> (diffn_AD_ForwardN 2 x =~ diffn_Symbolic 2 x)

