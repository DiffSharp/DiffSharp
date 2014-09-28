//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, National University of Ireland Maynooth.
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

#light

/// Various utility functions used all over the library
module DiffSharp.Util.General

/// First and third terms of a 3-tuple
let fsttrd (f, _, t) = (f, t)

/// Second and third terms of a 3-tuple
let sndtrd (_, s, t) = (s, t)

/// Tail of a 3-tuple
let trd (_, _, t) = t
    
/// Matrix transpose of a 2d array `m`
let transpose (m:_[,]) = Array2D.init (m.GetLength 1) (m.GetLength 0) (fun x y -> m.[y,x])

/// Matrix transpose of a list of lists
let rec transposeList = function
    | (_::_)::_ as M -> List.map List.head M :: transposeList (List.map List.tail M)
    | _ -> []

/// Apply `f` to a range of integers `i0` to `i1`, and accumulate the result
let sum i0 i1 f =
    let mutable t = 0.
    for i in i0..i1 do
        t <- t + f i
    t

/// Matrix trace of a 2d array `m`
let trace (m:_[,]) = 
    let sum i0 i1 f =
        let mutable t = 0.
        for i in i0..i1 do
            t <- t + f i
        t
    if m.GetLength 0 = m.GetLength 1 then sum 0 ((m.GetLength 0) - 1) (fun i -> m.[i, i]) else failwith "Trace not defined for nonsquare matrix."

/// Fill the elements of a symmetrix matrix from the upper triangular part given in 2d array `t`
let symmetricFromUpperTriangular (t:float[,]) =
    let m = t.GetLength 0
    if m = t.GetLength 1 then 
        if m = 1 then
            Array2D.create 1 1 t.[0, 0]
        else
            for i = 1 to m - 1 do
                for j = 0 to i - 1 do
                    t.[i, j] <- t.[j, i]
            Array2D.init m m (fun i j -> t.[i, j])
    else failwith "Expecting a square 2d array."

/// Global step size for numerical approximations
let eps = 0.000001

/// Two times eps
let deps = eps * 2.

/// Square of eps
let epssq = eps * eps

