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

/// Get the tail of a 3-tuple
let inline trd (_, _, t) = t

/// Get the first and third terms of a 3-tuple
let inline fsttrd (f, _, t) = (f, t)

/// Get the second and third terms of a 3-tuple
let inline sndtrd (_, s, t) = (s, t)

/// Checks whether float[,] `m` has the same number of elements in both dimensions
let (|Square|) (m:float[,]) =
    match m with
    | m when m.GetLength 0 = m.GetLength 1 -> m
    | _ -> invalidArg "m" "Expecting a square float[,]"

/// Get the transpose of float[,] `m`
let inline transpose (m:float[,]) = Array2D.init (m.GetLength 1) (m.GetLength 0) (fun i j -> m.[j, i])

/// Get a float[] containing the diagonal elements of float[,] `m`
let inline diagonal (Square m:float[,]) = Array.init (m.GetLength 0) (fun i -> m.[i, i])

/// Get the trace of the square matrix given in float[,] `m`
let inline trace (m:float[,]) = Array.sum (diagonal m)

/// Copy the upper triangular elements of the square matrix given in float[,] `m` to the lower triangular part
let inline copyupper (Square m:float[,]) =
    let rows = m.GetLength 0
    if rows > 1 then
        for i = 1 to rows - 1 do
            for j = 0 to i - 1 do
                m.[i, j] <- m.[j, i]
    m

/// Global step size for numerical approximations
let eps = 0.00001

/// Two times eps
let deps = eps * 2.

/// Square of eps
let epssq = eps * eps

