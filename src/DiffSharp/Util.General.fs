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

/// Checks whether 'a[,] `m` has the same number of elements in both dimensions
let (|Square|) (m:'a[,]) =
    match m with
    | m when m.GetLength 0 = m.GetLength 1 -> m
    | _ -> invalidArg "m" "Expecting a square float[,]"

/// Get the transpose of 'a[,] `m`
let inline transpose (m:'a[,]) = Array2D.init (m.GetLength 1) (m.GetLength 0) (fun i j -> m.[j, i])

/// Get a 'a[] containing the diagonal elements of 'a[,] `m`
let inline diagonal (Square m:'a[,]) = Array.init (m.GetLength 0) (fun i -> m.[i, i])

/// Get the trace of the square matrix given in 'a[,] `m`
let inline trace (m:'a[,]) = Array.sum (diagonal m)

/// Copy the upper triangular elements of the square matrix given in 'a[,] `m` to the lower triangular part
let inline copyupper (Square m:'a[,]) =
    let r = Array2D.copy m
    let rows = r.GetLength 0
    if rows > 1 then
        for i = 1 to rows - 1 do
            for j = 0 to i - 1 do
                r.[i, j] <- r.[j, i]
    r

/// Get a string representation of a float[] that can be pasted into a Mathematica notebook
let MathematicaVector (v:float[]) =
    let sb = System.Text.StringBuilder()
    sb.Append("{") |> ignore
    for i = 0 to v.Length - 1 do
        sb.Append(sprintf "%.2f" v.[i]) |> ignore
        if i < v.Length - 1 then sb.Append(", ") |> ignore
    sb.Append("}") |> ignore
    sb.ToString()

///Get a string representation of a float[] that can be pasted into MATLAB
let MatlabVector (v:float[]) =
    let sb = System.Text.StringBuilder()
    sb.Append("[") |> ignore
    for i = 0 to v.Length - 1 do
        sb.Append(sprintf "%.2f" v.[i]) |> ignore
        if i < v.Length - 1 then sb.Append(" ") |> ignore
    sb.Append("]") |> ignore
    sb.ToString()

/// Get a string representation of a float[,] than can be pasted into a Mathematica notebook
let MathematicaMatrix (m:float[,]) =
    let rows = m.GetLength 0
    let cols = m.GetLength 1
    let sb = System.Text.StringBuilder()
    sb.Append("{") |> ignore
    for i = 0 to rows - 1 do
        sb.Append(MathematicaVector(Array.init cols (fun j -> m.[i, j]))) |> ignore
        if i < rows - 1 then sb.Append(", ") |> ignore
    sb.Append("}") |> ignore
    sb.ToString()

/// Get a string representation of a float[,] that can be pasted into MATLAB
let MatlabMatrix (m:float[,]) =
    let rows = m.GetLength 0
    let cols = m.GetLength 1
    let sb = System.Text.StringBuilder()
    sb.Append("[") |> ignore
    for i = 0 to rows - 1 do
        sb.Append(MatlabVector(Array.init cols (fun j -> m.[i, j]))) |> ignore
        if i < rows - 1 then sb.Append("; ") |> ignore
    sb.Append("]") |> ignore
    sb.ToString()

/// Compute a combined hash code for the objects in array `o`
let inline hash (o:obj[]) =
    Array.map (fun a -> a.GetHashCode()) o
    |> Seq.fold (fun acc elem -> acc * 23 + elem) 17

/// Global step size for numerical approximations
let eps = 0.00001

/// Two times eps
let deps = eps * 2.

/// Square of eps
let epssq = eps * eps

