//
// This file is part of
// DiffSharp: Automatic Differentiation Library
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

#light

/// Various utility functions used all over the library
module DiffSharp.Util.General

/// Gets the first term of a 3-tuple
let inline fst3 (f, _, _) = f

/// Gets the second term of a 3-tuple
let inline snd3 (_, s, _) = s

/// Gets the tail of a 3-tuple
let inline trd (_, _, t) = t

/// Gets the first and third terms of a 3-tuple
let inline fsttrd (f, _, t) = (f, t)

/// Gets the second and third terms of a 3-tuple
let inline sndtrd (_, s, t) = (s, t)

/// Checks whether 'a[,] `m` has the same number of elements in both dimensions
let (|Square|) (m:'a[,]) =
    match m with
    | m when m.GetLength 0 = m.GetLength 1 -> m
    | _ -> invalidArg "m" "Expecting a square float[,]"

/// Gets the transpose of 'a[,] `m`
let inline transpose (m:'a[,]) = Array2D.init (m.GetLength 1) (m.GetLength 0) (fun i j -> m.[j, i])

/// Gets a 'a[] containing the diagonal elements of 'a[,] `m`
let inline diagonal (Square m:'a[,]) = Array.init (m.GetLength 0) (fun i -> m.[i, i])

/// Gets the trace of the square matrix given in 'a[,] `m`
let inline trace (m:'a[,]) = Array.sum (diagonal m)

/// Gets an array of size `n`, where the `i`-th element is 1 and the rest of the elements are 0
let inline standardBasis (n:int) (i:int) = Array.init n (fun j -> if i = j then 1. else 0.)

/// Copies the upper triangular elements of the square matrix given in 'a[,] `m` to the lower triangular part
let inline copyUpperToLower (Square m:'a[,]) =
    let r = Array2D.copy m
    let rows = r.GetLength 0
    if rows > 1 then
        for i = 1 to rows - 1 do
            for j = 0 to i - 1 do
                r.[i, j] <- r.[j, i]
    r

/// Finds an array that, when multiplied by an LU matrix `lu`, gives array `b`
let inline matrixSolveHelper (lu:'a[,]) (b:'a[]) =
    let n = lu.GetLength 0
    let x = Array.copy b
    for i = 1 to n - 1 do
        let mutable sum = x.[i]
        for j = 0 to i - 1 do
            sum <- sum - lu.[i, j] * x.[j]
        x.[i] <- sum
    x.[n - 1] <- x.[n - 1] / lu.[n - 1, n - 1]
    for i in (n - 2) .. -1 .. 0 do
        let mutable sum = x.[i]
        for j = i + 1 to n - 1 do
            sum <- sum - lu.[i, j] * x.[j]
        x.[i] <- sum / lu.[i, i]
    x

/// Gets a string representation of float[] `v` that can be pasted into a Mathematica notebook
let MathematicaVector (v:float[]) =
    let sb = System.Text.StringBuilder()
    sb.Append("{") |> ignore
    for i = 0 to v.Length - 1 do
        sb.Append(sprintf "%.2f" v.[i]) |> ignore
        if i < v.Length - 1 then sb.Append(", ") |> ignore
    sb.Append("}") |> ignore
    sb.ToString()

/// Gets a string representation of float[] `v` that can be pasted into MATLAB
let MatlabVector (v:float[]) =
    let sb = System.Text.StringBuilder()
    sb.Append("[") |> ignore
    for i = 0 to v.Length - 1 do
        sb.Append(sprintf "%.2f" v.[i]) |> ignore
        if i < v.Length - 1 then sb.Append(" ") |> ignore
    sb.Append("]") |> ignore
    sb.ToString()

/// Gets a string representation of float[,] `m` than can be pasted into a Mathematica notebook
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

/// Gets a string representation of float[,] `m` that can be pasted into MATLAB
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

/// Computes a combined hash code for the objects in array `o`
let inline hash (o:obj[]) =
    Array.map (fun a -> a.GetHashCode()) o
    |> Seq.fold (fun acc elem -> acc * 23 + elem) 17

/// Checks whether a float contains an integer value
let isInteger a = a = float (int a)

/// Checks whether a float is halfway between two integers
let isHalfway a = abs (a % 1.) = 0.5

/// Value of log 10.
let log10val = log 10.

/// Global step size for numerical approximations
let eps = 0.00001

/// Two times eps
let deps = eps * 2.

/// Square of eps
let epssq = eps * eps

let invalidArgLog() = invalidArg "" "The derivative of log(x) is not defined for x <= 0."
let invalidArgLog10() = invalidArg "" "The derivative of log10(x) is not defined for x <= 0."
let invalidArgTan() = invalidArg "" "The derivative of tan(x) is not defined for x such that cos(x) = 0."
let invalidArgSqrt() = invalidArg "" "The derivative of sqrt(x) is not defined for x <= 0."
let invalidArgAsin() = invalidArg "" "The derivative of asin(x) is not defined for x such that abs(x) >= 1."
let invalidArgAcos() = invalidArg "" "The derivative of acos(x) is not defined for x such that abs(x) >= 1."
let invalidArgAbs() = invalidArg "" "The derivative of abs(x) is not defined for x = 0."
let invalidArgFloor() = invalidArg "" "The derivative of floor(x) is not defined for integer values of x."
let invalidArgCeil() = invalidArg "" "The derivative of ceil(x) is not defined for integer values of x."
let invalidArgRound() = invalidArg "" "The derivative of round(x) is not defined for values of x halfway between integers."
