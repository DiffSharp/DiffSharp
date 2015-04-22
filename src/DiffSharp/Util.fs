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

#light

/// Various utility functions used all over the library
module DiffSharp.Util

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

/// Checks whether the 2d array `m` has the same number of elements in both dimensions
let (|Square|) (m:_[,]) =
    match m with
    | m when m.GetLength 0 = m.GetLength 1 -> m
    | _ -> invalidArg "m" "Expecting a square 2d array"

/// Gets the transpose of the 2d array `m`
let inline transpose (m:_[,]) = Array2D.init (m.GetLength 1) (m.GetLength 0) (fun i j -> m.[j, i])

/// Gets an array containing the diagonal elements of the square 2d array `m`
let inline diagonal (Square m:_[,]) = Array.init (m.GetLength 0) (fun i -> m.[i, i])

/// Gets the trace of the square matrix given in the 2d array `m`
let inline trace (m:_[,]) = Array.sum (diagonal m)

/// Gets an array of size `n`, where the `i`-th element is 1 and the rest of the elements are 0
let inline standardBasis (n:int) (i:int) = Array.init n (fun j -> if i = j then LanguagePrimitives.GenericOne else LanguagePrimitives.GenericZero)

/// Copies the upper triangular elements of the square matrix given in the 2d array `m` to the lower triangular part
let inline copyUpperToLower (Square m:_[,]) =
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
let mutable StepSize = 0.00001

/// Vector-to-scalar to scalar-to-scalar function transform. Given a vector-to-scalar function `f` and an evaluation point `x`, returns a scalar-to-scalar version of `f`, where the `i`-th variable is free and the rest of the variables have the constant values given in `x`.
let inline fVStoSS i f x =
    let xc = Array.copy x
    fun xx ->
        xc.[i] <- xx
        f xc

/// Vector-to-vector to scalar-to-vector function transform. Given a vector-to-vector function `f` and an evaluation point `x`, returns a scalar-to-vector version of `f`, where the `i`-th variable is free and the rest of the variables have the constant values given in `x`.
let inline fVVtoSV i (f:_[]->_[]) x =
    let xc = Array.copy x
    fun xx ->
        xc.[i] <- xx
        f xc

/// Vector-to-vector to vector-to-scalar function transform. Given a vector-to-vector function `f`, returns a vector-to-scalar version of `f` supplying only the `i`-th output.
let inline fVVtoVS i (f:_[]->_[]) =
    fun xx -> (f xx).[i]

/// Vector-to-vector to scalar-to-scalar function transform. Given a vector-to-vector function `f`, returns a scalar-to-scalar version of `f`, where the `i`-th variable is free and the rest of the variables have the constant values given in `x`, supplying only the `j`-th output.
let inline fVVtoSS i j (f:'a[]->'b[]) (x:'a[]) =
    let xc = Array.copy x
    fun xx ->
        xc.[i] <- xx
        (f xc).[j]

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
let invalidArgCurl() = invalidArg "" "Curl is supported only for functions with a three-by-three Jacobian matrix."
let invalidArgDiv() = invalidArg "" "Div is defined only for functions with a square Jacobian matrix."
let invalidArgCurlDiv() = invalidArg "" "Curldiv is supported only for functions with a three-by-three Jacobian matrix."
let invalidArgDiffn() = invalidArg "" "Order of differentiation cannot be negative."


/// Tagger for generating incremental integers
type Tagger =
    val mutable LastTag : uint32
    new(t) = {LastTag = t}
    member t.Next() = t.LastTag <- t.LastTag + 1u; t.LastTag

/// Global tagger for nested D operations
type GlobalTagger() =
    static let T = new Tagger(0u)
    static member Next = T.Next()
    static member Reset = T.LastTag <- 0u