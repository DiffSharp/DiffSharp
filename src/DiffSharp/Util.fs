//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
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


/// Various utility functions
module DiffSharp.Util

open System.Threading.Tasks


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

/// Value of log 10.
let log10ValFloat64 = log 10.
let log10ValFloat32 = log 10.f

/// Computes a combined hash code for the objects in array `o`
let inline hash (o:obj[]) =
    Array.map (fun a -> a.GetHashCode()) o
    |> Seq.fold (fun acc elem -> acc * 23 + elem) 17

/// Gets an array of size `n`, where the `i`-th element is 1 and the rest of the elements are zero
let inline standardBasis (n:int) (i:int) = 
    let s = Array.zeroCreate n
    s.[i] <- LanguagePrimitives.GenericOne
    s

/// Gets an array of size `n`, where the `i`-th element has value `v` and the rest of the elements are zero
let inline standardBasisVal (n:int) (i:int) v = 
    let s = Array.zeroCreate n
    s.[i] <- v
    s

/// Copies the upper triangular elements of the square matrix given in the 2d array `m` to the lower triangular part
let inline copyUpperToLower (m:_[,]) =
    if (Array2D.length1 m) <> (Array2D.length2 m) then invalidArg "" "Expecting a square matrix."
    let r = Array2D.copy m
    let rows = r.GetLength 0
    if rows > 1 then
        Parallel.For(1, rows, fun i ->
            Parallel.For(0, i, fun j ->
                r.[i, j] <- r.[j, i]) |> ignore) |> ignore
    r
            
let inline signummod x =
    if x < LanguagePrimitives.GenericZero then -LanguagePrimitives.GenericOne
    elif x > LanguagePrimitives.GenericZero then LanguagePrimitives.GenericOne
    else LanguagePrimitives.GenericZero

let inline signum (x:'a) = (^a : (static member Sign : ^a -> ^a) x)

let inline logsumexp (x:^a) = (^a : (static member LogSumExp : ^a -> ^b) x)
let inline softplus (x:^a) = (^a : (static member SoftPlus : ^a -> ^a) x)
let inline softsign (x:^a) = (^a : (static member SoftSign : ^a -> ^a) x)
let inline sigmoid (x:^a) = (^a : (static member Sigmoid : ^a -> ^a) x)
let inline reLU (x:^a) = (^a : (static member ReLU : ^a -> ^a) x)
let inline softmax (x:^a) = (^a : (static member SoftMax : ^a -> ^a) x)
let inline maximum (x: ^a) (y:^b) : ^c = ((^a or ^b) : (static member Max : ^a * ^b -> ^c) x, y)
let inline minimum (x: ^a) (y:^b) : ^c = ((^a or ^b) : (static member Min : ^a * ^b -> ^c) x, y)

//type System.Single with
//    static member LogSumExp(x:float32) = x
//    static member SoftPlus(x) = log (1.f + exp x)
//    static member SoftSign(x) = x / (1.f + abs x)
//    static member Sigmoid(x) = 1.f / (1.f + exp -x)
//    static member ReLU(x) = max 0.f x
//
//type System.Double with
//    static member LogSumExp(x:float) = x
//    static member SoftPlus(x) = log (1. + exp x)
//    static member SoftSign(x) = x / (1. + abs x)
//    static member Sigmoid(x) = 1. / (1. + exp -x)
//    static member ReLU(x) = max 0. x

module ErrorMessages =
    let InvalidArgDiffn() = invalidArg "" "Order of differentiation cannot be negative."
    let InvalidArgSolve() = invalidArg "" "Given system of linear equations has no solution."
    let InvalidArgCurl() = invalidArg "" "Curl is supported only for functions with a three-by-three Jacobian matrix."
    let InvalidArgDiv() = invalidArg "" "Div is defined only for functions with a square Jacobian matrix."
    let InvalidArgCurlDiv() = invalidArg "" "Curldiv is supported only for functions with a three-by-three Jacobian matrix."
    let InvalidArgInverse() = invalidArg "" "Cannot compute the inverse of the given matrix."
    let InvalidArgDet() = invalidArg "" "Cannot compute the determinant of the given matrix."

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

/// Extensions for the FSharp.Collections.Array module
module Array =
    module Parallel =
        let map2 f (a1:_[]) (a2:_[]) =
            let n = min a1.Length a2.Length
            Array.Parallel.init n (fun i -> f a1.[i] a2.[i])

/// Extensions for the FSharp.Collections.Array2D module
module Array2D =
    let copyFast (array : 'T[,]) =  array.Clone() :?> 'T[,]
    let empty<'T> = Array2D.zeroCreate<'T> 0 0
    let isEmpty (array : 'T[,]) = (array.Length = 0)
    let toArray (array : 'T[,]) = array |> Seq.cast<'T> |> Seq.toArray
    let find (predicate : 'T -> bool) (array : 'T[,]) = array |> toArray |> Array.find predicate
    let tryFind (predicate : 'T -> bool) (array : 'T[,]) = array |> toArray |> Array.tryFind predicate
    let map2 f (a1:_[,]) (a2:_[,]) = 
        let m = min (Array2D.length1 a1) (Array2D.length1 a2)
        let n = min (Array2D.length2 a1) (Array2D.length2 a2)
        Array2D.init m n (fun i j -> f a1.[i, j] a2.[i, j])

    module Parallel =
        let init m n f =
            let a = Array2D.zeroCreate m n
            // Nested parallel fors caused problems with mutable variables
            //Parallel.For(0, m, fun i ->
            //    Parallel.For(0, n, fun j -> a.[i, j] <- f i j) |> ignore) |> ignore
            for i = 0 to m - 1 do
                Parallel.For(0, n, fun j -> a.[i, j] <- f i j) |> ignore
            a
        let map f (a:_[,]) =
            init (Array2D.length1 a) (Array2D.length2 a) (fun i j -> f a.[i, j])
        let map2 f (a1:_[,]) (a2:_[,]) =
            let m = min (Array2D.length1 a1) (Array2D.length1 a2)
            let n = min (Array2D.length2 a1) (Array2D.length2 a2)
            init m n (fun i j -> f a1.[i, j] a2.[i, j])