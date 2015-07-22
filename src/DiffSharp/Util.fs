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
let log10val = log 10.

/// Gets an array of size `n`, where the `i`-th element is 1 and the rest of the elements are 0
let inline standardBasis (n:int) (i:int) = Array.init n (fun j -> if i = j then LanguagePrimitives.GenericOne else LanguagePrimitives.GenericZero)

module ErrorMessages =
    let invalidArgDiffn() = invalidArg "" "Order of differentiation cannot be negative."
    let invalidArgSolve() = invalidArg "" "Given system of linear equations has no solution."

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

module Array2D =
    let empty<'T> = Array2D.zeroCreate<'T> 0 0
    let isEmpty (array : 'T[,]) = (array.Length = 0)
    let toArray (array : 'T [,]) = array |> Seq.cast<'T> |> Seq.toArray
