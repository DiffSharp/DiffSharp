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

module DiffSharp.Extensions

open System.Threading.Tasks

/// Extensions for the FSharp.Collections.Array module
module Array =
    module Parallel =
        let map2 f (a1:_[]) (a2:_[]) =
            let n = min a1.Length a2.Length
            Array.Parallel.init n (fun i -> f a1.[i] a2.[i])

/// Extensions for the FSharp.Collections.Array2D module
module Array2D =
    let empty<'T> = Array2D.zeroCreate<'T> 0 0
    let isEmpty (array : 'T[,]) = (array.Length = 0)
    let toArray (array : 'T [,]) = array |> Seq.cast<'T> |> Seq.toArray

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
