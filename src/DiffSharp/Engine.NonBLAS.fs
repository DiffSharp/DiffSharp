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

namespace DiffSharp.Engine

open System.Threading.Tasks
open DiffSharp.Util


// TODO: reimplement some of these, calling another library, GPU, or custom unsafe C# code
module NonBLAS =

    let v_mul_hadamard(a:float[], b:float[]) =
        if Array.isEmpty a || Array.isEmpty b then
            Array.empty
        else
            Array.init a.Length (fun i -> a.[i] * b.[i])
    
    let v_div_hadamard(a:float[], b:float[]) =
        if Array.isEmpty a then
            Array.empty
        elif Array.isEmpty b then
            invalidArg "" "Attempted Hadamard division by a zero vector."
        else
            Array.init a.Length (fun i -> a.[i] / b.[i])

    let v_sum(a:float[]) =
        if Array.isEmpty a then
            0.
        else
            Array.sum a

    let v_abs(a:float[]) =
        if Array.isEmpty a then
            Array.empty
        else
            Array.Parallel.map abs a

    let v_sign(a:float[]) =
        if Array.isEmpty a then
            Array.empty
        else
            Array.Parallel.map (sign>>float) a

    let vs_add(alpha:float, a:float[]) =
        Array.Parallel.map (fun v -> v + alpha) a

    let m_add(a:float[,], b:float[,]) =
        if Array2D.isEmpty a then
            Array2D.copy b
        elif Array2D.isEmpty b then
            Array2D.copy a
        else
            Array2D.init (Array2D.length1 a) (Array2D.length2 a) (fun i j -> a.[i, j] + b.[i, j])

    let m_sub(a:float[,], b:float[,]) =
        if Array2D.isEmpty a then
            OpenBLAS.m_scale(-1., b)
        elif Array2D.isEmpty b then
            Array2D.copy a
        else
            Array2D.init (Array2D.length1 a) (Array2D.length2 a) (fun i j -> a.[i, j] - b.[i, j])

    let m_mul_hadamard(a:float[,], b:float[,]) =
        if Array2D.isEmpty a || Array2D.isEmpty b then
            Array2D.empty
        else
            Array2D.init (Array2D.length1 a) (Array2D.length2 a) (fun i j -> a.[i, j] * b.[i, j])

    let m_div_hadamard(a:float[,], b:float[,]) =
        if Array2D.isEmpty a then
            Array2D.empty
        elif Array2D.isEmpty b then
            invalidArg "" "Attempted Hadamard division by a zero matrix."
        else
            Array2D.init (Array2D.length1 a) (Array2D.length2 a) (fun i j -> a.[i, j] / b.[i, j])

    let m_sum(a:float[,]) =
        if Array2D.isEmpty a then
            0.
        else
            [|for i = 0 to (Array2D.length1 a) - 1 do yield [|for j = 0 to (Array2D.length2 a) - 1 do yield a.[i, j]|]|]
            |> Array.Parallel.map Array.sum |> Array.sum

    let m_transpose(a:float[,]) =
        if Array2D.isEmpty a then
            Array2D.empty
        else
            Array2D.init (Array2D.length2 a) (Array2D.length1 a) (fun i j -> a.[j, i])
