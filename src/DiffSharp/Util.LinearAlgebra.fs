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

/// Lightweight Vector and Matrix types
module DiffSharp.Util.LinearAlgebra

open DiffSharp.Util.General

/// Lightweight vector type
type Vector =
    val V : float[]
    new(v) = {V = v}
    member v.Length = v.V.Length
    member v.Item i = v.V.[i]
    interface System.IComparable with
        override v.CompareTo(other) =
            match other with
            | :? Vector as v2 -> compare (v.GetNorm()) (v2.GetNorm())
            | _ -> failwith "Cannot compare this Vector with another type of object."
    override v.Equals(other) = 
        match other with
        | :? Vector as v2 -> v.GetNorm() = v2.GetNorm()
        | _ -> false
    override v.GetHashCode() = v.V.GetHashCode()
    override v.ToString() = sprintf "Vector %A" v.V
    /// Get the Euclidean norm of the Vector
    member v.GetNorm() = sqrt (Array.sumBy (fun x -> x * x) v.V)
    /// Builds a new Vector whose elements are the results of applying function `f` to each of the elements of Vector `v`
    static member map f (v:Vector) = Vector.Create(Array.map f v.V)
    /// Create Vector from array `v`
    static member Create(v) = Vector(v)
    /// Create Vector with dimension `n` and a generator function `f` to compute the elements 
    static member Create(n, f) = Vector(Array.init n f)
    /// Create Vector with dimension `n` and all elements having value `v`
    static member Create(n, v) = Vector(Array.create n v)
    /// Create Vector with dimension `n`, the element with index `i` having value `v`, and the rest of the elements 0
    static member Create(n, i, v) = Vector.Create(n, (fun j -> if j = i then v else 0.))
    /// 0-dimension Vector
    static member Zero = Vector([||])
    /// Add Vector `a` to Vector `b`
    static member (+) (a:Vector, b:Vector) = Vector(Array.init a.Length (fun i -> a.[i] + b.[i]))
    /// Subtract Vector `b` from Vector `a`
    static member (-) (a:Vector, b:Vector) = Vector(Array.init a.Length (fun i -> a.[i] - b.[i]))
    /// Multiply Vector `a` and Vector `b` element-wise (Hadamard product)
    static member (*) (a:Vector, b:Vector) = Vector(Array.init a.Length (fun i -> a.[i] * b.[i]))
    /// Multiply Vector `a` by float `b`
    static member (*) (a:Vector, b:float) = Vector(Array.init a.Length (fun i -> a.[i] * b))
    /// Multiply Vector `b` by float `a`
    static member (*) (a:float, b:Vector) = Vector(Array.init b.Length (fun i -> a * b.[i]))
    /// Divide Vector `a` by Vector `b` element-wise
    static member (/) (a:Vector, b:Vector) = Vector(Array.init a.Length (fun i -> a.[i] / b.[i]))
    /// Divide Vector `a` by float `b`
    static member (/) (a:Vector, b:float) = Vector(Array.init a.Length (fun i -> a.[i] / b))
    /// Create Vector whose elements are float `a` divided by the corresponding element of Vector `b`
    static member (/) (a:float, b:Vector) = Vector(Array.init b.Length (fun i -> a / b.[i]))
    /// Negative of Vector `a`
    static member (~-) (a:Vector) = Vector(Array.init a.Length (fun i -> -a.[i]))


/// Lightweight matrix type
type Matrix =
    val M : float[,]
    new(m) = {M = m}
    member m.Rows = m.M.GetLength 0
    member m.Cols = m.M.GetLength 1
    member m.Item(i, j) = m.M.[i, j]
    override m.ToString() = sprintf "Matrix %A" m.M
    /// Get a string representation of this Matrix that can be pasted into a Mathematica notebook
    member m.ToMathematicaString() = 
        let sb = System.Text.StringBuilder()
        sb.Append("{") |> ignore
        for i = 0 to m.Rows - 1 do
            sb.Append("{") |> ignore
            for j = 0 to m.Cols - 1 do
                sb.Append(sprintf "%.2f" m.M.[i, j]) |> ignore
                if j <> m.Cols - 1 then sb.Append(", ") |> ignore
            sb.Append("}") |> ignore
            if i <> m.Rows - 1 then sb.Append(", ") |> ignore
        sb.Append("}") |> ignore
        sb.ToString()
    /// Get the trace of this Matrix
    member m.GetTrace() = trace m.M
    /// Get the transpose of this Matrix
    member m.GetTranspose() = Matrix(transpose m.M)
    /// Builds a new Matrix whose elements are the results of applying function `f` to each of the elements of Matrix `m`
    static member map f (m:Matrix) = Matrix(Array2D.map f m.M)
    /// Create Matrix from given 2d array `m`
    static member Create(m) = Matrix(m)
    /// Create Matrix with `m` rows, `n` columns, and a generator function `f` to compute the elements
    static member Create(m, n, f) = Matrix(Array2D.init m n f)
    /// Create Matrix with `m` rows, `n` columns, and all elements having value `v`
    static member Create(m, n, v) = Matrix(Array2D.create m n v)
    /// Create Matrix with `m` rows and a generator function `f` that gives each row as a float[]
    static member Create(m, (f:int->float[])) = Matrix(array2D (Array.init m f))
    /// Create Matrix with `m` rows and a generator function `f` that gives each row as a Vector
    static member Create(m, (f:int->Vector)) =
        let a = Array.init m f
        Matrix(Array2D.init m (a.[0].Length) (fun i j -> a.[i].V.[j]))
    /// Create Matrix with rows given in Vector[] `v`
    static member Create(v:Vector[]) = Matrix.Create(v.Length, fun i -> v.[i])
    /// Create Matrix with `m` rows and all rows equal to float[] `v`
    static member Create(m, (v:float[])) = Matrix.Create(m, fun i -> v)
    /// Create zero Matrix
    static member Zero = Matrix.Create(0, 0, 0.)
    /// Add Matrix `a` to Matrix `b`
    static member (+) (a:Matrix, b:Matrix) = Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] + b.[i, j]))
    /// Subtract Matrix `b` from Matrix `a`
    static member (-) (a:Matrix, b:Matrix) = Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] - b.[i, j]))
    /// Matrix product of Matrix `a` and Matrix `b`
    static member (*) (a:Matrix, b:Matrix) = Matrix(Array2D.init a.Rows a.Cols (fun i j -> sum 0 (b.Rows - 1) (fun k -> a.[i, k] * b.[k, j])))
    /// Multiply Matrix `a` by float `b`
    static member (*) (a:Matrix, b:float) = Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] * b))
    /// Multiply Matrix `b` by float `a`
    static member (*) (a:float, b:Matrix) = Matrix(Array2D.init b.Rows b.Cols (fun i j -> a * b.[i, j]))
    /// Divide Matrix `a` by float `b`
    static member (/) (a:Matrix, b:float) = Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] / b))
    /// Negative of Matrix `a`
    static member (~-) (a:Matrix) = Matrix(Array2D.init a.Rows a.Cols (fun i j -> -a.[i, j]))


/// Convert float[] `v` into Vector
let vector v = Vector.Create(v)
/// Get the Euclidean norm of Vector `v`
let norm (v:Vector) = v.GetNorm()
/// Convert Vector `v` into float[]
let array (v:Vector) = v.V
/// Convert float[,] `m` into Matrix
let matrix (m:float[,]) = Matrix.Create(m)
/// Convert Matrix m into float[,]
let array2d (m:Matrix) = m.M
/// Get the trace of Matrix `m`
let trace (m:Matrix) = m.GetTrace()
/// Get the transpose of Matrix `m`
let transpose (m:Matrix) = m.GetTranspose()