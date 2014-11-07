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
type Vector(v:float[]) =
    member this.V = v
    member v.Length = v.V.Length
    member v.Item i = if Vector.IsZero(v) then 0. else v.V.[i]
    member v.FirstItem = if Vector.IsZero(v) then 0. else v.V.[0]
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
    /// Get a string representation of this Vector that can be pasted into a Mathematica notebook
    member v.ToMathematicaString() = 
        let sb = System.Text.StringBuilder()
        sb.Append("{") |> ignore
        for i = 0 to v.Length - 1 do
            sb.Append(sprintf "%.2f" v.[i]) |> ignore
            if i <> v.Length - 1 then sb.Append(", ") |> ignore
        sb.Append("}") |> ignore
        sb.ToString()
    /// Get a string representation of this Vector that can be pasted into MATLAB
    member v.ToMatlabString() = 
        let sb = System.Text.StringBuilder()
        sb.Append("[") |> ignore
        for i = 0 to v.Length - 1 do
            sb.Append(sprintf "%.2f" v.[i]) |> ignore
            if i < v.Length - 1 then sb.Append(" ") |> ignore
        sb.Append("]") |> ignore
        sb.ToString()
    /// Get the Euclidean norm of this vector
    member v.GetNorm() = if Vector.IsZero(v) then 0. else sqrt (Array.sumBy (fun x -> x * x) v.V)
    /// Get the unit vector codirectional with this vector
    member v.GetUnitVector() = if Vector.IsZero(v) then v else let n = v.GetNorm() in Vector.Create(Array.init v.Length (fun i -> v.[i] / n))
    /// Builds a new Vector whose elements are the results of applying function `f` to each of the elements of Vector `v`
    static member map f (v:Vector) = Vector.Create(Array.map f v.V)
    /// Returns the sum of all the elements in vector `v`
    static member sum (v:Vector) = Array.sum v.V
    /// Create Vector from array `v`
    static member Create(v) = Vector(v)
    /// Create Vector with dimension `n` and a generator function `f` to compute the elements
    static member Create(n, f) = Vector(Array.init n f)
    /// Create Vector with dimension `n` and all elements having value `v`
    static member Create(n, v) = Vector(Array.create n v)
    /// Create Vector with dimension `n`, the element with index `i` having value `v`, and the rest of the elements 0
    static member Create(n, i, v) = Vector.Create(n, (fun j -> if j = i then v else 0.))
    /// Vector with infinite dimension and all elements 0
    static member Zero = ZeroVector()
    /// Check whether Vector `a` is an instance of ZeroVector
    static member IsZero(a:Vector) =
        match a with
        | :? ZeroVector -> true
        | _ -> false
    /// Add Vector `a` to Vector `b`
    static member (+) (a:Vector, b:Vector) = 
        match Vector.IsZero(a), ZeroVector.IsZero(b) with
        | true, true -> a
        | true, false -> b
        | false, true -> a
        | false, false -> Vector(Array.init a.Length (fun i -> a.[i] + b.[i]))
    /// Subtract Vector `b` from Vector `a`
    static member (-) (a:Vector, b:Vector) =
        match Vector.IsZero(a), ZeroVector.IsZero(b) with
        | true, true -> a
        | true, false -> -b
        | false, true -> a
        | false, false -> Vector(Array.init a.Length (fun i -> a.[i] - b.[i]))
    /// Multiply Vector `a` and Vector `b` element-wise (Hadamard product)
    static member (*) (a:Vector, b:Vector) =
        match Vector.IsZero(a), ZeroVector.IsZero(b) with
        | true, true -> a
        | true, false -> a
        | false, true -> b
        | false, false -> Vector(Array.init a.Length (fun i -> a.[i] * b.[i]))
    /// Divide Vector `a` by Vector `b` element-wise
    static member (/) (a:Vector, b:Vector) =
        match Vector.IsZero(a), ZeroVector.IsZero(b) with
        | true, true -> raise (new System.DivideByZeroException("Attempted to divide a ZeroVector by a ZeroVector."))
        | true, false -> a
        | false, true -> raise (new System.DivideByZeroException("Attempted to divide a Vector by a ZeroVector."))
        | false, false -> Vector(Array.init a.Length (fun i -> a.[i] / b.[i]))
    /// Multiply Vector `a` by float `b`
    static member (*) (a:Vector, b:float) =
        match Vector.IsZero(a) with
        | true -> a
        | false -> Vector(Array.init a.Length (fun i -> a.[i] * b))
    /// Multiply Vector `b` by float `a`
    static member (*) (a:float, b:Vector) =
        match Vector.IsZero(b) with
        | true -> b
        | false -> Vector(Array.init b.Length (fun i -> a * b.[i]))
    /// Divide Vector `a` by float `b`
    static member (/) (a:Vector, b:float) =
        match Vector.IsZero(a) with
        | true -> a
        | false -> Vector(Array.init a.Length (fun i -> a.[i] / b))
    /// Create Vector whose elements are float `a` divided by the elements of Vector `b`
    static member (/) (a:float, b:Vector) =
        match Vector.IsZero(b) with
        | true -> raise (new System.DivideByZeroException("Attempted division by a ZeroVector."))
        | false -> Vector(Array.init b.Length (fun i -> a / b.[i]))
    /// Negative of Vector `a`
    static member (~-) (a:Vector) =
        match Vector.IsZero(a) with
        | true -> a
        | false -> Vector(Array.init a.Length (fun i -> -a.[i]))

/// A vector with infinite length and all elements 0
and ZeroVector() =
    inherit Vector([||])
    override v.ToString() = "ZeroVector"



/// Lightweight matrix type
type Matrix(m:float[,]) =
    member this.M = m
    member m.Rows = m.M.GetLength 0
    member m.Cols = m.M.GetLength 1
    member m.Item(i, j) = if Matrix.IsZero(m) then 0. else m.M.[i, j]
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
    /// Get a string representation of this Matrix that can be pasted into MATLAB
    member m.ToMatlabString() = 
        let sb = System.Text.StringBuilder()
        sb.Append("[") |> ignore
        for i = 0 to m.Rows - 1 do
            for j = 0 to m.Cols - 1 do
                sb.Append(sprintf "%.2f" m.M.[i, j]) |> ignore
                if j < m.Cols - 1 then sb.Append(" ") |> ignore
            if i < m.Rows - 1 then sb.Append("; ") |> ignore
        sb.Append("]") |> ignore
        sb.ToString()
    /// Get the trace of this Matrix
    member m.GetTrace() = if Matrix.IsZero(m) then 0. else trace m.M
    /// Get the transpose of this Matrix
    member m.GetTranspose() = if Matrix.IsZero(m) then m else Matrix(transpose m.M)
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
    /// Create symmetric Matrix with `m` rows and columns and a generator function `f` to compute the elements
    static member SymmetricCreate(m, f) =
        let s = Array2D.zeroCreate<float> m m
        for i = 0 to m - 1 do
            for j = i to m - 1 do
                s.[i, j] <- f i j
        Matrix(copyupper s)
    static member Zero = ZeroMatrix()
    /// Check whether Matrix `a` is an instance of ZeroMatrix
    static member IsZero(a:Matrix) =
        match a with
        | :? ZeroMatrix -> true
        | _ -> false
    /// Symmetric binary operation `f` on Matrix `a` and Matrix `b`
    static member SymmetricOp (a:Matrix, b:Matrix, f) =
        match Matrix.IsZero(a), Matrix.IsZero(b) with
        | true, true -> a
        | true, false -> Matrix.SymmetricCreate(b.Rows, f)
        | false, true -> Matrix.SymmetricCreate(a.Rows, f)
        | false, false -> Matrix.SymmetricCreate(a.Rows, f)
    /// Symmetric unary operation `f` on Matrix `a`
    static member SymmetricOp (a:Matrix, f) =
        match Matrix.IsZero(a) with
        | true -> a
        | false -> Matrix.SymmetricCreate(a.Rows, f)
    /// Add Matrix `a` to Matrix `b`
    static member (+) (a:Matrix, b:Matrix) =
        match Matrix.IsZero(a), Matrix.IsZero(b) with
        | true, true -> a
        | true, false -> b
        | false, true -> a
        | false, false -> Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] + b.[i, j]))
    /// Subtract Matrix `b` from Matrix `a`
    static member (-) (a:Matrix, b:Matrix) =
        match Matrix.IsZero(a), Matrix.IsZero(b) with
        | true, true -> a
        | true, false -> -b
        | false, true -> a
        | false, false -> Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] - b.[i, j]))  
    /// Matrix product of Matrix `a` and Matrix `b`
    static member (*) (a:Matrix, b:Matrix) =
        match Matrix.IsZero(a), Matrix.IsZero(b) with
        | true, true -> a
        | true, false -> a
        | false, true -> b
        | false, false -> Matrix(Array2D.init a.Rows a.Cols (fun i j -> Array.sumBy (fun k -> a.[i, k] * b.[k, j]) [|0..(b.Rows - 1)|] ))
    /// Multiply Matrix `a` by float `b`
    static member (*) (a:Matrix, b:float) =
        match Matrix.IsZero(a) with
        | true -> a
        | false -> Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] * b))
    /// Multiply Matrix `b` by float `a`
    static member (*) (a:float, b:Matrix) =
        match Matrix.IsZero(b) with
        | true -> b
        | false -> Matrix(Array2D.init b.Rows b.Cols (fun i j -> a * b.[i, j]))
    /// Divide Matrix `a` by float `b`
    static member (/) (a:Matrix, b:float) =
        match Matrix.IsZero(a) with
        | true -> a
        | false -> Matrix(Array2D.init a.Rows a.Cols (fun i j -> a.[i, j] / b))
    /// Create Matrix whose elements are float `a` divided by the element of Matrix `b`
    static member (/) (a:float, b:Matrix) =
        match Matrix.IsZero(b) with
        | true ->  raise (new System.DivideByZeroException("Attempted division by a zero matrix."))
        | false -> Matrix(Array2D.init b.Rows b.Cols (fun i j -> a / b.[i, j]))
    /// Negative of Matrix `a`
    static member (~-) (a:Matrix) =
        match Matrix.IsZero(a) with
        | true -> a
        | false -> Matrix(Array2D.init a.Rows a.Cols (fun i j -> -a.[i, j]))

/// A matrix with infinite dimensions and all elements 0
and ZeroMatrix() =
    inherit Matrix(Array2D.zeroCreate 0 0)
    override m.ToString() = "ZeroMatrix"


/// Convert float[] `v` into Vector
let vector v = Vector.Create(v)
/// Get the Euclidean norm of Vector `v`
let norm (v:Vector) = v.GetNorm()
/// Get the unit vector codirectional with Vector `v`
let unitVector (v:Vector) = v.GetUnitVector()
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