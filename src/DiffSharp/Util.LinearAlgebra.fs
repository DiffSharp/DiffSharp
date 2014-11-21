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
[<NoEquality; NoComparison>]
type Vector =
    | ZeroVector
    | Vector of float[]
    with
    member v.Item i =
        match v with
        | ZeroVector -> 0.
        | Vector v -> v.[i]
    member v.FirstItem =
        match v with
        | ZeroVector -> 0.
        | Vector v -> v.[0]
    member v.Length =
        match v with
        | ZeroVector -> 0
        | Vector v -> v.Length
    /// Get the Euclidean norm of this vector
    member v.GetNorm() =
        match v with
        | ZeroVector -> 0.
        | Vector v -> sqrt (Array.sumBy (fun x -> x * x) v)
    /// Get the unit vector codirectional with this vector
    member v.GetUnitVector() =
        match v with
        | ZeroVector -> ZeroVector
        | Vector vv -> let n = v.GetNorm() in Vector (Array.init vv.Length (fun i -> vv.[i] / n))
    override v.ToString() =
        match v with
        | ZeroVector -> "ZeroVector"
        | Vector v -> sprintf "Vector %A" v
    /// Get a string representation of this Vector that can be pasted into a Mathematica notebook
    member v.ToMathematicaString() = 
        match v with
        | ZeroVector -> MathematicaVector (Array.zeroCreate 0)
        | Vector v -> MathematicaVector v
    /// Get a string representation of this Vector that can be pasted into MATLAB
    member v.ToMatlabString() = 
        match v with
        | ZeroVector -> MatlabVector (Array.zeroCreate 0)
        | Vector v -> MatlabVector v
    /// Create Vector from array `v`
    static member Create(v) = Vector v
    /// Create Vector with dimension `n` and a generator function `f` to compute the elements
    static member Create(n, f) = Vector (Array.init n f)
    /// Create Vector with dimension `n` and all elements having value `v`
    static member Create(n, v) = Vector (Array.create n v)
    /// Create Vector with dimension `n`, the element with index `i` having value `v`, and the rest of the elements 0
    static member Create(n, i, v) = Vector.Create(n, fun j -> if j = i then v else 0.)
    /// Returns the sum of all the elements in vector `v`
    static member sum v = 
        match v with
        | ZeroVector -> 0.
        | Vector v -> Array.sum v
    /// Builds a new Vector whose elements are the results of applying function `f` to each of the elements of Vector `v`
    static member map f v =
        match v with
        | ZeroVector -> ZeroVector
        | Vector v -> Vector (Array.map f v)
    /// Vector with infinite dimension and all elements 0
    static member Zero = ZeroVector
    /// Convert Vector `v` to float[]
    static member op_Explicit(v:Vector) =
        match v with
        | ZeroVector -> [|0.|]
        | Vector v -> v
    /// Add Vector `a` to Vector `b`
    static member (+) (a, b) =
        match a, b with
        | ZeroVector, ZeroVector -> ZeroVector
        | ZeroVector, Vector vb -> Vector vb
        | Vector va, ZeroVector -> Vector va
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] + vb.[i])
    /// Subtract Vector `b` from Vector `a`
    static member (-) (a, b) =
        match a, b with
        | ZeroVector, ZeroVector -> ZeroVector
        | ZeroVector, Vector vb -> Vector.Create(vb.Length, fun i -> -vb.[i])
        | Vector va, ZeroVector -> Vector va
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] - vb.[i])
    /// Multiply Vector `a` and Vector `b` element-wise (Hadamard product)
    static member (*) (a, b) =
        match a, b with
        | ZeroVector, ZeroVector -> ZeroVector
        | ZeroVector, Vector _ -> ZeroVector
        | Vector _, ZeroVector -> ZeroVector
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] * vb.[i])
    /// Divide Vector `a` by Vector `b` element-wise
    static member (/) (a, b) =
        match a, b with
        | ZeroVector, ZeroVector -> raise (new System.DivideByZeroException("Attempted to divide a ZeroVector by a ZeroVector."))
        | ZeroVector, Vector _ -> ZeroVector
        | Vector _, ZeroVector -> raise (new System.DivideByZeroException("Attempted to divide a Vector by a ZeroVector."))
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] / vb.[i])
    /// Multiply Vector `a` by float `b`
    static member (*) (a, b) =
        match a with
        | ZeroVector -> ZeroVector
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] * b)
    /// Multiply Vector `b` by float `a`
    static member (*) (a, b) =
        match b with
        | ZeroVector -> ZeroVector
        | Vector vb -> Vector.Create(vb.Length, fun i -> a * vb.[i])
    /// Divide Vector `a` by float `b`
    static member (/) (a, b) =
        match a with
        | ZeroVector -> ZeroVector
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] / b)
    /// Create Vector whose elements are float `a` divided by the elements of Vector `b`
    static member (/) (a, b) =
        match b with
        | ZeroVector -> raise (new System.DivideByZeroException("Attempted division by a ZeroVector."))
        | Vector vb -> Vector.Create(vb.Length, fun i -> a / vb.[i])
    /// Negative of Vector `a`
    static member (~-) a =
        match a with
        | ZeroVector -> ZeroVector
        | Vector va -> Vector.Create(va.Length, fun i -> -va.[i])


/// Lightweight matrix type
[<NoEquality; NoComparison>]
type Matrix =
    | ZeroMatrix
    | Matrix of float[,]
    | SymmetricMatrix of float[,]
    with
    member m.Item(i, j) =
        match m with
        | ZeroMatrix -> 0.
        | Matrix m -> m.[i, j]
        | SymmetricMatrix m -> if j >= i then m.[i, j] else m.[j, i]
    member m.Rows =
        match m with
        | ZeroMatrix -> 0
        | Matrix m -> m.GetLength 0
        | SymmetricMatrix m -> m.GetLength 0
    member m.Cols =
        match m with
        | ZeroMatrix -> 0
        | Matrix m -> m.GetLength 1
        | SymmetricMatrix m -> m.GetLength 1
    override m.ToString() =
        match m with
        | ZeroMatrix -> "ZeroMatrix"
        | Matrix m -> sprintf "Matrix %A" m
        | SymmetricMatrix m -> sprintf "SymmetricMatrix %A" (copyupper m)
    /// Get a string representation of this Matrix that can be pasted into a Mathematica notebook
    member m.ToMathematicaString() = 
        match m with
        | ZeroMatrix -> MathematicaMatrix (Array2D.zeroCreate 0 0)
        | Matrix m -> MathematicaMatrix m
        | SymmetricMatrix m -> MathematicaMatrix (copyupper m)
    /// Get a string representation of this Matrix that can be pasted into MATLAB
    member m.ToMatlabString() =
        match m with
        | ZeroMatrix -> MatlabMatrix (Array2D.zeroCreate 0 0)
        | Matrix m -> MatlabMatrix m
        | SymmetricMatrix m -> MatlabMatrix (copyupper m)
    /// Get the trace of this Matrix
    member m.GetTrace() =
        match m with
        | ZeroMatrix -> 0.
        | Matrix m -> trace m
        | SymmetricMatrix m -> trace m
    /// Get the transpose of this Matrix
    member m.GetTranspose() =
        match m with
        | ZeroMatrix -> ZeroMatrix
        | Matrix m -> Matrix (transpose m)
        | SymmetricMatrix m -> SymmetricMatrix m
    /// Create Matrix from given float[,] `m`
    static member Create(m) = Matrix m
    /// Create Matrix with `m` rows, `n` columns, and all elements having value `v`. If m = n, a SymmetricMatrix is created.
    static member Create(m, n, v) = if m = n then SymmetricMatrix (Array2D.create m m v) else Matrix (Array2D.create m n v)
    /// Create Matrix with `m` rows, `n` columns, and a generator function `f` to compute the elements
    static member Create(m, n, f) = Matrix (Array2D.init m n f)
    /// Create Matrix with `m` rows and all rows equal to float[] `v`
    static member Create(m, v:float[]) = Matrix.Create(m, fun _ -> v)
    /// Create Matrix with `m` rows and a generator function `f` that gives each row as a float[]
    static member Create(m, f:int->float[]) = Matrix (array2D (Array.init m f))
    /// Create Matrix with `m` rows and a generator function `f` that gives each row as a Vector
    static member Create(m, f:int->Vector) =
        let a = Array.init m f
        Matrix (Array2D.init m (a.[0].Length) (fun i j -> a.[i].[j]))
    /// Create Matrix with rows given in Vector[] `v`
    static member Create(v:Vector[]) = Matrix.Create(v.Length, fun i -> v.[i])
    /// Create SymmetricMatrix with `m` rows and columns and a generator function `f` to compute the elements
    static member CreateSymmetric(m, f) =
        let s = Array2D.zeroCreate<float> m m
        for i = 0 to m - 1 do
            for j = i to m - 1 do
                s.[i, j] <- f i j
        SymmetricMatrix s
    /// Matrix with infinite number of rows and columns and all entries 0
    static member Zero = ZeroMatrix
    /// Convert Matrix `m` to float[,]
    static member op_Explicit(m:Matrix) =
        match m with
        | ZeroMatrix -> Array2D.zeroCreate 0 0
        | Matrix m -> m
        | SymmetricMatrix m -> copyupper m
    /// Symmetric unary operation `f` on Matrix `a`
    static member SymmetricOp(a, f:int->int->float) =
        match a with
        | ZeroMatrix -> ZeroMatrix
        | Matrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
    /// Symmetric binary operation `f` on Matrix `a` and Matrix `b`
    static member SymmetricOp(a, b, f:int->int->float) =
        match a, b with
        | ZeroMatrix, ZeroMatrix -> ZeroMatrix
        | ZeroMatrix, Matrix _ -> Matrix.CreateSymmetric(b.Rows, f)
        | ZeroMatrix, SymmetricMatrix _ -> Matrix.CreateSymmetric(b.Rows, f)
        | Matrix _, ZeroMatrix -> Matrix.CreateSymmetric(a.Rows, f)
        | Matrix _, Matrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | Matrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, ZeroMatrix -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, Matrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
    /// Add Matrix `a` to Matrix `b`
    static member (+) (a, b) =
        match a, b with
        | ZeroMatrix, ZeroMatrix -> ZeroMatrix
        | ZeroMatrix, Matrix bm -> Matrix bm
        | ZeroMatrix, SymmetricMatrix bm -> SymmetricMatrix bm
        | Matrix am, ZeroMatrix -> Matrix am
        | Matrix am, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] + bm.[i, j])
        | Matrix am, SymmetricMatrix _ -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] + b.[i, j])
        | SymmetricMatrix a, ZeroMatrix -> SymmetricMatrix a
        | SymmetricMatrix _, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] + bm.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] + b.[i, j])
    /// Subtract Matrix `b` from Matrix `a`
    static member (-) (a, b) =
        match a, b with
        | ZeroMatrix, ZeroMatrix -> ZeroMatrix
        | ZeroMatrix, Matrix bm -> Matrix.Create(b.Rows, b.Cols, fun i j -> -bm.[i ,j])
        | ZeroMatrix, SymmetricMatrix _ -> Matrix.CreateSymmetric(b.Rows, fun i j -> -b.[i ,j])
        | Matrix am, ZeroMatrix -> Matrix am
        | Matrix am, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] - bm.[i, j])
        | Matrix am, SymmetricMatrix _ -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] - b.[i, j])
        | SymmetricMatrix a, ZeroMatrix -> SymmetricMatrix a
        | SymmetricMatrix _, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] - bm.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] - b.[i, j])
    /// Matrix product of Matrix `a` and Matrix `b`
    static member (*) (a, b) =
        match a, b with
        | ZeroMatrix, ZeroMatrix -> ZeroMatrix
        | ZeroMatrix, Matrix bm -> ZeroMatrix
        | ZeroMatrix, SymmetricMatrix bm -> ZeroMatrix
        | Matrix am, ZeroMatrix -> ZeroMatrix
        | Matrix am, Matrix bm -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> am.[i, k] * bm.[k, j]) [|0..(b.Rows - 1)|] )
        | Matrix am, SymmetricMatrix _ -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> am.[i, k] * b.[k, j]) [|0..(b.Rows - 1)|] )
        | SymmetricMatrix a, ZeroMatrix -> ZeroMatrix
        | SymmetricMatrix _, Matrix bm -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> a.[i, k] * bm.[k, j]) [|0..(b.Rows - 1)|] )
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> a.[i, k] * b.[k, j]) [|0..(b.Rows - 1)|] )
    /// Multiply Matrix `a` by float `b`
    static member (*) (a, b) =
        match a with
        | ZeroMatrix -> ZeroMatrix
        | Matrix am -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] * b)
        | SymmetricMatrix am -> Matrix.CreateSymmetric(a.Rows, fun i j -> am.[i, j] * b)
    /// Multiply Matrix `b` by float `a`
    static member (*) (a, b) =
        match b with
        | ZeroMatrix -> ZeroMatrix
        | Matrix bm -> Matrix.Create(b.Rows, b.Cols, fun i j -> a * bm.[i, j])
        | SymmetricMatrix bm -> Matrix.CreateSymmetric(b.Rows, fun i j -> a * bm.[i, j])
    /// Divide Matrix `a` by float `b`
    static member (/) (a, b) =
        match a with
        | ZeroMatrix -> ZeroMatrix
        | Matrix am -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] / b)
        | SymmetricMatrix am -> Matrix.CreateSymmetric(a.Rows, fun i j -> am.[i, j] / b)
    /// Create Matrix whose elements are float `a` divided by the element of Matrix `b`
    static member (/) (a, b) =
        match b with
        | ZeroMatrix -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
        | Matrix bm -> Matrix.Create(b.Rows, b.Cols, fun i j -> a / bm.[i, j])
        | SymmetricMatrix bm -> Matrix.CreateSymmetric(b.Rows, fun i j -> a / bm.[i, j])
    /// Negative of Matrix `a`
    static member (~-) a =
        match a with
        | ZeroMatrix -> ZeroMatrix
        | Matrix am -> Matrix.Create(a.Rows, a.Cols, fun i j -> -am.[i, j])
        | SymmetricMatrix am -> Matrix.CreateSymmetric(a.Rows, fun i j -> -am.[i, j])


/// Convert float[] `v` into Vector
let vector v = Vector v
/// Get the Euclidean norm of Vector `v`
let norm (v:Vector) = v.GetNorm()
/// Get the unit vector codirectional with Vector `v`
let unitVector (v:Vector) = v.GetUnitVector()
/// Convert Vector `v` into float[]
let array v = Vector.op_Explicit(v)
/// Convert float[,] `m` into Matrix
let matrix (m:float[,]) = Matrix.Create(m)
/// Convert Matrix `m` into float[,]
let array2d m = Matrix.op_Explicit(m)
/// Get the trace of Matrix `m`
let trace (m:Matrix) = m.GetTrace()
/// Get the transpose of Matrix `m`
let transpose (m:Matrix) = m.GetTranspose()

