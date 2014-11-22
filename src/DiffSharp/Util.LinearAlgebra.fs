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
type Vector<'T when 'T : (static member Zero : 'T)
                and 'T : (static member (+) : 'T * 'T -> 'T)
                and 'T : (static member (-) : 'T * 'T -> 'T)
                and 'T : (static member (*) : 'T * 'T -> 'T)
                and 'T : (static member (/) : 'T * 'T -> 'T)
                and 'T : (static member (~-) : 'T -> 'T)
                and 'T : (static member Sqrt : 'T -> 'T)
                and 'T : (static member op_Explicit : 'T -> float)> =
    /// Vector with infinite dimension whose elements are all 0
    | ZeroVector of 'T
    /// Vector with finite dimension
    | Vector of 'T[]
    with
    /// Gets the element of this Vector at the given position `i`
    member inline v.Item i =
        match v with
        | ZeroVector z -> z
        | Vector v -> v.[i]
    /// Gets the first element of this Vector
    member inline v.FirstItem =
        match v with
        | ZeroVector z -> z
        | Vector v -> v.[0]
    /// Gets the total number of elements of this Vector
    member inline v.Length =
        match v with
        | ZeroVector _ -> 0
        | Vector v -> v.Length
    /// Converts this Vector to an array, e.g. from Vector<float> to float[]
    member inline v.ToArray() =
        match v with
        | ZeroVector _ -> [||]
        | Vector v -> v
    /// Gets the Euclidean norm of this Vector
    member inline v.GetNorm() =
        match v with
        | ZeroVector z -> z
        | Vector v -> sqrt (Array.sumBy (fun x -> x * x) v)
    /// Gets the unit Vector codirectional with this Vector    
    member inline v.GetUnitVector() =
        match v with
        | ZeroVector z -> ZeroVector z
        | Vector vv -> let n = v.GetNorm() in Vector (Array.init vv.Length (fun i -> vv.[i] / n))
    /// Gets a string representation of this Vector that can be pasted into a Mathematica notebook
    member inline v.ToMathematicaString() = 
        let sb = System.Text.StringBuilder()
        sb.Append("{") |> ignore
        for i = 0 to v.Length - 1 do
            sb.Append(sprintf "%.2f" (float v.[i])) |> ignore
            if i < v.Length - 1 then sb.Append(", ") |> ignore
        sb.Append("}") |> ignore
        sb.ToString()
    /// Gets a string representation of this Vector that can be pasted into MATLAB
    member inline v.ToMatlabString() =
        let sb = System.Text.StringBuilder()
        sb.Append("[") |> ignore
        for i = 0 to v.Length - 1 do
            sb.Append(sprintf "%.2f" (float v.[i])) |> ignore
            if i < v.Length - 1 then sb.Append(" ") |> ignore
        sb.Append("]") |> ignore
        sb.ToString()
    /// Creates a Vector from the array `v`
    static member inline Create(v) : Vector<'T> = Vector v
    /// Creates a Vector with dimension `n` and a generator function `f` to compute the elements
    static member inline Create(n, f) : Vector<'T> = Vector (Array.init n f)
    /// Creates a Vector with dimension `n` whose elements are all initally the given value `v`
    static member inline Create(n, v) : Vector<'T> = Vector (Array.create n v)
    /// Creates a Vector with dimension `n` where the element with index `i` has value `v` and the rest of the elements have value 0
    static member inline Create(n, i, v) : Vector<'T> = Vector.Create(n, fun j -> if j = i then v else LanguagePrimitives.GenericZero<'T>)
    /// Returns the sum of all the elements in Vector `v`
    static member inline sum (v:Vector<'T>) = 
        match v with
        | ZeroVector z -> z
        | Vector v -> Array.sum v
    /// Creates a Vector with dimension `n` and a generator function `f` to compute the eleements
    static member inline init (n:int) (f:int->'T) = Vector.Create(n, f)
    /// ZeroVector
    static member inline Zero = ZeroVector LanguagePrimitives.GenericZero<'T>
    /// Converts Vector `v` to float[]
    static member inline op_Explicit(v:Vector<'T>) =
        match v with
        | ZeroVector _ -> [||]
        | Vector v -> Array.map float v
    /// Adds Vector `a` to Vector `b`
    static member inline (+) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector vb -> Vector vb
        | Vector va, ZeroVector _ -> Vector va
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] + vb.[i])
    /// Subtracts Vector `b` from Vector `a`
    static member inline (-) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector vb -> Vector.Create(vb.Length, fun i -> -vb.[i])
        | Vector va, ZeroVector _ -> Vector va
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] - vb.[i])
    /// Multiplies Vector `a` and Vector `b` element-wise (Hadamard product)
    static member inline (*) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector _ -> Vector.Zero
        | Vector _, ZeroVector _ -> Vector.Zero
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] * vb.[i])
    /// Divides Vector `a` by Vector `b` element-wise
    static member inline (/) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> raise (new System.DivideByZeroException("Attempted to divide a ZeroVector by a ZeroVector."))
        | ZeroVector _, Vector _ -> Vector.Zero
        | Vector _, ZeroVector _-> raise (new System.DivideByZeroException("Attempted to divide a Vector by a ZeroVector."))
        | Vector va, Vector vb -> Vector.Create(va.Length, fun i -> va.[i] / vb.[i])
    /// Multiplies Vector `a` by number `b`
    static member inline (*) (a, b) =
        match a with
        | ZeroVector _ -> Vector.Zero
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] * b)
    /// Multiples Vector `b` by number `a`
    static member inline (*) (a, b) =
        match b with
        | ZeroVector _ -> Vector.Zero
        | Vector vb -> Vector.Create(vb.Length, fun i -> a * vb.[i])
    /// Divides Vector `a` by number `b`
    static member inline (/) (a, b) =
        match a with
        | ZeroVector _ -> Vector.Zero
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] / b)
    /// Creates a Vector whose elements are number `a` divided by the elements of Vector `b`
    static member inline (/) (a, b) =
        match b with
        | ZeroVector _ -> raise (new System.DivideByZeroException("Attempted division by a ZeroVector."))
        | Vector vb -> Vector.Create(vb.Length, fun i -> a / vb.[i])
    /// Gets the negative of Vector `a`
    static member inline (~-) (a:Vector<'T>) =
        match a with
        | ZeroVector _ -> Vector.Zero
        | Vector va -> Vector.Create(va.Length, fun i -> -va.[i])

/// Lightweight matrix type
[<NoEquality; NoComparison>]
type Matrix<'T when 'T : (static member Zero : 'T)
                and 'T : (static member (+) : 'T * 'T -> 'T)
                and 'T : (static member (-) : 'T * 'T -> 'T)
                and 'T : (static member (*) : 'T * 'T -> 'T)
                and 'T : (static member (/) : 'T * 'T -> 'T)
                and 'T : (static member (~-) : 'T -> 'T)
                and 'T : (static member Sqrt : 'T -> 'T)
                and 'T : (static member op_Explicit : 'T -> float)> =
    /// Matrix with infinite number of rows and columns whose entries are all 0
    | ZeroMatrix of 'T
    /// Matrix with finite number of rows and columns
    | Matrix of 'T[,]
    /// Symmetric square matrix that is equal to its transpose
    | SymmetricMatrix of 'T[,]
    with
    /// Gets the entry of this Matrix at row `i` and column `j`
    member inline m.Item(i, j) =
        match m with
        | ZeroMatrix z -> z
        | Matrix m -> m.[i, j]
        | SymmetricMatrix m -> if j >= i then m.[i, j] else m.[j, i]
    /// Gets the number of rows of this Matrix
    member inline m.Rows =
        match m with
        | ZeroMatrix _ -> 0
        | Matrix m -> m.GetLength 0
        | SymmetricMatrix m -> m.GetLength 0
    /// Gets the number of columns of thisMatrix
    member inline m.Cols =
        match m with
        | ZeroMatrix _ -> 0
        | Matrix m -> m.GetLength 1
        | SymmetricMatrix m -> m.GetLength 1
    /// Converts this Matrix to a 2d array, e.g. from Matrix<float> to float[,]
    member inline m.ToArray2d() =
        match m with
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
        | Matrix m -> m
        | SymmetricMatrix m -> copyupper m
    /// Gets a string representation of this Matrix that can be pasted into a Mathematica notebook
    member inline m.ToMathematicaString() =
        let sb = System.Text.StringBuilder()
        sb.Append("{") |> ignore
        for i = 0 to m.Rows - 1 do
            sb.Append("{") |> ignore
            for j = 0 to m.Cols - 1 do
                sb.Append(sprintf "%.2f" (float m.[i, j])) |> ignore
                if j <> m.Cols - 1 then sb.Append(", ") |> ignore
            sb.Append("}") |> ignore
            if i <> m.Rows - 1 then sb.Append(", ") |> ignore
        sb.Append("}") |> ignore
        sb.ToString()
    /// Gets a string representation of this Matrix that can be pasted into MATLAB
    member inline m.ToMatlabString() =
        let sb = System.Text.StringBuilder()
        sb.Append("[") |> ignore
        for i = 0 to m.Rows - 1 do
            for j = 0 to m.Cols - 1 do
                sb.Append(sprintf "%.2f" (float m.[i, j])) |> ignore
                if j < m.Cols - 1 then sb.Append(" ") |> ignore
            if i < m.Rows - 1 then sb.Append("; ") |> ignore
        sb.Append("]") |> ignore
        sb.ToString()
    /// Gets the trace of this Matrix
    member inline m.GetTrace() =
        match m with
        | ZeroMatrix z -> z
        | Matrix m -> trace m
        | SymmetricMatrix m -> trace m
    /// Gets the transpose of this Matrix
    member inline m.GetTranspose() =
        match m with
        | ZeroMatrix z -> ZeroMatrix z
        | Matrix m -> Matrix (transpose m)
        | SymmetricMatrix m -> SymmetricMatrix m
    /// Creates a Matrix from the given 2d array `m`
    static member inline Create(m):Matrix<'T> = Matrix m
    /// Creates a Matrix with `m` rows, `n` columns, and all elements having value `v`. If m = n, a SymmetricMatrix is created.
    static member inline Create(m, n, v):Matrix<'T> = if m = n then SymmetricMatrix (Array2D.create m m v) else Matrix (Array2D.create m n v)
    /// Creates a Matrix with `m` rows, `n` columns, and a generator function `f` to compute the elements
    static member inline Create(m, n, f):Matrix<'T> = Matrix (Array2D.init m n f)
    /// Creates a Matrix with `m` rows and a generator function `f` that gives each row as a an array
    static member inline Create(m, f:int->'T[]):Matrix<'T> = Matrix (array2D (Array.init m f))
    /// Creates a Matrix with `m` rows and all rows equal to array `v`
    static member inline Create(m, v:'T[]):Matrix<'T> = Matrix.Create(m, fun _ -> v)
    /// Creates a Matrix with `m` rows and a generator function `f` that gives each row as a Vector
    static member inline Create(m, f:int->Vector<'T>):Matrix<'T> =
        let a = Array.init m f
        Matrix (Array2D.init m (a.[0].Length) (fun i j -> a.[i].[j]))
    /// Creates a Matrix with rows given in Vector[] `v`
    static member inline Create(v:Vector<'T>[]):Matrix<'T> = Matrix.Create(v.Length, fun i -> v.[i])
    /// Creates a SymmetricMatrix with `m` rows and `m` columns and a generator function `f` to compute the elements. Function `f` is used only for populating the upper triangular part of the Matrix, the lower triangular part will be the reflection.
    static member inline CreateSymmetric(m, f):Matrix<'T> =
        let s = Array2D.zeroCreate<'T> m m
        for i = 0 to m - 1 do
            for j = i to m - 1 do
                s.[i, j] <- f i j
        SymmetricMatrix s
    /// ZeroMatrix
    static member inline Zero = ZeroMatrix LanguagePrimitives.GenericZero<'T>
    /// Converts Matrix `m` to float[,]
    static member inline op_Explicit(m:Matrix<'T>) =
        match m with
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
        | Matrix m -> Array2D.map float m
        | SymmetricMatrix m -> copyupper (Array2D.map float m)
    /// Perfomrs a symmetric unary operation `f` on Matrix `a`
    static member inline SymmetricOp(a:Matrix<'T>, f:int->int->'T):Matrix<'T> =
        match a with
        | ZeroMatrix z -> ZeroMatrix z
        | Matrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
    /// Performs a symmetric binary operation `f` on Matrix `a` and Matrix `b`
    static member inline SymmetricOp(a:Matrix<'T>, b:Matrix<'T>, f:int->int->'T):Matrix<'T> =
        match a, b with
        | ZeroMatrix _, ZeroMatrix z -> ZeroMatrix z
        | ZeroMatrix _, Matrix _ -> Matrix.CreateSymmetric(b.Rows, f)
        | ZeroMatrix _ , SymmetricMatrix _ -> Matrix.CreateSymmetric(b.Rows, f)
        | Matrix _, ZeroMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | Matrix _, Matrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | Matrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, ZeroMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, Matrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
    /// Adds Matrix `a` to Matrix `b`
    static member inline (+) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroMatrix z -> ZeroMatrix z
        | ZeroMatrix _, Matrix bm -> Matrix bm
        | ZeroMatrix _, SymmetricMatrix bm -> SymmetricMatrix bm
        | Matrix am, ZeroMatrix _ -> Matrix am
        | Matrix am, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] + bm.[i, j])
        | Matrix am, SymmetricMatrix _ -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] + b.[i, j])
        | SymmetricMatrix a, ZeroMatrix _ -> SymmetricMatrix a
        | SymmetricMatrix _, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] + bm.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] + b.[i, j])
    /// Subtracts Matrix `b` from Matrix `a`
    static member inline (-) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroMatrix z -> ZeroMatrix z
        | ZeroMatrix _, Matrix bm -> Matrix.Create(b.Rows, b.Cols, fun i j -> -bm.[i ,j])
        | ZeroMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(b.Rows, fun i j -> -b.[i ,j])
        | Matrix am, ZeroMatrix _ -> Matrix am
        | Matrix am, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] - bm.[i, j])
        | Matrix am, SymmetricMatrix _ -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] - b.[i, j])
        | SymmetricMatrix a, ZeroMatrix _ -> SymmetricMatrix a
        | SymmetricMatrix _, Matrix bm -> Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] - bm.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] - b.[i, j])
    /// Calculates the matrix product of Matrix `a` and Matrix `b`
    static member inline (*) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix z, ZeroMatrix _ -> ZeroMatrix z
        | ZeroMatrix z, Matrix _ -> ZeroMatrix z
        | ZeroMatrix z, SymmetricMatrix _ -> ZeroMatrix z
        | Matrix am, ZeroMatrix z -> ZeroMatrix z
        | Matrix am, Matrix bm -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> am.[i, k] * bm.[k, j]) [|0..(b.Rows - 1)|] )
        | Matrix am, SymmetricMatrix _ -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> am.[i, k] * b.[k, j]) [|0..(b.Rows - 1)|] )
        | SymmetricMatrix _, ZeroMatrix z -> ZeroMatrix z
        | SymmetricMatrix _, Matrix bm -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> a.[i, k] * bm.[k, j]) [|0..(b.Rows - 1)|] )
        | SymmetricMatrix _, SymmetricMatrix _ -> Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> a.[i, k] * b.[k, j]) [|0..(b.Rows - 1)|] )
    /// Multiplies Matrix `a` by number `b`
    static member inline (*) (a, b) =
        match a with
        | ZeroMatrix z -> ZeroMatrix z
        | Matrix am -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] * b)
        | SymmetricMatrix am -> Matrix.CreateSymmetric(a.Rows, fun i j -> am.[i, j] * b)
    /// Multiplies Matrix `b` by number `a`
    static member inline (*) (a, b) =
        match b with
        | ZeroMatrix z -> ZeroMatrix z
        | Matrix bm -> Matrix.Create(b.Rows, b.Cols, fun i j -> a * bm.[i, j])
        | SymmetricMatrix bm -> Matrix.CreateSymmetric(b.Rows, fun i j -> a * bm.[i, j])
    /// Divides Matrix `a` by number `b`
    static member inline (/) (a, b) =
        match a with
        | ZeroMatrix z -> ZeroMatrix z
        | Matrix am -> Matrix.Create(a.Rows, a.Cols, fun i j -> am.[i, j] / b)
        | SymmetricMatrix am -> Matrix.CreateSymmetric(a.Rows, fun i j -> am.[i, j] / b)
    /// Creates a Matrix whose elements are number `a` divided by the element of Matrix `b`
    static member inline (/) (a, b) =
        match b with
        | ZeroMatrix _ -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
        | Matrix bm -> Matrix.Create(b.Rows, b.Cols, fun i j -> a / bm.[i, j])
        | SymmetricMatrix bm -> Matrix.CreateSymmetric(b.Rows, fun i j -> a / bm.[i, j])
    /// Gets the negative of Matrix `a`
    static member inline (~-) (a:Matrix<'T>) =
        match a with
        | ZeroMatrix z -> ZeroMatrix z
        | Matrix am -> Matrix.Create(a.Rows, a.Cols, fun i j -> -am.[i, j])
        | SymmetricMatrix am -> Matrix.CreateSymmetric(a.Rows, fun i j -> -am.[i, j])



/// Converts array `v` into a Vector
let inline vector v = Vector v
/// Gets the Euclidean norm of Vector `v`
let inline norm (v:Vector<'T>) = v.GetNorm()
/// Gets the unit vector codirectional with Vector `v`
let inline unitVector (v:Vector<'T>) = v.GetUnitVector()
/// Converts Vector `v` into array
let inline array (v:Vector<'T>) = v.ToArray()
/// Converts 2d array `m` into a Matrix
let inline matrix (m:'T[,]) = Matrix.Create(m)
/// Converts Matrix `m` into a 2d array
let inline array2d (m:Matrix<'T>) = m.ToArray2d()
/// Gets the trace of Matrix `m`
let inline trace (m:Matrix<'T>) = m.GetTrace()
/// Gets the transpose of Matrix `m`
let inline transpose (m:Matrix<'T>) = m.GetTranspose()