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
                and ^T : (static member (/) : ^T * ^T -> ^T)
                and 'T : (static member (~-) : 'T -> 'T)
                and 'T : (static member Sqrt : 'T -> 'T)
                and 'T : (static member Abs : 'T -> 'T)
                and ^T : (static member op_Explicit : ^T -> float)
                and 'T : comparison> =
    /// Vector with infinite dimension whose elements are all 0
    | ZeroVector of 'T
    /// Vector with finite dimension
    | Vector of 'T[]
    with
    /// Gets the element of this Vector at the given position `i`
    member inline v.Item
        with get i =
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
    /// Converts the elements of this Vector to another type, using the given conversion function `f`
    member inline v.Convert(f:'T->'a):Vector<'a> =
        match v with
        | ZeroVector z -> ZeroVector LanguagePrimitives.GenericZero<'a>
        | Vector v -> Vector (Array.map f v)
    /// Creates a Vector from the array `v`
    static member inline Create(v) : Vector<'T> = Vector v
    /// Creates a Vector with dimension `n` and a generator function `f` to compute the elements
    static member inline Create(n, f) : Vector<'T> = Vector (Array.init n f)
    /// Creates a Vector with dimension `n` whose elements are all initally the given value `v`
    static member inline Create(n, v) : Vector<'T> = Vector (Array.create n v)
    /// Creates a Vector with dimension `n` where the element with index `i` has value `v` and the rest of the elements have value 0
    static member inline Create(n, i, v) : Vector<'T> = Vector.Create(n, fun j -> if j = i then v else LanguagePrimitives.GenericZero<'T>)
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
        | Vector va, Vector vb ->
            if va.Length <> vb.Length then invalidArg "b" "Cannot add two Vectors with different dimensions."
            Vector.Create(va.Length, fun i -> va.[i] + vb.[i])
    /// Subtracts Vector `b` from Vector `a`
    static member inline (-) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector vb -> Vector.Create(vb.Length, fun i -> -vb.[i])
        | Vector va, ZeroVector _ -> Vector va
        | Vector va, Vector vb -> 
            if va.Length <> vb.Length then invalidArg "b" "Cannot subtract two Vectors with different dimensions."
            Vector.Create(va.Length, fun i -> va.[i] - vb.[i])
    /// Computes the inner product of Vector `a` and Vector `b` (dot / scalar product)
    static member inline (*) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> LanguagePrimitives.GenericZero<'T>
        | ZeroVector _, Vector _ -> LanguagePrimitives.GenericZero<'T>
        | Vector _, ZeroVector _ -> LanguagePrimitives.GenericZero<'T>
        | Vector va, Vector vb ->
            if va.Length <> vb.Length then invalidArg "b" "Cannot multiply two Vectors with different dimensions."
            Array.sumBy (fun (x, y) -> x * y) (Array.zip va vb)
    /// Computes the cross product of Vector `a` and Vector `b` (three-dimensional)
    static member inline (%*) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector _ -> Vector.Zero
        | Vector _, ZeroVector _ -> Vector.Zero
        | Vector va, Vector vb ->
            if (a.Length <> 3) || (b.Length <> 3) then invalidArg "b" "The cross product is only defined for three-dimensional vectors."
            Vector [|va.[1] * vb.[2] - va.[2] * vb.[1]; va.[2] * vb.[0] - va.[0] * vb.[2]; va.[0] * vb.[1] - va.[1] * vb.[0]|]
    /// Multiplies Vector `a` and Vector `b` element-wise (Hadamard product)
    static member inline (.*) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector _ -> Vector.Zero
        | Vector _, ZeroVector _ -> Vector.Zero
        | Vector va, Vector vb ->
            if va.Length <> vb.Length then invalidArg "b" "Cannot multiply two Vectors with different dimensions."
            Vector.Create(va.Length, fun i -> va.[i] * vb.[i])
    /// Divides Vector `a` by Vector `b` element-wise (Hadamard division)
    static member inline (./) (a:Vector<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroVector _, ZeroVector _ -> raise (new System.DivideByZeroException("Attempted to divide a ZeroVector by a ZeroVector."))
        | ZeroVector _, Vector _ -> Vector.Zero
        | Vector _, ZeroVector _-> raise (new System.DivideByZeroException("Attempted to divide a Vector by a ZeroVector."))
        | Vector va, Vector vb -> 
            if va.Length <> vb.Length then invalidArg "b" "Cannot divide two Vectors with different dimensions."
            Vector.Create(va.Length, fun i -> va.[i] / vb.[i])
    /// Adds scalar `b` to each element of Vector `a`
    static member inline (+) (a:Vector<'T>, b:'T) =
        match a with
        | ZeroVector _ -> invalidArg "a" "Unsupported operation. Cannot add a scalar to a ZeroVector."
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] + b)
    /// Adds scalar `a` to each element of Vector `b`
    static member inline (+) (a:'T, b:Vector<'T>) =
        match b with
        | ZeroVector _ -> invalidArg "b" "Unsupported operation. Cannot add a scalar to a ZeroVector."
        | Vector vb -> Vector.Create(vb.Length, fun i -> a + vb.[i])
    /// Subtracts scalar `b` from each element of Vector `a`
    static member inline (-) (a:Vector<'T>, b:'T) =
        match a with
        | ZeroVector _ -> invalidArg "a" "Unsupported operation. Cannot subtract a scalar from a ZeroVector."
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] - b)
    /// Subtracts each element of Vector `b` from scalar `a`
    static member inline (-) (a:'T, b:Vector<'T>) =
        match b with
        | ZeroVector _ -> invalidArg "b" "Unsupported operation. Cannot add subtract a ZeroVector from a scalar."
        | Vector vb -> Vector.Create(vb.Length, fun i -> a - vb.[i])
    /// Multiplies each element of Vector `a` by scalar `b`
    static member inline (*) (a:Vector<'T>, b:'T) =
        match a with
        | ZeroVector _ -> Vector.Zero
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] * b)
    /// Multiples each element of Vector `b` by scalar `a`
    static member inline (*) (a:'T, b:Vector<'T>) =
        match b with
        | ZeroVector _ -> Vector.Zero
        | Vector vb -> Vector.Create(vb.Length, fun i -> a * vb.[i])
    /// Divides each element of Vector `a` by scalar `b`
    static member inline (/) (a:Vector<'T>, b:'T) =
        match a with
        | ZeroVector _ -> Vector.Zero
        | Vector va -> Vector.Create(va.Length, fun i -> va.[i] / b)
    /// Creates a Vector whose elements are scalar `a` divided by each element of Vector `b`
    static member inline (/) (a:'T, b:Vector<'T>) =
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
                and 'T : (static member One : 'T)
                and 'T : (static member (+) : 'T * 'T -> 'T)
                and 'T : (static member (-) : 'T * 'T -> 'T)
                and 'T : (static member (*) : 'T * 'T -> 'T)
                and 'T : (static member (/) : 'T * 'T -> 'T)
                and 'T : (static member (~-) : 'T -> 'T)
                and 'T : (static member Sqrt : 'T -> 'T)
                and 'T : (static member Abs : 'T -> 'T)
                and 'T : (static member op_Explicit : 'T -> float)
                and 'T : comparison> =
    /// Matrix with infinite number of rows and columns whose entries are all 0
    | ZeroMatrix of 'T
    /// Matrix with finite number of rows and columns
    | Matrix of 'T[,]
    /// Symmetric square matrix that is equal to its transpose
    | SymmetricMatrix of 'T[,]
    with
    /// Gets the entry of this Matrix at row `i` and column `j`
    member inline m.Item
        with get (i, j) =
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
    /// Converts this Matrix into a 2d array
    member inline m.ToArray2d() =
        match m with
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
        | Matrix m -> m
        | SymmetricMatrix m -> copyupper m
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
    /// Gets the array of the diagonal elements of this Matrix
    member inline m.GetDiagonal() =
        match m with
        | ZeroMatrix z -> [||]
        | Matrix mm -> 
            if m.Rows <> m.Cols then failwith "Cannot get the diagonal entries of a nonsquare matrix."
            Array.init m.Rows (fun i -> mm.[i, i])
        | SymmetricMatrix mm -> Array.init m.Rows (fun i -> mm.[i, i])
    /// Returns the LU decomposition of this Matrix. The return values are the LU matrix, pivot indices, and a toggle value indicating the number of row exchanges during the decomposition, which is +1 if the number of exchanges were even, -1 if odd.
    member inline m.GetLUDecomposition() =
        match m with
        | ZeroMatrix z -> ZeroMatrix z, [||], LanguagePrimitives.GenericZero<'T>
        | Matrix _ | SymmetricMatrix _ ->
            if (m.Rows <> m.Cols) then failwith "Cannot compute the LU decomposition of a nonsquare matrix."
            let res = Array2D.copy (m.ToArray2d())
            let perm = Array.init m.Rows (fun i -> i)
            let mutable toggle = LanguagePrimitives.GenericOne<'T>
            for j = 0 to m.Rows - 2 do
                let mutable colmax:'T = abs res.[j, j]
                let mutable prow = j
                for i = j + 1 to m.Rows - 1 do
                    let absresij = abs res.[i, j]
                    if absresij > colmax then
                        colmax <- absresij
                        prow <- i
                if prow <> j then
                    let tmprow = res.[prow, 0..]
                    res.[prow, 0..] <- res.[j, 0..]
                    res.[j, 0..] <- tmprow
                    let tmp = perm.[prow]
                    perm.[prow] <- perm.[j]
                    perm.[j] <- tmp
                    toggle <- -toggle
                for i = j + 1 to m.Rows - 1 do
                    res.[i, j] <- res.[i, j] / res.[j, j]
                    for k = j + 1 to m.Rows - 1 do
                        res.[i, k] <- res.[i, k] - res.[i, j] * res.[j, k]
            Matrix res, perm, toggle
    /// Gets the determinant of this Matrix
    member inline m.GetDeterminant() =
        match m with
        | ZeroMatrix z -> z
        | Matrix _ | SymmetricMatrix _ ->
            if (m.Rows <> m.Cols) then failwith "Cannot compute the determinant of a nonsquare matrix."
            let lu, _, toggle = m.GetLUDecomposition()
            toggle * Array.fold (fun s x -> s * x) LanguagePrimitives.GenericOne<'T> (lu.GetDiagonal())
    /// Gets the inverse of this Matrix
    member inline m.GetInverse() =
        match m with
        | ZeroMatrix z -> ZeroMatrix z
        | Matrix _ | SymmetricMatrix _ ->
            if (m.Rows <> m.Cols) then failwith "Cannot compute the inverse of a nonsquare matrix."
            let res = Array2D.copy (m.ToArray2d())
            let lu, perm, _ = m.GetLUDecomposition()
            let b:'T[] = Array.zeroCreate m.Rows
            for i = 0 to m.Rows - 1 do
                for j = 0 to m.Rows - 1 do
                    if i = perm.[j] then
                        b.[j] <- LanguagePrimitives.GenericOne<'T>
                    else
                        b.[j] <- LanguagePrimitives.GenericZero<'T>
                let x = matrixSolveHelper (lu.ToArray2d()) b
                res.[0.., i] <- x
            Matrix res
    /// Creates a Matrix from the given 2d array `m`
    static member inline Create(m):Matrix<'T> = Matrix m
    /// Creates a Matrix with `m` rows, `n` columns, and all elements having value `v`. If m = n, a SymmetricMatrix is created.
    static member inline Create(m, n, v:'T):Matrix<'T> = if m = n then SymmetricMatrix (Array2D.create m m v) else Matrix (Array2D.create m n v)
    /// Creates a Matrix with `m` rows, `n` columns, and a generator function `f` to compute the elements
    static member inline Create(m:int, n:int, f:int->int->'T):Matrix<'T> = Matrix (Array2D.init m n f)
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
    /// Creates a SymmetricMatrix with `m` rows and columns and a generator function `f` to compute the elements. Function `f` is used only for populating the upper triangular part of the Matrix, the lower triangular part will be the reflection.
    static member inline CreateSymmetric(m, f):Matrix<'T> =
        let s = Array2D.zeroCreate<'T> m m
        for i = 0 to m - 1 do
            for j = i to m - 1 do
                s.[i, j] <- f i j
        SymmetricMatrix s
    /// Creates the identity matrix with `m` rows and columns
    static member inline CreateIdentity(m):Matrix<'T> =
        let s = Array2D.zeroCreate<'T> m m
        for i = 0 to m - 1 do s.[i, i] <- LanguagePrimitives.GenericOne<'T>
        SymmetricMatrix s
    /// ZeroMatrix
    static member inline Zero = ZeroMatrix LanguagePrimitives.GenericZero<'T>
    /// Converts Matrix `m` to float[,]
    static member inline op_Explicit(m:Matrix<'T>) =
        match m with
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
        | Matrix m -> Array2D.map float m
        | SymmetricMatrix m -> copyupper (Array2D.map float m)
    /// Solves a system of linear equations ax = b, where the coefficients are given in Matrix `a` and the result vector is Vector `b`
    static member inline Solve(a:Matrix<'T>, b:Vector<'T>) =
        if a.Cols <> b.Length then invalidArg "b" "Cannot solve the system of equations using a matrix and a vector of incompatible sizes."
        let lu, perm, _ = a.GetLUDecomposition()
        let bp = Array.init a.Rows (fun i -> b.[perm.[i]])
        Vector (matrixSolveHelper (lu.ToArray2d()) bp)
    /// Perfomrs a symmetric unary operation `f` on Matrix `a`
    static member inline SymmetricOp(a:Matrix<'T>, f:int->int->'T):Matrix<'T> =
        match a with
        | ZeroMatrix _ -> Matrix.Zero
        | Matrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
    /// Performs a symmetric binary operation `f` on Matrix `a` and Matrix `b`
    static member inline SymmetricOp(a:Matrix<'T>, b:Matrix<'T>, f:int->int->'T):Matrix<'T> =
        match a, b with
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix _ -> 
            if (b.Rows <> b.Cols) then invalidArg "b" "Cannot perform symmetric binary operation with a nonsquare matrix."
            Matrix.CreateSymmetric(b.Rows, f)
        | ZeroMatrix _ , SymmetricMatrix _ -> Matrix.CreateSymmetric(b.Rows, f)
        | Matrix _, ZeroMatrix _ -> 
            if (a.Rows <> a.Cols) then invalidArg "a" "Cannot perform symmetric binary operation with a nonsquare matrix."
            Matrix.CreateSymmetric(a.Rows, f)
        | Matrix _, Matrix _ -> 
            if (a.Rows <> a.Cols) || (b.Rows <> b.Cols) then invalidArg "a || b" "Cannot perform symmetric binary operation with a nonsquare matrix."
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot perform symmetric binary operation between matrices of different size."
            Matrix.CreateSymmetric(a.Rows, f)
        | Matrix _, SymmetricMatrix _ ->
            if (a.Rows <> a.Cols) then invalidArg "a" "Cannot perform symmetric binary operation with a nonsquare matrix."
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot perform symmetric binary operation between matrices of different size."
            Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, ZeroMatrix _ -> Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, Matrix _ -> 
            if (b.Rows <> b.Cols) then invalidArg "b" "Cannot perform symmetric binary operation with a nonsquare matrix."
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot perform symmetric binary operation between matrices of different size."
            Matrix.CreateSymmetric(a.Rows, f)
        | SymmetricMatrix _, SymmetricMatrix _ -> 
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot perform symmetric binary operation between matrices of different size."
            Matrix.CreateSymmetric(a.Rows, f)
    /// Adds Matrix `a` to Matrix `b`
    static member inline (+) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix mb -> Matrix mb
        | ZeroMatrix _, SymmetricMatrix mb -> SymmetricMatrix mb
        | Matrix ma, ZeroMatrix _ -> Matrix ma
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "b" "Cannot add matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] + mb.[i, j])
        | Matrix ma, SymmetricMatrix _ -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Rows) then invalidArg "b" "Cannot add matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] + b.[i, j])
        | SymmetricMatrix a, ZeroMatrix _ -> SymmetricMatrix a
        | SymmetricMatrix _, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Rows <> b.Cols) then invalidArg "b" "Cannot add matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] + mb.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ -> 
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot add matrices of different size."
            Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] + b.[i, j])
    /// Subtracts Matrix `b` from Matrix `a`
    static member inline (-) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix mb -> Matrix.Create(b.Rows, b.Cols, fun i j -> -mb.[i ,j])
        | ZeroMatrix _, SymmetricMatrix _ -> Matrix.CreateSymmetric(b.Rows, fun i j -> -b.[i ,j])
        | Matrix ma, ZeroMatrix _ -> Matrix ma
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "b" "Cannot subtract matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] - mb.[i, j])
        | Matrix ma, SymmetricMatrix _ -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Rows) then invalidArg "b" "Cannot subtract matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] - b.[i, j])
        | SymmetricMatrix a, ZeroMatrix _ -> SymmetricMatrix a
        | SymmetricMatrix _, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Rows <> b.Cols) then invalidArg "b" "Cannot subtract matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] - mb.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ -> 
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot subtract matrices of different size."
            Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] - b.[i, j])
    /// Multiplies Matrix `a` and Matrix `b` (matrix product)
    static member inline (*) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
        | ZeroMatrix _, SymmetricMatrix _ -> Matrix.Zero
        | Matrix _, ZeroMatrix _ -> Matrix.Zero
        | Matrix ma, Matrix mb ->
            if (a.Cols <> b.Rows) then invalidArg "b" "Cannot multiply two matrices with incompatible sizes."
            Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> ma.[i, k] * mb.[k, j]) [|0..(b.Rows - 1)|] )
        | Matrix ma, SymmetricMatrix _ -> 
            if (a.Cols <> b.Rows) then invalidArg "b" "Cannot multiply two matrices with incompatible sizes."
            Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> ma.[i, k] * b.[k, j]) [|0..(b.Rows - 1)|] )
        | SymmetricMatrix _, ZeroMatrix z -> ZeroMatrix z
        | SymmetricMatrix _, Matrix mb ->
            if (a.Cols <> b.Rows) then invalidArg "b" "Cannot multiply two matrices with incompatible sizes."
            Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> a.[i, k] * mb.[k, j]) [|0..(b.Rows - 1)|] )
        | SymmetricMatrix _, SymmetricMatrix _ -> 
            if (a.Cols <> b.Rows) then invalidArg "b" "Cannot multiply two matrices with incompatible sizes."
            Matrix.Create(a.Rows, b.Cols, fun i j -> Array.sumBy (fun k -> a.[i, k] * b.[k, j]) [|0..(b.Rows - 1)|] )
    /// Multiplies Matrix `a` and Matrix `b` element-wise (Hadamard product)
    static member inline (.*) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
        | ZeroMatrix _, SymmetricMatrix _ -> Matrix.Zero
        | Matrix _, ZeroMatrix _ -> Matrix.Zero
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "b" "Cannot multiply matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] * mb.[i, j])
        | Matrix ma, SymmetricMatrix _ -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Rows) then invalidArg "b" "Cannot multiply matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] * b.[i, j])
        | SymmetricMatrix _, ZeroMatrix z -> ZeroMatrix z
        | SymmetricMatrix _, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Rows <> b.Cols) then invalidArg "b" "Cannot multiply matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] * mb.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ ->
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot multiply matrices of different size."
            Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] * b.[i, j])
    /// Divides Matrix `a` by Matrix `b` element-wise (Hadamard division)
    static member inline (./) (a:Matrix<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroMatrix z -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
        | ZeroMatrix _, SymmetricMatrix _ -> Matrix.Zero
        | Matrix _, ZeroMatrix z -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "b" "Cannot divide matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] / mb.[i, j])
        | Matrix ma, SymmetricMatrix _ -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Rows) then invalidArg "b" "Cannot divide matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] / b.[i, j])
        | SymmetricMatrix _, ZeroMatrix z ->  raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
        | SymmetricMatrix _, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Rows <> b.Cols) then invalidArg "b" "Cannot divide matrices of different size."
            Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] / mb.[i, j])
        | SymmetricMatrix _, SymmetricMatrix _ ->
            if (a.Rows <> b.Rows) then invalidArg "b" "Cannot divide matrices of different size."
            Matrix.CreateSymmetric(a.Rows, fun i j -> a.[i, j] / b.[i, j])
    /// Computes the matrix-vector product of Matrix `a` and Vector `b`
    static member inline (*) (a:Matrix<'T>, b:Vector<'T>) =
        match a, b with
        | ZeroMatrix _, ZeroVector z -> ZeroVector z
        | ZeroMatrix z, Vector _ -> ZeroVector z
        | Matrix _, ZeroVector z -> ZeroVector z
        | Matrix ma, Vector vb ->
            if (a.Cols <> b.Length) then invalidArg "b" "Cannot compute the matrix-vector product of a matrix and a vector with incompatible sizes."
            Vector.Create(a.Rows, fun i -> Array.sumBy (fun j -> ma.[i, j] * vb.[j]) [|0..(b.Length - 1)|] )
        | SymmetricMatrix _, ZeroVector z -> ZeroVector z
        | SymmetricMatrix _, Vector vb ->
            if (a.Cols <> b.Length) then invalidArg "b" "Cannot compute the matrix-vector product of a matrix and a vector with incompatible sizes."
            Vector.Create(a.Rows, fun i -> Array.sumBy (fun j -> a.[i, j] * vb.[j]) [|0..(b.Length - 1)|] )
    /// Computes the vector-matrix product of Vector `a` and Matrix `b`
    static member inline (*) (a:Vector<'T>, b:Matrix<'T>) =
        match a, b with
        | ZeroVector z, ZeroMatrix _ -> ZeroVector z
        | ZeroVector z, Matrix _ -> ZeroVector z
        | ZeroVector z, SymmetricMatrix _ -> ZeroVector z
        | Vector _, ZeroMatrix z -> ZeroVector z
        | Vector va, Matrix mb ->
            if (a.Length <> b.Rows) then invalidArg "b" "Cannot compute the vector-matrix product of a vector and matrix with incompatible sizes."
            Vector.Create(b.Cols, fun i -> Array.sumBy (fun j -> va.[j] * mb.[j, i]) [|0..(a.Length - 1)|])
        | Vector va, SymmetricMatrix _ ->
            if (a.Length <> b.Rows) then invalidArg "b" "Cannot compute the vector-matrix product of a vector and matrix with incompatible sizes."
            Vector.Create(b.Cols, fun i -> Array.sumBy (fun j -> va.[j] * b.[j, i]) [|0..(a.Length - 1)|])
    /// Adds scalar `b` to each element of Matrix `a`
    static member inline (+) (a:Matrix<'T>, b:'T) =
        match a with
        | ZeroMatrix z -> invalidArg "a" "Unsupported operation. Cannot add a scalar to a ZeroMatrix."
        | Matrix ma -> Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] + b)
        | SymmetricMatrix ma -> Matrix.CreateSymmetric(a.Rows, fun i j -> ma.[i, j] + b)
    /// Adds scalar `a` to each element of Matrix `b`
    static member inline (+) (a:'T, b:Matrix<'T>) =
        match b with
        | ZeroMatrix z -> invalidArg "a" "Unsupported operation. Cannot add a scalar to a ZeroMatrix."
        | Matrix mb -> Matrix.Create(b.Rows, b.Cols, fun i j -> a + mb.[i, j])
        | SymmetricMatrix mb -> Matrix.CreateSymmetric(b.Rows, fun i j -> a + mb.[i, j])
    /// Subtracts scalar `b` from each element of Matrix `a`
    static member inline (-) (a:Matrix<'T>, b:'T) =
        match a with
        | ZeroMatrix z -> invalidArg "a" "Unsupported operation. Cannot subtract a scalar from a ZeroMatrix."
        | Matrix ma -> Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] - b)
        | SymmetricMatrix ma -> Matrix.CreateSymmetric(a.Rows, fun i j -> ma.[i, j] - b)
    /// Subtracts each element of of Matrix `b` from scalar `a`
    static member inline (-) (a:'T, b:Matrix<'T>) =
        match b with
        | ZeroMatrix z -> invalidArg "a" "Unsupported operation. Cannot subtract a ZeroMatrix from a scalar."
        | Matrix mb -> Matrix.Create(b.Rows, b.Cols, fun i j -> a - mb.[i, j])
        | SymmetricMatrix mb -> Matrix.CreateSymmetric(b.Rows, fun i j -> a - mb.[i, j])
    /// Multiplies each element of Matrix `a` by scalar `b`
    static member inline (*) (a:Matrix<'T>, b:'T) =
        match a with
        | ZeroMatrix _ -> Matrix.Zero
        | Matrix ma -> Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] * b)
        | SymmetricMatrix ma -> Matrix.CreateSymmetric(a.Rows, fun i j -> ma.[i, j] * b)
    /// Multiplies each element of Matrix `b` by scalar `a`
    static member inline (*) (a:'T, b:Matrix<'T>) =
        match b with
        | ZeroMatrix _ -> Matrix.Zero
        | Matrix mb -> Matrix.Create(b.Rows, b.Cols, fun i j -> a * mb.[i, j])
        | SymmetricMatrix mb -> Matrix.CreateSymmetric(b.Rows, fun i j -> a * mb.[i, j])
    /// Divides each element of Matrix `a` by scalar `b`
    static member inline (/) (a:Matrix<'T>, b:'T) =
        match a with
        | ZeroMatrix _ -> Matrix.Zero
        | Matrix ma -> Matrix.Create(a.Rows, a.Cols, fun i j -> ma.[i, j] / b)
        | SymmetricMatrix ma -> Matrix.CreateSymmetric(a.Rows, fun i j -> ma.[i, j] / b)
    /// Creates a Matrix whose elements are scalar `a` divided by each element of Matrix `b`
    static member inline (/) (a:'T, b:Matrix<'T>) =
        match b with
        | ZeroMatrix _ -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
        | Matrix mb -> Matrix.Create(b.Rows, b.Cols, fun i j -> a / mb.[i, j])
        | SymmetricMatrix mb -> Matrix.CreateSymmetric(b.Rows, fun i j -> a / mb.[i, j])
    /// Gets the negative of Matrix `a`
    static member inline (~-) (a:Matrix<'T>) =
        match a with
        | ZeroMatrix _ -> Matrix.Zero
        | Matrix ma -> Matrix.Create(a.Rows, a.Cols, fun i j -> -ma.[i, j])
        | SymmetricMatrix ma -> Matrix.CreateSymmetric(a.Rows, fun i j -> -ma.[i, j])

/// Provides basic operations on Vector types. (Implementing functionality similar to Microsoft.FSharp.Collections.Array)
module Vector =
    let inline create (n:int) (v:'T) = Vector.Create(n, v)
    /// Creates a Vector with dimension `n` and a generator function `f` to compute the eleements
    let inline init (n:int) (f:int->'T) = Vector.Create(n, f)
    let inline length (v:Vector<_>) = v.Length
    /// Creates a Vector from sequence `s`
    let inline ofSeq (s:seq<_>) = Vector.Create(Array.ofSeq s)
    /// Returns the sum of all the elements in Vector `v`
    let inline sum (v:Vector<_>) = 
        match v with
        | ZeroVector z -> z
        | Vector v -> Array.sum v
    /// Converts Vector `v` to an array, e.g. from Vector<float> to float[]
    let inline toArray (v:Vector<_>) =
        match v with
        | ZeroVector _ -> [||]
        | Vector v -> v

/// Provides basic operations on Matrix types. (Implementing functionality similar to Microsoft.FSharp.Collections.Array2D)
module Matrix =
    let inline create (m:int) (n:int) (v:'T) = Matrix.Create(m, n, v)
    let inline init (m:int) (n:int) (f:int->int->'T) = Matrix.Create(m, n, f)
    let inline length1 (m:Matrix<_>) = m.Rows
    let inline length2 (m:Matrix<_>) = m.Cols
    /// Creates a Matrix from 2d array `m`
    let inline ofArray2d (m:'T[,]) = Matrix.Create(m)
    /// Creates a Matrix from sequence `s`
    let inline ofSeq (s:seq<seq<'T>>) =
        let a = Array.ofSeq s
        let b = Array.ofSeq a
        let c = array2D b
        Matrix.Create(c)
    /// Converts Matrix `m` to a 2d array, e.g. from Matrix<float> to float[,]
    let inline toArray2d (m:Matrix<_>) =
        match m with
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
        | Matrix m -> m
        | SymmetricMatrix m -> copyupper m    /// Converts Matrix `m` to a jagged array, e.g. from Matrix<float> to float[][]
    let inline toArray (m:Matrix<_>) =
        let a = toArray2d m
        [|for i = 0 to m.Rows - 1 do yield [|for j = 0 to m.Cols - 1 do yield a.[i, j]|]|]


/// Linear algebra operations module (automatically opened)
[<AutoOpen>]
module LinearAlgebraOps =
    /// Converts array, list, or sequence `v` into a Vector
    let inline vector v = Vector.ofSeq v
    /// Gets the Euclidean norm of Vector `v`
    let inline norm (v:Vector<_>) = v.GetNorm()
    /// Gets the unit vector codirectional with Vector `v`
    let inline unitVector (v:Vector<_>) = v.GetUnitVector()
    /// Converts 2d array `m` into a Matrix
    let inline matrix m = Matrix.ofSeq m
    /// Converts Vector `v` into array
    let inline array (v:Vector<_>) = Vector.toArray v
    /// Converts Matrix `m` into a 2d array
    let inline array2d (m:Matrix<_>) = Matrix.toArray2d m
    /// Gets the trace of Matrix `m`
    let inline trace (m:Matrix<_>) = m.GetTrace()
    /// Gets the transpose of Matrix `m`
    let inline transpose (m:Matrix<_>) = m.GetTranspose()
    /// Gets the determinant of Matrix `m`
    let inline det (m:Matrix<_>) = m.GetDeterminant()
    /// Gets the inverse of Matrix `m`
    let inline inverse (m:Matrix<_>) = m.GetInverse()
