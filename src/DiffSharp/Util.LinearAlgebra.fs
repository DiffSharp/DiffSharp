//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
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
//   Brain and Computation Lab
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

#light

/// Generic Vector and Matrix types
module DiffSharp.Util.LinearAlgebra

open DiffSharp.Util.General


/// Generic vector type
[<NoEquality; NoComparison>]
type Vector<'T when 'T : (static member Zero : 'T)
                and 'T : (static member One : 'T)
                and 'T : (static member (+) : 'T * 'T -> 'T)
                and 'T : (static member (-) : 'T * 'T -> 'T)
                and 'T : (static member (*) : 'T * 'T -> 'T)
                and 'T : (static member (/) : 'T * 'T -> 'T)
                and 'T : (static member (~-) : 'T -> 'T)
                and 'T : (static member Abs : 'T -> 'T)
                and 'T : (static member Pow : 'T * 'T -> 'T)
                and 'T : (static member Sqrt : 'T -> 'T)
                and 'T : (static member op_Explicit : 'T -> float)
                and 'T : comparison> =
    | ZeroVector of 'T
    | Vector of 'T[]
    /// ZeroVector
    static member inline Zero = ZeroVector LanguagePrimitives.GenericZero<'T>
    /// Converts Vector `v` to float[]
    static member inline op_Explicit(v:Vector<'T>) =
        match v with
        | Vector v -> Array.map float v
        | ZeroVector _ -> [||]
    /// Gets the element of this Vector at the given position `i`
    member inline v.Item
        with get i =
            match v with
            | Vector v -> v.[i]
            | ZeroVector z -> z
    /// Gets the first element of this Vector
    member inline v.FirstItem =
        match v with
        | Vector v -> v.[0]
        | ZeroVector z -> z
    /// Gets the total number of elements of this Vector
    member inline v.Length =
        match v with
        | Vector v -> v.Length
        | ZeroVector _ -> 0
    /// Gets the L1 (Manhattan) norm of this Vector
    member inline v.GetL1Norm() =
        match v with
        | Vector v -> Array.sumBy abs v
        | ZeroVector z -> z
    /// Gets the L2 (Euclidean) norm of this Vector
    member inline v.GetL2Norm() =
        match v with
        | Vector v -> sqrt (Array.sumBy (fun x -> x * x) v)
        | ZeroVector z -> z
    /// Gets the squared L2 (Euclidean) norm of this Vector
    member inline v.GetL2NormSq() =
        match v with
        | Vector v -> Array.sumBy (fun x -> x * x) v
        | ZeroVector z -> z
    /// Gets the Lp norm (or p-norm) of this Vector, with the given `p`
    member inline v.GetLPNorm(p:'T):'T =
        match v with
        | Vector v -> (Array.sumBy (fun x -> (abs x) ** p) v) ** (LanguagePrimitives.GenericOne<'T> / p)
        | ZeroVector z -> z
    /// Gets the minimum element of this Vector
    member inline v.GetMin() =
        match v with
        | Vector v -> Array.min v
        | ZeroVector z -> z
    /// Gets the minimum element of this Vector, compared by using Operators.min on the result of function `f`
    member inline v.GetMinBy(f) =
        match v with
        | Vector v -> Array.minBy f v
        | ZeroVector z -> z
    /// Gets the maximum element of this Vector
    member inline v.GetMax() =
        match v with
        | Vector v -> Array.max v
        | ZeroVector z -> z
    /// Gets the maximum element of this Vector, compared by using Operators.max on the result of function `f`
    member inline v.GetMaxBy(f) =
        match v with
        | Vector v -> Array.maxBy f v
        | ZeroVector z -> z
    /// Gets the unit Vector codirectional with this Vector
    member inline v.GetUnitVector() =
        match v with
        | Vector vv -> let n = v.GetL2Norm() in Vector (Array.map (fun x -> x / n) vv)
        | ZeroVector z -> ZeroVector z
    /// Returns a sequence of Vectors that are obtained by splitting this Vector into `n` subvectors of equal length. The length of this Vector must be an integer multiple of `n`, otherwise ArgumentException is raised.
    member inline v.Split(n:int) =
        match v with
        | Vector v ->
            if n <= 0 then invalidArg "" "For splitting this Vector, n should be a positive integer."
            let l = (float v.Length) / (float n)
            if not (isInteger l) then invalidArg "" "Cannot split Vector into n equal pieces when length of Vector is not an integer multiple of n."
            seq {for i in 0 .. (int l) .. (v.Length - 1) do yield Vector (Array.sub v i (int l))}
        | ZeroVector _ -> seq {yield v}
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
        | Vector v -> Vector (Array.map f v)
        | ZeroVector _ -> ZeroVector LanguagePrimitives.GenericZero<'a>
    /// Creates a copy of this Vector
    member inline v.Copy() =
        match v with
        | Vector v -> Vector (Array.copy v)
        | ZeroVector z -> ZeroVector z
    /// Converts this Vector to an array
    member inline v.ToArray() =
        match v with
        | Vector v -> v
        | ZeroVector _ -> [||]
    /// Converts this Vector to a sequence
    member inline v.ToSeq() =
        match v with
        | Vector v -> Array.toSeq v
        | ZeroVector _ -> Seq.empty
    /// Creates a new Vector that contains the given subrange of elements, specified by start index `s` and count `c`
    member inline v.GetSubVector(s, c) =
        match v with
        | Vector v -> Vector (Array.sub v s c)
        | ZeroVector _ -> Vector.Zero
    /// Adds Vector `a` to Vector `b`
    static member inline (+) (a:Vector<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Vector a, Vector b -> try Vector (Array.map2 (+) a b) with | _ -> invalidArg "" "Cannot add two Vectors of different dimensions."
        | Vector _, ZeroVector _ -> a
        | ZeroVector _, Vector _ -> b
        | ZeroVector _, ZeroVector _ -> Vector.Zero
    /// Subtracts Vector `b` from Vector `a`
    static member inline (-) (a:Vector<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Vector a, Vector b -> try Vector (Array.map2 (-) a b) with | _ -> invalidArg "" "Cannot subtract two Vectors of different dimensions."
        | Vector _, ZeroVector _ -> a
        | ZeroVector _, Vector b -> Vector (Array.map (~-) b)
        | ZeroVector _, ZeroVector _ -> Vector.Zero
    /// Computes the inner product (dot / scalar product) of Vector `a` and Vector `b`
    static member inline (*) (a:Vector<'T>, b:Vector<'T>):'T =
        match a, b with
        | Vector a, Vector b -> try Array.map2 (*) a b |> Array.sum with | _ -> invalidArg "" "Cannot multiply two Vectors of different dimensions."
        | Vector _, ZeroVector _ -> LanguagePrimitives.GenericZero<'T>
        | ZeroVector _, Vector _ -> LanguagePrimitives.GenericZero<'T>
        | ZeroVector _, ZeroVector _ -> LanguagePrimitives.GenericZero<'T>
    /// Computes the cross product of Vector `a` and Vector `b` (three-dimensional)
    static member inline (%*) (a:Vector<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Vector va, Vector vb ->
            if (a.Length <> 3) || (b.Length <> 3) then invalidArg "" "The cross product is only defined for three-dimensional vectors."
            Vector [|va.[1] * vb.[2] - va.[2] * vb.[1]; va.[2] * vb.[0] - va.[0] * vb.[2]; va.[0] * vb.[1] - va.[1] * vb.[0]|]
        | Vector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector _ -> Vector.Zero
        | ZeroVector _, ZeroVector _ -> Vector.Zero
    /// Multiplies Vector `a` and Vector `b` element-wise (Hadamard product)
    static member inline (.*) (a:Vector<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Vector a, Vector b -> try Vector (Array.map2 (*) a b) with | _ -> invalidArg "" "Cannot multiply two Vectors of different dimensions."
        | Vector _, ZeroVector _ -> Vector.Zero
        | ZeroVector _, Vector _ -> Vector.Zero
        | ZeroVector _, ZeroVector _ -> Vector.Zero
    /// Divides Vector `a` by Vector `b` element-wise (Hadamard division)
    static member inline (./) (a:Vector<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Vector a, Vector b -> try Vector (Array.map2 (/) a b) with | _ -> invalidArg "" "Cannot divide two Vectors of different dimensions."
        | Vector _, ZeroVector _-> raise (new System.DivideByZeroException("Attempted to divide a Vector by a ZeroVector."))
        | ZeroVector _, Vector _ -> Vector.Zero
        | ZeroVector _, ZeroVector _ -> raise (new System.DivideByZeroException("Attempted to divide a ZeroVector by a ZeroVector."))
    /// Adds scalar `b` to each element of Vector `a`
    static member inline (+) (a:Vector<'T>, b:'T):Vector<'T> =
        match a with
        | Vector a -> Vector (Array.map ((+) b) a)
        | ZeroVector _ -> invalidArg "" "Unsupported operation. Cannot add a scalar to a ZeroVector."
    /// Adds scalar `a` to each element of Vector `b`
    static member inline (+) (a:'T, b:Vector<'T>):Vector<'T> =
        match b with
        | Vector b -> Vector (Array.map ((+) a) b)
        | ZeroVector _ -> invalidArg "" "Unsupported operation. Cannot add a scalar to a ZeroVector."
    /// Subtracts scalar `b` from each element of Vector `a`
    static member inline (-) (a:Vector<'T>, b:'T):Vector<'T> =
        match a with
        | Vector a -> Vector (Array.map (fun x -> x - b) a)
        | ZeroVector _ -> invalidArg "" "Unsupported operation. Cannot subtract a scalar from a ZeroVector."
    /// Subtracts each element of Vector `b` from scalar `a`
    static member inline (-) (a:'T, b:Vector<'T>):Vector<'T> =
        match b with
        | Vector b -> Vector (Array.map ((-) a) b)
        | ZeroVector _ -> invalidArg "" "Unsupported operation. Cannot add subtract a ZeroVector from a scalar."
    /// Multiplies each element of Vector `a` by scalar `b`
    static member inline (*) (a:Vector<'T>, b:'T):Vector<'T> =
        match a with
        | Vector a -> Vector (Array.map ((*) b) a)
        | ZeroVector _ -> Vector.Zero
    /// Multiplies each element of Vector `b` by scalar `a`
    static member inline (*) (a:'T, b:Vector<'T>):Vector<'T> =
        match b with
        | Vector b -> Vector (Array.map ((*) a) b)
        | ZeroVector _ -> Vector.Zero
    /// Divides each element of Vector `a` by scalar `b`
    static member inline (/) (a:Vector<'T>, b:'T):Vector<'T> =
        match a with
        | Vector a -> Vector (Array.map (fun x -> x / b) a)
        | ZeroVector _ -> Vector.Zero
    /// Divides scalar `a` by each element of Vector `b`
    static member inline (/) (a:'T, b:Vector<'T>):Vector<'T> =
        match b with
        | Vector b -> Vector (Array.map ((/) a) b)
        | ZeroVector _ -> raise (new System.DivideByZeroException("Attempted division by a ZeroVector."))
    /// Gets the negative of Vector `a`
    static member inline (~-) (a:Vector<'T>):Vector<'T> =
        match a with
        | Vector a -> Vector (Array.map (~-) a)
        | ZeroVector _ -> Vector.Zero


/// Provides basic operations on Vector types. (Implementing functionality similar to Microsoft.FSharp.Collections.Array)
[<RequireQualifiedAccess>]
module Vector =
    /// Creates a Vector from sequence `s`
    let inline ofSeq s = Vector (Array.ofSeq s)
    /// Converts Vector `v` to an array
    let inline toArray (v:Vector<_>) = v.ToArray()
    /// Returns Vector `v` as a sequence
    let inline toSeq (v:Vector<_>) = v.ToSeq()
    /// Builds a new Vector that contains the elements of each of the given sequence of Vectors `v`
    let inline concat (v:seq<Vector<_>>) = Seq.map toArray v |> Array.concat |> ofSeq
    /// Creates a copy of Vector `v`
    let inline copy (v:Vector<_>) = v.Copy()
    /// Creates a Vector with `n` elements, all having value `v`
    let inline create n v = Vector (Array.create n v)
    /// Creates a Vector with `n` elements, where the element with index `i` has value `v` and the rest of the elements have value 0
    let inline createBasis n i v = Vector (Array.init n (fun j -> if j = i then v else LanguagePrimitives.GenericZero))
    /// Tests if any element of Vector `v` satisfies predicate `p`
    let inline exists p (v:Vector<_>) = v |> toArray |> Array.exists p
    /// Returns the first element of Vector `v` for which predicate `p` is true
    let inline find p (v:Vector<_>) = v |> toArray |> Array.find p
    /// Returns the index of the first element of Vector `v` for which predicate `p` is true
    let inline findIndex p (v:Vector<_>) = v |> toArray |> Array.findIndex p
    /// Applies function `f` to each element of Vector `v`, threading an accumulator (with initial state `s`) through the computation. If the input function is f and the elements are i0...iN then computes f (... (f s i0)...) iN.
    let inline fold f s (v:Vector<_>) = v |> toArray |> Array.fold f s
    /// Applies function `f` to each element of Vector `v`, threading an accumulator (with initial state `s`) through the computation. If the input function is f and the elements are i0...iN then computes f i0 (...(f iN s)).
    let inline foldBack f s (v:Vector<_>) = v |> toArray |> Array.foldBack f s
    /// Tests if all elements of Vector `v` satisfy predicate `p`
    let inline forall p (v:Vector<_>) = v |> toArray |> Array.forall p
    /// Creates a Vector with dimension `n` and a generator function `f` to compute the elements
    let inline init n f = Vector (Array.init n f)
    /// Applies function `f` to each element of Vector `v`
    let inline iter f (v:Vector<_>) = v |> toArray |> Array.iter f
    /// Applies function `f` to each element of Vector `v`. The integer passed to function `f` indicates the index of element.
    let inline iteri f (v:Vector<_>) = v |> toArray |> Array.iteri f
    /// Gets the L1 (Manhattan) norm of Vector `v`
    let inline l1norm (v:Vector<_>) = v.GetL1Norm()
    /// Gets the L2 (Euclidean) norm of Vector `v`. This is the same with `Vector.norm`.
    let inline l2norm (v:Vector<_>) = v.GetL2Norm()
    /// Gets the squared L2 (Euclidean) norm of Vector `v`. This is the same with `Vector.normSq`.
    let inline l2normSq (v:Vector<_>) = v.GetL2NormSq()
    /// Returns the length of Vector `v`
    let inline length (v:Vector<_>) = v.Length
    /// Gets the Lp norm (or p-norm) of Vector `v`, with the given `p`
    let inline lpnorm p (v:Vector<_>) = v.GetLPNorm(p)
    /// Creates a Vector whose elements are the results of applying function `f` to each element of Vector `v`
    let inline map f (v:Vector<_>) = v |> toArray |> Array.map f |> Vector
    /// Creates a Vector whose elements are the results of applying function `f` to each element of Vector `v`. An element index is also supplied to function `f`.
    let inline mapi f (v:Vector<_>) = v |> toArray |> Array.mapi f |> Vector
    /// Returns the maximum of all elements of Vector `v`
    let inline max (v:Vector<_>) = v.GetMax()
    /// Returns the maximum of all elements of Vector `v`, compared by using Operators.max on the result of function `f`
    let inline maxBy f (v:Vector<_>) = v.GetMaxBy(f)
    /// Returns the minimum of all elements of Vector `v`
    let inline min (v:Vector<_>) = v.GetMin()
    /// Returns the minimum of all elements of Vector `v`, compared by using Operators.min on the result of function `f`
    let inline minBy f (v:Vector<_>) = v.GetMinBy(f)
    /// Gets the L2 (Euclidean) norm of Vector `v`. This is the same with `Vector.l2norm`.
    let inline norm v = l2norm v
    /// Gets the squared L2 (Euclidean) norm of Vector `v`. This is the same with `Vector.l2normSq`.
    let inline normSq v = l2normSq v
    /// Applies function `f` to each element of Vector `v`, threading an accumulator argument through the computation. If the input function is f and the elements are i0...iN, then computes f (... (f i0 i1)...) iN.
    let inline reduce f (v:Vector<_>) = v |> toArray |> Array.reduce f
    /// Applies function `f` to each element of Vector `v`, threading an accumulator argument through the computation. If the input function is f and the elements are i0...iN then computes f i0 (...(f iN-1 iN)).
    let inline reduceBack f (v:Vector<_>) = v |> toArray |> Array.reduceBack f
    /// Like Vector.fold, but returns the intermediate and final results
    let inline scan f s (v:Vector<_>) = v |> toArray |> Array.scan f s
    /// Like Vector.foldBack, but returns both the intermediate and final results
    let inline scanBack f s (v:Vector<_>) = v |> toArray |> Array.scanBack f s
    /// Returns a sequence of Vectors that are obtained by splitting Vector `v` into `n` subvectors of equal length. The length of Vector `v` must be an integer multiple of `n`, otherwise ArgumentException is raised.
    let inline split n (v:Vector<_>) = v.Split(n)
    /// Creates a Vector with `n` elements, where the `i`-th element is 1 and the rest of the elements are 0
    let inline standardBasis n i = createBasis n i LanguagePrimitives.GenericOne
    /// Creates a new Vector that contains the given subrange of Vector `v`, specified by start index `s` and count `c`
    let inline sub (v:Vector<_>) s c = v.GetSubVector(s, c)
    /// Returns the sum of all the elements in Vector `v`
    let inline sum (v:Vector<_>) = v |> toArray |> Array.sum
    /// Returns the sum of the results generated by applying function `f` to each element of Vector `v`
    let inline sumBy f (v:Vector<_>) = v |> toArray |> Array.sumBy f
    /// Gets the unit vector codirectional with Vector `v`
    let inline unitVector (v:Vector<_>) = v.GetUnitVector()


/// Generic matrix type
[<NoEquality; NoComparison>]
type Matrix<'T when 'T : (static member Zero : 'T)
                and 'T : (static member One : 'T)
                and 'T : (static member (+) : 'T * 'T -> 'T)
                and 'T : (static member (-) : 'T * 'T -> 'T)
                and 'T : (static member (*) : 'T * 'T -> 'T)
                and 'T : (static member (/) : 'T * 'T -> 'T)
                and 'T : (static member (~-) : 'T -> 'T)
                and 'T : (static member Abs : 'T -> 'T)
                and 'T : (static member Pow : 'T * 'T -> 'T)
                and 'T : (static member Sqrt : 'T -> 'T)
                and 'T : (static member op_Explicit : 'T -> float)
                and 'T : comparison> =
    | ZeroMatrix of 'T
    | Matrix of 'T[,]
    /// ZeroMatrix
    static member inline Zero = ZeroMatrix LanguagePrimitives.GenericZero<'T>
    /// Converts Matrix `m` to float[,]
    static member inline op_Explicit(m:Matrix<'T>) =
        match m with
        | Matrix m -> Array2D.map float m
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
    /// Gets the number of rows of this Matrix
    member inline m.Rows =
        match m with
        | Matrix m -> m.GetLength 0
        | ZeroMatrix _ -> 0
    /// Gets the number of columns of thisMatrix
    member inline m.Cols =
        match m with
        | Matrix m -> m.GetLength 1
        | ZeroMatrix _ -> 0
    /// Gets the entry of this Matrix at row `i` and column `j`
    member inline m.Item
        with get (i, j) =
            match m with
            | Matrix m -> m.[i, j]
            | ZeroMatrix z -> z
    /// Gets a submatrix of this Matrix with the bounds given in `rowStart`, `rowFinish`, `colStart`, `colFinish`
    member inline m.GetSlice(rowStart, rowFinish, colStart, colFinish) =
        match m with
        | Matrix mm ->
            let rowStart = defaultArg rowStart 0
            let rowFinish = defaultArg rowFinish (m.Rows - 1)
            let colStart = defaultArg colStart 0
            let colFinish = defaultArg colFinish (m.Cols - 1)
            Matrix mm.[rowStart..rowFinish, colStart..colFinish]
        | ZeroMatrix z -> invalidArg "" "Cannot get slice of a ZeroMatrix."
    /// Gets a row subvector of this Matrix with the given row index `row` and column bounds `colStart` and `colFinish`
    member inline m.GetSlice(row, colStart, colFinish) =
        match m with
        | Matrix mm ->
            let colStart = defaultArg colStart 0
            let colFinish = defaultArg colFinish (m.Cols - 1)
            Vector mm.[row, colStart..colFinish]
        | ZeroMatrix z -> invalidArg "" "Cannot get slice of a ZeroMatrix."
    /// Gets a column subvector of this Matrix with the given column index `col` and row bounds `rowStart` and `rowFinish`
    member inline m.GetSlice(rowStart, rowFinish, col) =
        match m with
        | Matrix mm ->
            let rowStart = defaultArg rowStart 0
            let rowFinish = defaultArg rowFinish (m.Rows - 1)
            Vector mm.[rowStart..rowFinish, col]
        | ZeroMatrix z -> invalidArg "" "Cannot get slice of a ZeroMatrix."
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
    member inline m.ToArray2D() =
        match m with
        | Matrix m -> m
        | ZeroMatrix _ -> Array2D.zeroCreate 0 0
    /// Converts this Matrix into a jagged array, e.g. from Matrix<float> to float[][]
    member inline m.ToArray() =
        let a = m.ToArray2D()
        [|for i = 0 to m.Rows - 1 do yield [|for j = 0 to m.Cols - 1 do yield a.[i, j]|]|]
    /// Creates a copy of this Matrix
    member inline m.Copy() = 
        match m with
        | Matrix m -> Matrix (Array2D.copy m)
        | ZeroMatrix z -> ZeroMatrix z
    /// Gets the trace of this Matrix
    member inline m.GetTrace() =
        match m with
        | Matrix m -> trace m
        | ZeroMatrix z -> z
    /// Gets the transpose of this Matrix
    member inline m.GetTranspose() =
        match m with
        | Matrix m -> Matrix (transpose m)
        | ZeroMatrix z -> ZeroMatrix z
    /// Gets a Vector of the diagonal elements of this Matrix
    member inline m.GetDiagonal() =
        match m with
        | Matrix mm -> 
            if m.Rows <> m.Cols then invalidArg "" "Cannot get the diagonal entries of a nonsquare matrix."
            Array.init m.Rows (fun i -> mm.[i, i])
        | ZeroMatrix z -> [||]
    /// Returns the LU decomposition of this Matrix. The return values are the LU matrix, pivot indices, and a toggle value indicating the number of row exchanges during the decomposition, which is +1 if the number of exchanges were even, -1 if odd.
    member inline m.GetLUDecomposition() =
        match m with
        | Matrix mm ->
            if (m.Rows <> m.Cols) then invalidArg "" "Cannot compute the LU decomposition of a nonsquare matrix."
            let res = Array2D.copy mm
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
        | ZeroMatrix z -> ZeroMatrix z, [||], LanguagePrimitives.GenericZero<'T>
    /// Gets the determinant of this Matrix
    member inline m.GetDeterminant() =
        match m with
        | Matrix _ ->
            if (m.Rows <> m.Cols) then invalidArg "" "Cannot compute the determinant of a nonsquare matrix."
            let lu, _, toggle = m.GetLUDecomposition()
            toggle * Array.fold (fun s x -> s * x) LanguagePrimitives.GenericOne<'T> (lu.GetDiagonal())
        | ZeroMatrix z -> z
    /// Gets the inverse of this Matrix
    member inline m.GetInverse() =
        match m with
        | Matrix mm ->
            if (m.Rows <> m.Cols) then invalidArg "" "Cannot compute the inverse of a nonsquare matrix."
            let res = Array2D.copy mm
            let lu, perm, _ = m.GetLUDecomposition()
            let b:'T[] = Array.zeroCreate m.Rows
            for i = 0 to m.Rows - 1 do
                for j = 0 to m.Rows - 1 do
                    if i = perm.[j] then
                        b.[j] <- LanguagePrimitives.GenericOne<'T>
                    else
                        b.[j] <- LanguagePrimitives.GenericZero<'T>
                let x = matrixSolveHelper (lu.ToArray2D()) b
                res.[0.., i] <- x
            Matrix res
        | ZeroMatrix z -> ZeroMatrix z
    /// Adds Matrix `a` to Matrix `b`
    static member inline (+) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "" "Cannot add matrices of different sizes."
            Matrix (Array2D.init a.Rows a.Cols (fun i j -> ma.[i, j] + mb.[i, j]))
        | Matrix _, ZeroMatrix _ -> a
        | ZeroMatrix _, Matrix _ -> b
        | ZeroMatrix z, ZeroMatrix _ -> ZeroMatrix z
    /// Subtracts Matrix `b` from Matrix `a`
    static member inline (-) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "" "Cannot subtract matrices of different sizes."
            Matrix (Array2D.init a.Rows a.Cols (fun i j -> ma.[i, j] - mb.[i, j]))
        | Matrix _, ZeroMatrix _ -> a
        | ZeroMatrix _, Matrix b -> Matrix (Array2D.map (~-) b)
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
    /// Multiplies Matrix `a` and Matrix `b` (matrix product)
    static member inline (*) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb ->
            if (a.Cols <> b.Rows) then invalidArg "" "Cannot multiply two matrices of incompatible sizes."
            Matrix (Array2D.init a.Rows b.Cols (fun i j -> Array.sumBy (fun k -> ma.[i, k] * mb.[k, j]) [|0..(b.Rows - 1)|] ))
        | Matrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
    /// Multiplies Matrix `a` and Matrix `b` element-wise (Hadamard product)
    static member inline (.*) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "" "Cannot multiply matrices of different sizes."
            Matrix (Array2D.init a.Rows a.Cols (fun i j -> ma.[i, j] * mb.[i, j]))
        | Matrix _, ZeroMatrix _ -> Matrix.Zero
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
        | ZeroMatrix _, ZeroMatrix _ -> Matrix.Zero
    /// Divides Matrix `a` by Matrix `b` element-wise (Hadamard division)
    static member inline (./) (a:Matrix<'T>, b:Matrix<'T>):Matrix<'T> =
        match a, b with
        | Matrix ma, Matrix mb -> 
            if (a.Rows <> b.Rows) || (a.Cols <> b.Cols) then invalidArg "" "Cannot divide matrices of different sizes."
            Matrix (Array2D.init a.Rows a.Cols (fun i j -> ma.[i, j] / mb.[i, j]))
        | Matrix _, ZeroMatrix _ -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
        | ZeroMatrix _, Matrix _ -> Matrix.Zero
        | ZeroMatrix _, ZeroMatrix _ -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
    /// Computes the matrix-vector product of Matrix `a` and Vector `b`
    static member inline (*) (a:Matrix<'T>, b:Vector<'T>):Vector<'T> =
        match a, b with
        | Matrix ma, Vector vb ->
            if (a.Cols <> b.Length) then invalidArg "" "Cannot compute the matrix-vector product of a matrix and a vector of incompatible sizes."
            Vector (Array.init a.Rows (fun i -> Array.sumBy (fun j -> ma.[i, j] * vb.[j]) [|0..(b.Length - 1)|] ))
        | Matrix _, ZeroVector _ -> Vector.Zero
        | ZeroMatrix _, Vector _ -> Vector.Zero
        | ZeroMatrix _, ZeroVector _ -> Vector.Zero
    /// Computes the vector-matrix product of Vector `a` and Matrix `b`
    static member inline (*) (a:Vector<'T>, b:Matrix<'T>):Vector<'T> =
        match a, b with
        | Vector va, Matrix mb ->
            if (a.Length <> b.Rows) then invalidArg "" "Cannot compute the vector-matrix product of a vector and matrix of incompatible sizes."
            Vector (Array.init b.Cols (fun i -> Array.sumBy (fun j -> va.[j] * mb.[j, i]) [|0..(a.Length - 1)|]))
        | Vector _, ZeroMatrix _ -> Vector.Zero
        | ZeroVector _, Matrix _ -> Vector.Zero
        | ZeroVector _, ZeroMatrix _ -> Vector.Zero
    /// Adds scalar `b` to each element of Matrix `a`
    static member inline (+) (a:Matrix<'T>, b:'T):Matrix<'T> =
        match a with
        | Matrix a -> Matrix (Array2D.map ((+) b) a)
        | ZeroMatrix z -> invalidArg "" "Unsupported operation. Cannot add a scalar to a ZeroMatrix."
    /// Adds scalar `a` to each element of Matrix `b`
    static member inline (+) (a:'T, b:Matrix<'T>):Matrix<'T> =
        match b with
        | Matrix b -> Matrix (Array2D.map ((+) a) b)
        | ZeroMatrix z -> invalidArg "" "Unsupported operation. Cannot add a scalar to a ZeroMatrix."
    /// Subtracts scalar `b` from each element of Matrix `a`
    static member inline (-) (a:Matrix<'T>, b:'T):Matrix<'T> =
        match a with
        | Matrix a -> Matrix (Array2D.map (fun x -> x - b) a)
        | ZeroMatrix z -> invalidArg "" "Unsupported operation. Cannot subtract a scalar from a ZeroMatrix."
    /// Subtracts each element of of Matrix `b` from scalar `a`
    static member inline (-) (a:'T, b:Matrix<'T>):Matrix<'T> =
        match b with
        | Matrix b -> Matrix (Array2D.map ((-) a) b)
        | ZeroMatrix z -> invalidArg "" "Unsupported operation. Cannot subtract a ZeroMatrix from a scalar."
    /// Multiplies each element of Matrix `a` by scalar `b`
    static member inline (*) (a:Matrix<'T>, b:'T):Matrix<'T> =
        match a with
        | Matrix a -> Matrix (Array2D.map ((*) b) a)
        | ZeroMatrix _ -> Matrix.Zero
    /// Multiplies each element of Matrix `b` by scalar `a`
    static member inline (*) (a:'T, b:Matrix<'T>):Matrix<'T> =
        match b with
        | Matrix b -> Matrix (Array2D.map ((*) a) b)
        | ZeroMatrix _ -> Matrix.Zero
    /// Divides each element of Matrix `a` by scalar `b`
    static member inline (/) (a:Matrix<'T>, b:'T):Matrix<'T> =
        match a with
        | Matrix a -> Matrix (Array2D.map (fun x -> x / b) a)
        | ZeroMatrix _ -> Matrix.Zero
    /// Creates a Matrix whose elements are scalar `a` divided by each element of Matrix `b`
    static member inline (/) (a:'T, b:Matrix<'T>):Matrix<'T> =
        match b with
        | Matrix b -> Matrix (Array2D.map ((/) a) b)
        | ZeroMatrix _ -> raise (new System.DivideByZeroException("Attempted division by a ZeroMatrix."))
    /// Gets the negative of Matrix `a`
    static member inline (~-) (a:Matrix<'T>) =
        match a with
        | Matrix a -> Matrix (Array2D.map (~-) a)
        | ZeroMatrix _ -> Matrix.Zero
    /// Returns the QR decomposition of this Matrix
    member inline m.GetQRDecomposition() =
        match m with
        | ZeroMatrix z -> failwith "Cannot compute the QR decomposition of ZeroMatrix."
        | Matrix mm ->
            let minor (m:_[,]) (d) =
                let rows = Array2D.length1 m
                let cols = Array2D.length2 m
                let ret = Array2D.zeroCreate rows cols
                for i = 0 to d - 1 do
                    ret.[i, i] <- LanguagePrimitives.GenericOne
                Array2D.blit m d d ret d d (rows - d) (cols - d)
                ret
            let identity d = Array2D.init d d (fun i j -> if i = j then LanguagePrimitives.GenericOne else LanguagePrimitives.GenericZero)
            // Householder
            let kmax = -1 + min (m.Rows - 1) m.Cols
            let mutable z = m.Copy()
            let q = Array.create m.Rows Matrix.Zero
            for k = 0 to kmax do
                z <- Matrix (minor (z.ToArray2D()) k)
                let x = z.[*, k]
                let mutable a = x.GetL2Norm()
                if mm.[k, k] > LanguagePrimitives.GenericZero then a <- -a
                let e = (x + Vector.createBasis m.Rows k a).GetUnitVector()
                q.[k] <- Matrix (identity m.Rows) + Matrix (Array2D.init m.Rows m.Rows (fun i j -> -(e.[i] * e.[j] + e.[i] * e.[j])))
                z <- q.[k] * z
            let mutable q' = q.[0]
            for i = 1 to kmax do
                q' <- q.[i] * q'
            q'.GetTranspose(), q' * m
    /// Returns the eigenvalues of this Matrix. (Experimental code, complex eigenvalues are not supported.)
    member inline m.GetEigenvalues() =
        let mutable m' = m.Copy()
        for i = 0 to 20 do
            let q, r = m'.GetQRDecomposition()
            m' <- r * q
        m'.GetDiagonal()

/// Provides basic operations on Matrix types. (Implementing functionality similar to Microsoft.FSharp.Collections.Array2D)
[<RequireQualifiedAccess>]
module Matrix =
    /// Creates a Matrix from 2d array `m`
    let inline ofArray2D (m:_[,]) = Matrix m
    /// Creates a Matrix from sequence `s`
    let inline ofSeq (s:seq<seq<_>>) = s |> Array.ofSeq |> Array.ofSeq |> array2D |> Matrix
    /// Converts Matrix `m` to a 2d array, e.g. from Matrix<float> to float[,]
    let inline toArray2D (m:Matrix<_>) = m.ToArray2D()
    /// Converts Matrix `m` to a jagged array, e.g. from Matrix<float> to float[][]
    let inline toArray (m:Matrix<_>) = m.ToArray()
    /// Returns the number of columns in Matrix `m`. This is the same with `Matrix.length2`.
    let inline cols (m:Matrix<_>) = m.Cols
    /// Creates a copy of Matrix `m`
    let inline copy (m:Matrix<_>) = m.Copy()
    /// Creates a Matrix with `m` rows, `n` columns, and all entries having value `v`
    let inline create m n v = Matrix (Array2D.create m n v)
    /// Creates a Matrix with `m` rows and all rows equal to array `v`
    let inline createRows (m:int) (v:_[]) = Matrix (array2D (Array.init m (fun _ -> v)))
    /// Gets the determinant of Matrix `m`
    let inline det (m:Matrix<_>) = m.GetDeterminant()
    /// Creates the identity matrix with `m` rows and columns
    let inline identity m =
        Matrix (Array2D.init m m (fun i j -> if i = j then LanguagePrimitives.GenericOne<'T> else LanguagePrimitives.GenericZero<'T>))
    /// Gets the eigenvalues of Matrix `m`
    let inline eigenvalues (m:Matrix<_>) = m.GetEigenvalues()
    /// Creates a Matrix with `m` rows, `n` columns and a generator function `f` to compute the entries
    let inline init m n f = Matrix (Array2D.init m n f)
    /// Creates a Matrix with `m` rows and a generator function `f` that gives each row as a an array
    let inline initRows (m:int) (f:int->_[]) = Matrix (array2D (Array.init m f))
    /// Creates a square Matrix with `m` rows and columns and a generator function `f` to compute the elements. Function `f` is used only for populating the diagonal and the upper triangular part of the Matrix, the lower triangular part will be the reflection.
    let inline initSymmetric m f =
        if m = 0 then 
            Matrix.Zero
        else
            let s = Array2D.zeroCreate<'T> m m
            for i = 0 to m - 1 do
                for j = i to m - 1 do
                    s.[i, j] <- f i j
            Matrix (copyUpperToLower s)
    /// Gets the inverse of Matrix `m`
    let inline inverse (m:Matrix<_>) = m.GetInverse()
    /// Returns the number of rows in Matrix `m`. This is the same with `Matrix.rows`.
    let inline length1 (m:Matrix<_>) = m.Rows
    /// Returns the number of columns in Matrix `m`. This is the same with `Matrix.cols`.
    let inline length2 (m:Matrix<_>) = m.Cols
    /// Creates a Matrix whose entries are the results of applying function `f` to each entry of Matrix `m`
    let inline map f (m:Matrix<_>) = m |> toArray2D |> Array2D.map f |> Matrix
    /// Creates a Matrix whose entries are the results of applying function `f` to each entry of Matrix `m`. An element index is also supplied to function `f`.
    let inline mapi f (m:Matrix<_>) = m |> toArray2D |> Array2D.mapi f |> Matrix
    /// Returns the number of rows in Matrix `m`. This is the same with `Matrix.length1`.
    let inline rows (m:Matrix<_>) = m.Rows
    /// Solves a system of linear equations ax = b, where the coefficients are given in Matrix `a` and the result vector is Vector `b`. The returned vector will correspond to x.
    let inline solve (a:Matrix<'T>) (b:Vector<'T>) =
        if a.Cols <> b.Length then invalidArg "" "Cannot solve the system of equations using a matrix and a vector of incompatible sizes."
        let lu, perm, _ = a.GetLUDecomposition()
        let bp = Array.init a.Rows (fun i -> b.[perm.[i]])
        Vector (matrixSolveHelper (lu.ToArray2D()) bp)
    /// Gets the trace of Matrix `m`
    let inline trace (m:Matrix<_>) = m.GetTrace()
    /// Gets the transpose of Matrix `m`
    let inline transpose (m:Matrix<_>) = m.GetTranspose()


/// Linear algebra operations module (automatically opened)
[<AutoOpen>]
module LinearAlgebraOps =
    /// Converts array, list, or sequence `v` into a Vector
    let inline vector v = Vector.ofSeq v
    /// Converts array, list, or sequence `v` into a Vector, first passing the elements through a conversion function `f`
    let inline vectorBy f v = Vector.map f (Vector.ofSeq v)
    /// Converts 2d array `m` into a Matrix
    let inline matrix m = Matrix.ofSeq m
    /// Converts 2d array `m` into a Matrix, first passing the elements through a conversion function `f`
    let inline matrixBy f m = Matrix.map f (Matrix.ofSeq m)