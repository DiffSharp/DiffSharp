//
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright 2014 National University of Ireland Maynooth.
// All rights reserved.
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
//   Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//


#light

namespace DiffSharp.Util

open DiffSharp.Util.General

/// Lightweight matrix type for internal usage
type Matrix =
    val M : float [,]
    new(m) = {M = m}
    member m.Rows = m.M.GetLength 0
    member m.Cols = m.M.GetLength 1
    member m.Item(i, j) = m.M.[i, j]
    override m.ToString() = sprintf "Matrix %A" m.M
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
        Matrix(Array2D.init m (a.[0].Dim) (fun i j -> a.[i].V.[j]))
    /// Create Matrix with `m` rows and all rows equal to float[] `v`
    static member Create(m, (v:float[])) = Matrix.Create(m, fun i -> v)
    /// Create zero Matrix
    static member Zero = Matrix.Create(0, 0, 0.)
    /// Add Matrix `a` to Matrix `b`
    static member (+) (a:Matrix, b:Matrix) = Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] + b.[i, j])
    /// Subtract Matrix `b` from Matrix `a`
    static member (-) (a:Matrix, b:Matrix) = Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] - b.[i, j])
    /// Matrix product of Matrix `a` and Matrix `b`
    static member (*) (a:Matrix, b:Matrix) = Matrix.Create(a.Rows, b.Cols, fun i j -> sum 0 (b.Rows - 1) (fun k -> a.[i, k] * b.[k, j]))
    /// Multiply Matrix `a` by float `b`
    static member (*) (a:Matrix, b:float) = Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] * b)
    /// Multiply Matrix `b` by float `a`
    static member (*) (a:float, b:Matrix) = Matrix.Create(b.Rows, b.Cols, fun i j -> a * b.[i, j])
    /// Divide Matrix `a` by float `b`
    static member (/) (a:Matrix, b:float) = Matrix.Create(a.Rows, a.Cols, fun i j -> a.[i, j] / b)
    /// Negative of Matrix `a`
    static member (~-) (a:Matrix) = Matrix.Create(a.Rows, a.Cols, fun i j -> -a.[i, j])