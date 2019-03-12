// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

namespace DiffSharp.Backend

/// Interface for DiffSharp backends
type Backend<'T> =
    // Scalar valued
    abstract member Mul_Dot_V_V : 'T[] * 'T[] -> 'T
    abstract member L1Norm_V : 'T[] -> 'T
    abstract member L2Norm_V : 'T[] -> 'T
    abstract member SupNorm_V : 'T[] -> 'T
    abstract member Sum_V : 'T[] -> 'T
    abstract member Sum_M : 'T[,] -> 'T

    // Vector valued, in-place.
    abstract member Add_V_V_Inplace : 'T[] * dest: 'T[] -> unit

    // Vector valued
    abstract member Add_V_V : 'T[] * 'T[] -> 'T[]
    abstract member Add_S_V : 'T * 'T[] -> 'T[]
    abstract member Sub_V_V : 'T[] * 'T[] -> 'T[]
    abstract member Sub_S_V : 'T * 'T[] -> 'T[]
    abstract member Sub_V_S : 'T[] * 'T -> 'T[]
    abstract member Mul_S_V : 'T * 'T[] -> 'T[]
    abstract member Mul_M_V : 'T[,] * 'T[] -> 'T[]
    abstract member Mul_M_V_Add_V : 'T[,] * 'T[] * 'T[] -> 'T[]
    abstract member Mul_V_M : 'T[] * 'T[,] -> 'T[]
    abstract member Solve_M_V : 'T[,] * 'T[] -> 'T[] option
    abstract member SolveSymmetric_M_V : 'T[,] * 'T[] -> 'T[] option
    abstract member Diagonal_M : 'T[,] -> 'T[]
    abstract member Map_F_V : ('T -> 'T) * 'T[] -> 'T[]
    abstract member Map2_F_V_V : ('T -> 'T -> 'T) * 'T[] * 'T[] -> 'T[]
    abstract member ReshapeCopy_MRows_V : 'T[,] -> 'T[]

    // Matrix valued, in-place. Accumulate to ``acc``
    abstract member AlphaAdd_M_M_Inplace : alpha:'T * x:'T[,] * acc: 'T[,] -> unit

    // Matrix valued
    abstract member Mul_Out_V_V : 'T[] * 'T[] -> 'T[,]
    abstract member Add_M_M : 'T[,] * 'T[,] -> 'T[,]
    abstract member Add_S_M : 'T * 'T[,] -> 'T[,]
    abstract member Add_V_MCols : 'T[] * 'T[,] -> 'T[,]
    abstract member Sub_M_M : 'T[,] * 'T[,] -> 'T[,]
    abstract member Sub_M_S : 'T[,] * 'T -> 'T[,]
    abstract member Sub_S_M : 'T * 'T[,] -> 'T[,]
    abstract member Mul_M_M : 'T[,] * 'T[,] -> 'T[,]
    abstract member Mul_S_M : 'T * 'T[,] -> 'T[,]
    abstract member Mul_M_M_Add_V_MCols : 'T[,] * 'T[,] * 'T[] -> 'T[,]
    abstract member Mul_Had_M_M : 'T[,] * 'T[,] -> 'T[,]
    abstract member Inverse_M : 'T[,] -> 'T[,] option
    abstract member Det_M : 'T[,] -> 'T option
    abstract member Transpose_M : 'T[,] -> 'T[,]
    abstract member Map_F_M : ('T -> 'T) * 'T[,] -> 'T[,]
    abstract member Map2_F_M_M : ('T -> 'T -> 'T) * 'T[,] * 'T[,] -> 'T[,]
    abstract member ReshapeCopy_V_MRows : int * 'T[] -> 'T[,]
    abstract member RepeatReshapeCopy_V_MRows : int * 'T[] -> 'T[,]
    abstract member RepeatReshapeCopy_V_MCols : int * 'T[] -> 'T[,]
