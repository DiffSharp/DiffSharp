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

namespace DiffSharp.Backend

/// Interface for DiffSharp backends
type Backend<'T> =
    // Scalar valued
    abstract member Mul_Dot_V_V : 'T[] * 'T[] -> 'T
    abstract member L1Norm_V : ('T[]) -> 'T
    abstract member L2Norm_V : ('T[]) -> 'T
    abstract member SupNorm_V : ('T[]) -> 'T
    abstract member Sum_V : ('T[]) -> 'T
    abstract member Sum_M : ('T[,]) -> 'T
    
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