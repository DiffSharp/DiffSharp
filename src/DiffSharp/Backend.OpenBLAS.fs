﻿//
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

#nowarn "9"
#nowarn "51"

namespace DiffSharp.Backend

open System
open System.Runtime.InteropServices
open FSharp.NativeInterop
open System.Security
open DiffSharp.Util

/// Backend using OpenBLAS library for BLAS and LAPACK operations, and parallel threads for non-BLAS operations
module OpenBLAS =
    type PinnedArray<'T when 'T : unmanaged> (array : 'T[]) =
        let h = GCHandle.Alloc(array, GCHandleType.Pinned)
        let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
        member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
        interface IDisposable with
            member this.Dispose() = h.Free()

    type PinnedArray2D<'T when 'T : unmanaged> (array : 'T[,]) =
        let h = GCHandle.Alloc(array, GCHandleType.Pinned)
        let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
        member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
        interface IDisposable with
            member this.Dispose() = h.Free()

    module BLAS =
        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="isamax_")>]
        extern int isamax_(int *n, float32 *x, int *incx);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="saxpy_")>]
        extern void saxpy_(int *n, float32 *a, float32 *x, int *incx, float32 *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sscal_")>]
        extern void sscal_(int *n, float32 *alpha, float32 *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sdot_")>]
        extern float32 sdot_(int *n, float32 *x, int *incx, float32 *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sger_")>]
        extern void sger_(int *m, int *n, float32 *alpha, float32 *x, int *incx, float32 *y, int *incy, float32 *a, int *lda)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sasum_")>]
        extern float32 sasum_(int *n, float32 *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="snrm2_")>]
        extern float32 snrm2_(int *n, float32 *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sgemm_")>]
        extern void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float32 *alpha, float32 *a, int *lda, float32 *b, int *ldb, float32 *beta, float32 *c, int *ldc);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sgemv_")>]
        extern void sgemv_(char *trans, int *m, int *n, float32 *alpha, float32 *a, int *lda, float32 *x, int *incx, float32 *beta, float32 *y, int *incy)

        let isamax(x:float32[]) =
            let mutable arg_n = x.Length
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float32>(x)
            isamax_(&&arg_n, arg_x.Ptr, &&arg_incx)

        // y <- alpha * x + y
        let saxpy(alpha:float32, x:float32[], y:float32[]) =
            let mutable arg_n = min x.Length y.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            use arg_x = new PinnedArray<float32>(x)
            use arg_y = new PinnedArray<float32>(y)
            saxpy_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)

        // Y <- alpha * X + Y
        let saxpy'(alpha:float32, x:float32[,], y:float32[,]) =
            let mutable arg_n = min x.Length y.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            use arg_x = new PinnedArray2D<float32>(x)
            use arg_y = new PinnedArray2D<float32>(y)
            saxpy_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)

        // x <- alpha * x
        let sscal(alpha:float32, x:float32[]) =
            let mutable arg_n = x.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float32>(x)
            sscal_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx)

        // X <- alpha * X
        let sscal'(alpha:float32, x:float32[,]) =
            let mutable arg_n = x.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            use arg_x = new PinnedArray2D<float32>(x)
            sscal_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx)

        let sdot(x:float32[], y:float32[]) =
            let mutable arg_n = min x.Length y.Length
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            use arg_x = new PinnedArray<float32>(x)
            use arg_y = new PinnedArray<float32>(y)
            sdot_(&&arg_n, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)

        // A <- A + alpha * x * yT
        let sger(alpha:float32, x:float32[], y:float32[], a:float32[,]) =
            // Order modified to work with row-major matrices and eliminate the need for transposing the result
            let mutable arg_m = y.Length
            let mutable arg_n = x.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            let mutable arg_lda = arg_m
            use arg_x = new PinnedArray<float32>(y)
            use arg_y = new PinnedArray<float32>(x)
            use arg_a = new PinnedArray2D<float32>(a)
            sger_(&&arg_m, &&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy, arg_a.Ptr, &&arg_lda)

        let sasum(x:float32[]) =
            let mutable arg_n = x.Length
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float32>(x)
            sasum_(&&arg_n, arg_x.Ptr, &&arg_incx)

        let snrm2(x:float32[]) =
            let mutable arg_n = x.Length
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float32>(x)
            snrm2_(&&arg_n, arg_x.Ptr, &&arg_incx)

        // C <- alpha * A * B + beta * C
        let sgemm(alpha:float32, a:float32[,], b:float32[,], beta:float32, c:float32[,]) =
            // Order modified to work with row-major matrices and eliminate the need for transposing the result
            let m = Array2D.length1 a
            let n = Array2D.length2 b
            let k = Array2D.length1 b
            let mutable arg_transa = 'N'
            let mutable arg_transb = 'N'
            let mutable arg_m = n
            let mutable arg_n = m
            let mutable arg_k = k
            let mutable arg_alpha = alpha
            let mutable arg_lda = n
            let mutable arg_ldb = k
            let mutable arg_beta = beta
            let mutable arg_ldc = n
            use arg_a = new PinnedArray2D<float32>(b)
            use arg_b = new PinnedArray2D<float32>(a)
            use arg_c = new PinnedArray2D<float32>(c)
            sgemm_(&&arg_transa, &&arg_transb, &&arg_m, &&arg_n, &&arg_k, &&arg_alpha, arg_a.Ptr, &&arg_lda, arg_b.Ptr, &&arg_ldb, &&arg_beta, arg_c.Ptr, &&arg_ldc)

        // y <- alpha * A * x + beta * y
        let sgemv(alpha:float32, a:float32[,], x:float32[], beta:float32, y:float32[]) =
            let mutable arg_trans = 'T'
            let mutable arg_m = Array2D.length2 a
            let mutable arg_n = Array2D.length1 a
            let mutable arg_alpha = alpha
            let mutable arg_lda = arg_m
            let mutable arg_incx = 1
            let mutable arg_beta = beta
            let mutable arg_incy = 1
            use arg_a = new PinnedArray2D<float32>(a)
            use arg_x = new PinnedArray<float32>(x)
            use arg_y = new PinnedArray<float32>(y)
            sgemv_(&&arg_trans, &&arg_m, &&arg_n, &&arg_alpha, arg_a.Ptr, &&arg_lda, arg_x.Ptr, &&arg_incx, &&arg_beta, arg_y.Ptr, &&arg_incy)

        // y <- alpha * x * A + beta * y
        let sgemv'(alpha:float32, a:float32[,], x:float32[], beta:float32, y:float32[]) =
            let mutable arg_trans = 'N'
            let mutable arg_m = Array2D.length2 a
            let mutable arg_n = Array2D.length1 a
            let mutable arg_alpha = alpha
            let mutable arg_lda = arg_m
            let mutable arg_incx = 1
            let mutable arg_beta = beta
            let mutable arg_incy = 1
            use arg_a = new PinnedArray2D<float32>(a)
            use arg_x = new PinnedArray<float32>(x)
            use arg_y = new PinnedArray<float32>(y)
            sgemv_(&&arg_trans, &&arg_m, &&arg_n, &&arg_alpha, arg_a.Ptr, &&arg_lda, arg_x.Ptr, &&arg_incx, &&arg_beta, arg_y.Ptr, &&arg_incy)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="idamax_")>]
        extern int idamax_(int *n, float *x, int *incx);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="daxpy_")>]
        extern void daxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dscal_")>]
        extern void dscal_(int *n, float *alpha, float *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="ddot_")>]
        extern float ddot_(int *n, float *x, int *incx, float *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dger_")>]
        extern void dger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dasum_")>]
        extern float dasum_(int *n, float *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dnrm2_")>]
        extern float dnrm2_(int *n, float *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dgemm_")>]
        extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dgemv_")>]
        extern void dgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy)

        let idamax(x:float[]) =
            let mutable arg_n = x.Length
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float>(x)
            idamax_(&&arg_n, arg_x.Ptr, &&arg_incx)

        // y <- alpha * x + y
        let daxpy(alpha:float, x:float[], y:float[]) =
            let mutable arg_n = min x.Length y.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            use arg_x = new PinnedArray<float>(x)
            use arg_y = new PinnedArray<float>(y)
            daxpy_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)

        // Y <- alpha * X + Y
        let daxpy'(alpha:float, x:float[,], y:float[,]) =
            let mutable arg_n = min x.Length y.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            use arg_x = new PinnedArray2D<float>(x)
            use arg_y = new PinnedArray2D<float>(y)
            daxpy_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)

        // x <- alpha * x
        let dscal(alpha:float, x:float[]) =
            let mutable arg_n = x.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float>(x)
            dscal_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx)

        // X <- alpha * X
        let dscal'(alpha:float, x:float[,]) =
            let mutable arg_n = x.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            use arg_x = new PinnedArray2D<float>(x)
            dscal_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx)

        let ddot(x:float[], y:float[]) =
            let mutable arg_n = min x.Length y.Length
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            use arg_x = new PinnedArray<float>(x)
            use arg_y = new PinnedArray<float>(y)
            ddot_(&&arg_n, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)

        // A <- A + alpha * x * yT
        let dger(alpha:float, x:float[], y:float[], a:float[,]) =
            // Order modified to work with row-major matrices and eliminate the need for transposing the result
            let mutable arg_m = y.Length
            let mutable arg_n = x.Length
            let mutable arg_alpha = alpha
            let mutable arg_incx = 1
            let mutable arg_incy = 1
            let mutable arg_lda = arg_m
            use arg_x = new PinnedArray<float>(y)
            use arg_y = new PinnedArray<float>(x)
            use arg_a = new PinnedArray2D<float>(a)
            dger_(&&arg_m, &&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy, arg_a.Ptr, &&arg_lda)

        let dasum(x:float[]) =
            let mutable arg_n = x.Length
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float>(x)
            dasum_(&&arg_n, arg_x.Ptr, &&arg_incx)

        let dnrm2(x:float[]) =
            let mutable arg_n = x.Length
            let mutable arg_incx = 1
            use arg_x = new PinnedArray<float>(x)
            dnrm2_(&&arg_n, arg_x.Ptr, &&arg_incx)

        // C <- alpha * A * B + beta * C
        let dgemm(alpha:float, a:float[,], b:float[,], beta:float, c:float[,]) =
            // Order modified to work with row-major matrices and eliminate the need for transposing the result
            let m = Array2D.length1 a
            let n = Array2D.length2 b
            let k = Array2D.length1 b
            let mutable arg_transa = 'N'
            let mutable arg_transb = 'N'
            let mutable arg_m = n
            let mutable arg_n = m
            let mutable arg_k = k
            let mutable arg_alpha = alpha
            let mutable arg_lda = n
            let mutable arg_ldb = k
            let mutable arg_beta = beta
            let mutable arg_ldc = n
            use arg_a = new PinnedArray2D<float>(b)
            use arg_b = new PinnedArray2D<float>(a)
            use arg_c = new PinnedArray2D<float>(c)
            dgemm_(&&arg_transa, &&arg_transb, &&arg_m, &&arg_n, &&arg_k, &&arg_alpha, arg_a.Ptr, &&arg_lda, arg_b.Ptr, &&arg_ldb, &&arg_beta, arg_c.Ptr, &&arg_ldc)
        
        // y <- alpha * A * x + beta * y
        let dgemv(alpha:float, a:float[,], x:float[], beta:float, y:float[]) =
            let mutable arg_trans = 'T'
            let mutable arg_m = Array2D.length2 a
            let mutable arg_n = Array2D.length1 a
            let mutable arg_alpha = alpha
            let mutable arg_lda = arg_m
            let mutable arg_incx = 1
            let mutable arg_beta = beta
            let mutable arg_incy = 1
            use arg_a = new PinnedArray2D<float>(a)
            use arg_x = new PinnedArray<float>(x)
            use arg_y = new PinnedArray<float>(y)
            dgemv_(&&arg_trans, &&arg_m, &&arg_n, &&arg_alpha, arg_a.Ptr, &&arg_lda, arg_x.Ptr, &&arg_incx, &&arg_beta, arg_y.Ptr, &&arg_incy)

        // y <- alpha * x * A + beta * y
        let dgemv'(alpha:float, a:float[,], x:float[], beta:float, y:float[]) =
            let mutable arg_trans = 'N'
            let mutable arg_m = Array2D.length2 a
            let mutable arg_n = Array2D.length1 a
            let mutable arg_alpha = alpha
            let mutable arg_lda = arg_m
            let mutable arg_incx = 1
            let mutable arg_beta = beta
            let mutable arg_incy = 1
            use arg_a = new PinnedArray2D<float>(a)
            use arg_x = new PinnedArray<float>(x)
            use arg_y = new PinnedArray<float>(y)
            dgemv_(&&arg_trans, &&arg_m, &&arg_n, &&arg_alpha, arg_a.Ptr, &&arg_lda, arg_x.Ptr, &&arg_incx, &&arg_beta, arg_y.Ptr, &&arg_incy)

    module BLASExtensions =
        type Layout = // cblas.h: typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
            | R = 101
            | C = 102

        type Transpose = // cblas.h: typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
            | NT = 111
            | T = 112
            | CT = 113

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="cblas_somatcopy")>]
        extern void cblas_somatcopy(int ordering, int trans, int rows, int cols, float32 alpha, float32 *a, int lda, float32 *b, int ldb)

        // B <- alpha * transpose(A)
        let somatcopyT(alpha:float32, a:float32[,], b:float32[,]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let arg_ordering =  Layout.R |> int
            let arg_trans = Transpose.T |> int
            let arg_rows = m
            let arg_cols = n
            let arg_alpha = alpha
            use arg_a = new PinnedArray2D<float32>(a)
            let arg_lda = n
            use arg_b = new PinnedArray2D<float32>(b)
            let arg_ldb = m
            cblas_somatcopy(arg_ordering, arg_trans, arg_rows, arg_cols, arg_alpha, arg_a.Ptr, arg_lda, arg_b.Ptr, arg_ldb)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="cblas_domatcopy")>]
        extern void cblas_domatcopy(int ordering, int trans, int rows, int cols, float alpha, float *a, int lda, float *b, int ldb)

        // B <- alpha * transpose(A)
        let domatcopyT(alpha:float, a:float[,], b:float[,]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let arg_ordering =  Layout.R |> int
            let arg_trans = Transpose.T |> int
            let arg_rows = m
            let arg_cols = n
            let arg_alpha = alpha
            use arg_a = new PinnedArray2D<float>(a)
            let arg_lda = n
            use arg_b = new PinnedArray2D<float>(b)
            let arg_ldb = m
            cblas_domatcopy(arg_ordering, arg_trans, arg_rows, arg_cols, arg_alpha, arg_a.Ptr, arg_lda, arg_b.Ptr, arg_ldb)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="cblas_simatcopy")>]
        extern void cblas_simatcopy(int ordering, int trans, int rows, int cols, float32 alpha, float32 *ab, int lda, int ldb)

        // AB <- alpha * transpose(AB)
        let simatcopyT(alpha:float32, ab:float32[,]) =
            let m = Array2D.length1 ab
            let n = Array2D.length2 ab
            let arg_ordering =  Layout.R |> int
            let arg_trans = Transpose.T |> int
            let arg_rows = m
            let arg_cols = n
            let arg_alpha = alpha
            use arg_ab = new PinnedArray2D<float32>(ab)
            let arg_lda = n
            let arg_ldb = m
            cblas_simatcopy(arg_ordering, arg_trans, arg_rows, arg_cols, arg_alpha, arg_ab.Ptr, arg_lda, arg_ldb)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="cblas_dimatcopy")>]
        extern void cblas_dimatcopy(int ordering, int trans, int rows, int cols, float alpha, float *ab, int lda, int ldb)

        // B <- alpha * transpose(A)
        let dimatcopyT(alpha:float, ab:float[,]) =
            let m = Array2D.length1 ab
            let n = Array2D.length2 ab
            let arg_ordering =  Layout.R |> int
            let arg_trans = Transpose.T |> int
            let arg_rows = m
            let arg_cols = n
            let arg_alpha = alpha
            use arg_ab = new PinnedArray2D<float>(ab)
            let arg_lda = n
            let arg_ldb = m
            cblas_dimatcopy(arg_ordering, arg_trans, arg_rows, arg_cols, arg_alpha, arg_ab.Ptr, arg_lda, arg_ldb)

    module LAPACK =
        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sgesv_")>]
        extern void sgesv_(int *n, int *nrhs, float32 *a, int *lda, int *ipiv, float32 *b, int *ldb, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="ssysv_")>]
        extern void ssysv_(char *uplo, int *n, int *nrhs, float32 *a, int *lda, int *ipiv, float32 *b, int *ldb, float32 *work, int *lwork, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sgetrf_")>]
        extern void sgetrf_(int *m, int *n, float32 *a, int *lda, int *ipiv, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="sgetri_")>]
        extern void sgetri_(int *n, float32 *a, int *lda, int *ipiv, float32 *work, int *lwork, int *info)

        // This function works fine, but it is anomalous as it backs up the arguments inside the body unlike the others. See comments for ssysv.
        let sgesv(a:float32[,], b:float32[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            if m <> n then failwith "Matrix is not square in gesv (float32)."

            let a' = Array2D.zeroCreate m n
            BLASExtensions.somatcopyT(1.0f,a,a') // Transpose

            let b' = Array.copy b
            let ipiv = Array.zeroCreate n
            let mutable arg_n = n
            let mutable arg_nrhs = 1
            let mutable arg_lda = n
            let mutable arg_ldb = n
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float32>(a')
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_b = new PinnedArray<float32>(b')
            sgesv_(&&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, &&arg_info)
            if arg_info = 0 then
                Some(b')
            else
                None

        // This function is broken. The function mutates a and never uses b'. -Fixed. Modified it to be immutable.
        // Also it needs a trasposition step for a to transform it into column major. -Fixed.
        // As a consequence it no longer mutates a.

        // Also unlike the potrf, it performs no boundary checking at all like the rest of the Openblas functions.
        // More importantly it performs no check for symmetry. Be warned. -Possible TODO.

        // Note: As I do not see the values of this function being backed up before the call and as like sgesv above it needed a traspose step
        // that was missing, I made it immutable. Was that correct?
        let ssysv(a:float32[,], b:float32[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "Matrix is not square in sysv (float32)."
            if m <> b.Length then failwith "Length of b does not equal the dimensions of a in sysv (float32)."

            let b' = Array.copy b
            let a' = Array2D.zeroCreate n n
            BLASExtensions.somatcopyT(1.0f,a,a') // Transpose. Replace with simatcopy to make it mutable. Also remove b'.

            let ipiv = Array.zeroCreate n
            let work = Array.zeroCreate 1
            let mutable arg_uplo = 'U' // Assume upper triangular. TODO: check if LAPACK implementation requires the lower triangle to be zeroed. 
                                        // Marko's Note: No. Unlike the cuBLAS variant it will not check for symetry at all.

            let mutable arg_n = n
            let mutable arg_nrhs = 1
            let mutable arg_lda = n
            let mutable arg_ldb = n
            let mutable arg_lwork = 1
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float32>(a')
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_b = new PinnedArray<float32>(b') // Remove |> Array.copy if mutability is the intended behavior.
            use arg_work = new PinnedArray<float32>(work)
            ssysv_(&&arg_uplo, &&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                Some(b')
            else
                None

        let sgetrf(a:float32[,]) =
            // It needs an explicit transpose to col major.
            // For this function, its arguments are backed before the call so I allowed it to mutate a directly.
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "Matrix is not square in getrf (float32)."

            BLASExtensions.simatcopyT(1.0f,a) // Transpose inplace to col major

            let ipiv = Array.zeroCreate (min m n)
            let mutable arg_m = m
            let mutable arg_n = n
            let mutable arg_lda = m
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float32>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            sgetrf_(&&arg_m, &&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, &&arg_info)
            if arg_info = 0 then
                BLASExtensions.simatcopyT(1.0f,a) // Transpose inplace back to row major
                Some(ipiv)
            else
                None

        // Has arguments backed up before the call.
        let sgetri(a:float32[,], ipiv:int[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let work = Array.zeroCreate (n * n)

            if m <> n then failwith "Matrix is not square in getri (float32)."
            if m <> ipiv.Length then failwith "Length of ipiv does not equal the dimensions of a in getri (float32)."

            BLASExtensions.simatcopyT(1.0f,a) // Transpose inplace to col major

            let mutable arg_n = n
            let mutable arg_lda = n
            let mutable arg_lwork = n * n
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float32>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_work = new PinnedArray<float32>(work)
            sgetri_(&&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                BLASExtensions.simatcopyT(1.0f,a) // Transpose inplace back to row major
                Some(a)
            else
                None

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dgesv_")>]
        extern void dgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dsysv_")>]
        extern void dsysv_(char *uplo, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, float *work, int *lwork, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dgetrf_")>]
        extern void dgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas", EntryPoint="dgetri_")>]
        extern void dgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info)

        // This function works fine, but it is anomalous as it backs up the arguments inside the body unlike the others. See comments for ssysv.
        let dgesv(a:float[,], b:float[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            if m <> n then failwith "Matrix is not square in gesv (float)."

            let a' = Array2D.zeroCreate m n
            BLASExtensions.domatcopyT(1.0,a,a') // Transpose

            let b' = Array.copy b
            let ipiv = Array.zeroCreate n
            let mutable arg_n = n
            let mutable arg_nrhs = 1
            let mutable arg_lda = n
            let mutable arg_ldb = n
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float>(a')
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_b = new PinnedArray<float>(b')
            dgesv_(&&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, &&arg_info)
            if arg_info = 0 then
                Some(b')
            else
                None

        // This function is broken. The function mutates a and never uses b'. -Fixed. Modified it to be immutable.
        // Also it needs a trasposition step for a to transform it into column major. -Fixed.
        // As a consequence it no longer mutates a.

        // Also unlike the potrf, it performs no boundary checking at all like the rest of the Openblas functions.
        // More importantly it performs no check for symmetry. Be warned. -Possible TODO.

        // Note: As I do not see the values of this function being backed up before the call and as like sgesv above it needed a traspose step
        // that was missing, I made it immutable. Was that correct?
        let dsysv(a:float[,], b:float[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "Matrix is not square in sysv (float)."
            if m <> b.Length then failwith "Length of b does not equal the dimensions of a in sysv (float)."

            let b' = Array.copy b
            let a' = Array2D.zeroCreate n n
            BLASExtensions.domatcopyT(1.0,a,a') // Transpose. Replace with simatcopy to make it mutable. Also remove b'.

            let ipiv = Array.zeroCreate n
            let work = Array.zeroCreate 1
            let mutable arg_uplo = 'U' // Assume upper triangular. TODO: check if LAPACK implementation requires the lower triangle to be zeroed. 
                                        // Marko's Note: No. Unlike the cuBLAS variant it will not check for symetry at all.

            let mutable arg_n = n
            let mutable arg_nrhs = 1
            let mutable arg_lda = n
            let mutable arg_ldb = n
            let mutable arg_lwork = 1
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float>(a')
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_b = new PinnedArray<float>(b') // Remove |> Array.copy if mutability is the intended behavior.
            use arg_work = new PinnedArray<float>(work)
            dsysv_(&&arg_uplo, &&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                Some(b')
            else
                None

        let dgetrf(a:float[,]) =
            // It needs an explicit transpose to col major.
            // For this function, its arguments are backed before the call so I allowed it to mutate a directly.
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "Matrix is not square in getrf (float)."

            BLASExtensions.dimatcopyT(1.0,a) // Transpose inplace to col major

            let ipiv = Array.zeroCreate (min m n)
            let mutable arg_m = m
            let mutable arg_n = n
            let mutable arg_lda = m
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            dgetrf_(&&arg_m, &&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, &&arg_info)
            if arg_info = 0 then
                BLASExtensions.dimatcopyT(1.0,a) // Transpose inplace back to row major
                Some(ipiv)
            else
                None

        // Has arguments backed up before the call.
        let dgetri(a:float[,], ipiv:int[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let work = Array.zeroCreate (n * n)

            if m <> n then failwith "Matrix is not square in getri (float)."
            if m <> ipiv.Length then failwith "Length of ipiv does not equal the dimensions of a in getri (float)."

            BLASExtensions.dimatcopyT(1.0,a) // Transpose inplace to col major

            let mutable arg_n = n
            let mutable arg_lda = n
            let mutable arg_lwork = n * n
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_work = new PinnedArray<float>(work)
            dgetri_(&&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                BLASExtensions.dimatcopyT(1.0,a) // Transpose inplace back to row major
                Some(a)
            else
                None


    type Float32Backend() =
        interface Backend<float32> with
            // BLAS
            member o.Add_V_V(x, y) =
                if Array.isEmpty x then
                    Array.copy y
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let y' = Array.copy y
                    BLAS.saxpy(1.f, x, y')
                    y'
            // BLAS
            member o.Add_S_V(x, y) =
                if Array.isEmpty y then
                    Array.empty
                else
                    let x' = Array.create y.Length x
                    BLAS.saxpy(1.f, y, x')
                    x'
            // BLAS
            member o.Mul_S_V(alpha, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    let x' = Array.copy x
                    BLAS.sscal(alpha, x')
                    x'
            // BLAS
            member o.Sub_V_V(x, y) =
                if Array.isEmpty x then
                    (o :> Backend<float32>).Mul_S_V(-1.f, y)
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let x' = Array.copy x
                    BLAS.saxpy(-1.f, y, x')
                    x'
            // BLAS
            member o.Mul_Dot_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    0.f
                else
                    BLAS.sdot(x, y)
            // BLAS
            member o.Mul_Out_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate x.Length y.Length
                    BLAS.sger(1.f, x, y, z)
                    z
            // Non-BLAS
            member o.Sub_S_V(alpha, x) =
                if alpha = 0.f then 
                    (o :> Backend<float32>).Mul_S_V(-1.f, x)
                else
                    (o :> Backend<float32>).Map_F_V((fun v -> alpha - v), x)
            // Non-BLAS
            member o.Sub_V_S(x, alpha) =
                if alpha = 0.f then
                    x
                else
                    (o :> Backend<float32>).Map_F_V((fun v -> v - alpha), x)
            // Non-BLAS
            member o.Sub_S_M(alpha, x) =
                if alpha = 0.f then 
                    (o :> Backend<float32>).Mul_S_M(-1.f, x)
                else
                    (o :> Backend<float32>).Map_F_M((fun v -> alpha - v), x)
            // Non-BLAS
            member o.Sub_M_S(x, alpha) =
                if alpha = 0.f then
                    x
                else
                    (o :> Backend<float32>).Map_F_M((fun v -> v - alpha), x)
            // Non-BLAS
            member o.Map_F_V(f, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    Array.map f x
            // Non-BLAS
            member o.Map2_F_V_V(f, x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array.empty
                else
                    Array.map2 f x y
            // Non-BLAS
            member o.Map_F_M(f, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    Array2D.map f x
            // Non-BLAS
            member o.Map2_F_M_M(f, x, y) =
                if Array2D.isEmpty x || Array2D.isEmpty y then
                    Array2D.empty
                else
                    Array2D.map2 f x y
            // BLAS
            member o.L1Norm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                   BLAS.sasum(x)
            // BLAS
            member o.L2Norm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    BLAS.snrm2(x)
            // BLAS
            member o.SupNorm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    let i = BLAS.isamax(x)
                    abs x.[i - 1]
            // Non-BLAS
            member o.Sum_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    Array.sum x
            // BLAS
            member o.Add_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.copyFast y
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let y' = Array2D.copyFast y
                    BLAS.saxpy'(1.f, x, y')
                    y'
            // BLAS
            member o.Add_S_M(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                else
                    let x' = Array2D.create (Array2D.length1 y) (Array2D.length2 y) x
                    BLAS.saxpy'(1.f, y, x')
                    x'
            // BLAS
            member o.Add_V_MCols(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                elif Array.isEmpty x then
                    y
                else
                    let x' = (o :> Backend<float32>).RepeatReshapeCopy_V_MCols(Array2D.length2 y, x)
                    BLAS.saxpy'(1.f, y, x')
                    x'
            // BLAS
            member o.Mul_S_M(alpha, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let x' = Array2D.copyFast x
                    BLAS.sscal'(alpha, x')
                    x'
            // BLAS
            member o.Sub_M_M(x, y) =
                if Array2D.isEmpty x then
                    (o :> Backend<float32>).Mul_S_M(-1.f, y)
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let x' = Array2D.copyFast x
                    BLAS.saxpy'(-1.f, y, x')
                    x'
            // BLAS
            member o.Mul_M_M(x, y) =
                if (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 y)
                    BLAS.sgemm(1.f, x, y, 0.f, z)
                    z
            // BLAS
            member o.Mul_M_M_Add_V_MCols(x, y, z) =
                if Array.isEmpty z then
                    (o :> Backend<float32>).Mul_M_M(x, y)
                elif (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let n = (Array2D.length2 y)
                    let z' = (o :> Backend<float32>).RepeatReshapeCopy_V_MCols(n, z)
                    BLAS.sgemm(1.f, x, y, 1.f, z')
                    z'
            // Non-BLAS
            member o.Mul_Had_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.zeroCreate (Array2D.length1 y) (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 x)
                else
                    (o :> Backend<float32>).Map2_F_M_M((*), x, y)
            // BLAS
            member o.Mul_M_V(x, y) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                else
                    let z = Array.zeroCreate (Array2D.length1 x)
                    BLAS.sgemv(1.f, x, y, 0.f, z)
                    z
            // BLAS
            member o.Mul_M_V_Add_V(x, y, z) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                elif Array.isEmpty z then
                    (o :> Backend<float32>).Mul_M_V(x, y)
                else
                    let z' = Array.copy z
                    BLAS.sgemv(1.f, x, y, 1.f, z')
                    z'
            // BLAS
            member o.Mul_V_M(x, y) =
                if Array.isEmpty x then
                    Array.zeroCreate (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array.empty
                else
                    let z = Array.zeroCreate (Array2D.length2 y)
                    BLAS.sgemv'(1.f, y, x, 0.f, z)
                    z
            // BLAS extension
            member o.Transpose_M(x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let m = Array2D.length1 x
                    let n = Array2D.length2 x
                    let x' = Array2D.zeroCreate<float32> n m
                    BLASExtensions.somatcopyT(1.f, x, x')
                    x'
            // Non-BLAS
            member o.Sum_M(x) =
                if Array2D.isEmpty x then
                    0.f
                else
                    (o :> Backend<float32>).ReshapeCopy_MRows_V(x) |> Array.sum
            // LAPACK
            member o.Solve_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.sgesv(x, y)
            // LAPACK
            member o.SolveSymmetric_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.ssysv(x, y)
            // Non-BLAS
            member o.Diagonal_M(x) =
                if Array2D.isEmpty x then
                    Array.empty
                else
                    let n = min (Array2D.length1 x) (Array2D.length2 x)
                    Array.init n (fun i -> x.[i, i])
            // LAPACK
            member o.Inverse_M(x) =
                if Array2D.isEmpty x then
                    Some(Array2D.empty)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = LAPACK.sgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let inv = LAPACK.sgetri(x', ipiv)
                        match inv with
                        | Some(inv) -> Some(inv)
                        | _ -> None
                    | _ -> None
            // LAPACK
            member o.Det_M(x) =
                if Array2D.isEmpty x then
                    Some(0.f)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = LAPACK.sgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let n = Array2D.length1 x
                        let mutable det = 1.f
                        for i = 0 to n - 1 do
                            if ipiv.[i] <> (i + 1) then
                                det <- -det * x'.[i, i]
                            else
                                det <- det * x'.[i, i]
                        Some(det)
                    | _ -> None
            // Non-BLAS
            member o.ReshapeCopy_MRows_V(x) =
                if Array2D.isEmpty x then
                    Array.empty<float32>
                else
                    let r = Array.zeroCreate<float32> x.Length
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float32>)
                    r
            // Non-BLAS
            member o.ReshapeCopy_V_MRows(m, x) =
                if Array.isEmpty x then
                    Array2D.empty<float32>
                else
                    let n = x.Length / m
                    let r = Array2D.zeroCreate<float32> m n
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float32>)
                    r
            // Non-BLAS
            member o.RepeatReshapeCopy_V_MRows(m, x) =
                if Array.isEmpty x then
                    Array2D.empty<float32>
                else
                    let n = x.Length
                    let r = Array2D.zeroCreate<float32> m n
                    let xbytes = n * sizeof<float32>
                    for i = 0 to m - 1 do
                        Buffer.BlockCopy(x, 0, r, i * xbytes, xbytes)
                    r
            // Non-BLAS
            member o.RepeatReshapeCopy_V_MCols(n, x) =
                if Array.isEmpty x then
                    Array2D.empty<float32>
                else
                    let m = x.Length
                    let r = Array2D.zeroCreate<float32> m n
                    let rows = Array.init m (fun i -> Array.create n x.[i])
                    let xbytes = n * sizeof<float32>
                    for i = 0 to m - 1 do
                        Buffer.BlockCopy(rows.[i], 0, r, i * xbytes, xbytes)
                    r

    type Float64Backend() =
        interface Backend<float> with
            // BLAS
            member o.Add_V_V(x, y) =
                if Array.isEmpty x then
                    Array.copy y
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let y' = Array.copy y
                    BLAS.daxpy(1., x, y')
                    y'
            // BLAS
            member o.Add_S_V(x, y) =
                if Array.isEmpty y then
                    Array.empty
                else
                    let x' = Array.create y.Length x
                    BLAS.daxpy(1., y, x')
                    x'
            // BLAS
            member o.Mul_S_V(alpha, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    let x' = Array.copy x
                    BLAS.dscal(alpha, x')
                    x'
            // BLAS
            member o.Sub_V_V(x, y) =
                if Array.isEmpty x then
                    (o :> Backend<float>).Mul_S_V(-1., y)
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let x' = Array.copy x
                    BLAS.daxpy(-1., y, x')
                    x'
            // BLAS
            member o.Mul_Dot_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    0.
                else
                    BLAS.ddot(x, y)
            // BLAS
            member o.Mul_Out_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate x.Length y.Length
                    BLAS.dger(1., x, y, z)
                    z
            // Non-BLAS
            member o.Sub_S_V(alpha, x) =
                if alpha = 0. then 
                    (o :> Backend<float>).Mul_S_V(-1., x)
                else
                    (o :> Backend<float>).Map_F_V((fun v -> alpha - v), x)
            // Non-BLAS
            member o.Sub_V_S(x, alpha) =
                if alpha = 0. then
                    x
                else
                    (o :> Backend<float>).Map_F_V((fun v -> v - alpha), x)
            // Non-BLAS
            member o.Sub_S_M(alpha, x) =
                if alpha = 0. then 
                    (o :> Backend<float>).Mul_S_M(-1., x)
                else
                    (o :> Backend<float>).Map_F_M((fun v -> alpha - v), x)
            // Non-BLAS
            member o.Sub_M_S(x, alpha) =
                if alpha = 0. then
                    x
                else
                    (o :> Backend<float>).Map_F_M((fun v -> v - alpha), x)
            // Non-BLAS
            member o.Map_F_V(f, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    Array.map f x
            // Non-BLAS
            member o.Map2_F_V_V(f, x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array.empty
                else
                    Array.map2 f x y
            // Non-BLAS
            member o.Map_F_M(f, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    Array2D.map f x
            // Non-BLAS
            member o.Map2_F_M_M(f, x, y) =
                if Array2D.isEmpty x || Array2D.isEmpty y then
                    Array2D.empty
                else
                    Array2D.map2 f x y
            // BLAS
            member o.L1Norm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                   BLAS.dasum(x)
            // BLAS
            member o.L2Norm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    BLAS.dnrm2(x)
            // BLAS
            member o.SupNorm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    let i = BLAS.idamax(x)
                    abs x.[i - 1]
            // Non-BLAS
            member o.Sum_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    Array.sum x
            // BLAS
            member o.Add_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.copyFast y
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let y' = Array2D.copyFast y
                    BLAS.daxpy'(1., x, y')
                    y'
            // BLAS
            member o.Add_S_M(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                else
                    let x' = Array2D.create (Array2D.length1 y) (Array2D.length2 y) x
                    BLAS.daxpy'(1., y, x')
                    x'
            // BLAS
            member o.Add_V_MCols(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                elif Array.isEmpty x then
                    y
                else
                    let x' = (o :> Backend<float>).RepeatReshapeCopy_V_MCols(Array2D.length2 y, x)
                    BLAS.daxpy'(1., y, x')
                    x'
            // BLAS
            member o.Mul_S_M(alpha, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let x' = Array2D.copyFast x
                    BLAS.dscal'(alpha, x')
                    x'
            // BLAS
            member o.Sub_M_M(x, y) =
                if Array2D.isEmpty x then
                    (o :> Backend<float>).Mul_S_M(-1., y)
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let x' = Array2D.copyFast x
                    BLAS.daxpy'(-1., y, x')
                    x'
            // BLAS
            member o.Mul_M_M(x, y) =
                if (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 y)
                    BLAS.dgemm(1., x, y, 0., z)
                    z
            // BLAS
            member o.Mul_M_M_Add_V_MCols(x, y, z) =
                if Array.isEmpty z then
                    (o :> Backend<float>).Mul_M_M(x, y)
                elif (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let n = (Array2D.length2 y)
                    let z' = (o :> Backend<float>).RepeatReshapeCopy_V_MCols(n, z)
                    BLAS.dgemm(1., x, y, 1., z')
                    z'
            // Non-BLAS
            member o.Mul_Had_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.zeroCreate (Array2D.length1 y) (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 x)
                else
                    (o :> Backend<float>).Map2_F_M_M((*), x, y)
            // BLAS
            member o.Mul_M_V(x, y) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                else
                    let z = Array.zeroCreate (Array2D.length1 x)
                    BLAS.dgemv(1., x, y, 0., z)
                    z
            // BLAS
            member o.Mul_M_V_Add_V(x, y, z) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                elif Array.isEmpty z then
                    (o :> Backend<float>).Mul_M_V(x, y)
                else
                    let z' = Array.copy z
                    BLAS.dgemv(1., x, y, 1., z')
                    z'
            // BLAS
            member o.Mul_V_M(x, y) =
                if Array.isEmpty x then
                    Array.zeroCreate (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array.empty
                else
                    let z = Array.zeroCreate (Array2D.length2 y)
                    BLAS.dgemv'(1., y, x, 0., z)
                    z
            // BLAS extension
            member o.Transpose_M(x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let m = Array2D.length1 x
                    let n = Array2D.length2 x
                    let x' = Array2D.zeroCreate<float> n m
                    BLASExtensions.domatcopyT(1., x, x')
                    x'
            // Non-BLAS
            member o.Sum_M(x) =
                if Array2D.isEmpty x then
                    0.
                else
                    (o :> Backend<float>).ReshapeCopy_MRows_V(x) |> Array.sum
            // LAPACK
            member o.Solve_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.dgesv(x, y)
            // LAPACK
            member o.SolveSymmetric_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.dsysv(x, y)
            // Non-BLAS
            member o.Diagonal_M(x) =
                if Array2D.isEmpty x then
                    Array.empty
                else
                    let n = min (Array2D.length1 x) (Array2D.length2 x)
                    Array.init n (fun i -> x.[i, i])
            // LAPACK
            member o.Inverse_M(x) =
                if Array2D.isEmpty x then
                    Some(Array2D.empty)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = LAPACK.dgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let inv = LAPACK.dgetri(x', ipiv)
                        match inv with
                        | Some(inv) -> Some(inv)
                        | _ -> None
                    | _ -> None
            // LAPACK
            member o.Det_M(x) =
                if Array2D.isEmpty x then
                    Some(0.)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = LAPACK.dgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let n = Array2D.length1 x
                        let mutable det = 1.
                        for i = 0 to n - 1 do
                            if ipiv.[i] <> (i + 1) then
                                det <- -det * x'.[i, i]
                            else
                                det <- det * x'.[i, i]
                        Some(det)
                    | _ -> None
            // Non-BLAS
            member o.ReshapeCopy_MRows_V(x) =
                if Array2D.isEmpty x then
                    Array.empty<float>
                else
                    let r = Array.zeroCreate<float> x.Length
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float>)
                    r
            // Non-BLAS
            member o.ReshapeCopy_V_MRows(m, x) =
                if Array.isEmpty x then
                    Array2D.empty<float>
                else
                    let n = x.Length / m
                    let r = Array2D.zeroCreate<float> m n
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float>)
                    r
            // Non-BLAS
            member o.RepeatReshapeCopy_V_MRows(m, x) =
                if Array.isEmpty x then
                    Array2D.empty<float>
                else
                    let n = x.Length
                    let r = Array2D.zeroCreate<float> m n
                    let xbytes = n * sizeof<float>
                    for i = 0 to m - 1 do
                        Buffer.BlockCopy(x, 0, r, i * xbytes, xbytes)
                    r
            // Non-BLAS
            member o.RepeatReshapeCopy_V_MCols(n, x) =
                if Array.isEmpty x then
                    Array2D.empty<float>
                else
                    let m = x.Length
                    let r = Array2D.zeroCreate<float> m n
                    let rows = Array.init m (fun i -> Array.create n x.[i])
                    let xbytes = n * sizeof<float>
                    for i = 0 to m - 1 do
                        Buffer.BlockCopy(rows.[i], 0, r, i * xbytes, xbytes)
                    r