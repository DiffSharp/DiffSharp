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

#nowarn "9"
#nowarn "51"

namespace DiffSharp.BackEnd

open System
open System.Runtime.InteropServices
open FSharp.NativeInterop
open System.Security
open System.Threading.Tasks
open DiffSharp.Util


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
        [<DllImport("libopenblas.dll", EntryPoint="isamax_")>]
        extern int isamax_(int *n, float32 *x, int *incx);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="saxpy_")>]
        extern void saxpy_(int *n, float32 *a, float32 *x, int *incx, float32 *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sscal_")>]
        extern void sscal_(int *n, float32 *alpha, float32 *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sdot_")>]
        extern float32 sdot_(int *n, float32 *x, int *incx, float32 *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sger_")>]
        extern void sger_(int *m, int *n, float32 *alpha, float32 *x, int *incx, float32 *y, int *incy, float32 *a, int *lda)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sasum_")>]
        extern float32 sasum_(int *n, float32 *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="snrm2_")>]
        extern float32 snrm2_(int *n, float32 *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sgemm_")>]
        extern void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float32 *alpha, float32 *a, int *lda, float32 *b, int *ldb, float32 *beta, float32 *c, int *ldc);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sgemv_")>]
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
        [<DllImport("libopenblas.dll", EntryPoint="idamax_")>]
        extern int idamax_(int *n, float *x, int *incx);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="daxpy_")>]
        extern void daxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dscal_")>]
        extern void dscal_(int *n, float *alpha, float *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="ddot_")>]
        extern float ddot_(int *n, float *x, int *incx, float *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dger_")>]
        extern void dger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dasum_")>]
        extern float dasum_(int *n, float *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dnrm2_")>]
        extern float dnrm2_(int *n, float *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgemm_")>]
        extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgemv_")>]
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

    module LAPACK =
        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sgesv_")>]
        extern void sgesv_(int *n, int *nrhs, float32 *a, int *lda, int *ipiv, float32 *b, int *ldb, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="ssysv_")>]
        extern void ssysv_(char *uplo, int *n, int *nrhs, float32 *a, int *lda, int *ipiv, float32 *b, int *ldb, float32 *work, int *lwork, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sgetrf_")>]
        extern void sgetrf_(int *m, int *n, float32 *a, int *lda, int *ipiv, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="sgetri_")>]
        extern void sgetri_(int *n, float32 *a, int *lda, int *ipiv, float32 *work, int *lwork, int *info)

        let sgesv(a:float32[,], b:float32[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let a' = Array2D.zeroCreate n m
            Parallel.For(0, n, fun i -> Parallel.For(0, m, fun j -> a'.[i, j] <- a.[j, i]) |> ignore) |> ignore
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

        let ssysv(a:float32[,], b:float32[]) =
            let n = Array2D.length1 a
            let b' = Array.copy b
            let ipiv = Array.zeroCreate n
            let work = Array.zeroCreate 1
            let mutable arg_uplo = 'U' // Assume upper triangular. TODO: check if LAPACK implementation requires the lower triangle to be zeroed
            let mutable arg_n = n
            let mutable arg_nrhs = 1
            let mutable arg_lda = n
            let mutable arg_ldb = n
            let mutable arg_lwork = 1
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float32>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_b = new PinnedArray<float32>(b)
            use arg_work = new PinnedArray<float32>(work)
            ssysv_(&&arg_uplo, &&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                Some(b')
            else
                None

        let sgetrf(a:float32[,]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let ipiv = Array.zeroCreate (min m n)
            let mutable arg_m = m
            let mutable arg_n = n
            let mutable arg_lda = m
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float32>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            sgetrf_(&&arg_m, &&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, &&arg_info)
            if arg_info = 0 then
                Some(ipiv)
            else
                None

        let sgetri(a:float32[,], ipiv:int[]) =
            let n = Array2D.length1 a
            let work = Array.zeroCreate (n * n)
            let mutable arg_n = n
            let mutable arg_lda = n
            let mutable arg_lwork = n * n
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float32>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_work = new PinnedArray<float32>(work)
            sgetri_(&&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                Some(a)
            else
                None

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgesv_")>]
        extern void dgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dsysv_")>]
        extern void dsysv_(char *uplo, int *n, int *nrhs, float *a, int *lda, int *ipiv, float *b, int *ldb, float *work, int *lwork, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgetrf_")>]
        extern void dgetrf_(int *m, int *n, float *a, int *lda, int *ipiv, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgetri_")>]
        extern void dgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int *lwork, int *info)

        let dgesv(a:float[,], b:float[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let a' = Array2D.zeroCreate n m
            Parallel.For(0, n, fun i -> Parallel.For(0, m, fun j -> a'.[i, j] <- a.[j, i]) |> ignore) |> ignore
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

        let dsysv(a:float[,], b:float[]) =
            let n = Array2D.length1 a
            let b' = Array.copy b
            let ipiv = Array.zeroCreate n
            let work = Array.zeroCreate 1
            let mutable arg_uplo = 'U' // Assume upper triangular. TODO: check if LAPACK implementation requires the lower triangle to be zeroed
            let mutable arg_n = n
            let mutable arg_nrhs = 1
            let mutable arg_lda = n
            let mutable arg_ldb = n
            let mutable arg_lwork = 1
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_b = new PinnedArray<float>(b)
            use arg_work = new PinnedArray<float>(work)
            dsysv_(&&arg_uplo, &&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                Some(b')
            else
                None

        let dgetrf(a:float[,]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let ipiv = Array.zeroCreate (min m n)
            let mutable arg_m = m
            let mutable arg_n = n
            let mutable arg_lda = m
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            dgetrf_(&&arg_m, &&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, &&arg_info)
            if arg_info = 0 then
                Some(ipiv)
            else
                None

        let dgetri(a:float[,], ipiv:int[]) =
            let n = Array2D.length1 a
            let work = Array.zeroCreate (n * n)
            let mutable arg_n = n
            let mutable arg_lda = n
            let mutable arg_lwork = n * n
            let mutable arg_info = 0
            use arg_a = new PinnedArray2D<float>(a)
            use arg_ipiv = new PinnedArray<int>(ipiv)
            use arg_work = new PinnedArray<float>(work)
            dgetri_(&&arg_n, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_work.Ptr, &&arg_lwork, &&arg_info)
            if arg_info = 0 then
                Some(a)
            else
                None

    type Float32BackEnd() =
        interface BackEnd<float32> with
            member o.Add_V_V(x, y) =
                if Array.isEmpty x then
                    Array.copy y
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let y' = Array.copy y
                    BLAS.saxpy(1.f, x, y')
                    y'
            member o.Mul_S_V(alpha, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    let x' = Array.copy x
                    BLAS.sscal(alpha, x')
                    x'
            member o.Sub_V_V(x, y) =
                if Array.isEmpty x then
                    (o :> BackEnd<float32>).Mul_S_V(-1.f, y)
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let x' = Array.copy x
                    BLAS.saxpy(-1.f, y, x')
                    x'
            member o.Mul_Dot_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    0.f
                else
                    BLAS.sdot(x, y)
            member o.Mul_Out_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate x.Length y.Length
                    BLAS.sger(1.f, x, y, z)
                    z
            member o.Sub_S_V(alpha, x) = // Non-BLAS
                if alpha = 0.f then 
                    (o :> BackEnd<float32>).Mul_S_V(-1.f, x)
                else
                    (o :> BackEnd<float32>).Map_F_V((fun v -> alpha - v), x)
            member o.Sub_V_S(x, alpha) = // Non-BLAS
                if alpha = 0.f then
                    x
                else
                    (o :> BackEnd<float32>).Map_F_V((fun v -> v - alpha), x)
            member o.Sub_S_M(alpha, x) = // Non-BLAS
                if alpha = 0.f then 
                    (o :> BackEnd<float32>).Mul_S_M(-1.f, x)
                else
                    (o :> BackEnd<float32>).Map_F_M((fun v -> alpha - v), x)
            member o.Sub_M_S(x, alpha) = // Non-BLAS
                if alpha = 0.f then
                    x
                else
                    (o :> BackEnd<float32>).Map_F_M((fun v -> v - alpha), x)
            member o.Map_F_V(f, x) = // Non-BLAS
                if Array.isEmpty x then
                    Array.empty
                else
                    Array.Parallel.map f x
            member o.Map2_F_V_V(f, x, y) = // Non-BLAS
                if Array.isEmpty x || Array.isEmpty y then
                    Array.empty
                else
                    Array.Parallel.map2 f x y
            member o.Map_F_M(f, x) = // Non-BLAS
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    Array2D.Parallel.map f x
            member o.Map2_F_M_M(f, x, y) = // Non-BLAS
                if Array2D.isEmpty x || Array2D.isEmpty y then
                    Array2D.empty
                else
                    Array2D.Parallel.map2 f x y
            member o.L1Norm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                   BLAS.sasum(x)
            member o.L2Norm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    BLAS.snrm2(x)
            member o.SupNorm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    let i = BLAS.isamax(x)
                    abs x.[i - 1]
            member o.Sum_V(x) = // Non-BLAS
                if Array.isEmpty x then
                    0.f
                else
                    Array.sum x
            member o.Add_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.copy y
                elif Array2D.isEmpty y then
                    Array2D.copy x
                else
                    let y' = Array2D.copy y
                    BLAS.saxpy'(1.f, x, y')
                    y'
            member o.Mul_S_M(alpha, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let x' = Array2D.copy x
                    BLAS.sscal'(alpha, x')
                    x'
            member o.Sub_M_M(x, y) =
                if Array2D.isEmpty x then
                    (o :> BackEnd<float32>).Mul_S_M(-1.f, y)
                elif Array2D.isEmpty y then
                    Array2D.copy x
                else
                    let x' = Array2D.copy x
                    BLAS.saxpy'(-1.f, y, x')
                    x'
            member o.Mul_M_M(x, y) =
                if (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 y)
                    BLAS.sgemm(1.f, x, y, 0.f, z)
                    z
            member o.Mul_Had_M_M(x, y) = // Non-BLAS
                if Array2D.isEmpty x then
                    Array2D.zeroCreate (Array2D.length1 y) (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 x)
                else
                    (o :> BackEnd<float32>).Map2_F_M_M((*), x, y)
            member o.Mul_M_V(x, y) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                else
                    let z = Array.zeroCreate (Array2D.length1 x)
                    BLAS.sgemv(1.f, x, y, 0.f, z)
                    z
            member o.Mul_V_M(x, y) =
                if Array.isEmpty x then
                    Array.zeroCreate (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array.empty
                else
                    let z = Array.zeroCreate (Array2D.length2 y)
                    BLAS.sgemv'(1.f, y, x, 0.f, z)
                    z
            member o.Transpose_M(x) = // Non-BLAS
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let m = Array2D.length2 x
                    let n = Array2D.length1 x
                    Array2D.Parallel.init m n (fun i j -> x.[j, i])
            member o.Sum_M(x) = // Non-BLAS
                if Array2D.isEmpty x then
                    0.f
                else
                    Array.Parallel.init (Array2D.length1 x) (fun i -> x.[i, *])
                    |> Array.Parallel.map Array.sum
                    |> Array.sum
            member o.Solve_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.sgesv(x, y)
            member o.SolveSymmetric_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.ssysv(x, y)
            member o.Diagonal_M(x) =
                if Array2D.isEmpty x then
                    Array.empty
                else
                    let n = min (Array2D.length1 x) (Array2D.length2 x)
                    Array.Parallel.init n (fun i -> x.[i, i])
            member o.Inverse_M(x) =
                if Array2D.isEmpty x then
                    Some(Array2D.empty)
                else
                    let x' = Array2D.copy x
                    let ipiv = LAPACK.sgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let inv = LAPACK.sgetri(x', ipiv)
                        match inv with
                        | Some(inv) -> Some(inv)
                        | _ -> None
                    | _ -> None
            member o.Det_M(x) =
                if Array2D.isEmpty x then
                    Some(0.f)
                else
                    let x' = Array2D.copy x
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

    type Float64BackEnd() =
        interface BackEnd<float> with
            member o.Add_V_V(x, y) =
                if Array.isEmpty x then
                    Array.copy y
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let y' = Array.copy y
                    BLAS.daxpy(1., x, y')
                    y'
            member o.Mul_S_V(alpha, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    let x' = Array.copy x
                    BLAS.dscal(alpha, x')
                    x'
            member o.Sub_V_V(x, y) =
                if Array.isEmpty x then
                    (o :> BackEnd<float>).Mul_S_V(-1., y)
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let x' = Array.copy x
                    BLAS.daxpy(-1., y, x')
                    x'
            member o.Mul_Dot_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    0.
                else
                    BLAS.ddot(x, y)
            member o.Mul_Out_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate x.Length y.Length
                    BLAS.dger(1., x, y, z)
                    z
            member o.Sub_S_V(alpha, x) = // Non-BLAS
                if alpha = 0. then 
                    (o :> BackEnd<float>).Mul_S_V(-1., x)
                else
                    (o :> BackEnd<float>).Map_F_V((fun v -> alpha - v), x)
            member o.Sub_V_S(x, alpha) = // Non-BLAS
                if alpha = 0. then
                    x
                else
                    (o :> BackEnd<float>).Map_F_V((fun v -> v - alpha), x)
            member o.Sub_S_M(alpha, x) = // Non-BLAS
                if alpha = 0. then 
                    (o :> BackEnd<float>).Mul_S_M(-1., x)
                else
                    (o :> BackEnd<float>).Map_F_M((fun v -> alpha - v), x)
            member o.Sub_M_S(x, alpha) = // Non-BLAS
                if alpha = 0. then
                    x
                else
                    (o :> BackEnd<float>).Map_F_M((fun v -> v - alpha), x)
            member o.Map_F_V(f, x) = // Non-BLAS
                if Array.isEmpty x then
                    Array.empty
                else
                    Array.Parallel.map f x
            member o.Map2_F_V_V(f, x, y) = // Non-BLAS
                if Array.isEmpty x || Array.isEmpty y then
                    Array.empty
                else
                    Array.Parallel.map2 f x y
            member o.Map_F_M(f, x) = // Non-BLAS
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    Array2D.Parallel.map f x
            member o.Map2_F_M_M(f, x, y) = // Non-BLAS
                if Array2D.isEmpty x || Array2D.isEmpty y then
                    Array2D.empty
                else
                    Array2D.Parallel.map2 f x y
            member o.L1Norm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                   BLAS.dasum(x)
            member o.L2Norm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    BLAS.dnrm2(x)
            member o.SupNorm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    let i = BLAS.idamax(x)
                    abs x.[i - 1]
            member o.Sum_V(x) = // Non-BLAS
                if Array.isEmpty x then
                    0.
                else
                    Array.sum x
            member o.Add_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.copy y
                elif Array2D.isEmpty y then
                    Array2D.copy x
                else
                    let y' = Array2D.copy y
                    BLAS.daxpy'(1., x, y')
                    y'
            member o.Mul_S_M(alpha, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let x' = Array2D.copy x
                    BLAS.dscal'(alpha, x')
                    x'
            member o.Sub_M_M(x, y) =
                if Array2D.isEmpty x then
                    (o :> BackEnd<float>).Mul_S_M(-1., y)
                elif Array2D.isEmpty y then
                    Array2D.copy x
                else
                    let x' = Array2D.copy x
                    BLAS.daxpy'(-1., y, x')
                    x'
            member o.Mul_M_M(x, y) =
                if (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 y)
                    BLAS.dgemm(1., x, y, 0., z)
                    z
            member o.Mul_Had_M_M(x, y) = // Non-BLAS
                if Array2D.isEmpty x then
                    Array2D.zeroCreate (Array2D.length1 y) (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 x)
                else
                    (o :> BackEnd<float>).Map2_F_M_M((*), x, y)
            member o.Mul_M_V(x, y) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                else
                    let z = Array.zeroCreate (Array2D.length1 x)
                    BLAS.dgemv(1., x, y, 0., z)
                    z
            member o.Mul_V_M(x, y) =
                if Array.isEmpty x then
                    Array.zeroCreate (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array.empty
                else
                    let z = Array.zeroCreate (Array2D.length2 y)
                    BLAS.dgemv'(1., y, x, 0., z)
                    z
            member o.Transpose_M(x) = // Non-BLAS
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let m = Array2D.length2 x
                    let n = Array2D.length1 x
                    Array2D.Parallel.init m n (fun i j -> x.[j, i])
            member o.Sum_M(x) = // Non-BLAS
                if Array2D.isEmpty x then
                    0.
                else
                    Array.Parallel.init (Array2D.length1 x) (fun i -> x.[i, *])
                    |> Array.Parallel.map Array.sum
                    |> Array.sum
            member o.Solve_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.dgesv(x, y)
            member o.SolveSymmetric_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    LAPACK.dsysv(x, y)
            member o.Diagonal_M(x) =
                if Array2D.isEmpty x then
                    Array.empty
                else
                    let n = min (Array2D.length1 x) (Array2D.length2 x)
                    Array.Parallel.init n (fun i -> x.[i, i])
            member o.Inverse_M(x) =
                if Array2D.isEmpty x then
                    Some(Array2D.empty)
                else
                    let x' = Array2D.copy x
                    let ipiv = LAPACK.dgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let inv = LAPACK.dgetri(x', ipiv)
                        match inv with
                        | Some(inv) -> Some(inv)
                        | _ -> None
                    | _ -> None
            member o.Det_M(x) =
                if Array2D.isEmpty x then
                    Some(0.)
                else
                    let x' = Array2D.copy x
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