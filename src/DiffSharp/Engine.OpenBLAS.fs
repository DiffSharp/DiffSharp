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

namespace DiffSharp.Engine


open System
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop
open System.Security
open System.Threading.Tasks
open DiffSharp.Util

type PinnedArray<'T when 'T : unmanaged> (array : 'T[]) =
    let h = GCHandle.Alloc (array, GCHandleType.Pinned)
    let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
    member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
    interface IDisposable with
        member this.Dispose () = h.Free ()

type PinnedArray2D<'T when 'T : unmanaged> (array : 'T[,]) =
    let h = GCHandle.Alloc (array, GCHandleType.Pinned)
    let ptr = Marshal.UnsafeAddrOfPinnedArrayElement(array, 0)
    member this.Ptr = NativePtr.ofNativeInt<'T>(ptr)
    interface IDisposable with
        member this.Dispose () = h.Free ()


module OpenBLAS =
    module BLAS =
        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="daxpy_")>]
        extern void daxpy_(int *n, double *a, double *x, int *incx, double *y, int *incy);

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="ddot_")>]
        extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);
    
        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dnrm2_")>]
        extern double dnrm2_(int *n, double *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dasum_")>]
        extern double dasum_(int *n, double *x, int *incx)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dscal_")>]
        extern void dscal_(int *n, double *alpha, double *x, int *incx)
  
        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dger_")>]
        extern void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgemv_")>]
        extern void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgemm_")>]
        extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);

    module LAPACK =
        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dgesv_")>]
        extern void dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info)

        [<SuppressUnmanagedCodeSecurity>]
        [<DllImport("libopenblas.dll", EntryPoint="dsysv_")>]
        extern void dsysv_(char *uplo, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, double *work, int *lwork, int *info)

    let mv_solve(a:float[,], b:float[]) =
        let a = Array2D.init (Array2D.length2 a) (Array2D.length1 a) (fun i j -> a.[j, i])
        let n = Array2D.length1 a
        let nrhs = 1
        let ipiv = Array.zeroCreate n
        let b = Array.copy b

        let mutable arg_n = n
        let mutable arg_nrhs = nrhs
        let mutable arg_lda = n
        let mutable arg_ldb = n
        let mutable arg_info = 0

        use arg_a = new PinnedArray2D<float>(a)
        use arg_ipiv = new PinnedArray<int>(ipiv)
        use arg_b = new PinnedArray<float>(b)

        LAPACK.dgesv_(&&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, &&arg_info)

        if arg_info = 0 then
            Some(b)
        else
            None

    let mv_solve_symmetric(a:float[,], b:float[]) =
        let a = Array2D.init (Array2D.length2 a) (Array2D.length1 a) (fun i j -> a.[j, i])
        let n = Array2D.length1 a
        let nrhs = 1
        let ipiv = Array.zeroCreate n
        let b = Array.copy b
        let work = Array.zeroCreate 1

        let mutable arg_uplo = 'U' // Assume upper triangular. TODO: check if LAPACK implementation requires the lower triangle to be zeroed
        let mutable arg_n = n
        let mutable arg_nrhs = nrhs
        let mutable arg_lda = n
        let mutable arg_ldb = n
        let mutable arg_lwork = 1
        let mutable arg_info = 0

        use arg_a = new PinnedArray2D<float>(a)
        use arg_ipiv = new PinnedArray<int>(ipiv)
        use arg_b = new PinnedArray<float>(b)
        use arg_work = new PinnedArray<float>(work)

        LAPACK.dsysv_(&&arg_uplo, &&arg_n, &&arg_nrhs, arg_a.Ptr, &&arg_lda, arg_ipiv.Ptr, arg_b.Ptr, &&arg_ldb, arg_work.Ptr, &&arg_lwork, &&arg_info)

        if arg_info = 0 then
            Some(b)
        else
            None

    let v_l1norm(a:float[]) =
        if Array.isEmpty a then
            0.
        else
            let n = a.Length
    
            let mutable arg_n = n
            let mutable arg_incx = 1

            use arg_x = new PinnedArray<float>(a)

            BLAS.dasum_(&&arg_n, arg_x.Ptr, &&arg_incx)

    let v_l2norm(a:float[]) =
        if Array.isEmpty a then
            0.
        else
            let n = a.Length
    
            let mutable arg_n = n
            let mutable arg_incx = 1

            use arg_x = new PinnedArray<float>(a)

            BLAS.dnrm2_(&&arg_n, arg_x.Ptr, &&arg_incx)

    // c <- c + alpha * a * b'
    let v_mul_outer_replace(alpha:float, a:float[], b:float[], c:float[,]) =
        let m = a.Length
        let n = b.Length

        let mutable arg_m = m
        let mutable arg_n = n
        let mutable arg_alpha = alpha
        let mutable arg_incx = 1
        let mutable arg_incy = 1
        let mutable arg_lda = m

        use arg_x = new PinnedArray<float>(a)
        use arg_y = new PinnedArray<float>(b)
        use arg_a = new PinnedArray2D<float>(c)

        BLAS.dger_(&&arg_m, &&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy, arg_a.Ptr, &&arg_lda)
        
        Array2D.init m n (fun i j -> c.[j,i])

    // c <- a * b'
    let v_mul_outer(a:float[], b:float[]) =
        if Array.isEmpty a || Array.isEmpty b then
            Array2D.empty
        else
            v_mul_outer_replace(1., a, b, Array2D.zeroCreate b.Length a.Length)

    // a <- alpha * a
    let v_scale_replace(alpha:float, a:float[]) =
        let n = a.Length

        let mutable arg_n = n
        let mutable arg_alpha = alpha
        let mutable arg_incx = 1

        use arg_x = new PinnedArray<float>(a)

        BLAS.dscal_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx)
        a

    // c <- alpha * a
    let v_scale(alpha:float, a:float[]) =
        if Array.isEmpty a then
            Array.empty
        else
            v_scale_replace(alpha, Array.copy a)     

    // b <- alpha * a + b
    let v_add_mul_replace(alpha:float, a:float[], b:float[]) =
        let n = a.Length

        let mutable arg_n = n
        let mutable arg_a = alpha
        let mutable arg_incx = 1
        let mutable arg_incy = 1

        use arg_x = new PinnedArray<float>(a)
        use arg_y = new PinnedArray<float>(b)
        
        BLAS.daxpy_(&&arg_n, &&arg_a, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)
        b

    // b <- a + b
    let v_add_replace(a:float[], b:float[]) =
        v_add_mul_replace(1., a, b)

    // c <- a + b
    let v_add(a:float[], b:float[]) =
        if Array.isEmpty a then 
            Array.copy b
        elif Array.isEmpty b then 
            Array.copy a
        else
            v_add_mul_replace(1., a, Array.copy b)

    let v_sub(a:float[], b:float[]) =
        if Array.isEmpty a then
            v_scale_replace(-1., Array.copy b)
        elif Array.isEmpty b then
            Array.copy a
        else
            v_add_mul_replace(-1., b, Array.copy a)

    let v_dot(a:float[], b:float[]) =
        if Array.isEmpty a || Array.isEmpty b then
            0.
        else
            let n = a.Length

            let mutable arg_n = n
            let mutable arg_incx = 1
            let mutable arg_incy = 1

            use arg_x = new PinnedArray<float>(a)
            use arg_y = new PinnedArray<float>(b)

            BLAS.ddot_(&&arg_n, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)

    // c <- alpha * a * b + beta * c
    let mv_mul_replace(alpha:float, a:float[,], b:float[], beta:float, c:float[]) =
        let m = Array2D.length2 a
        let n = Array2D.length1 a

        let mutable arg_trans = 't'
        let mutable arg_m = m
        let mutable arg_n = n
        let mutable arg_alpha = alpha
        let mutable arg_lda = m
        let mutable arg_incx = 1
        let mutable arg_beta = beta
        let mutable arg_incy = 1

        use arg_a = new PinnedArray2D<float>(a)
        use arg_x = new PinnedArray<float>(b)
        use arg_y = new PinnedArray<float>(c)

        BLAS.dgemv_(&&arg_trans, &&arg_m, &&arg_n, &&arg_alpha, arg_a.Ptr, &&arg_lda, arg_x.Ptr, &&arg_incx, &&arg_beta, arg_y.Ptr, &&arg_incy)
        c

    let mv_mul(a:float[,], b:float[]) =
        if Array2D.isEmpty a then
            Array.empty
        elif Array.isEmpty b then
            Array.empty
        else
            mv_mul_replace(1., a, b, 0., Array.zeroCreate (Array2D.length1 a))

    // c <- alpha * a * b + beta * c
    let m_mul_replace(alpha:float, a:float[,], b:float[,], beta:float, c:float[,]) =
        let m = Array2D.length1 a
        let k = Array2D.length2 a
        let n = Array2D.length2 b
 
        // Declare arguments for the call
        let mutable arg_transa = 't'
        let mutable arg_transb = 't'
        let mutable arg_m = m
        let mutable arg_n = n
        let mutable arg_k = k
        let mutable arg_alpha = alpha
        let mutable arg_ldk = k
        let mutable arg_ldn = n
        let mutable arg_beta = beta
        let mutable arg_ldm = m

        // Temporarily pin the arrays
        use arg_a = new PinnedArray2D<float>(a)
        use arg_b = new PinnedArray2D<float>(b)
        use arg_c = new PinnedArray2D<float>(c)

        // Invoke the native routine
        BLAS.dgemm_(&&arg_transa, &&arg_transb, &&arg_m, &&arg_n, &&arg_k, &&arg_alpha, arg_a.Ptr, &&arg_ldk, arg_b.Ptr, &&arg_ldn, &&arg_beta, arg_c.Ptr, &&arg_ldm)

        // Transpose the result to get m*n matrix 
        Array2D.init m n (fun i j -> c.[j,i])

    // Matrix multiply a and b
    let m_mul(a:float[,], b:float[,]) =
        if Array2D.isEmpty a || Array2D.isEmpty b then
            Array2D.empty
        else
            m_mul_replace(1., a, b, 1., Array2D.zeroCreate (Array2D.length1 a) (Array2D.length2 b))

    // b <- alpha * a + b
    let m_add_mul_replace(alpha:float, a:float[,], b:float[,]) =

        let n = (Array2D.length1 a) * (Array2D.length2 a)

        let mutable arg_n = n
        let mutable arg_a = alpha
        let mutable arg_incx = 1
        let mutable arg_incy = 1

        use arg_x = new PinnedArray2D<float>(a)
        use arg_y = new PinnedArray2D<float>(b)
        
        BLAS.daxpy_(&&arg_n, &&arg_a, arg_x.Ptr, &&arg_incx, arg_y.Ptr, &&arg_incy)
        b
        
    let m_add(a:float[,], b:float[,]) =
        m_add_mul_replace(1., a, Array2D.copy b)

    let m_sub(a:float[,], b:float[,]) =
        m_add_mul_replace(-1., b, Array2D.copy a)

    // a <- alpha * a
    let m_scale_replace(alpha:float, a:float[,]) =
        let n = (Array2D.length1 a) * (Array2D.length2 a)

        let mutable arg_n = n
        let mutable arg_alpha = alpha
        let mutable arg_incx = 1

        use arg_x = new PinnedArray2D<float>(a)

        BLAS.dscal_(&&arg_n, &&arg_alpha, arg_x.Ptr, &&arg_incx)
        a

    // c <- alpha * a
    let m_scale(alpha:float, a:float[,]) =
        if Array2D.isEmpty a then
            Array2D.empty
        else
            m_scale_replace(alpha, Array2D.copy a)     
