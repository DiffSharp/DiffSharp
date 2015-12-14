#nowarn "9"
#nowarn "51"

namespace DiffSharp.Backend

open System
open System.Runtime.InteropServices
open FSharp.NativeInterop
open System.Security
open DiffSharp.Util

open ManagedCuda
open ManagedCuda.VectorTypes
open ManagedCuda.BasicTypes
open ManagedCuda.NVRTC
open ManagedCuda.CudaBlas

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

module Cuda =
    /// The cuBlas handle.
    let ctx = new CudaContext()
    printfn "Context=%A" ctx.Context
    printfn "Device=%A" ctx.Device
    printfn "DeviceId=%A" ctx.DeviceId
    printfn "Flags=%A" ctx.Flags
    printfn "IsContextOwner=%A" ctx.IsContextOwner

    let c = CudaBlas()
    let s = new CudaSolve.CudaSolveDense()

    module Numerics =
        let inline to_dev (host_ar: 't []) =
            let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
            d_a.CopyToDevice(host_ar)
            d_a

        let inline to_dev' (host_ar: 't [,]) =
            let d_a = new CudaDeviceVariable<'t>(SizeT host_ar.Length)    
            d_a.CopyToDevice(host_ar)
            d_a

        let inline to_host (dev_ar: CudaDeviceVariable<'t>) =
            let h_a = Array.zeroCreate<'t> (int dev_ar.Size)
            dev_ar.CopyToHost(h_a)
            h_a

        let new_dev<'t when 't: struct and 't: (new: unit -> 't) and 't:> System.ValueType> (n: int) =
            new CudaDeviceVariable<'t>(SizeT n)

        type Layout = // cblas.h: typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
            | R = 101
            | C = 102

        type Transpose = // cblas.h: typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
            | NT = 111
            | T = 112
            | CT = 113

        let nT = Transpose.NT
        let T = Transpose.T

        let inline sgeam2 transa transb (alpha: float32) (A:float32[,]) (beta: float32) (B:float32[,]) (C:float32[,]) =
            let a_row = if transa = nT then A |> Array2D.length1 else A |> Array2D.length2
            let a_col = if transa = nT then A |> Array2D.length2 else A |> Array2D.length1
            let b_row = if transb = nT then B |> Array2D.length1 else B |> Array2D.length2
            let b_col = if transb = nT then B |> Array2D.length2 else B |> Array2D.length1
        
            if a_row <> b_row then (failwithf "a_row <> b_row in sgeam2 (domatcopyT)! %i <> %i" a_row b_row)
            if a_col <> b_col then (failwithf "a_col <> b_col in sgeam2 (domatcopyT)! %i <> %i" a_col b_col)

            // For row major format, I invert the rows and columns.
            let lda = if transa = nT then a_col else a_row
            let ldb = if transa = nT then b_col else b_row
            let ldc = a_col

            use d_a = to_dev' A
            use d_b = if A = B then d_a else to_dev' B
            use d_c = to_dev' C

            let transa = if transa = nT then Operation.NonTranspose else Operation.Transpose
            let transb = if transb = nT then Operation.NonTranspose else Operation.Transpose

            // I also swap a_col and a_row in the call below.
            c.Geam(transa, transb, a_col, a_row, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc)
            d_c.CopyToHost(C)

        let inline dgeam2 transa transb (alpha: float) (A:float[,]) (beta: float) (B:float[,]) (C:float[,]) =
            let a_row = if transa = nT then A |> Array2D.length1 else A |> Array2D.length2
            let a_col = if transa = nT then A |> Array2D.length2 else A |> Array2D.length1
            let b_row = if transb = nT then B |> Array2D.length1 else B |> Array2D.length2
            let b_col = if transb = nT then B |> Array2D.length2 else B |> Array2D.length1
        
            if a_row <> b_row then (failwithf "a_row <> b_row in dgeam2 (domatcopyT)! %i <> %i" a_row b_row)
            if a_col <> b_col then (failwithf "a_col <> b_col in dgeam2 (domatcopyT)! %i <> %i" a_col b_col)

            // For row major format, I invert the rows and columns.
            let lda = if transa = nT then a_col else a_row
            let ldb = if transa = nT then b_col else b_row
            let ldc = a_col

            use d_a = to_dev' A
            use d_b = if A = B then d_a else to_dev' B
            use d_c = to_dev' C

            let transa = if transa = nT then Operation.NonTranspose else Operation.Transpose
            let transb = if transb = nT then Operation.NonTranspose else Operation.Transpose

            // I also swap a_col and a_row in the call below.
            c.Geam(transa, transb, a_col, a_row, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc)
            d_c.CopyToHost(C)

        // B <- alpha * transpose(A)
        let inline somatcopyT(alpha:float32, a:float32[,], b:float32[,]) = sgeam2 T T alpha a 0.0f a b

        // B <- alpha * transpose(A)
        let inline domatcopyT(alpha:float, a:float[,], b:float[,]) = dgeam2 T T alpha a 0.0 a b

        let isamax(x:float32[]) =
            use d_x = to_dev x
            let arg_incx = 1
            c.Max(d_x,arg_incx)

        // y <- alpha * x + y
        let saxpy(alpha:float32, x:float32[], y:float32[]) =
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_incx = 1
            let arg_incy = 1
            c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
            d_y.CopyToHost(y)

        // Y <- alpha * X + Y
        let saxpy'(alpha:float32, x:float32[,], y:float32[,]) =
            use d_x = to_dev' x
            use d_y = to_dev' y
            let arg_incx = 1
            let arg_incy = 1
            c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
            d_y.CopyToHost(y)

        let sscal(alpha: float32, x: float32[]) =
            use d_x = to_dev x
            let arg_incx = 1
            let arg_incy = 1
            c.Scale(alpha,d_x,arg_incx)
            d_x.CopyToHost(x)

        let sscal'(alpha: float32, x: float32[,]) =
            use d_x = to_dev' x
            let arg_incx = 1
            c.Scale(alpha,d_x,arg_incx)
            d_x.CopyToHost(x)

        let sdot(x: float32[], y: float32[]) =
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_incx = 1
            let arg_incy = 1
            c.Dot(d_x,arg_incx,d_y,arg_incy)

        let sger(alpha: float32, x:float32[], y:float32[], a:float32[,]) =
            use d_x = to_dev x
            use d_y = to_dev y
            use d_a = to_dev' a
            let arg_m = y.Length
            let arg_n = x.Length
            let arg_alpha = alpha
            let arg_incx = 1
            let arg_incy = 1
            let arg_lda = arg_m

            // As it happens, the wrapped c.Ger does not allow reversing the m and n arguments, so I have taken it out here.
            // This is an adjustment so it works for row major matrices.
            let _blasHandle = c.CublasHandle
            let _status = CudaBlasNativeMethods.cublasSger_v2(_blasHandle, arg_m, arg_n, ref alpha, d_x.DevicePointer, arg_incx, d_y.DevicePointer, arg_incy, d_a.DevicePointer, arg_lda)
            if (_status <> CublasStatus.Success) then raise (new CudaBlasException(_status))
            d_a.CopyToHost(a)

        let sasum(x:float32[]) =
            use d_x = to_dev x
            let arg_incx = 1
            c.AbsoluteSum(d_x,arg_incx)

        let snrm2(x:float32[]) =
            use d_x = to_dev x
            let arg_incx = 1
            c.Norm2(d_x,arg_incx)

        // O <- alpha * A * B + beta * O
        let sgemm(alpha:float32, a:float32[,], b:float32[,], beta:float32, o:float32[,]) =
            let d_a = to_dev' a
            let d_b = to_dev' b
            let d_o = to_dev' o
            // Order modified to work with row-major matrices and eliminate the need for transposing the result
            let m = Array2D.length1 a
            let n = Array2D.length2 b
            let k = Array2D.length1 b
            let arg_transa = Operation.NonTranspose
            let arg_transb = Operation.NonTranspose
            let arg_m = n
            let arg_n = m
            let arg_k = k
            let arg_alpha = alpha
            let arg_lda = n
            let arg_ldb = k
            let arg_beta = beta
            let arg_ldc = n
            c.Gemm(arg_transa,arg_transb,arg_m, arg_n, arg_k, arg_alpha, d_a, arg_lda, d_b, arg_ldb, arg_beta, d_o, arg_ldc)
            d_o.CopyToHost(o)

        // y <- alpha * A * x + beta * y
        let sgemv(alpha:float32, a:float32[,], x:float32[], beta:float32, y:float32[]) =
            use d_a = to_dev' a
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_trans = Operation.Transpose
            let arg_m = Array2D.length2 a
            let arg_n = Array2D.length1 a
            let arg_alpha = alpha
            let arg_lda = arg_m
            let arg_incx = 1
            let arg_beta = beta
            let arg_incy = 1
            c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
            d_y.CopyToHost(y)

        // y <- alpha * x * A + beta * y
        let sgemv'(alpha:float32, a:float32[,], x:float32[], beta:float32, y:float32[]) =
            use d_a = to_dev' a
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_trans = Operation.NonTranspose
            let arg_m = Array2D.length2 a
            let arg_n = Array2D.length1 a
            let arg_alpha = alpha
            let arg_lda = arg_m
            let arg_incx = 1
            let arg_beta = beta
            let arg_incy = 1
            c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
            d_y.CopyToHost(y)


        let idamax(x:float[]) =
            use d_x = to_dev x
            let arg_incx = 1
            c.Max(d_x,arg_incx)

        // y <- alpha * x + y
        let daxpy(alpha:float, x:float[], y:float[]) =
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_incx = 1
            let arg_incy = 1
            c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
            d_y.CopyToHost(y)

        // Y <- alpha * X + Y
        let daxpy'(alpha:float, x:float[,], y:float[,]) =
            use d_x = to_dev' x
            use d_y = to_dev' y
            let arg_incx = 1
            let arg_incy = 1
            c.Axpy(alpha,d_x,arg_incx,d_y,arg_incy)
            d_y.CopyToHost(y)

        let dscal(alpha: float, x: float[]) =
            use d_x = to_dev x
            let arg_incx = 1
            let arg_incy = 1
            c.Scale(alpha,d_x,arg_incx)
            d_x.CopyToHost(x)

        let dscal'(alpha: float, x: float[,]) =
            use d_x = to_dev' x
            let arg_incx = 1
            c.Scale(alpha,d_x,arg_incx)
            d_x.CopyToHost(x)

        let ddot(x: float[], y: float[]) =
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_incx = 1
            let arg_incy = 1
            c.Dot(d_x,arg_incx,d_y,arg_incy)

        let dger(alpha: float, x:float[], y:float[], a:float[,]) =
            use d_x = to_dev x
            use d_y = to_dev y
            use d_a = to_dev' a
            let arg_m = y.Length
            let arg_n = x.Length
            let arg_alpha = alpha
            let arg_incx = 1
            let arg_incy = 1
            let arg_lda = arg_m

            // As it happens, the wrapped c.Ger does not allow reversing the m and n arguments, so I have taken it out here.
            // This is an adjustment so it works for row major matrices.
            let _blasHandle = c.CublasHandle
            let _status = CudaBlasNativeMethods.cublasDger_v2(_blasHandle, arg_m, arg_n, ref alpha, d_x.DevicePointer, arg_incx, d_y.DevicePointer, arg_incy, d_a.DevicePointer, arg_lda)
            if (_status <> CublasStatus.Success) then raise (new CudaBlasException(_status))
            d_a.CopyToHost(a)

        let dasum(x:float[]) =
            use d_x = to_dev x
            let arg_incx = 1
            c.AbsoluteSum(d_x,arg_incx)

        let dnrm2(x:float[]) =
            use d_x = to_dev x
            let arg_incx = 1
            c.Norm2(d_x,arg_incx)

        // O <- alpha * A * B + beta * O
        let dgemm(alpha:float, a:float[,], b:float[,], beta:float, o:float[,]) =
            let d_a = to_dev' a
            let d_b = to_dev' b
            let d_o = to_dev' o
            // Order modified to work with row-major matrices and eliminate the need for transposing the result
            let m = Array2D.length1 a
            let n = Array2D.length2 b
            let k = Array2D.length1 b
            let arg_transa = Operation.NonTranspose
            let arg_transb = Operation.NonTranspose
            let arg_m = n
            let arg_n = m
            let arg_k = k
            let arg_alpha = alpha
            let arg_lda = n
            let arg_ldb = k
            let arg_beta = beta
            let arg_ldc = n
            c.Gemm(arg_transa,arg_transb,arg_m, arg_n, arg_k, arg_alpha, d_a, arg_lda, d_b, arg_ldb, arg_beta, d_o, arg_ldc)
            d_o.CopyToHost(o)

        // y <- alpha * A * x + beta * y
        let dgemv(alpha:float, a:float[,], x:float[], beta:float, y:float[]) =
            use d_a = to_dev' a
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_trans = Operation.Transpose
            let arg_m = Array2D.length2 a
            let arg_n = Array2D.length1 a
            let arg_alpha = alpha
            let arg_lda = arg_m
            let arg_incx = 1
            let arg_beta = beta
            let arg_incy = 1
            c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
            d_y.CopyToHost(y)

        // y <- alpha * x * A + beta * y
        let dgemv'(alpha:float, a:float[,], x:float[], beta:float, y:float[]) =
            use d_a = to_dev' a
            use d_x = to_dev x
            use d_y = to_dev y
            let arg_trans = Operation.NonTranspose
            let arg_m = Array2D.length2 a
            let arg_n = Array2D.length1 a
            let arg_alpha = alpha
            let arg_lda = arg_m
            let arg_incx = 1
            let arg_beta = beta
            let arg_incy = 1
            c.Gemv(arg_trans, arg_m, arg_n, arg_alpha, d_a, arg_lda, d_x, arg_incx, arg_beta, d_y, arg_incy)
            d_y.CopyToHost(y)

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

        // Compared to the LAPACK sgesv, there are slight numerical deviations on the order of 1e-4 relative to the size of the inputs for this function.
        // To get the equivalent of sgesv, getrf (LU factorization) and getrs (solver) have to be called seperately.

        // There is also a call to geam (matrix matrix add function) purely for the sake of transposition to column major format.
        // It might be worth considering going to column major fully as both OpenBlas, Lapack and now these Cuda library functions use column major natively.
        let sgesv(a:float32[,], b:float32[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "The matrix is not square"
            if m <> b.Length then failwith "The length of b does not equal the dimensions of a"

            let b' = Array.copy b

            use d_b = to_dev b
            let ipiv = Array.zeroCreate n
            use d_ipiv = new_dev<int> n

            let arg_n = n
            let arg_nrhs = 1
            let arg_lda = n
            let arg_ldb = n
    
            use d_nta = to_dev' a
            use d_a = new_dev<float32> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

            let Lwork = s.GetrfBufferSize(m,n,d_a,arg_lda)
            use workspace = new_dev<float32> Lwork
    
            use d_info = to_dev [|0|]
            s.Getrf(m,n,d_a,arg_lda,workspace,d_ipiv,d_info)

            let factorization_par = d_info.[SizeT 0]
            if factorization_par <> 0 then failwithf "Parameter %i in sgesv is incorrect." factorization_par
    
            s.Getrs(Operation.NonTranspose,arg_n,arg_nrhs,d_a,arg_lda,d_ipiv,d_b,arg_ldb,d_info)
            d_b.CopyToHost(b')
    
            if d_info.[SizeT 0] = 0 then
                Some(b')
            else
                None

        let ssysv(a:float32[,], b:float32[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "The matrix is not square"
            if m <> b.Length then failwith "The length of b does not equal the dimensions of a"

            let ipiv = Array.zeroCreate<float32> n
            let work = Array.zeroCreate<float32> 1

            use d_nta = to_dev' a
            use d_a = new_dev<float32> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,n,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

            use d_ipiv = new_dev<int> n
            use d_b = to_dev b

            let arg_n = n
            let arg_nrhs = 1
            let arg_lda = n
            let arg_ldb = n

            let Lwork = s.PotrfBufferSize(FillMode.Upper,n,d_nta,arg_lda)
            use d_work = new_dev<float32> Lwork
            use d_info = to_dev [|0|]

            s.Potrf(FillMode.Upper,arg_n,d_a,arg_lda,d_work,Lwork,d_info)

            let factorization_par = d_info.[SizeT 0]
            if factorization_par <> 0 then failwithf "Parameter %i in ssysv is incorrect." factorization_par

            s.Potrs(FillMode.Upper,arg_n,arg_nrhs,d_a,arg_lda,d_b,arg_ldb,d_info)

            if d_info.[SizeT 0] = 0 then
                Some(to_host d_b)
            else
                None

        let sgetrf(a:float32[,]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "The matrix is not square"

            use d_ipiv = new_dev<int> (min m n)

            let arg_m = m
            let arg_n = n
            let arg_lda = m
    
            use d_nta = to_dev' a
            use d_a = new_dev<float32> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

            let Lwork = s.GetrfBufferSize(m,n,d_a,arg_lda)
            use workspace = new_dev<float32> Lwork
    
            use d_info = to_dev [|0|]
            s.Getrf(m,n,d_a,arg_lda,workspace,d_ipiv,d_info)

            if d_info.[SizeT 0] = 0 then
                c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0f,d_a,n,d_a,n,0.0f,d_nta,n) // Transpose using geam.
                d_nta.CopyToHost(a)
                Some(to_host d_ipiv)
            else
                None


        // Strangely enough, cuSolver does not have a matrix inverse, but cuBlas does.
        // Has no transpose step as I assume it is intended to be called after sgetrf.

        // Does this function intend to mutate a and ipiv?
        // Given that it returns a value I am assuming that it does not.
        let sgetri(a:float32[,], ipiv:int[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "The matrix is not square"
            if m <> ipiv.Length then failwith "The length of ipiv does not equal the dimensions of a"

            use d_nta = to_dev' a
            use d_a = new_dev<float32> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,n,n,1.0f,d_nta,n,d_nta,n,0.0f,d_a,n) // Transpose using geam.

            use d_ar_a = new_dev<CUdeviceptr> 1
            d_ar_a.[SizeT 0] <- d_a.DevicePointer
            use d_ipiv = to_dev ipiv

            use d_work = new_dev<float32> (n * n)
            use d_ar_work = new_dev<CUdeviceptr> 1
            d_ar_work.[SizeT 0] <- d_work.DevicePointer

            let arg_n = n
            let arg_lda = n
            let arg_ldc = n
            let arg_lwork = n * n
            use d_info = to_dev [|0|]
            c.GetriBatchedS(arg_n,d_ar_a,arg_lda,d_ipiv,d_ar_work,arg_ldc,d_info,1)
            if d_info.[SizeT 0] = 0 then
                c.Geam(Operation.Transpose,Operation.Transpose,n,n,1.0f,d_work,n,d_work,n,0.0f,d_a,n) // Transpose using geam.
                d_a.CopyToHost(a)
                Some a
            else
                None



        // Compared to the LAPACK sgesv, there are slight numerical deviations on the order of 1e-4 relative to the size of the inputs for this function.
        // To get the equivalent of sgesv, getrf (LU factorization) and getrs (solver) have to be called seperately.

        // There is also a call to geam (matrix matrix add function) purely for the sake of transposition to column major format.
        // It might be worth considering going to column major fully as both OpenBlas, Lapack and now these Cuda library functions use column major natively.
        let dgesv(a:float[,], b:float[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "The matrix is not square"
            if m <> b.Length then failwith "The length of b does not equal the dimensions of a"

            let b' = Array.copy b

            use d_b = to_dev b
            let ipiv = Array.zeroCreate n
            use d_ipiv = new_dev<int> n

            let arg_n = n
            let arg_nrhs = 1
            let arg_lda = n
            let arg_ldb = n
    
            use d_nta = to_dev' a
            use d_a = new_dev<float> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0,d_nta,n,d_nta,n,0.0,d_a,n) // Transpose using geam.

            let Lwork = s.GetrfBufferSize(m,n,d_a,arg_lda)
            use workspace = new_dev<float> Lwork
    
            use d_info = to_dev [|0|]
            s.Getrf(m,n,d_a,arg_lda,workspace,d_ipiv,d_info)

            let factorization_par = d_info.[SizeT 0]
            if factorization_par <> 0 then failwithf "Parameter %i in sgesv is incorrect." factorization_par
    
            s.Getrs(Operation.NonTranspose,arg_n,arg_nrhs,d_a,arg_lda,d_ipiv,d_b,arg_ldb,d_info)
            d_b.CopyToHost(b')
    
            if d_info.[SizeT 0] = 0 then
                Some(b')
            else
                None

        let dsysv(a:float[,], b:float[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a
            let ipiv = Array.zeroCreate<float> n
            let work = Array.zeroCreate<float> 1

            if m <> n then failwith "The matrix is not square"
            if m <> b.Length then failwith "The length of b does not equal the dimensions of a"

            use d_nta = to_dev' a
            use d_a = new_dev<float> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,n,n,1.0,d_nta,n,d_nta,n,0.0,d_a,n) // Transpose using geam.

            use d_ipiv = new_dev<int> n
            use d_b = to_dev b

            let arg_n = n
            let arg_nrhs = 1
            let arg_lda = n
            let arg_ldb = n

            let Lwork = s.PotrfBufferSize(FillMode.Upper,n,d_nta,arg_lda)
            use d_work = new_dev<float> Lwork
            use d_info = to_dev [|0|]

            s.Potrf(FillMode.Upper,arg_n,d_a,arg_lda,d_work,Lwork,d_info)

            let factorization_par = d_info.[SizeT 0]
            if factorization_par <> 0 then failwithf "Parameter %i in ssysv is incorrect." factorization_par

            s.Potrs(FillMode.Upper,arg_n,arg_nrhs,d_a,arg_lda,d_b,arg_ldb,d_info)

            if d_info.[SizeT 0] = 0 then
                Some(to_host d_b)
            else
                None

        let dgetrf(a:float[,]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "The matrix is not square"

            use d_ipiv = new_dev<int> (min m n)

            let arg_m = m
            let arg_n = n
            let arg_lda = m
    
            use d_nta = to_dev' a
            use d_a = new_dev<float> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0,d_nta,n,d_nta,n,0.0,d_a,n) // Transpose using geam.

            let Lwork = s.GetrfBufferSize(m,n,d_a,arg_lda)
            use workspace = new_dev<float> Lwork
    
            use d_info = to_dev [|0|]
            s.Getrf(m,n,d_a,arg_lda,workspace,d_ipiv,d_info)

            if d_info.[SizeT 0] = 0 then
                c.Geam(Operation.Transpose,Operation.Transpose,m,n,1.0,d_a,n,d_a,n,0.0,d_nta,n) // Transpose using geam.
                d_nta.CopyToHost(a)
                Some(to_host d_ipiv)
            else
                None


        // Strangely enough, cuSolver does not have a matrix inverse, but cuBlas does.
        // Has no transpose step as I assume it is intended to be called after sgetrf.

        // Does this function intend to mutate a and ipiv?
        // Given that it returns a value I am assuming that it does not.
        let dgetri(a:float[,], ipiv:int[]) =
            let m = Array2D.length1 a
            let n = Array2D.length2 a

            if m <> n then failwith "The matrix is not square"
            if m <> ipiv.Length then failwith "The length of ipiv does not equal the dimensions of a"

            use d_nta = to_dev' a
            use d_a = new_dev<float> a.Length
            c.Geam(Operation.Transpose,Operation.Transpose,n,n,1.0,d_nta,n,d_nta,n,0.0,d_a,n) // Transpose using geam.

            use d_ar_a = new_dev<CUdeviceptr> 1
            d_ar_a.[SizeT 0] <- d_a.DevicePointer
            use d_ipiv = to_dev ipiv

            use d_work = new_dev<float> (n * n)
            use d_ar_work = new_dev<CUdeviceptr> 1
            d_ar_work.[SizeT 0] <- d_work.DevicePointer

            let arg_n = n
            let arg_lda = n
            let arg_ldc = n
            let arg_lwork = n * n
            use d_info = to_dev [|0|]
            c.GetriBatchedS(arg_n,d_ar_a,arg_lda,d_ipiv,d_ar_work,arg_ldc,d_info,1)
            if d_info.[SizeT 0] = 0 then
                c.Geam(Operation.Transpose,Operation.Transpose,n,n,1.0,d_work,n,d_work,n,0.0,d_a,n) // Transpose using geam.
                d_a.CopyToHost(a)
                Some a
            else
                None

    type Float32Backend() =
        do (ctx.Context |> ignore)

        interface Backend<float32> with
            // Numerics
            member o.Add_V_V(x, y) =
                if Array.isEmpty x then
                    Array.copy y
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let y' = Array.copy y
                    Numerics.saxpy(1.f, x, y')
                    y'
            // Numerics
            member o.Add_S_V(x, y) =
                if Array.isEmpty y then
                    Array.empty
                else
                    let x' = Array.create y.Length x
                    Numerics.saxpy(1.f, y, x')
                    x'
            // Numerics
            member o.Mul_S_V(alpha, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    let x' = Array.copy x
                    Numerics.sscal(alpha, x')
                    x'
            // Numerics
            member o.Sub_V_V(x, y) =
                if Array.isEmpty x then
                    (o :> Backend<float32>).Mul_S_V(-1.f, y)
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let x' = Array.copy x
                    Numerics.saxpy(-1.f, y, x')
                    x'
            // Numerics
            member o.Mul_Dot_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    0.f
                else
                    Numerics.sdot(x, y)
            // Numerics
            member o.Mul_Out_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate x.Length y.Length
                    Numerics.sger(1.f, x, y, z)
                    z

            // Non-Numerics
            member o.Sub_S_V(alpha, x) =
                if alpha = 0.f then 
                    (o :> Backend<float32>).Mul_S_V(-1.f, x)
                else
                    (o :> Backend<float32>).Map_F_V((fun v -> alpha - v), x)
            // Non-Numerics
            member o.Sub_V_S(x, alpha) =
                if alpha = 0.f then
                    x
                else
                    (o :> Backend<float32>).Map_F_V((fun v -> v - alpha), x)
            // Non-Numerics
            member o.Sub_S_M(alpha, x) =
                if alpha = 0.f then 
                    (o :> Backend<float32>).Mul_S_M(-1.f, x)
                else
                    (o :> Backend<float32>).Map_F_M((fun v -> alpha - v), x)
            // Non-Numerics
            member o.Sub_M_S(x, alpha) =
                if alpha = 0.f then
                    x
                else
                    (o :> Backend<float32>).Map_F_M((fun v -> v - alpha), x)
            // Non-Numerics
            member o.Map_F_V(f, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    Array.map f x
            // Non-Numerics
            member o.Map2_F_V_V(f, x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array.empty
                else
                    Array.map2 f x y
            // Non-Numerics
            member o.Map_F_M(f, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    Array2D.map f x
            // Non-Numerics
            member o.Map2_F_M_M(f, x, y) =
                if Array2D.isEmpty x || Array2D.isEmpty y then
                    Array2D.empty
                else
                    Array2D.map2 f x y
            // Numerics
            member o.L1Norm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    Numerics.sasum(x)
            // Numerics
            member o.L2Norm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    Numerics.snrm2(x)
            // Numerics
            member o.SupNorm_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    let i = Numerics.isamax(x)
                    abs x.[i - 1]
            // Non-Numerics
            member o.Sum_V(x) =
                if Array.isEmpty x then
                    0.f
                else
                    Array.sum x
            // Numerics
            member o.Add_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.copyFast y
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let y' = Array2D.copyFast y
                    Numerics.saxpy'(1.f, x, y')
                    y'
            // Numerics
            member o.Add_S_M(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                else
                    let x' = Array2D.create (Array2D.length1 y) (Array2D.length2 y) x
                    Numerics.saxpy'(1.f, y, x')
                    x'
            // Numerics
            member o.Add_V_MCols(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                elif Array.isEmpty x then
                    y
                else
                    let x' = (o :> Backend<float32>).RepeatReshapeCopy_V_MCols(Array2D.length2 y, x)
                    Numerics.saxpy'(1.f, y, x')
                    x'
            // Numerics
            member o.Mul_S_M(alpha, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let x' = Array2D.copyFast x
                    Numerics.sscal'(alpha, x')
                    x'
            // Numerics
            member o.Sub_M_M(x, y) =
                if Array2D.isEmpty x then
                    (o :> Backend<float32>).Mul_S_M(-1.f, y)
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let x' = Array2D.copyFast x
                    Numerics.saxpy'(-1.f, y, x')
                    x'
            // Numerics
            member o.Mul_M_M(x, y) =
                if (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 y)
                    Numerics.sgemm(1.f, x, y, 0.f, z)
                    z
            // Numerics
            member o.Mul_M_M_Add_V_MCols(x, y, z) =
                if Array.isEmpty z then
                    (o :> Backend<float32>).Mul_M_M(x, y)
                elif (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let n = (Array2D.length2 y)
                    let z' = (o :> Backend<float32>).RepeatReshapeCopy_V_MCols(n, z)
                    Numerics.sgemm(1.f, x, y, 1.f, z')
                    z'
            // Non-Numerics
            member o.Mul_Had_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.zeroCreate (Array2D.length1 y) (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 x)
                else
                    (o :> Backend<float32>).Map2_F_M_M((*), x, y)
            // Numerics
            member o.Mul_M_V(x, y) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                else
                    let z = Array.zeroCreate (Array2D.length1 x)
                    Numerics.sgemv(1.f, x, y, 0.f, z)
                    z
            // Numerics
            member o.Mul_M_V_Add_V(x, y, z) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                elif Array.isEmpty z then
                    (o :> Backend<float32>).Mul_M_V(x, y)
                else
                    let z' = Array.copy z
                    Numerics.sgemv(1.f, x, y, 1.f, z')
                    z'
            // Numerics
            member o.Mul_V_M(x, y) =
                if Array.isEmpty x then
                    Array.zeroCreate (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array.empty
                else
                    let z = Array.zeroCreate (Array2D.length2 y)
                    Numerics.sgemv'(1.f, y, x, 0.f, z)
                    z
            // Numerics extension
            member o.Transpose_M(x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let m = Array2D.length1 x
                    let n = Array2D.length2 x
                    let x' = Array2D.zeroCreate<float32> n m
                    Numerics.somatcopyT(1.f, x, x')
                    x'
            // Non-Numerics
            member o.Sum_M(x) =
                if Array2D.isEmpty x then
                    0.f
                else
                    (o :> Backend<float32>).ReshapeCopy_MRows_V(x) |> Array.sum
            // Numerics
            member o.Solve_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    Numerics.sgesv(x, y)
            // Numerics
            member o.SolveSymmetric_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    Numerics.ssysv(x, y)
            // Non-Numerics
            member o.Diagonal_M(x) =
                if Array2D.isEmpty x then
                    Array.empty
                else
                    let n = min (Array2D.length1 x) (Array2D.length2 x)
                    Array.init n (fun i -> x.[i, i])
            // Numerics
            member o.Inverse_M(x) =
                if Array2D.isEmpty x then
                    Some(Array2D.empty)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = Numerics.sgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let inv = Numerics.sgetri(x', ipiv)
                        match inv with
                        | Some(inv) -> Some(inv)
                        | _ -> None
                    | _ -> None
            // Numerics
            member o.Det_M(x) =
                if Array2D.isEmpty x then
                    Some(0.f)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = Numerics.sgetrf(x')
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
            // Non-Numerics
            member o.ReshapeCopy_MRows_V(x) =
                if Array2D.isEmpty x then
                    Array.empty<float32>
                else
                    let r = Array.zeroCreate<float32> x.Length
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float32>)
                    r
            // Non-Numerics
            member o.ReshapeCopy_V_MRows(m, x) =
                if Array.isEmpty x then
                    Array2D.empty<float32>
                else
                    let n = x.Length / m
                    let r = Array2D.zeroCreate<float32> m n
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float32>)
                    r
            // Non-Numerics
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
            // Non-Numerics
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
        do (ctx.Context |> ignore)
        
        interface Backend<float> with
            // Numerics
            member o.Add_V_V(x, y) =
                if Array.isEmpty x then
                    Array.copy y
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let y' = Array.copy y
                    Numerics.daxpy(1., x, y')
                    y'
            // Numerics
            member o.Add_S_V(x, y) =
                if Array.isEmpty y then
                    Array.empty
                else
                    let x' = Array.create y.Length x
                    Numerics.daxpy(1., y, x')
                    x'
            // Numerics
            member o.Mul_S_V(alpha, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    let x' = Array.copy x
                    Numerics.dscal(alpha, x')
                    x'
            // Numerics
            member o.Sub_V_V(x, y) =
                if Array.isEmpty x then
                    (o :> Backend<float>).Mul_S_V(-1., y)
                elif Array.isEmpty y then
                    Array.copy x
                else
                    let x' = Array.copy x
                    Numerics.daxpy(-1., y, x')
                    x'
            // Numerics
            member o.Mul_Dot_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    0.
                else
                    Numerics.ddot(x, y)
            // Numerics
            member o.Mul_Out_V_V(x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate x.Length y.Length
                    Numerics.dger(1., x, y, z)
                    z
            // Non-Numerics
            member o.Sub_S_V(alpha, x) =
                if alpha = 0. then 
                    (o :> Backend<float>).Mul_S_V(-1., x)
                else
                    (o :> Backend<float>).Map_F_V((fun v -> alpha - v), x)
            // Non-Numerics
            member o.Sub_V_S(x, alpha) =
                if alpha = 0. then
                    x
                else
                    (o :> Backend<float>).Map_F_V((fun v -> v - alpha), x)
            // Non-Numerics
            member o.Sub_S_M(alpha, x) =
                if alpha = 0. then 
                    (o :> Backend<float>).Mul_S_M(-1., x)
                else
                    (o :> Backend<float>).Map_F_M((fun v -> alpha - v), x)
            // Non-Numerics
            member o.Sub_M_S(x, alpha) =
                if alpha = 0. then
                    x
                else
                    (o :> Backend<float>).Map_F_M((fun v -> v - alpha), x)
            // Non-Numerics
            member o.Map_F_V(f, x) =
                if Array.isEmpty x then
                    Array.empty
                else
                    Array.map f x
            // Non-Numerics
            member o.Map2_F_V_V(f, x, y) =
                if Array.isEmpty x || Array.isEmpty y then
                    Array.empty
                else
                    Array.map2 f x y
            // Non-Numerics
            member o.Map_F_M(f, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    Array2D.map f x
            // Non-Numerics
            member o.Map2_F_M_M(f, x, y) =
                if Array2D.isEmpty x || Array2D.isEmpty y then
                    Array2D.empty
                else
                    Array2D.map2 f x y
            // Numerics
            member o.L1Norm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    Numerics.dasum(x)
            // Numerics
            member o.L2Norm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    Numerics.dnrm2(x)
            // Numerics
            member o.SupNorm_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    let i = Numerics.idamax(x)
                    abs x.[i - 1]
            // Non-Numerics
            member o.Sum_V(x) =
                if Array.isEmpty x then
                    0.
                else
                    Array.sum x
            // Numerics
            member o.Add_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.copyFast y
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let y' = Array2D.copyFast y
                    Numerics.daxpy'(1., x, y')
                    y'
            // Numerics
            member o.Add_S_M(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                else
                    let x' = Array2D.create (Array2D.length1 y) (Array2D.length2 y) x
                    Numerics.daxpy'(1., y, x')
                    x'
            // Numerics
            member o.Add_V_MCols(x, y) =
                if Array2D.isEmpty y then
                    Array2D.empty
                elif Array.isEmpty x then
                    y
                else
                    let x' = (o :> Backend<float>).RepeatReshapeCopy_V_MCols(Array2D.length2 y, x)
                    Numerics.daxpy'(1., y, x')
                    x'
            // Numerics
            member o.Mul_S_M(alpha, x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let x' = Array2D.copyFast x
                    Numerics.dscal'(alpha, x')
                    x'
            // Numerics
            member o.Sub_M_M(x, y) =
                if Array2D.isEmpty x then
                    (o :> Backend<float>).Mul_S_M(-1., y)
                elif Array2D.isEmpty y then
                    Array2D.copyFast x
                else
                    let x' = Array2D.copyFast x
                    Numerics.daxpy'(-1., y, x')
                    x'
            // Numerics
            member o.Mul_M_M(x, y) =
                if (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let z = Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 y)
                    Numerics.dgemm(1., x, y, 0., z)
                    z
            // Numerics
            member o.Mul_M_M_Add_V_MCols(x, y, z) =
                if Array.isEmpty z then
                    (o :> Backend<float>).Mul_M_M(x, y)
                elif (Array2D.isEmpty x) || (Array2D.isEmpty y) then
                    Array2D.empty
                else
                    let n = (Array2D.length2 y)
                    let z' = (o :> Backend<float>).RepeatReshapeCopy_V_MCols(n, z)
                    Numerics.dgemm(1., x, y, 1., z')
                    z'
            // Non-Numerics
            member o.Mul_Had_M_M(x, y) =
                if Array2D.isEmpty x then
                    Array2D.zeroCreate (Array2D.length1 y) (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array2D.zeroCreate (Array2D.length1 x) (Array2D.length2 x)
                else
                    (o :> Backend<float>).Map2_F_M_M((*), x, y)
            // Numerics
            member o.Mul_M_V(x, y) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                else
                    let z = Array.zeroCreate (Array2D.length1 x)
                    Numerics.dgemv(1., x, y, 0., z)
                    z
            // Numerics
            member o.Mul_M_V_Add_V(x, y, z) =
                if Array2D.isEmpty x then
                    Array.empty
                elif Array.isEmpty y then
                    Array.zeroCreate (Array2D.length1 x)
                elif Array.isEmpty z then
                    (o :> Backend<float>).Mul_M_V(x, y)
                else
                    let z' = Array.copy z
                    Numerics.dgemv(1., x, y, 1., z')
                    z'
            // Numerics
            member o.Mul_V_M(x, y) =
                if Array.isEmpty x then
                    Array.zeroCreate (Array2D.length2 y)
                elif Array2D.isEmpty y then
                    Array.empty
                else
                    let z = Array.zeroCreate (Array2D.length2 y)
                    Numerics.dgemv'(1., y, x, 0., z)
                    z
            // Numerics extension
            member o.Transpose_M(x) =
                if Array2D.isEmpty x then
                    Array2D.empty
                else
                    let m = Array2D.length1 x
                    let n = Array2D.length2 x
                    let x' = Array2D.zeroCreate<float> n m
                    Numerics.domatcopyT(1., x, x')
                    x'
            // Non-Numerics
            member o.Sum_M(x) =
                if Array2D.isEmpty x then
                    0.
                else
                    (o :> Backend<float>).ReshapeCopy_MRows_V(x) |> Array.sum
            // Numerics
            member o.Solve_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    Numerics.dgesv(x, y)
            // Numerics
            member o.SolveSymmetric_M_V(x, y) =
                if Array2D.isEmpty x || Array.isEmpty y then
                    None
                else
                    Numerics.dsysv(x, y)
            // Non-Numerics
            member o.Diagonal_M(x) =
                if Array2D.isEmpty x then
                    Array.empty
                else
                    let n = min (Array2D.length1 x) (Array2D.length2 x)
                    Array.init n (fun i -> x.[i, i])
            // Numerics
            member o.Inverse_M(x) =
                if Array2D.isEmpty x then
                    Some(Array2D.empty)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = Numerics.dgetrf(x')
                    match ipiv with
                    | Some(ipiv) ->
                        let inv = Numerics.dgetri(x', ipiv)
                        match inv with
                        | Some(inv) -> Some(inv)
                        | _ -> None
                    | _ -> None
            // Numerics
            member o.Det_M(x) =
                if Array2D.isEmpty x then
                    Some(0.)
                else
                    let x' = Array2D.copyFast x
                    let ipiv = Numerics.dgetrf(x')
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
            // Non-Numerics
            member o.ReshapeCopy_MRows_V(x) =
                if Array2D.isEmpty x then
                    Array.empty<float>
                else
                    let r = Array.zeroCreate<float> x.Length
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float>)
                    r
            // Non-Numerics
            member o.ReshapeCopy_V_MRows(m, x) =
                if Array.isEmpty x then
                    Array2D.empty<float>
                else
                    let n = x.Length / m
                    let r = Array2D.zeroCreate<float> m n
                    Buffer.BlockCopy(x, 0, r, 0, x.Length * sizeof<float>)
                    r
            // Non-Numerics
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
            // Non-Numerics
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
