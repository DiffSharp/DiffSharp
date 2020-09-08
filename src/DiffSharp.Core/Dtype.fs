namespace DiffSharp

/// <summary>
///   Represents the type of a device. 
/// </summary>
///
/// <remarks>
///   The numeric values used are as for LibTorch.
/// </remarks>
///
/// <namespacedoc>
///   <summary>Contains fundamental types for the DiffSharp tensor programming model, including Tensor and Shape, and the <c>dsharp</c> API.</summary>
/// </namespacedoc>
type DeviceType =
    | CPU = 0
    | CUDA = 1 // CUDA.
    | MKLDNN = 2 // Reserved for explicit MKLDNN
    | OPENGL = 3 // OpenGL
    | OPENCL = 4 // OpenCL
    | IDEEP = 5 // IDEEP.
    | HIP = 6 // AMD HIP
    | FPGA = 7 // FPGA
    | MSNPU = 8 // MSNPU
    | XLA = 9 // XLA / TPU


/// Represents a device specification.
[<Struct>]
type Device =
    | Device of DeviceType * int
    member x.DeviceType = (let (Device(a,_)) = x in a)
    member x.DeviceIndex = (let (Device(_,b)) = x in b)
    static member CPU = Device(DeviceType.CPU, -1)
    static member GPU = Device(DeviceType.CUDA, 0)

    member internal x.Code = (int x.DeviceType <<< 4) + x.DeviceIndex

    member internal x.Name =
       (match x.DeviceType with
        | DeviceType.CPU -> "cpu"
        | DeviceType.CUDA -> "cuda"
        | DeviceType.MKLDNN -> "mkldnn"
        | DeviceType.OPENGL -> "opengl"
        | DeviceType.OPENCL -> "opencl"
        | DeviceType.IDEEP -> "ideep"
        | DeviceType.HIP -> "hip"
        | DeviceType.FPGA -> "fpga"
        | DeviceType.MSNPU -> "msnpu"
        | DeviceType.XLA -> "xla"
        | _ -> failwith "unknown device type") + string x.DeviceIndex

/// Contains functions and settings related to device specifications.
module Device = 

    /// Get or set the default device used when creating tensors.  Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default : Device = Device(DeviceType.CPU, 0)

/// Represents a backend for DiffSharp tensors
[<RequireQualifiedAccess>]
type Backend =
    /// The reference backend 
    | Reference
    /// The LibTorch backend 
    | Torch
    /// Reserved for future use
    | Other of name: string * code: int

    member internal x.Code = 
        match x with 
        | Reference -> 0x000
        | Torch -> 0x0100
        | Other (_name, code) -> (code + 3) <<< 8

    /// Get the name of the backend
    member x.Name = 
        match x with 
        | Reference -> "Reference"
        | Torch -> "Torch"
        | Other (name, _) -> name

/// Contains functions and settings related to backend specifications.
module Backend = 
    let internal count = ref 0
    let internal codes = System.Collections.Concurrent.ConcurrentDictionary<string,Backend>()

    /// Register a new backend
    let Register name = codes.GetOrAdd(name, (fun _ -> incr count; Backend.Other(name, count.Value)))

    /// Get or set the default backend used when creating tensors.  Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default = Backend.Reference

/// Represents a storage type for elements of a tensor
[<Struct>]
type Dtype =
    //| Float16
    /// Store elements as 32-bit floating point numbers
    | Float32
    /// Store elements as 64-bit floating point numbers
    | Float64
    /// Store elements as 8-bit integers
    | Int8
    /// Store elements as 8-bit unsigned integers
    | Byte
    /// Store elements as 16-bit signed integers
    | Int16
    /// Store elements as 32-bit signed integers
    | Int32
    /// Store elements as 64-bit signed integers
    | Int64
    /// Store elements as booleans
    | Bool

    member internal x.Code =
        match x with
        //| Float16 -> 0x10000
        | Float32 -> 0x20000
        | Float64 -> 0x30000
        | Int8 -> 0x40000
        | Byte -> 0x50000
        | Int16 -> 0x60000
        | Int32 -> 0x70000
        | Int64 -> 0x80000
        | Bool -> 0x90000

    member internal x.Name =
        match x with
        //| Float16 -> "Float16"
        | Float32 -> "Float32"
        | Float64 -> "Float64"
        | Int8 -> "Int8"
        | Byte -> "Byte"
        | Int16 -> "Int16"
        | Int32 -> "Int32"
        | Int64 -> "Int64"
        | Bool -> "Bool"

    /// Get the .NET type that corresponds to this type when data is transferred to .NET
    member x.AsType () =
        match x with
        //| Float16 -> typeof<single>
        | Float32 -> typeof<single>
        | Float64 -> typeof<double>
        | Int8 -> typeof<int8>
        | Byte -> typeof<byte>
        | Int16 -> typeof<int16>
        | Int32 -> typeof<int32>
        | Int64 -> typeof<int64>
        | Bool -> typeof<bool>

    /// Gets the natural result of the Sum(), SumToSize() and Sum(dim) operation on this dtype
    member t.SummationType =
        match t with
        | Bool | Byte | Int8 | Int16 | Int32 | Int64 -> Dtype.Int64
        | dt -> dt

/// Contains functions and settings related to tensor element types
module Dtype =
    /// Matches all floating point tensor element types
    let (|FloatingPoint|_|) x =
        match x with
        | Float32 | Float64 -> Some()
        | _ -> None

    /// Matches all integral tensor element types
    let (|Integral|_|) x =
        match x with
        | Byte | Int8 | Int16 | Int32 | Int64 -> Some()
        | _ -> None

    /// Matches all integral or boolean tensor element types
    let (|IntegralOrBool|_|) x =
        match x with
        | Integral | Bool -> Some()
        | _ -> None

    /// Find the Dtype into which dtype1 and dtype2 can be widened
    let widen (dtype1: Dtype) (dtype2: Dtype) =
        if dtype1 = dtype2 then Some dtype1
        else
            match dtype1, dtype2 with 
            | Float64, _ | _, Float64 -> Some Float64
            | Float32, _ | _, Float32 -> Some Float32
            | Int64, _ | _, Int64 -> Some Int64
            | Int32, _ | _, Int32 -> Some Int32
            | Int16, _ | _, Int16 -> Some Int16
            | Int8, Bool | Bool, Int8 -> Some Int8
            | Byte, Bool | Bool, Byte -> Some Byte
            | Int8, Int8 -> Some Int8
            | Byte, Byte -> Some Byte
            | Bool, Bool -> Some Bool
            | Int8, Byte | Byte, Int8  -> None

    /// Convert System.Type to Dtype
    let ofType (ty: System.Type) =
        if ty.Equals(typeof<int32>) then Dtype.Int32
        elif ty.Equals(typeof<double>) then Dtype.Float64
        elif ty.Equals(typeof<single>) then Dtype.Float32
        elif ty.Equals(typeof<int64>) then Dtype.Int64
        elif ty.Equals(typeof<int16>) then Dtype.Int16
        elif ty.Equals(typeof<int8>) then Dtype.Int8
        elif ty.Equals(typeof<byte>) then Dtype.Byte
        elif ty.Equals(typeof<bool>) then Dtype.Bool
        else failwithf "unknown type '%A' used as tensor type" ty

    /// Get or set the default element type used when creating tensors.  Note, use <c>dsharp.config(...)</c> instead.
    let mutable Default = Dtype.Float32

/// Contains global functions and settings related to tensor element types, used when writing backends.
[<AutoOpen>]
module DtypeAutoOpens =

    /// Raise an exception indicating the given operation is not supported for the given tensor element type.
    let opNotSupported msg (dtype: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A" msg dtype)

    /// Raise an exception indicating the given operation is not supported for the given tensor device type.
    let opNotSupportedOnDeviceType msg (dtype: Dtype) (deviceType: DeviceType) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type %A on device type %A" msg dtype deviceType)

    /// Raise an exception indicating the given binary operation is not supported for the two given tensor element types.
    let opNotSupported2 msg (dtype1: Dtype) (dtype2: Dtype) =
        invalidOp (sprintf "operation '%s' not permitted on tensors of type (%A, %A)" msg dtype1 dtype2)

