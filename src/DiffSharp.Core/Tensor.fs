// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

open DiffSharp.Backends
open DiffSharp.Util
open System

#nowarn "1182" // turn off compiler-generated unused variable warnings in this file only

/// <summary>
///   Represents a multi-dimensional data type containing elements of a single data type.
/// </summary>
///
/// <example>
///   A tensor can be constructed from a list or sequence using <see cref="M:DiffSharp.dsharp.tensor(System.Object)" />
///
///  <code>
///    let t = dsharp.tensor([[1.; -1.]; [1.; -1.]])
///  </code>
/// </example>
[<CustomEquality; CustomComparison>]
type Tensor = 
    internal 
    | TensorC of primalRaw:RawTensor
    | TensorF of primal:Tensor * derivative:Tensor * nestingTag:uint32
    | TensorR of primal:Tensor * derivative:(Tensor ref) * parentOp:TensorOp * fanout:(uint32 ref) * nestingTag:uint32

    /// Gets the value of the tensor ignoring its first derivative
    member t.primal =
        match t with
        | TensorC(_) -> t
        | TensorF(tp,_,_) -> tp
        | TensorR(tp,_,_,_,_) -> tp

    /// Gets the value of the tensor ignoring all its derivatives
    member t.primalDeep =
        match t with
        | TensorC(_) -> t
        | TensorF(tp,_,_) -> tp.primalDeep
        | TensorR(tp,_,_,_,_) -> tp.primalDeep

    /// Gets the raw value of the tensor ignoring all its derivatives
    member t.primalRaw =
        match t with
        | TensorC(tp) -> tp
        | TensorF(tp,_,_) -> tp.primalRaw
        | TensorR(tp,_,_,_,_) -> tp.primalRaw

    /// Gets the differentiation nesting tag of the tensor
    member t.nestingTag =
        match t with
        | TensorC(_) -> failwithf "Cannot get nesting tag of constant tensor"
        | TensorF(_,_,tt) -> tt
        | TensorR(_,_,_,_,tt) -> tt

    /// Converts the tensor to a new tensor with the given <see cref="T:DiffSharp.Dtype"/>
    member t.cast(dtype) =
        if t.dtype = dtype then t else
        match t with
        | TensorC(tp) -> TensorC(tp.Cast(dtype))
        | TensorF(_) -> failwith "Cannot cast TensorF - do not cast during differentiation"
        | TensorR(_) -> failwith "Cannot cast TensorR - do not cast during differentiation"

    /// Converts the tensor to a new tensor with the given system type
    member t.cast<'T>() =
        match box Unchecked.defaultof<'T> with
        | :? float32 -> t.cast(Dtype.Float32)
        | :? double -> t.cast(Dtype.Float64)
        | :? int32 -> t.cast(Dtype.Int32)
        | :? int64 -> t.cast(Dtype.Int64)
        | :? int16 -> t.cast(Dtype.Int16)
        | :? int8 -> t.cast(Dtype.Int8)
        | :? byte -> t.cast(Dtype.Byte)
        | :? bool -> t.cast(Dtype.Bool)
        | _ -> failwithf "Cannot cast tensor with type %A to given type %A" t.dtype typeof<'T>

    /// Returns a new tensor with the same contents moved to the given backend
    member t.move(backend: Backend) =
        // If a backend move is needed then first move to the CPU
        let t = 
            if t.backend = backend then t
            elif t.device = Device.CPU then t
            else t.move(Device.CPU)

        if t.backend = backend then t else
        match t with
        | TensorC(tp) -> 
            let tpflat = tp.ViewT([|tp.Nelement|])
            let tpflatValues = tpflat.ToValues()
            TensorC(tp.CreateLike(tpflatValues, backend=backend).ViewT(tp.Shape))
        | TensorF(_) -> failwith "Cannot move TensorF - do not move during differentiation"
        | TensorR(_) -> failwith "Cannot move TensorR - do not move during differentiation"

    /// Returns a new tensor with the same contents moved to the given device
    member t.move(device: Device) =
        if t.device = device then t else
        match t with
        | TensorC(tp) -> TensorC(tp.MoveTo(device))
        | TensorF(_) -> failwith "Cannot move TensorF - do not move during differentiation"
        | TensorR(_) -> failwith "Cannot move TensorR - do not move during differentiation"

    /// Returns a new tensor with the same contents moved to the given configuration
    member t.move(?device:Device, ?dtype:Dtype, ?backend:Backend) =
        let t = match backend with None -> t | Some backend -> t.move(backend)
        let t = match dtype with None -> t | Some dtype -> t.cast(dtype)
        let t = match device with None -> t | Some device -> t.move(device)
        t

    member internal t.castAfterSummation(?dtype:Dtype) =
        match dtype with
        | None -> t
        | Some dt -> t.cast(dt)

    /// Returns a new tensor with the same contents moved to the CPU
    member t.cpu() = t.move(Device.CPU)

    /// Returns a new tensor with the same contents moved to the primary GPU device
    member t.gpu() = t.move(Device.GPU)

    /// Returns a new tensor with each element converted to type bool
    member t.bool() = t.cast(Dtype.Bool)

    /// Returns a new tensor with each element converted to type int8
    member t.int8() = t.cast(Dtype.Int8)

    /// Returns a new tensor with each element converted to type int16
    member t.int16() = t.cast(Dtype.Int16)

    /// Returns a new tensor with each element converted to type int32
    member t.int32() = t.cast(Dtype.Int32)

    /// Returns a new tensor with each element converted to type int32
    member t.int() = t.cast(Dtype.Int32)

    /// Returns a new tensor with each element converted to type int64
    member t.int64() = t.cast(Dtype.Int64)

    /// Returns a new tensor with each element converted to type float16
    member t.float16() = t.cast(Dtype.Float16)

    /// Returns a new tensor with each element converted to type bfloat16
    member t.bfloat16() = t.cast(Dtype.BFloat16)

    /// Returns a new tensor with each element converted to type float32
    member t.float32() = t.cast(Dtype.Float32)

    /// Returns a new tensor with each element converted to type float64
    member t.float64() = t.cast(Dtype.Float64)

    /// Returns a new tensor with each element converted to type float64
    member t.float() = t.cast(Dtype.Float64)

    /// Returns a new tensor with each element converted to type float64
    member t.double() = t.cast(Dtype.Float64)

    /// Returns a new tensor with each element converted to type float64
    member t.byte() = t.cast(Dtype.Byte)

    /// Gets the element type of the tensor
    member t.dtype = t.primalRaw.Dtype

    /// Gets the device of the tensor
    member t.device = t.primalRaw.Device

    /// Gets the device type of the tensor
    member t.deviceType = t.primalRaw.Device.DeviceType

    /// Gets the backend of the tensor
    member t.backend = t.primalRaw.Backend

    /// Gets the differentiation depth of the tensor
    member t.depth =
        let rec depth x d =
            match x with
            | TensorC(_) -> d
            | TensorF(tp,_,_) -> depth tp (d + 1)
            | TensorR(tp,_,_,_,_) -> depth tp (d + 1)
        depth t 0

    /// Gets the parent operation of a tensor used in reverse-mode differentiation
    member t.parentOp =
        match t with
        | TensorC(_) -> failwith "Cannot get parent operation of constant Tensor"
        | TensorF(_)-> failwith "Cannot get parent operation of TensorF"
        | TensorR(_,_,o,_,_) -> o

    /// Gets or sets the derivative of a tensor used in differentiation
    member t.derivative
        with get() =
            match t with
            | TensorC(_) -> failwith "Cannot get derivative of constant Tensor"
            | TensorF(_,td,_) -> td
            | TensorR(_,td,_,_,_) -> !td
        and set(value) =
            match t with
            | TensorC(_) -> failwith "Cannot set derivative of constant Tensor"
            | TensorF(_) -> failwith "Cannot set derivative of TensorF"
            | TensorR(_,td,_,_,_) -> td := value

    member t.derivativeDeep =
        match t with
        | TensorC(_) -> failwith "Cannot get derivative of constant Tensor"
        | TensorF(_,td,_) -> 
            match td with
            | TensorC(_) -> td
            | _ -> td.derivativeDeep
        | TensorR(_,td,_,_,_) -> 
            match !td with
            | TensorC(_) -> !td
            | _ -> (!td).derivativeDeep

    /// Gets the fanout of a tensor used in reverse-mode differentiation
    member t.fanout
        with get() =
            match t with
            | TensorC(_) -> failwith "Cannot get fanout of constant Tensor"
            | TensorF(_) -> failwith "Cannot get fanout of TensorF"
            | TensorR(_,_,_,f,_) -> !f
        and set(value) =
            match t with
            | TensorC(_) -> failwith "Cannot set fanout of constant Tensor"
            | TensorF(_) -> failwith "Cannot set fanout of TensorF"
            | TensorR(_,_,_,f,_) -> f := value

    /// <summary>
    ///  Returns the input tensor with added support for forward-mode automatic differentiation.
    /// </summary>
    /// <remarks>
    ///  Any tensors produced using this tensor will have attached derivatives for forward mode propagation.
    ///  The current global nesting level is used for nested differentiation.
    /// </remarks>
    member t.forwardDiff(derivative:Tensor, ?nestingTag:uint32) = 
        if not t.dtype.IsFloatingPoint then failwithf "Only tensors with floating dtype can be differentiated. Tensor has dtype %A." t.dtype
        let nestingTag = defaultArg nestingTag GlobalNestingLevel.Current
        if t.shape <> derivative.shape then
            failwithf "Expecting derivative of same shape with primal. primal: %A, derivative: %A" t derivative
        TensorF(t, derivative, nestingTag)

    /// <summary>
    ///  Returns the input tensor with added support for reverse-mode automatic differentiation.
    /// </summary>
    /// <param name="derivative">The derivative (adjoint) to assign to the new reverse-mode tensor. Defaults to an empty placeholder tensor.</param>
    /// <param name="nestingTag">The level nestingTag for nested differentiation. Defaults to the current global nesting level</param>
    /// <remarks>
    ///  Any tensors produced using this tensor will also support reverse-mode propagation. After the completion
    ///  of the corresponding <c>reverse</c> operation on the overall result tensor, the computed derivative
    ///  will be available. 
    /// </remarks>
    member t.reverseDiff(?derivative:Tensor, ?nestingTag:uint32) =
        if not t.dtype.IsFloatingPoint then failwithf "Only tensors with floating dtype can be differentiated. Tensor has dtype %A." t.dtype
        let derivative = defaultArg derivative (t.zerosLike([0]))
        if derivative.nelement <> 0 && derivative.shape <> t.shape then failwithf "Expecting derivative shape (%A) to match the tensor shape (%A)" derivative.shape t.shape
        let nestingTag = defaultArg nestingTag GlobalNestingLevel.Current
        TensorR(t, ref derivative, NewT, ref 0u, nestingTag)

    ///  Returns the input tensor but with any support for automatic differentiation removed.
    member t.noDiff() = t.primalDeep

    /// Indicates if a tensor is taking part in forward-mode differentiation
    member t.isForwardDiff =
        match t with
        | TensorF(_) -> true
        | _ -> false

    /// Indicates if a tensor is taking part in reverse-mode differentiation
    member t.isReverseDiff =
        match t with
        | TensorR(_) -> true
        | _ -> false

    /// Indicates if a tensor is a constant, meaning that it is not taking part in forward or reverse-mode differentiation
    member t.isNoDiff =
        match t with
        | TensorC(_) -> true
        | _ -> false

    /// Gets the shape of the tensor
    member t.shape = t.primalRaw.Shape

    member internal t.shapeFullBounds = shapeToFullBounds(t.shape)

    /// Gets the number of dimensions of the tensor
    member t.dim = t.primalRaw.Dim

    /// Gets the number of elements in the tensor
    member t.nelement = t.primalRaw.Nelement

    /// Returns the value of a scalar tensor as an object
    member t.toScalar() = t.primalRaw.ToScalar()

    /// Returns the value of a (non-scalar) tensor as an array
    member t.toArray() = t.primalRaw.ToArray()

    /// Returns the value of a 1D tensor as a 1D array
    member t.toArray1D<'T>() = 
        if t.dim <> 1 then failwithf "Cannot convert tensor with shape %A to 1D array" t.shape
        t.cast<'T>().toArray() :?> 'T[]

    /// Returns the value of a 2D tensor as a 2D array
    member t.toArray2D<'T>() = 
        if t.dim <> 2 then failwithf "Cannot convert tensor with shape %A to 2D array" t.shape
        t.cast<'T>().toArray() :?> 'T[,]

    /// Returns the value of a 3D tensor as a 3D array
    member t.toArray3D<'T>() = 
        if t.dim <> 3 then failwithf "Cannot convert tensor with shape %A to 3D array" t.shape
        t.cast<'T>().toArray() :?> 'T[,,]

    /// Returns the value of a 4D tensor as a 4D array
    member t.toArray4D<'T>() = 
        if t.dim <> 4 then failwithf "Cannot convert tensor with shape %A to 4D array" t.shape
        t.cast<'T>().toArray() :?> 'T[,,,]      

    /// Returns the value of a 5D tensor as a 5D array
    member t.toArray5D<'T>() = 
        if t.dim <> 5 then failwithf "Cannot convert tensor with shape %A to 5D array" t.shape
        t.cast<'T>().toArray()

    /// Returns the value of a 6D tensor as a 6D array
    member t.toArray6D<'T>() = 
        if t.dim <> 6 then failwithf "Cannot convert tensor with shape %A to 6D array" t.shape
        t.cast<'T>().toArray()

    /// Indicates if two tensors have the same differentiation type
    member t1.isSameDiffType(t2:Tensor) =
        match t1, t2 with
        | TensorC(_), TensorC(_) -> true
        | TensorC(_), TensorF(_) -> false
        | TensorC(_), TensorR(_) -> false
        | TensorF(_), TensorC(_) -> false
        | TensorF(_), TensorF(_) -> true
        | TensorF(_), TensorR(_) -> false
        | TensorR(_), TensorC(_) -> false
        | TensorR(_), TensorF(_) -> false
        | TensorR(_), TensorR(_) -> true

    /// <summary>Saves the tensor to the given file using a bespoke binary format.</summary>
    /// <remarks>
    ///   The binary format records the elements, backend, element type and shape. It does not record the device.
    ///   The format used may change from version to version of DiffSharp.
    /// </remarks>
    member t.save(fileName:string) = saveBinary t fileName

    /// <summary>Loads the tensor from the given file using the given element type and configuration.</summary>
    ///
    /// <param name="fileName">The file from which to load the tensor.</param>
    /// <param name="device">The device of the resulting tensor. Defaults to the current default device.</param>
    /// <param name="dtype">The element type of the resulting tensor. Defaults to the element type of the saved tensor.</param>
    /// <param name="backend">The device of the resulting tensor. Defaults to the current default backend.</param>
    ///
    /// <remarks>
    ///    The backend at the time of saving the tensor must be available when the tensor is reloaded.
    ///    The tensor is first loaded into that backend and then moved. As a result, intermediate tensors may be created
    ///    in the process of reloading.
    /// </remarks>
    static member load(fileName:string, ?device: Device, ?dtype: Dtype, ?backend: Backend):Tensor =
        let t : Tensor = loadBinary fileName
        let device = defaultArg device Device.Default
        let dtype = defaultArg dtype t.dtype
        let backend = defaultArg backend Backend.Default
        t.move(device=device, dtype=dtype, backend=backend)

    /// Returns the tensor after min-max scaling
    member t.normalize() =
        let min = t.min()
        let range = t.max() - min
        if range = t.zeroLike() then
            t.zerosLike()
        else
            (t - min) / range

    /// Returns the tensor after standardization (z-score normalization)
    member t.standardize() =
        let stddev:Tensor = t.stddev()
        if stddev = t.zeroLike() || stddev.hasnan() then
            t.zerosLike()
        else
            (t - t.mean()) / stddev

    /// Returns a string summarising the tensor
    member t.summary() =
        match t with
        | TensorC(_) -> sprintf "Tensor %A" t.shape
        | TensorF(_) -> sprintf "TensorF %A" t.shape
        | TensorR(_,_,o,_,_) -> 
            let c, _ = Reflection.FSharpValue.GetUnionFields(o, typeof<TensorOp>)
            let fields = c.GetFields()
            sprintf "TensorR %A %s" t.shape c.Name

    /// A debugging routine that returns the ancestors of a tensor involved in reverse-mode automatic differentiation
    member t.ancestors() =
        let mutable p = []
        let rec ancestors (t:obj) d =
            match t with
            | :? Tensor as t ->
                p <- p |> List.append [t]
                match t with
                | TensorC(_) -> sprintf "Tensor %A" t.shape
                | TensorF(_) -> sprintf "TensorF %A" t.shape
                | TensorR(_,_,o,_,_) -> 
                    let c, _ = Reflection.FSharpValue.GetUnionFields(o, typeof<TensorOp>)
                    let fields = c.GetFields()
                    let mutable ret = sprintf "TensorR %A %s" t.shape (o.ToString())
                    for field in fields do
                        let fv = field.GetValue(o)
                        if fv :? Tensor then 
                            ret <- ret + sprintf "\n%s%s" (String.replicate d " ") (ancestors fv (d+1))
                    ret
            | :? (Tensor array) as ts ->
                // p <- p |> List.append (ts |> Array.toList)
                let mutable ret = ""
                let mutable prefix = ""
                for t in ts do
                    ret <- ret + sprintf "%s%s%s" prefix (String.replicate d " ") (ancestors t (d+1))
                    prefix <- "\n"
                ret
            // | _ -> indentNewLines (sprintf "%A" t) d
            | _ -> ""
        let ps = ancestors t 1
        p |> List.rev, ps

    override t.ToString() = 
        let rec fmt postfix (t: Tensor) =
            match t with
            | TensorC(p) -> p.Print(postfix)
            | TensorF(tp,_,_) -> fmt (postfix + ":fwd") tp
            | TensorR(tp,_,_,_,_) -> fmt (postfix + ":rev") tp
        fmt "" t

    override t.Equals(other) =
        match other with
        | :? Tensor as tensor -> t.primalRaw.Equals(tensor.primalRaw)
        | _ -> false

    override t.GetHashCode() = hash t.primalRaw

    interface System.IComparable with
        override t.CompareTo(other) =
            match other with
            | :? Tensor as tensor -> 
                if t.dim = tensor.dim && t.dim = 0 then
                    (t.primalRaw :> System.IComparable).CompareTo(tensor.primalRaw)
                else
                    failwith "Cannot compare non-scalar Tensors"
            | _ -> failwith "Cannot compare Tensor with another type"

    /// Get the scalar zero tensor for the current configuration
    static member Zero = TensorC(RawTensor.Zero())

    /// Get the scalar one tensor for the current configuration
    static member One = TensorC(RawTensor.One())

    /// Convert a scalar tensor to a float32 value
    static member op_Explicit(tensor:Tensor):single = tensor.toScalar().toSingle()

    /// Convert a scalar tensor to a float64 value
    static member op_Explicit(tensor:Tensor):double = tensor.toScalar().toDouble()

    /// Convert a scalar tensor to a byte value
    static member op_Explicit(tensor:Tensor):byte = tensor.toScalar().toByte()

    /// Convert a scalar tensor to a signed byte value
    static member op_Explicit(tensor:Tensor):int8 = tensor.toScalar().toSByte()

    /// Convert a scalar tensor to an int16 value
    static member op_Explicit(tensor:Tensor):int16 = tensor.toScalar().toInt16()

    /// Convert a scalar tensor to an int32 value
    static member op_Explicit(tensor:Tensor):int32 = tensor.toScalar().toInt32()

    /// Convert a scalar tensor to an int64 value
    static member op_Explicit(tensor:Tensor):int64 = tensor.toScalar().toInt64()

    /// Convert a scalar tensor to a boolean value
    static member op_Explicit(tensor:Tensor):bool = tensor.toScalar().toBool()

    interface System.IConvertible with
        override t.GetTypeCode() =
            match t.dtype with 
            | Dtype.Byte -> TypeCode.Byte
            | Dtype.Int8 -> TypeCode.SByte
            | Dtype.Int16 -> TypeCode.Int16
            | Dtype.Int32 -> TypeCode.Int32
            | Dtype.Int64 -> TypeCode.Int64
            | Dtype.Float32 -> TypeCode.Single
            | Dtype.Float64 -> TypeCode.Double
            | Dtype.Bool -> TypeCode.Boolean
            | Dtype.BFloat16 -> TypeCode.Single
            | Dtype.Float16 -> TypeCode.Single

        override t.ToSingle(fmt) = t.toScalar().ToSingle(fmt)
        override t.ToDouble(fmt) = t.toScalar().ToDouble(fmt)
        override t.ToByte(fmt) = t.toScalar().ToByte(fmt)
        override t.ToSByte(fmt) = t.toScalar().ToSByte(fmt)
        override t.ToInt16(fmt) = t.toScalar().ToInt16(fmt)
        override t.ToInt32(fmt) = t.toScalar().ToInt32(fmt)
        override t.ToInt64(fmt) = t.toScalar().ToInt64(fmt)
        override t.ToBoolean(fmt) = t.toScalar().ToBoolean(fmt)
        override t.ToChar(fmt) = t.toScalar().ToChar(fmt)
        override t.ToDateTime(fmt) = t.toScalar().ToDateTime(fmt)
        override t.ToDecimal(fmt) = t.toScalar().ToDecimal(fmt)
        override t.ToString(fmt) = t.toScalar().ToString(fmt)
        override t.ToType(ty, fmt) = t.toScalar().ToType(ty, fmt)
        override t.ToUInt16(fmt) = t.toScalar().ToUInt16(fmt)
        override t.ToUInt32(fmt) = t.toScalar().ToUInt32(fmt)
        override t.ToUInt64(fmt) = t.toScalar().ToUInt64(fmt)

    /// Convert a scalar tensor to a float32 value
    member t.toSingle() = t.toScalar().toSingle()

    /// Convert a scalar tensor to a float64 value
    member t.toDouble() = t.toScalar().toDouble()

    /// Convert a scalar tensor to a byte value
    member t.toByte() = t.toScalar().toByte()

    /// Convert a scalar tensor to a signed byte value
    member t.toSByte() = t.toScalar().toSByte()

    /// Convert a scalar tensor to an int16 value
    member t.toInt16() = t.toScalar().toInt16()

    /// Convert a scalar tensor to an int32 value
    member t.toInt32() = t.toScalar().toInt32()

    /// Convert a scalar tensor to an int64 value
    member t.toInt64() = t.toScalar().toInt64()

    /// Convert a scalar tensor to a boolean value
    member t.toBool() = t.toScalar().toBool()

    /// Returns the size in bytes of an individual element in this tensor. Depending on dtype, backend configuration, this is not guaranteed to be correct and can behave differently in different runtime environments.
    member t.elementSize =
        let bitsPerElement =
            match t.backend, t.dtype with
            | Backend.Reference, Dtype.BFloat16 -> 32 // Backed by float32
            | Backend.Reference, Dtype.Float16 -> 32 // Backed by float32
            | Backend.Reference, Dtype.Float32 -> 32
            | Backend.Reference, Dtype.Float64 -> 64
            | Backend.Reference, Dtype.Int8 -> 8
            | Backend.Reference, Dtype.Byte -> 8
            | Backend.Reference, Dtype.Int16 -> 16
            | Backend.Reference, Dtype.Int32 -> 32
            | Backend.Reference, Dtype.Int64 -> 64
            | Backend.Reference, Dtype.Bool -> 8 // Not reliable https://stackoverflow.com/a/28515361
            | Backend.Torch, Dtype.BFloat16 -> 16
            | Backend.Torch, Dtype.Float16 -> 16
            | Backend.Torch, Dtype.Float32 -> 32
            | Backend.Torch, Dtype.Float64 -> 64
            | Backend.Torch, Dtype.Int8 -> 8
            | Backend.Torch, Dtype.Byte -> 8
            | Backend.Torch, Dtype.Int16 -> 16
            | Backend.Torch, Dtype.Int32 -> 32
            | Backend.Torch, Dtype.Int64 -> 64
            | Backend.Torch, Dtype.Bool -> 8 // https://github.com/pytorch/pytorch/issues/41571
            | _ -> failwithf "Unknown backend, dtype configuration to compute memory size"
        bitsPerElement / 8

    /// Returns the size in bytes of the total memory used by this tensor. Depending on dtype, backend configuration, this is not guaranteed to be correct and can behave differently in different runtime environments.
    member t.memorySize = t.nelement * t.elementSize

    /// Indicates if two tensors have the same shape and all corresponding elements are equal within the
    /// given tolerances.
    member t.allclose(tensor:Tensor, ?relativeTolerance, ?absoluteTolerance) =
        let relativeTolerance = defaultArg relativeTolerance 1e-5
        let absoluteTolerance = defaultArg absoluteTolerance 1e-8
        t.primalRaw.AllClose(tensor.primalRaw, relativeTolerance, absoluteTolerance)

    /// Returns a new tensor filled with '0' values for the given shape, element type and configuration, defaulting to the 
    /// shape and configuration of the input tensor.
    member a.zerosLike(?shape:seq<int>, ?device, ?dtype, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        TensorC(a.primalRaw.ZerosLike(shape |> Array.ofSeq, ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a new tensor filled with '1' values for the given shape, element type and configuration, defaulting to the 
    /// shape and configuration of the input tensor.
    member a.onesLike(?shape:seq<int>, ?device, ?dtype, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        TensorC(a.primalRaw.OnesLike(shape |> Array.ofSeq, ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a new tensor filled with the given scalar value for the given shape, element type and configuration, defaulting to the 
    /// shape and configuration of the input tensor.
    member a.fullLike(value:scalar, ?shape:seq<int>, ?device, ?dtype, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        TensorC(a.primalRaw.FullLike(shape |> Array.ofSeq, value, ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a new scalar tensor for the given shape, element type and configuration, defaulting to the 
    /// shape and configuration of the input tensor.
    member a.scalarLike(scalar:scalar, ?device, ?dtype, ?backend) = 
        a.fullLike(scalar, [], ?device=device, ?dtype=dtype, ?backend=backend)

    /// Returns a new tensor with random values drawn from the uniform distribution [0,1) for the
    /// given shape, element type and configuration, defaulting to the shape and configuration of the input tensor.
    member a.randLike(?shape:seq<int>, ?device, ?dtype, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        TensorC(a.primalRaw.RandomLike((shape |> Array.ofSeq), ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a new tensor with random values drawn from the standard normal distribution, for the

    /// given shape, element type and configuration, defaulting to the shape and configuration of the input tensor.
    member a.randnLike(?shape:seq<int>, ?device, ?dtype, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        TensorC(a.primalRaw.RandomNormalLike(shape |> Array.ofSeq, ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a new tensor with random integer values drawn from the given range, for the
    /// given shape, element type and configuration, defaulting to the shape and configuration of the input tensor.
    member a.randintLike(low:int, high:int, ?shape:seq<int>, ?device, ?dtype, ?backend) = 
        let shape = defaultArg shape (a.shape |> Array.toSeq)
        TensorC(a.primalRaw.RandomIntLike(shape |> Array.ofSeq, low, high, ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a scalar '0' tensor for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.zeroLike(?device, ?dtype, ?backend) = TensorC(a.primalRaw.ZeroLike(?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a scalar '1' tensor for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.oneLike(?device, ?dtype, ?backend) = TensorC(a.primalRaw.OneLike(?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a tensor in the manner of <see cref="M:DiffSharp.dsharp.arange"/> for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.arangeLike(endVal:float, ?startVal:float, ?step:float, ?device, ?dtype, ?backend) =
        let startVal = defaultArg startVal 0.
        let step = defaultArg step 1.
        let length = (endVal - startVal) / step |> ceil |> int
        let v = Array.init length (fun i -> startVal + float(i) * step)
        a.like(box v, ?device=device, ?dtype=dtype, ?backend=backend)

    /// Returns a tensor in the manner of <see cref="M:DiffSharp.dsharp.arange"/> for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.arangeLike(endVal:int, ?startVal:int, ?step:int, ?device, ?dtype, ?backend) =
        let endVal = endVal |> float
        let startVal = defaultArg startVal 0 |> float
        let step = defaultArg step 1 |> float
        let dtype = defaultArg dtype Dtype.Int32
        a.arangeLike(endVal=endVal, startVal=startVal, step=step, ?device=device, dtype=dtype, ?backend=backend)

    /// Returns a tensor in the manner of <see cref="M:DiffSharp.dsharp.linspace"/> for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.linspaceLike(startVal:float, endVal:float, steps:int, ?device, ?dtype, ?backend) =
        let stepVal = (endVal - startVal) / (float (steps - 1))
        let v = Array.init steps (fun i -> startVal + (float i) * stepVal)
        a.like(box v, ?device=device, ?dtype=dtype, ?backend=backend)

    /// Returns a tensor in the manner of <see cref="M:DiffSharp.dsharp.linspace"/> for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.linspaceLike(startVal:int, endVal:int, steps:int, ?device, ?dtype, ?backend) =
        a.linspaceLike(startVal |> float, endVal |> float, steps, ?device=device, ?dtype=dtype, ?backend=backend)

    /// Returns a tensor in the manner of <see cref="M:DiffSharp.dsharp.logspace"/> for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.logspaceLike(startVal:float, endVal:float, steps:int, ?baseVal:float, ?device, ?dtype, ?backend) =
        let baseVal = defaultArg baseVal 10.
        a.scalarLike(baseVal, ?device=device, ?dtype=dtype, ?backend=backend).pow(a.linspaceLike(startVal, endVal, steps, ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a tensor in the manner of <see cref="M:DiffSharp.dsharp.logspace"/> for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.logspaceLike(startVal:int, endVal:int, steps:int, ?baseVal:int, ?device, ?dtype, ?backend) =
        let baseVal = defaultArg baseVal 10
        a.logspaceLike(startVal |> float, endVal |> float, steps, baseVal |> float, ?device=device, ?dtype=dtype, ?backend=backend)

    /// <summary>
    ///  Returns a tensor from the .NET data in <c>value</c> for the given element type and configuration, defaulting to
    ///  the element type and configuration of the input tensor.
    /// </summary>
    member a.like(value, ?device, ?dtype, ?backend) = TensorC(a.primalRaw.CreateLike(value, ?device=device, ?dtype=dtype, ?backend=backend))

    /// Returns a new tensor with underlying storage copied.
    member a.clone() = TensorC(a.primalRaw.Clone())

    /// Returns a tensor in the manner of <see cref="M:DiffSharp.dsharp.onehot"/> for the given element type and configuration, defaulting to
    /// the element type and configuration of the input tensor.
    member a.onehotLike(length:int, hot:int, ?device, ?dtype, ?backend) =
        if hot < 0 || hot >= length then failwithf "Expecting 0 <= hot < length"
        a.zerosLike([|length|], ?device=device, ?dtype=dtype, ?backend=backend).addSlice([|hot|], a.onesLike([|1|], ?device=device, ?dtype=dtype, ?backend=backend))

    /// <summary>Computes element-wise \(a &lt; b\), returning a boolean tensor containing a <c>true</c> at each location where the comparison is true</summary>
    member a.lt(b:Tensor) = TensorC(a.primalRaw.LtTT(b.primalRaw))

    /// <summary>Computes element-wise \(a &gt; b\), returning a boolean tensor containing a <c>true</c> at each location where the comparison is true</summary>
    member a.gt(b:Tensor) = TensorC(a.primalRaw.GtTT(b.primalRaw))

    /// <summary>Computes element-wise \(a \leq b\), returning a boolean tensor containing a <c>true</c> at each location where the comparison is true</summary>
    member a.le(b:Tensor) =TensorC(a.primalRaw.LeTT(b.primalRaw))

    /// <summary>Computes element-wise \(a \geq b\), returning a boolean tensor containing a <c>true</c> at each location where the comparison is true</summary>
    member a.ge(b:Tensor) = TensorC(a.primalRaw.GeTT(b.primalRaw))

    /// <summary>Computes element-wise \(a = b\), returning a boolean tensor containing a <c>true</c> at each location where the comparison is true</summary>
    member a.eq(b:Tensor) = TensorC(a.primalRaw.EqTT(b.primalRaw))

    /// <summary>Computes element-wise \(a \neq b\), returning a boolean tensor containing a <c>true</c> at each location where the comparison is true</summary>
    member a.ne(b:Tensor) = let e = a.eq(b) in e.lt(e.onesLike()) // Implement "not equal" relying on "equal"

    /// <summary>Returns a new tensor with boolean elements representing if each element is +/-INF or not.</summary>
    member a.isinf() = TensorC(a.primalRaw.IsInfT())

    /// <summary>Returns a new tensor with boolean elements representing if each element is NaN or not. Complex values are considered NaN when either their real and/or imaginary part is NaN.</summary>
    member a.isnan() = TensorC(a.primalRaw.IsNaNT())

    /// Gets if any value in the tensor is +/- INF.
    member a.hasinf() = a.isinf().sum() > a.zeroLike(dtype=Dtype.Int64)

    /// Gets if any value in the tensor is NaN.
    member a.hasnan() = a.isnan().sum() > a.zeroLike(dtype=Dtype.Int64)

    /// Gets if any value in the tensor is NaN or +/- INF.
    member a.hasinfnan() = a.hasinf() || a.hasnan()

    /// Gets the index of a maximum value in the tensor.
    member a.argmax() =
        a.primalRaw.MaxIndexT()

    /// <summary>Returns the indexes of maximum values of the primal of the tensor, reducing the given dimension.</summary>
    /// <remarks>The resulting tensor does not participate in reverse or forward differentiation. It can be used as input to another operation such as <c>dsharp.gather</c>.</remarks>
    member a.argmax(dim:int, ?keepDim: bool) =
        let keepDim = defaultArg keepDim false
        Shape.checkCanMinMaxReduce dim keepDim a.shape |> ignore
        a.primalRaw.MaxReduceT(dim, keepdim=keepDim) |> snd |> TensorC

    /// Gets the index of a minimum value in the tensor.
    member a.argmin() =
        a.primalRaw.MinIndexT()

    /// <summary>Returns the indexes of minimum values of the primal of the tensor, reducing the given dimension.</summary>
    /// <remarks>The resulting tensor does not participate in reverse or forward differentiation. It can be used as input to another operation such as <c>dsharp.gather</c>.</remarks>
    member a.argmin(dim: int, ?keepDim: bool) =
        let keepDim = defaultArg keepDim false
        Shape.checkCanMinMaxReduce dim keepDim a.shape |> ignore
        a.primalRaw.MinReduceT(dim, keepdim=keepDim) |> snd |> TensorC

    /// Returns the maximum value along the given dimension of all elements in the input tensor.
    member a.max(dim:int, ?keepDim:bool) =
        let keepdim = defaultArg keepDim false
        let indices = a.argmax(dim=dim, keepDim=true)
        let ret:Tensor = a.gather(dim, indices)
        if keepdim then ret else ret.squeeze(dim)

    /// Returns the minimum value along the given dimension of all elements in the input tensor.
    member a.min(dim:int, ?keepDim:bool) =
        let keepdim = defaultArg keepDim false
        let indices = a.argmin(dim=dim, keepDim=true)
        let ret:Tensor = a.gather(dim, indices)
        if keepdim then ret else ret.squeeze(dim)

    /// Returns the maximum value of all elements in the input tensor.
    member a.max() = if a.dim = 0 then a else a[a.argmax()]

    /// Returns the minimum value of all elements in the input tensor.
    member a.min() = if a.dim = 0 then a else a[a.argmin()]

    /// Returns the element-wise maximum of the elements in the two tensors.
    member a.max(b:Tensor) = 
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "max" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast.max(bCast)
        elif a.dtype = Dtype.Byte || a.dtype = Dtype.Bool then
            let result:Tensor = a.cast(Dtype.Int16).max(b.cast(Dtype.Int16))
            result.cast(a.dtype)
        else
            let result:Tensor = ((a + b) + Tensor.Abs(b - a)) / 2
            if result.dtype <> a.dtype then result.cast(a.dtype) else result

    /// Returns the element-wise minimum of the elements in the two tensors.
    member a.min(b:Tensor) = 
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "min" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast.min(bCast)
        elif a.dtype = Dtype.Byte || a.dtype = Dtype.Bool then
            let result:Tensor = a.cast(Dtype.Int16).min(b.cast(Dtype.Int16))
            result.cast(a.dtype)
        else
            let result:Tensor = ((a + b) - Tensor.Abs(a - b)) / 2
            if result.dtype <> a.dtype then result.cast(a.dtype) else result

    /// <summary>
    ///  Returns a tensor with the diagonal elements with respect to <c>dim1</c> and <c>dim2</c>.
    ///  The argument offset controls which diagonal to consider.
    /// </summary>
    member a.diagonal(?offset:int, ?dim1:int, ?dim2:int) =
        if a.dim < 2 then failwithf "Tensor must be at least 2-dimensional"
        let offset = defaultArg offset 0
        let dim1 = defaultArg dim1 0
        let dim2 = defaultArg dim2 1
        let mutable finished = false
        let mutable d = []
        let mutable i = 0
        let mutable j = offset
        while not finished do
            if i >= a.shape[dim1] || j >= a.shape[dim2] then 
                finished <- true
            elif j >= 0 then
                // let bounds = array2D [[i0min; i0max; i0given]; [i1min; i1max; i1given]; [i2min; i2max; i2given]; [i3min; i3max; i3given]]
                let bounds = Array2D.init (a.dim) 3 (fun ii jj -> 
                                                        if ii = dim1 then
                                                            if jj < 2 then i else 1
                                                        elif ii = dim2 then
                                                            if jj < 2 then j else 1
                                                        else
                                                            if jj = 0 then 0
                                                            elif jj = 1 then a.shape[ii]-1
                                                            else 0
                                                        )
                d <- [a.GetSlice(bounds)] |> List.append d
            i <- i + 1
            j <- j + 1
        if d |> List.isEmpty then failwithf "Empty diagonal"
        Tensor.stack(d)

    /// <summary>Returns the sum of the elements of the diagonal of the input 2-D matrix.</summary>
    member a.trace() = let d:Tensor = a.diagonal() in d.sum()

    /// <summary>Returns a new view of the object tensor with singleton dimensions expanded to a larger size.</summary>
    /// <remarks>
    ///   <para>Passing -1 as the size for a dimension means not changing the size of that dimension.</para>
    ///   <para>The tensor can be also expanded to a larger number of dimensions, and the new ones will be appended 
    ///         at the front. For the new dimensions, the size cannot be set to -1.
    ///   </para>
    ///   <para>
    ///      Expanding a tensor does not allocate new memory, but only creates a new view on the existing tensor
    ///      where a dimension of size one is expanded to a larger size by setting the stride to 0. Any dimension
    ///      of size 1 can be expanded to an arbitrary value without allocating new memory.
    ///   </para>
    /// </remarks>
    member a.expand(newShape:seq<int>) =
        let newShape = newShape|>Shape.create
        if a.shape = newShape then a 
        else
            let newShape = Shape.completeExpand a.shape newShape  // Handles -1 semantics
            Shape.checkCanExpand a.shape newShape
            match a with
            | TensorC(ap) -> TensorC(ap.Expand(newShape))
            | TensorF(ap,ad,at) ->
                let fp = ap.expand(newShape)
                let fd = ad.expand(newShape)
                TensorF(fp,fd,at)
            | TensorR(ap,_,_,_,at) ->
                let fp = ap.expand(newShape)
                TensorR(fp, ref (a.zerosLike([0])), ExpandT(a), ref 0u, at)

    /// <summary>Expand this tensor to the same size as the other.</summary>
    member a.expandAs(b:Tensor) = a.expand(b.shape)

    /// <summary>Convert tensor to an image tensor with shape Channels x Height x Width</summary>
    member t.toImage(?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int) =
        let pixelMin = defaultArg pixelMin 0.
        let pixelMax = defaultArg pixelMax 1.
        let normalize = defaultArg normalize false
        if t.dim < 1 || t.dim > 4 then failwithf "Expecting the tensor 1 <= dim (%A) <= 4, received shape %A" t.dim t.shape

        if t.dim = 4 then // we make an image grid
            let mutable numItems = t.shape[0]
            let cols = defaultArg gridCols (int(ceil(sqrt(float(numItems)))))
            if cols < 1 || cols > numItems then failwithf "Expecting 1 <= gridCols (%A) <= %A" cols numItems
            let mutable rows = 0
            let mutable items = numItems
            while items > 0 do
                rows <- rows + 1
                items <- items - cols
            let c, h, w = t.shape[1], t.shape[2], t.shape[3]
            let mutable tgrid = t.zerosLike([h*rows; w*cols; c])
            // transform [n, c, h, w] to [n, h, w, c]
            let t:Tensor = t.transpose(1, 3)
            let t = t.transpose(2, 1)
            let mutable i = 0
            for row=0 to rows-1 do
                for col=0 to cols-1 do
                    if i < numItems then
                        tgrid <- tgrid.addSlice([row*h; col*w; 0], t[i])
                        i <- i + 1
            // transform [h, w, c] to [c, h, w]
            tgrid <- tgrid.transpose(0, 2)
            tgrid <- tgrid.transpose(1, 2)
            tgrid.toImage(pixelMin=pixelMin, pixelMax=pixelMax, normalize=normalize)
        else
            let mutable pixels = t
            if t.dim = 1 then
                pixels <- pixels.view([1; 1; t.nelement])
                pixels <- pixels.expand([3; -1; -1])
            elif t.dim = 2 then
                pixels <- pixels.view([1; t.shape[0]; t.shape[1]])
                pixels <- pixels.expand([3; -1; -1])
            else
                if t.shape[0] = 1 then
                    pixels <- pixels.expand([3; -1; -1])
                elif t.shape[0] <> 3 then 
                    failwithf "Expecting the number of channels (%A) to be 1 or 3" t.shape[0]
            if pixelMin < 0. || pixelMin > 1. then failwithf "Expecting 0 <= pixelMin (%A) <= 1" pixelMin
            if pixelMax < 0. || pixelMax > 1. then failwithf "Expecting 0 <= pixelMax (%A) <= 1" pixelMax
            let pixelRange = pixelMax - pixelMin
            if pixelRange <= 0. then failwithf "Expecting pixelMin (%A) < pixelMax (%A)" pixelMin pixelMax
            if normalize then
                pixels <- pixels.normalize()
            pixels <- pixelMin + pixels.mul(pixelRange)
            pixels

    /// <summary>Convert tensor to a grayscale image tensor and return a string representation approximating grayscale values</summary>
    member t.toImageString(?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int, ?asciiPalette:string) =
        let asciiPalette = defaultArg asciiPalette """ .'`,^:";~-_+<>i!lI?/\|()1{}[]rcvunxzjftLCJUYXZO0Qoahkbdpqwm*WMB8&%$#@"""
        let pixels:Tensor = t.toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols).mean(0) // make it grayscale
        let numToAscii (numZeroToOne:float) =
            let c = int (numZeroToOne * float(asciiPalette.Length)) - 1
            let c = min (asciiPalette.Length - 1) (max 0 c)
            asciiPalette[c]
        let h, w = pixels.shape[0], pixels.shape[1]
        let sb = System.Text.StringBuilder()
        for y=0 to h-1 do
            for x=0 to w-1 do
                sb.Append(numToAscii (float(pixels[y, x]))) |> ignore
            sb.AppendLine() |> ignore
        sb.ToString()

    member t.GetSlice(bounds:int[,]) =
        if t.dim = 0 then failwith "Cannot slice a scalar Tensor"
        let fullBounds = t.shapeFullBounds |> Array2D.copy
        bounds |> Array2D.iteri (fun i j v -> 
            if j=1 && v >= t.shape[i] then failwithf "Index outside the bounds of Tensor shape %A" t.shape
            fullBounds[i, j] <- v)
        if fullBounds = t.shapeFullBounds then t // We don't need to slice as the result of the slicing would be the same with this existing tensor
        else
        match t with
        | TensorC(ap) -> TensorC(ap.GetSlice(fullBounds))
        | TensorF(ap,ad,at) -> TensorF(ap.GetSlice(fullBounds), ad.GetSlice(fullBounds), at)
        | TensorR(ap,_,_,_,at) -> TensorR(ap.GetSlice(fullBounds), ref (ap.zerosLike([0])), SliceT(t, fullBounds), ref 0u, at)

    /// <summary>Get the item at the given index as a scalar tensor.</summary>
    member t.Item
        with get([<System.ParamArray>] index:int[]) =
            if t.dim = 0 then failwith "Cannot index a scalar Tensor"
            if index.Length > t.dim then failwithf "Expecting an index with <=%i dimensions" t.dim
            let bounds = Array2D.init index.Length 3 (fun i j -> if j=2 then 1 else index[i])
            t.GetSlice(bounds)

    /// <summary>
    /// Creates a new tensor from the raw tensor.
    /// </summary>
    /// <param name="rawTensor">The given raw tensor.</param>
    static member ofRawTensor(rawTensor: RawTensor) = TensorC rawTensor

    /// <summary>
    /// Creates a new tensor from the given data, using the given element type and configuration.
    /// </summary>
    /// <param name="value">The .NET object used to form the initial values for the tensor.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    /// <remarks>The fastest creation technique is a one dimensional array matching the desired dtype. Then use 'view' to reshape.</remarks>
    static member create(value:obj, ?device:Device, ?dtype:Dtype, ?backend:Backend) =
        // Fast paths to create directly from 1D array matching the dtype
        match value, defaultArg dtype Dtype.Default with
        | (:? (int32[]) as arr), Dtype.Int32 -> TensorC(RawTensor.CreateFromFlatArray(arr, shape=[| arr.Length |], ?device=device, ?dtype=dtype, ?backend=backend))
        | (:? (single[]) as arr), Dtype.Float32 -> TensorC(RawTensor.CreateFromFlatArray(arr, shape=[| arr.Length |], ?device=device, ?dtype=dtype, ?backend=backend))
        | (:? (double[]) as arr), Dtype.Float64 -> TensorC(RawTensor.CreateFromFlatArray(arr, shape=[| arr.Length |], ?device=device, ?dtype=dtype, ?backend=backend))
        | (:? (int16[]) as arr), Dtype.Int16 -> TensorC(RawTensor.CreateFromFlatArray(arr, shape=[| arr.Length |], ?device=device, ?dtype=dtype, ?backend=backend))
        | (:? (int64[]) as arr), Dtype.Int64 -> TensorC(RawTensor.CreateFromFlatArray(arr, shape=[| arr.Length |], ?device=device, ?dtype=dtype, ?backend=backend))
        // Extra type match check is needed to distinguish between arrays holding byte and int8, see https://github.com/dotnet/fsharp/issues/10202
        | (:? (byte[]) as arr), Dtype.Byte when DataConverter.typesMatch<byte> arr -> TensorC(RawTensor.CreateFromFlatArray(arr, shape=[| arr.Length |], ?device=device, ?dtype=dtype, ?backend=backend))
        | (:? (int8[]) as arr), Dtype.Int8 when DataConverter.typesMatch<int8> arr -> TensorC(RawTensor.CreateFromFlatArray(arr, shape=[| arr.Length |], ?device=device, ?dtype=dtype, ?backend=backend))
        | _ -> 
        // Empty tensor (no data, shape: [0])
        match value with
        | :? (seq<obj>) as v when Seq.isEmpty v -> 
            let result = TensorC(RawTensor.CreateFromFlatArray(Array.zeroCreate<float32> 0, shape=[|0|], ?device=device, dtype=Dtype.Float32, ?backend=backend))
            let dtype2 = defaultArg dtype Dtype.Default
            result.cast(dtype=dtype2)
        | _ ->
        // Create a new Tensor from a structure holding scalar Tensors. Maintains differentiability.
        let res = value |> DataConverter.tryFlatArrayAndShape<Tensor> 
        match res with
        | Some (tensors, shape) -> 
            let allScalar = tensors |> Array.forall (fun t -> t.dim = 0)
            if not allScalar then failwithf "Combining tensors in an array is only supported where all tensors in the array are scalar (zero-dimensional). Check other operations like stack, cat to combine tensors."
            Tensor.stack(tensors).view(shape)
        | None ->
        // General constant tensor
        TensorC(RawTensor.Create(value, ?device=device, ?dtype=dtype, ?backend=backend))        

    /// <summary>Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.</summary>
    static member eye(rows:int, ?cols:int, ?device:Device, ?dtype:Dtype, ?backend:Backend) =
        let cols = defaultArg cols rows
        if rows <= 0 || cols <= 0 then Tensor.create([], ?device=device, ?dtype=dtype, ?backend=backend)
        else
            let vals = Array2D.init rows cols (fun i j -> if i = j then 1 else 0)
            Tensor.create(vals, ?device=device, ?dtype=dtype, ?backend=backend)

    /// <summary>Concatenates sequence of tensors along a new dimension.</summary>
    /// <remarks>All tensors need to be of the same shape.</remarks>
    /// <param name="tensors">sequence of tensors to concatenate</param>
    /// <param name="dim">dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive)</param>
    static member stack(tensors:seq<Tensor>, ?dim:int) = 
        let dim = defaultArg dim 0 
        let tensors = tensors |> Seq.toArray
        let allSameDiffType = tensors |> Array.forall (fun t -> t.isSameDiffType(tensors[0]))
        if not allSameDiffType then failwithf "Cannot stack tensors with different differentiation type (TensorC, TensorF, TensorR)."
        if not tensors[0].isNoDiff then
            let allSameTag = tensors |> Array.forall (fun t -> t.nestingTag = tensors[0].nestingTag)
            if not allSameTag then failwithf "Cannot stack tensors with different nesting tags."
        let shapes = tensors |> Array.map (fun t -> t.shape)
        Shape.checkCanStack shapes dim |> ignore
        match Seq.head tensors with
        | TensorC(ap) -> TensorC(ap.StackTs((tensors |> Array.map (fun t -> t.primalRaw)), dim))
        | TensorF(_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let ad = tensors |> Seq.map (fun t -> t.derivative)
            TensorF(Tensor.stack(ap,dim=dim), Tensor.stack(ad,dim=dim), at)
        | TensorR(_,_,_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let fp = Tensor.stack(ap,dim=dim)
            TensorR(fp, ref (fp.zerosLike([0])), StackTs(tensors, dim), ref 0u, at)

    /// <summary>Removes a tensor dimension.</summary>
    /// <param name="dim">The dimension to remove, defaults to 0.</param>
    /// <returns>Returns an array of all slices along a given dimension.</returns>
    member a.unstack (?dim:int) =
        let dim = defaultArg dim 0 
        Shape.checkCanUnstack a.shape |> ignore
        match a with
        | TensorC(ap) -> ap.UnstackT(dim) |> Array.map TensorC
        | TensorF(ap,ad,at) -> Array.map2 (fun p d -> TensorF(p,d,at)) (ap.unstack(dim)) (ad.unstack(dim))
        | TensorR(ap,_,_,_,at) -> Array.mapi (fun i p -> TensorR(p, ref (p.zerosLike([0])), UnstackT(a, dim, i), ref 0u, at)) (ap.unstack(dim))

    /// <summary>Concatenates the given sequence of seq tensors in the given dimension.</summary>
    /// <remarks>All tensors must either have the same shape (except in the concatenating dimension) or be empty.</remarks>
    /// <param name="tensors">The tensors to concatenate.</param>
    /// <param name="dim">The dimension over which the tensors are concatenated, defaults to 0.</param>
    static member cat(tensors:seq<Tensor>, ?dim: int) = 
        let dim = defaultArg dim 0 
        let tensors = tensors |> Seq.toArray
        let allSameDiffType = tensors |> Array.forall (fun t -> t.isSameDiffType(tensors[0]))
        if not allSameDiffType then failwithf "Cannot cat tensors with different differentiation type (TensorC, TensorF, TensorR)."
        if not tensors[0].isNoDiff then
            let allSameTag = tensors |> Array.forall (fun t -> t.nestingTag = tensors[0].nestingTag)
            if not allSameTag then failwithf "Cannot cat tensors with different nesting tags."
        let shapes = tensors |> Array.map (fun t -> t.shape)
        Shape.checkCanCat shapes dim |> ignore
        match Seq.head tensors with
        | TensorC(ap) -> TensorC(ap.CatTs((tensors |> Array.map (fun t -> t.primalRaw)), dim))
        | TensorF(_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let ad = tensors |> Seq.map (fun t -> t.derivative)
            TensorF(Tensor.cat(ap, dim=dim), Tensor.cat(ad, dim=dim), at)
        | TensorR(_,_,_,_,at) ->
            let ap = tensors |> Seq.map (fun t -> t.primal)
            let fp = Tensor.cat(ap, dim=dim)
            TensorR(fp, ref (fp.zerosLike([0])), CatTs(tensors, dim), ref 0u, at)

    /// <summary>Splits the tensor into chunks. Each chunk is a view of the original tensor.</summary>
    /// <param name="sizes">List of sizes for each chunk</param>
    /// <param name="dim">The dimension along which to split the tensor, defaults to 0.</param>
    member a.split (sizes: seq<int>, ?dim: int) =
        let dim = defaultArg dim 0
        let sizes = sizes |> Seq.toArray
        match a with
        | TensorC(ap) -> ap.SplitT(sizes, dim=dim) |> Array.map TensorC
        | TensorF(ap,ad,at) -> Array.map2 (fun p d -> TensorF(p,d,at)) (ap.split(sizes, dim=dim)) (ad.split(sizes, dim=dim))
        | TensorR(ap,_,_,_,at) -> Array.mapi (fun i p -> TensorR(p, ref (p.zerosLike([0])), SplitT(a, sizes, dim, i), ref 0u, at)) (ap.split(sizes, dim=dim))

    /// <summary>Pipeline the tensor into a function.</summary>
    static member inline (-->) (t:Tensor, f:Tensor -> ^a) = f t

    static member inline internal OpUnary(a, fRaw:RawTensor->RawTensor, fTensor, dfFwd, dfRev) =
        match a with
        | TensorC(ap)           -> TensorC(fRaw(ap))
        | TensorF(ap,ad,at)    -> let fp = fTensor(ap) in TensorF(fp, dfFwd(ap,ad,fp), at)
        | TensorR(ap,_,_,_,at) -> let fp = fTensor(ap) in TensorR(fp, ref (a.zerosLike([0])), dfRev(a), ref 0u, at)

    static member inline internal OpBinary(a, b, fRaw: RawTensor * RawTensor -> RawTensor, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT) =
        match a, b with
        | TensorC(ap),          TensorC(bp)                     -> TensorC(fRaw(ap, bp))
        | TensorC(_),           TensorF(bp,bd,bt)               -> let fp = fTensor(a,bp)  in TensorF(fp, dfFwdCT(bp,bd,fp), bt)
        | TensorC(_),           TensorR(bp,_,_,_,bt)            -> let fp = fTensor(a,bp)  in TensorR(fp, ref (a.zerosLike([0])), dfRevCT(a,b), ref 0u, bt)
        | TensorF(ap,ad,at),    TensorC(_)                      -> let fp = fTensor(ap,b)  in TensorF(fp, dfFwdTC(ap,ad,fp), at)
        | TensorF(ap,ad,at),    TensorF(bp,bd,bt)    when at=bt -> let fp = fTensor(ap,bp) in TensorF(fp, dfFwdTT(ap,ad,bp,bd,fp), at)
        | TensorF(ap,ad,at),    TensorF(_,_,bt)      when at>bt -> let fp = fTensor(ap,b)  in TensorF(fp, dfFwdTC(ap,ad,fp), at)
        | TensorF(_,_,at),      TensorF(bp,bd,bt)    when at<bt -> let fp = fTensor(a,bp)  in TensorF(fp, dfFwdCT(bp,bd,fp), bt)
        | TensorF(_,_,at),      TensorR(_,_,_,_,bt)  when at=bt -> failwith "Cannot have TensorF and TensorR in the same nesting level"
        | TensorF(ap,ad,at),    TensorR(_,_,_,_,bt)  when at>bt -> let fp = fTensor(ap,b)  in TensorF(fp, dfFwdTC(ap,ad,fp), at)
        | TensorF(_,_,at),      TensorR(bp,_,_,_,bt) when at<bt -> let fp = fTensor(a,bp)  in TensorR(fp, ref (a.zerosLike([0])), dfRevCT(a,b), ref 0u, bt)
        | TensorR(ap,_,_,_,at), TensorC(_)                      -> let fp = fTensor(ap,b)  in TensorR(fp, ref (a.zerosLike([0])), dfRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(_,_,bt)      when at=bt -> failwith "Cannot have TensorR and TensorF in the same nesting level"
        | TensorR(ap,_,_,_,at), TensorF(_,_,bt)      when at>bt -> let fp = fTensor(ap,b) in TensorR(fp, ref (a.zerosLike([0])), dfRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorF(bp,bd,bt)    when at<bt -> let fp = fTensor(a,bp)  in TensorF(fp, dfFwdCT(bp,bd,fp), bt)
        | TensorR(ap,_,_,_,at), TensorR(bp,_,_,_,bt) when at=bt -> let fp = fTensor(ap,bp) in TensorR(fp, ref (a.zerosLike([0])), dfRevTT(a,b), ref 0u, at)
        | TensorR(ap,_,_,_,at), TensorR(_,_,_,_,bt)  when at>bt -> let fp = fTensor(ap,b)  in TensorR(fp, ref (a.zerosLike([0])), dfRevTC(a,b), ref 0u, at)
        | TensorR(_,_,_,_,at),  TensorR(bp,_,_,_,bt) when at<bt -> let fp = fTensor(a,bp)  in TensorR(fp, ref (a.zerosLike([0])), dfRevCT(a,b), ref 0u, bt)
        | _ -> failwith "Unexpected combination of Tensors" // Won't happen, added for suppressing "incomplete matches" warning

    /// <summary>Each element of the tensor <paramref name="a" /> is added to each corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    static member (+) (a:Tensor, b:Tensor) : Tensor =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "+" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast + bCast
        elif a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.AddTT(b)
            let inline fTensor(a,b) = a + b
            let inline dfFwdTT(ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor,fp:Tensor) = ad + bd
            let inline dfFwdTC(ap:Tensor,ad,fp:Tensor) = ad
            let inline dfFwdCT(bp:Tensor,bd:Tensor,fp:Tensor) = bd
            let inline dfRevTT(a,b) = AddTT(a,b)
            let inline dfRevTC(a,b:Tensor) = AddTTConst(a)
            let inline dfRevCT(a:Tensor,b) = AddTTConst(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded + bExpanded

    /// <summary>Each element of the tensor <paramref name="a" /> is added to the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    static member (+) (a:Tensor, b: scalar) =
        match tryWidenScalar a.dtype b with
        | ValueSome tnew ->
            let aCast = a.cast(tnew)
            let bCast = b.cast(tnew)
            aCast + bCast
        | ValueNone ->
            let inline fRaw(a:RawTensor) = a.AddTT0(b)
            let inline fTensor(a) = a + b
            let inline dfFwd(ap,ad,fp) = ad
            let inline dfRev(a) = AddTT0Const(a)
            Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>The scalar <paramref name="a" /> is added to each element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    static member (+) (a: scalar, b:Tensor) : Tensor = b + a

    /// <summary>Each element of the object tensor is added to each corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    member a.add(b:Tensor) : Tensor = a + b

    /// <summary>Each element of the object tensor is added to the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    member a.add(b:scalar) : Tensor = a + b

    /// <summary>Subtracts each element of the tensor <paramref name="b" /> from the corresponding element of the tensor <paramref name="a" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    static member (-) (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "-" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast - bCast
        elif a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.SubTT(b)
            let inline fTensor(a,b) = a - b
            let inline dfFwdTT(ap,ad,bp,bd,fp) = ad - bd
            let inline dfFwdTC(ap,ad,fp) = ad
            let inline dfFwdCT(bp,bd,fp) = -bd
            let inline dfRevTT(a,b) = SubTT(a,b)
            let inline dfRevTC(a,b) = SubTTConst(a)
            let inline dfRevCT(a,b) = SubTConstT(b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded - bExpanded

    /// <summary>Subtracts the scalar <paramref name="b" /> from the corresponding element of the tensor <paramref name="a" />. The resulting tensor is returned.</summary>
    static member (-) (a:Tensor, b:scalar) =
        match tryWidenScalar a.dtype b with
        | ValueSome tnew ->
            let aCast = a.cast(tnew)
            let bCast = b.cast(tnew)
            aCast - bCast
        | ValueNone ->
            let inline fRaw(a:RawTensor) = a.SubTT0(b)
            let inline fTensor(a) = a - b
            let inline dfFwd(ap,ad,fp) = ad
            let inline dfRev(a) = SubTT0Const(a)
            Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Subtracts each element of the tensore <paramref name="b" /> from the scalar <paramref name="a" />. The resulting tensor is returned.</summary>
    static member (-) (a:scalar, b:Tensor) : Tensor =
        match tryWidenScalar b.dtype a with
        | ValueSome tnew ->
            let aCast = a.cast(tnew)
            let bCast = b.cast(tnew)
            aCast * bCast
        | ValueNone ->
            let inline fRaw(b:RawTensor) = b.SubFromT0T(a)
            let inline fTensor(b) = a - b
            let inline dfFwd(bp,bd,fp) = -bd
            let inline dfRev(b) = SubT0ConstT(b)
            Tensor.OpUnary(b, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Subtracts each element of the object tensor from the corresponding element of the self tensor. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    member a.sub(b:Tensor) = a - b

    /// <summary>Subtracts the scalar <paramref name="b" /> from the corresponding element of the object tensor. The resulting tensor is returned.</summary>
    member a.sub(b:scalar) = a - b

    /// <summary>Multiplies each element of the tensor <paramref name="a" /> by the corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    static member (*) (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "*" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast * bCast
        elif a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.MulTT(b)
            let inline fTensor(a,b) = a * b
            let inline dfFwdTT(ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor,fp:Tensor) = (ad * bp) + (ap * bd)
            let inline dfFwdTC(ap:Tensor,ad:Tensor,fp:Tensor) = ad * b
            let inline dfFwdCT(bp:Tensor,bd:Tensor,fp:Tensor) = a * bd
            let inline dfRevTT(a,b) = MulTT(a,b)
            let inline dfRevTC(a,b) = MulTTConst(a,b)
            let inline dfRevCT(a,b) = MulTTConst(b,a)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded * bExpanded

    /// <summary>Multiplies each element of the tensor <paramref name="a" /> by the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    static member (*) (a:Tensor, b:scalar) =
        match tryWidenScalar a.dtype b with
        | ValueSome tnew ->
            let aCast = a.cast(tnew)
            let bCast = b.cast(tnew)
            aCast * bCast
        | ValueNone ->
            let inline fRaw(a:RawTensor) = a.MulTT0(b)
            let inline fTensor(a) = a * b
            let inline dfFwd(ap,ad,fp) = ad * b
            let inline dfRev(a) = MulTT0Const(a,b)
            Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Multiplies the scalar <paramref name="a" /> by each element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    static member (*) (a:scalar, b:Tensor) = b * a

    /// <summary>Multiplies each element of the object tensor by the corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    member a.mul(b:Tensor) = a * b

    /// <summary>Multiplies each element of the object tensor by the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    member a.mul(b: scalar) = a * b

    /// <summary>Divides each element of the tensor <paramref name="a" /> by the corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    static member (/) (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "/" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                aCast / bCast
        elif a.shape = b.shape then
            let outtype = Dtype.divisionType a.dtype b.dtype
            let a = a.cast(outtype)
            let b = b.cast(outtype)

            let inline fRaw(a:RawTensor,b) = a.DivTT(b)
            let inline fTensor(a,b) = a / b
            let inline dfFwdTT(ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor,fp:Tensor) = (ad - bd * fp) / bp
            let inline dfFwdTC(ap:Tensor,ad:Tensor,fp:Tensor) = ad / b
            let inline dfFwdCT(bp:Tensor,bd:Tensor,fp:Tensor) = -bd * fp / bp
            let inline dfRevTT(a,b) = DivTT(a,b)
            let inline dfRevTC(a,b) = DivTTConst(a,b)
            let inline dfRevCT(a,b) = DivTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            aExpanded / bExpanded

    /// <summary>Divides each element of the tensor <paramref name="a" /> by the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    static member (/) (a:Tensor, b:scalar) =
        let outtype = widenScalarForDivision a.dtype b.dtype
        let a = a.cast(outtype)
        let b = b.cast(outtype)

        let inline fRaw(a:RawTensor) = a.DivTT0(b)
        let inline fTensor(a) = a / b
        let inline dfFwd(ap,ad,fp) = ad / b
        let inline dfRev(a) = DivTT0Const(a,b)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Divides the scalar <paramref name="a" /> by the each element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    static member (/) (a:scalar, b:Tensor) =
        let outtype = widenScalarForDivision b.dtype a.dtype
        let a = a.cast(outtype)
        let b = b.cast(outtype)

        let inline fRaw(b:RawTensor) = b.DivFromT0T(a)
        let inline fTensor(b) = a / b
        let inline dfFwd(bp,bd,fp) = -bd * fp / bp
        let inline dfRev(b) = DivT0ConstT(a,b)
        Tensor.OpUnary(b, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Divides each element of the object tensor by the corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    member a.div(b:Tensor) = a / b

    /// <summary>Divides each element of the object tensor by the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    member a.div(b:scalar) = a / b

    static member internal powImpl (a:Tensor, b:Tensor) =
        if a.dtype <> b.dtype then
            match Dtype.widen a.dtype b.dtype with
            | None -> opNotSupported "Pow" a.dtype b.dtype 
            | Some tnew ->
                let aCast = a.cast(tnew)
                let bCast = b.cast(tnew)
                Tensor.Pow (aCast, bCast)
        elif a.shape = b.shape then
            let inline fRaw(a:RawTensor,b) = a.PowTT(b)
            let inline fTensor(a:Tensor,b:Tensor) = a ** b
            let inline dfFwdTT(ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor,fp:Tensor) = (ap ** (bp - 1.)) * (ad * bp + ap * bd * log ap)
            let inline dfFwdTC(ap,ad,fp) = ad * (ap ** (b - 1.)) * b
            let inline dfFwdCT(bp,bd,fp) = bd * fp * log a
            let inline dfRevTT(a,b) = PowTT(a,b)
            let inline dfRevTC(a,b) = PowTTConst(a,b)
            let inline dfRevCT(a,b) = PowTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)
        else
            let newShape = Shape.broadcast2 a.shape b.shape
            let aExpanded = a.expand(newShape)
            let bExpanded = b.expand(newShape)
            Tensor.Pow(aExpanded, bExpanded)

    static member internal powImpl (a:Tensor, b:scalar) =
        match tryWidenScalar a.dtype b with
        | ValueSome tnew ->
            let aCast = a.cast(tnew)
            let bCast = b.cast(tnew)
            Tensor.powImpl(aCast, bCast)
        | ValueNone ->
            let inline fRaw(a:RawTensor) = a.PowTT0(b)
            let inline fTensor(a) = Tensor.powImpl (a, b)
            let inline dfFwd(ap,ad,fp) = ad * (ap ** b.sub(1.)) * b
            let inline dfRev(a) = PowTT0Const(a,b)
            Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    static member internal powImpl (a:scalar, b:Tensor) =
        match tryWidenScalar b.dtype a with
        | ValueSome tnew ->
            let aCast = a.cast(tnew)
            let bCast = b.cast(tnew)
            Tensor.powImpl(aCast, bCast)
        | ValueNone ->
            let inline fRaw(b:RawTensor) = b.PowFromT0T(a)
            let inline fTensor(b) = Tensor.powImpl (a, b)
            let inline dfFwd(bp:Tensor,bd:Tensor,fp:Tensor) : Tensor = bd * fp * a.log()
            let inline dfRev(b) = PowT0ConstT(a,b)
            Tensor.OpUnary(b, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Raises each element of the tensor <paramref name="a" /> to the power of the corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    static member Pow (a:Tensor, b:Tensor) = Tensor.powImpl(a, b)

    /// <summary>Raises each element of the tensor <paramref name="a" /> to the power of the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    static member Pow (a:Tensor, b: scalar) = Tensor.powImpl(a, b)

    /// <summary>Raises each element of the tensor <paramref name="a" /> to the power of the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    static member Pow (a:Tensor, b:float) = Tensor.powImpl(a, (b :> scalar))

    /// <summary>Raises each element of the tensor <paramref name="a" /> to the power of the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    static member Pow (a:Tensor, b:int) = Tensor.powImpl(a, (b :> scalar))

    /// <summary>Raises the scalar <paramref name="a" /> to the power of each element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    static member Pow (a:scalar, b:Tensor) = Tensor.powImpl(a, b)

    /// <summary>Raises the scalar <paramref name="a" /> to the power of each element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    static member Pow (a:float, b:Tensor) = Tensor.powImpl((a :> scalar), b)

    /// <summary>Raises the scalar <paramref name="a" /> to the power of each element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    static member Pow (a:int, b:Tensor) = Tensor.powImpl((a :> scalar), b)

    /// <summary>Raises each element of the self tensor to the power of each corresponding element of the tensor <paramref name="b" />. The resulting tensor is returned.</summary>
    /// <remarks>The shapes of the two tensors must be broadcastable.</remarks>
    member a.pow(b:Tensor) = Tensor.powImpl(a, b)

    /// <summary>Raises each element of the self tensor to the power of the scalar <paramref name="b" />. The resulting tensor is returned.</summary>
    member a.pow(b: scalar) = Tensor.powImpl(a, b)

    /// <summary>Matrix product of two tensors.</summary>
    ///
    /// <remarks>
    /// <para>
    /// The behavior depends on the dimensionality of the tensors as follows:
    /// </para>
    /// 
    /// <para>
    /// If both tensors are 1-dimensional, the dot product (scalar) is returned.
    /// </para>
    /// 
    /// <para>
    /// If both arguments are 2-dimensional, the matrix-matrix product is returned.
    /// </para>
    /// 
    /// <para>
    /// If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
    /// </para>
    /// 
    /// <para>
    ///  If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
    /// </para>
    /// 
    /// <para>
    ///  If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a 
    ///  batched matrix multiply is returned. If the first argument is 1-dimensional, a 1 is prepended to its dimension for the
    ///  purpose of the batched matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is appended to
    ///  its dimension for the purpose of the batched matrix multiple and removed after. The non-matrix (i.e. batch) dimensions
    ///  are broadcasted (and thus must be broadcastable). For example, if input is a (j \times 1 \times n \times m)(j×1×n×m)
    ///  tensor and other is a (k \times m \times p)(k×m×p) tensor, out will be an (j \times k \times n \times p)(j×k×n×p)
    ///  tensor.
    /// </para>
    /// </remarks>
    member a.matmul (b:Tensor) : Tensor =
        if a.dim = 1 && b.dim = 1 then a.dot(b) 
        // Increase to at least 2x2
        elif a.dim = 1 && b.dim > 1 then a.unsqueeze(0).matmul(b).squeeze(b.dim-2)
        elif a.dim > 1 && b.dim = 1 then a.matmul(b.unsqueeze(1)).squeeze(a.dim-1)
        else
        let (aBatchPart, aMatrixPart), (bBatchPart, bMatrixPart) = Shape.checkCanMatmul a.shape b.shape
        if aBatchPart = bBatchPart then
            let inline fRaw(a:RawTensor,b) = a.MatMulTT(b)
            let inline fTensor(a:Tensor,b) = a.matmul(b)
            let inline dfFwdTT(ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor,fp) = ad.matmul(bp) + ap.matmul(bd)
            let inline dfFwdTC(ap,ad:Tensor,fp) = ad.matmul(b)
            let inline dfFwdCT(bp,bd,fp) = a.matmul(bd)
            let inline dfRevTT(a,b) = MatMulTT(a,b)
            let inline dfRevTC(a,b) = MatMulTTConst(a,b)
            let inline dfRevCT(a,b) = MatMulTConstT(a,b)
            Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)
        else
            let newBatchPart = Shape.broadcast2 aBatchPart bBatchPart
            let aNewShape = Array.append newBatchPart aMatrixPart
            let bNewShape = Array.append newBatchPart bMatrixPart
            let aExpanded = a.expand(aNewShape)
            let bExpanded = b.expand(bNewShape)
            aExpanded.matmul(bExpanded)

    /// <summary>Computes the dot product (inner product) of two vector (1d-tensors).</summary>
    /// <param name="b">The vector to multiply this tensor by (1d-tensor).</param>
    /// <remarks>This function does not broadcast and expects this tensor to be a vector (1d-tensor).   
    /// The tensors must have the same number of elements.
    /// </remarks>
    member a.dot(b:Tensor) =
        Shape.checkCanDot a.shape b.shape
        let a:Tensor = a.view([1;a.nelement])
        let b:Tensor = b.view([b.nelement;1])
        a.matmul(b).view([])

    /// <summary>Returns a new tensor with the negative of the elements of <paramref name="a" />.</summary>
    static member (~-) (a:Tensor) =
        let inline fRaw(a:RawTensor) = a.NegT()
        let inline fTensor(a) = -a
        let inline dfFwd(ap,ad,fp) = -ad
        let inline dfRev(a) = NegT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a new tensor with the negative of the elements of the object tensor.</summary>
    member a.neg() = -a

    /// <summary>Returns the sum of all elements in the input tensor.</summary>
    /// <param name="dtype">The desired data type of returned tensor.</param>
    member a.sum(?dtype: Dtype) =
        let inline fRaw(a:RawTensor) = a.SumT(?resultType=dtype)
        let inline fTensor(a:Tensor) = a.sum(?dtype=dtype)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.sum(?dtype=dtype)
        let inline dfRev(a) = SumT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns the sum of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.</summary>
    /// <remarks>If keepdim is <c>true</c>, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension.</remarks>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    /// <param name="dtype">The desired data type of returned tensor.</param>
    member a.sum(dim:int, ?keepDim:bool, ?dtype: Dtype) =
        let keepDim = defaultArg keepDim false
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        let res =
            if dim = 0 && a.dim = 0 then a
            else
               if dim >= a.dim || dim < 0 then failwithf "Expecting 0 < dim (%A) < %A" dim a.dim
               let inline fRaw(a:RawTensor) = a.SumTDim(dim=dim, ?resultType=dtype)
               let inline fTensor(a:Tensor) = a.sum(dim=dim, ?dtype=dtype)
               let inline dfFwd(ap,ad:Tensor,fp) = ad.sum(dim=dim, ?dtype=dtype)
               let inline dfRev(a) = SumTDim(a, dim)
               Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)
        let res2 = if keepDim then res.unsqueeze(dim) else res
        res2.castAfterSummation(?dtype=dtype)

    /// <summary>Sum this tensor to size <paramref name="newShape" />, which must be broadcastable to this tensor size.</summary>
    member a.sumToSize(newShape:int[], ?dtype: Dtype) =
        let oldShape = a.shape
        if oldShape = newShape then
            a.cast(defaultArg dtype a.dtype.SummationType)
        elif newShape.Length = 0 then
            a.sum(?dtype=dtype)
        else
            Shape.checkCanExpand newShape oldShape
            let trim = oldShape.Length - newShape.Length
            let mutable result = a.cast(a.dtype.SummationType)
            // collapse the eliminated dimensions
            for _dim in 0 .. trim-1 do 
                result <- result.sum(0, keepDim=false)
            // reduce the squeezed dimensions
            for dim in 0 .. newShape.Length-1 do 
                if oldShape[trim+dim] <> newShape[dim] then 
                    result <- result.sum(dim, keepDim=true)
            result.castAfterSummation(?dtype=dtype)

    /// <summary>Returns the mean value of all elements in the input tensor</summary>
    member a.mean() = a.sum() / a.nelement

    /// <summary>Returns the mean value of each row of the input tensor in the given dimension dim.</summary>
    /// <remarks>If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension.</remarks>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    member a.mean(dim:int, ?keepDim:bool) = 
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        if dim = 0 && a.dim = 0 then a
        else 
           let sm = a.sum(dim, ?keepDim=keepDim)
           let dv = sm / a.shape[dim]
           dv

    /// <summary>Returns the variance of all elements in the input tensor.</summary>
    /// <remarks>If unbiased is False, then the variance will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.</remarks>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    member a.variance(?unbiased:bool) = 
        // This is the two-pass algorithm, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        let unbiased = defaultArg unbiased true  // Use Bessel's correction if unbiased=true
        let n = if unbiased then a.nelement - 1 else a.nelement
        let a' = a - a.mean() in (a' * a').sum() / n

    /// <summary>Returns the variance of each row of the input tensor in the given dimension dim.</summary>
    /// <remarks>
    ///   <para>If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension(s).</para>
    ///   <para>If unbiased is False, then the variance will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.</para>
    /// </remarks>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    member a.variance(dim:int, ?keepDim:bool, ?unbiased:bool) =
        // This is the two-pass algorithm, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        let unbiased = defaultArg unbiased true  // Use Bessel's correction if unbiased=true
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        let n = if unbiased then a.shape[dim] - 1 else a.shape[dim]
        let a' = a - a.mean(dim=dim, keepDim=true) in (a' * a').sum(dim=dim, ?keepDim=keepDim) / n

    /// <summary>Returns the standard deviation of each row of the input tensor in the given dimension dim.</summary>
    /// <remarks>
    ///   <para>If keepdim is True, the output tensor is of the same size as input except in the dimension dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 fewer dimension(s).</para>
    ///   <para>If unbiased is False, then the standard deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.</para>
    /// </remarks>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    member a.stddev(dim, ?keepDim, ?unbiased) = a.variance(dim, ?keepDim=keepDim, ?unbiased=unbiased) |> Tensor.Sqrt

    /// <summary>Returns the standard deviation of all elements in the input tensor.</summary>
    /// <remarks>If unbiased is False, then the standard deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.</remarks>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    member a.stddev(?unbiased) = a.variance(?unbiased=unbiased) |> Tensor.Sqrt

    /// <summary>
    /// Estimates the covariance matrix of the given tensor. The tensor's first
    /// dimension should index variables and the second dimension should
    /// index observations for each variable.
    /// </summary>
    /// <remarks>
    /// If no weights are given, the covariance between variables \(x\) and \(y\) is
    ///  \[cov(x,y)= \frac{\sum^{N}_{i = 1}(x_{i} - \mu_x)(y_{i} - \mu_y)}{N~-~\text{correction}}\]
    /// where \(\mu_x\) and \(\mu_y\) are the sample means.
    /// 
    /// If there are fweights or aweights then the covariance is
    /// \[cov(x,y)=\frac{\sum^{N}_{i = 1}w_i(x_{i} - \mu_x^*)(y_{i} - \mu_y^*)}{\text{normalization factor}}\]
    /// where \(w\) is either fweights or aweights if one weight type is provided.
    /// If both weight types are provided \(w=\text{fweights}\times\text{aweights}\). 
    /// \(\mu_x^* = \frac{\sum^{N}_{i = 1}w_ix_{i} }{\sum^{N}_{i = 1}w_i}\)
    /// is the weighted mean of variables.
    /// The normalization factor is \(\sum^{N}_{i=1} w_i\) if only fweights are provided or if aweights are provided and <c>correction=0</c>. 
    /// Otherwise if aweights \(aw\) are provided the normalization factor is
    ///  \(\sum^N_{i=1} w_i - \text{correction}\times\frac{\sum^N_{i=1} w_i aw_i}{\sum^N_{i=1} w_i}\) 
    /// </remarks>
    /// <param name="correction">Difference between the sample size and the sample degrees of freedom. Defaults to 1 (Bessel's correction).</param>
    /// <param name="fweights">Frequency weights represent the number of times each observation was observed. 
    /// Should be given as a tensor of integers. Defaults to no weights.</param>
    /// <param name="aweights">Relative importance weights, larger weights for observations that
    /// should have a larger effect on the estimate. 
    /// Should be given as a tensor of floating point numbers. Defaults to no weights.</param>
    /// <returns>Returns a square tensor representing the covariance matrix.
    ///  Given a tensor with \(N\) variables \(X=[x_1,x_2,\ldots,x_N]\) the
    /// \(C_{i,j}\) entry on the covariance matrix is the covariance between
    /// \(x_i\) and \(x_j\).
    /// </returns>
    /// <example id="tensor-covariance1">
    /// <code lang="fsharp">
    /// let x = dsharp.tensor([0.0;3.4;5.0])
    /// let y = dsharp.tensor([1.0;2.3;-3.0])
    /// let xy = dsharp.stack([x;y])
    /// xy.covariance()
    /// </code>
    /// Evaluates to
    /// <code>
    /// tensor([[ 6.5200, -4.0100],
    ///         [-4.0100,  7.6300]])
    /// </code>
    /// </example>
    member a.covariance(?correction:int64, ?fweights:Tensor, ?aweights:Tensor) =
        if a.dim > 2 then 
            failwith $"Expected input to have two or fewer dimensions but input.dim is {a.dim}"
        if a.dtype = Dtype.Bool then failwith $"bool dtype is not supported for input"
        let mutable input = if a.dim < 2 then a.view([1;-1]) else a
        let correction = defaultArg correction (int64 1)
        let nObservations = input.[0].nelement
        let checkWeightDims name (w: Tensor) =
            if w.dim > 1 then
                failwith $"{name} should be scalar or 1D. {name}.dim is {w.dim}."
            if w.nelement <> nObservations then
                let error =
                    $"The number of columns in the input tensor should be the same as the number of elements in {name}." +
                    $"There are {nObservations} columns in input and {w.nelement} elements in {name}." 
                failwith error
            if w.nelement > 0 && w.min().le(w.zeroLike()).toBool() then failwith $"{name} cannot be negative"
        let fweights = 
            match fweights with
            | None -> None
            | Some fw ->
                checkWeightDims "fweights" fw
                match fw.dtype with
                | Dtype.Integral -> Some fw
                | _ -> failwith $"fweights.dtype should be integral but it is {fw.dtype}."
        let aweights = 
            match aweights with
            | None -> None
            | Some aw ->
                checkWeightDims "aweights" aw
                match aw.dtype with
                | Dtype.FloatingPoint -> Some aw
                | _ -> failwith $"aweights.dtype should be floating point but it is {aw.dtype}."
        let w =
            match fweights, aweights with
            | None, None -> None
            | Some fw, None -> Some fw
            | None, Some aw -> Some aw
            | Some fw, Some aw -> Some (fw * aw)
        let wSum =
            match w with
            | None -> Tensor.create(nObservations, device=input.device, dtype=input.dtype, backend=input.backend)
            | Some w -> w.sum()
        if w.IsSome && wSum.eq(wSum.zeroLike()).toBool() then 
            failwith "weights cannot be normalized because they sum to zero"
        let avg =
            match w with
            | None -> input.mean(dim=1)
            | Some w -> (input * w).sum(dim=1) / wSum
        let normFactor =
            let nf =
                match w, aweights, correction <> int64 0 with
                | Some w, Some aweights, true ->
                    wSum - correction * (w * aweights).sum() / wSum
                | _ -> wSum - correction
            if nf.le(nf.zeroLike()).toBool() then 
                printfn $"Warning: degress of freedom <= 0"
                nf.zeroLike() 
            else nf
        input <- input - avg.unsqueeze(1)
        let cov = 
            match w with
            | None -> input.matmul(input.transpose())
            | Some w -> input.matmul((input * w).transpose())
        cov.div(normFactor).squeeze()

    /// <summary>
    /// Estimates the Pearson correlation coefficient matrix for the given tensor. The tensor's first
    /// dimension should index variables and the second dimension should
    /// index observations for each variable.
    /// </summary>
    /// <returns>
    /// The correlation coefficient matrix \(R\) is computed from the covariance
    /// matrix 
    /// Returns a square tensor representing the correlation coefficient matrix.
    ///  Given a tensor with \(N\) variables \(X=[x_1,x_2,\ldots,x_N]\) the
    /// \(R_{i,j}\) entry on the correlation matrix is the correlation between
    /// \(x_i\) and \(x_j\).
    /// </returns>
    /// <remarks>
    /// The correlation between variables \(x\) and \(y\) is
    ///  \[cor(x,y)= \frac{\sum^{N}_{i = 1}(x_{i} - \mu_x)(y_{i} - \mu_y)}{\sigma_x \sigma_y (N ~-~1)}\]
    /// where \(\mu_x\) and \(\mu_y\) are the sample means and \(\sigma_x\) and \(\sigma_x\) are 
    /// the sample standard deviations.
    /// </remarks>
    /// <example id="tensor-correlation1">
    /// <code lang="fsharp">
    /// let x = dsharp.tensor([-0.2678; -0.0908; -0.3766;  0.2780])
    /// let y = dsharp.tensor([-0.5812;  0.1535;  0.2387;  0.2350])
    /// let xy = dsharp.stack([x;y])
    /// xy.corrcoef()
    /// </code>
    /// Evaluates to
    /// <code>
    /// tensor([[1.0000, 0.3582],
    ///         [0.3582, 1.0000]])
    /// </code>
    /// </example>
    member a.corrcoef() =
        if a.dim > 2 then failwith $"Expected to have fewer than 2 dimensions but tensor.dim is {a.dim}"
        let mutable c = a.covariance()
        if c.dim = 0 then 
            c / c
        else
            let stddev:Tensor = c.diagonal().sqrt()
            c <- c / stddev.view([-1;1])
            c <- c / stddev.view([1;-1])
            c.clamp(-1,1)

    /// <summary>Returns a tensor where each row contains numSamples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.</summary>
    /// <param name="numSamples">The number of samples to draw.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    /// <param name="normalize">Indicates where the probabilities should first be normalized by their sum.</param>
    member probs.multinomial(numSamples:int, ?normalize:bool, ?device:Device, ?dtype:Dtype, ?backend:Backend) =
        // TODO: the following may be implemented by RawTensor at a later point
        if probs.dim < 1 || probs.dim > 2 then failwithf "Expecting 1d or 2d probs, received shape %A" probs.shape
        let device = defaultArg device probs.device
        let dtype = defaultArg dtype Dtype.Int32
        let backend = defaultArg backend probs.backend
        let normalize = defaultArg normalize false
        let mutable probs = probs
        if normalize then probs <- probs / probs.sum(-1, keepDim=true)
        if probs.dim = 1 then
            let p = 
                match probs.dtype with
                | Dtype.Float16
                | Dtype.BFloat16
                | Dtype.Float32 -> probs.toArray() :?> float32[] |> Array.map Convert.ToDouble
                | Dtype.Float64 -> probs.toArray() :?> float[]
                | _ -> failwithf "Expecting probs to have dtype Float32 or Float64, received %A" probs.dtype
            Tensor.create(Random.Multinomial(p, numSamples), device=device, dtype=dtype, backend=backend)
        else
            let p = 
                match probs.dtype with
                | Dtype.BFloat16
                | Dtype.Float16
                | Dtype.Float32 -> probs.toArray() :?> float32[,] |> Array2D.map Convert.ToDouble
                | Dtype.Float64 -> probs.toArray() :?> float[,]
                | _ -> failwithf "Expecting probs to be floating point, received %A" probs.dtype
            Tensor.create(Random.Multinomial(p, numSamples), device=device, dtype=dtype, backend=backend)

    /// <summary>Draws binary random numbers (0 or 1) from a Bernoulli distribution</summary>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    member probs.bernoulli(?device:Device, ?dtype:Dtype, ?backend:Backend) =
        // TODO: the following may be implemented by RawTensor at a later point
        if not probs.dtype.IsFloatingPoint then failwithf "Expecting probs to be floating point, received %A" probs.dtype
        let device = defaultArg device probs.device
        let dtype = defaultArg dtype probs.dtype
        let backend = defaultArg backend probs.backend
        if probs.dim = 0 then
            let b = Random.Bernoulli (float probs)
            Tensor.create(b, device=device, dtype=dtype, backend=backend).view(probs.shape)
        else
            let p:Tensor = probs.float().flatten()
            let b = p.toArray() :?> float[] |> Array.map Random.Bernoulli
            Tensor.create(b, device=device, dtype=dtype, backend=backend).view(probs.shape)

    /// <summary>Randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution</summary>
    /// <param name="p">The probability of an element to be zeroed. Default: 0.5.</param>
    member a.dropout(?p:double) =
        let p = defaultArg p 0.5
        Shape.checkCanDropout p
        if p = 0. then
            a
        elif p = 1. then
            a * a.zerosLike()
        else
            let mask = a.fullLike(1.-p).bernoulli()
            a * mask

    /// <summary>Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j] ). Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution</summary>
    /// <param name="p">The probability of an element to be zeroed. Default: 0.5.</param>
    member a.dropout2d(?p:double) =
        let p = defaultArg p 0.5
        Shape.checkCanDropout2d a.shape p
        if p = 0. then
            a
        elif p = 1. then
            a * a.zerosLike()
        else
            let mask = a.fullLike(1.-p, Array.append a.shape[0..1] [|1;1|]).bernoulli()
            a * mask

    /// <summary>Randomly zero out entire channels (a channel is a 3D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 3D tensor \text{input}[i, j]input[i,j] ). Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.</summary>
    /// <param name="p">The probability of an element to be zeroed. Default: 0.5.</param>
    member a.dropout3d(?p:double) =
        let p = defaultArg p 0.5
        Shape.checkCanDropout3d a.shape p
        if p = 0. then
            a
        elif p = 1. then
            a * a.zerosLike()
        else
            let mask = a.fullLike(1.-p, Array.append a.shape[0..1] [|1;1;1|]).bernoulli()
            a * mask
    
    /// <summary>Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.</summary>
    /// <param name="dim0">The first dimension to be transposed.</param>
    /// <param name="dim1">The second dimension to be transposed.</param>
    member a.transpose(dim0:int, dim1:int) =
        let dim0 = Shape.completeDim a.dim dim0  // Handles -1 semantics
        let dim1 = Shape.completeDim a.dim dim1  // Handles -1 semantics
        Shape.checkCanTranspose a.shape dim0 dim1
        if dim0 = dim1 then
            a
        else
            let inline fRaw(a:RawTensor) = a.TransposeT(dim0, dim1)
            let inline fTensor(a:Tensor) = a.transpose(dim0, dim1)
            let inline dfFwd(ap,ad:Tensor,fp) = ad.transpose(dim0, dim1)
            let inline dfRev(a) = TransposeT(a, dim0, dim1)
            Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns the original tensor with its dimensions permuted.</summary>
    /// <param name="permutation">The desired ordering of dimensions.</param>
    member a.permute(permutation:seq<int>) =
        let permutation = Seq.toArrayQuick permutation
        let inversePermutation, _ = Shape.checkCanPermute a.shape permutation
        if permutation |> Array.foralli (fun i j -> i = j) then
            a
        else
            let inline fRaw(a:RawTensor) = a.PermuteT(permutation)
            let inline fTensor(a:Tensor) = a.permute(permutation)
            let inline dfFwd(ap,ad:Tensor,fp) = ad.permute(permutation)
            let inline dfRev(a) = PermuteT(a, inversePermutation)
            Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a tensor that is a transposed version of input with dimensions 0 and 1 swapped.</summary>
    member a.transpose() =
        Shape.checkCanTranspose2d a.dim
        let inline fRaw(a:RawTensor) = a.TransposeT2()
        let inline fTensor(a:Tensor) = a.transpose()
        let inline dfFwd(ap,ad:Tensor,fp) = ad.transpose()
        let inline dfRev(a) = TransposeT2(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a tensor with all the dimensions of input of size 1 removed.</summary>
    /// <remarks>If the tensor has a batch dimension of size 1, then squeeze(input) will also remove the batch dimension, which can lead to unexpected errors.</remarks>
    /// <param name="dim">If given, the input will be squeezed only in this dimension.</param>
    member a.squeeze(?dim:int) =
        let dim = defaultArg dim -1
        let inline fRaw(a:RawTensor) = a.SqueezeT(dim)
        let inline fTensor(a:Tensor) = a.squeeze(dim)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.squeeze(dim)
        let inline dfRev(a) = SqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a new tensor with a dimension of size one inserted at the specified position</summary>
    /// <param name="dim">The index at which to insert the singleton dimension.</param>
    member a.unsqueeze(dim:int) : Tensor =
        let dim = Shape.completeDimUnsqueeze a.dim dim
        let inline fRaw(a:RawTensor) = a.UnsqueezeT(dim)
        let inline fTensor(a:Tensor) = a.unsqueeze(dim)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.unsqueeze(dim)
        let inline dfRev(a) = UnsqueezeT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a new tensor with dimensions of size one appended to the end until the number of dimensions is the same as the other tensor.</summary>
    /// <param name="other">The other tensor.</param>
    member a.unsqueezeAs(other:Tensor) =
        if a.dim >= other.dim then a
        else
            let newShape = Array.create other.dim 1
            System.Array.Copy(a.shape, newShape, a.shape.Length)
            a.view(newShape)

    /// <summary>Reverse the order of a n-D tensor along given axis in dims</summary>
    /// <param name="dims">The axis to flip on.</param>
    member a.flip(dims:seq<int>) =
        let dims = dims |> Array.ofSeq
        Shape.checkCanFlip a.dim dims
        let inline fRaw(a:RawTensor) = a.FlipT(dims)
        let inline fTensor(a:Tensor) = a.flip(dims)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.flip(dims)
        let inline dfRev(a) = FlipT(a, dims)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Dilate the tensor in using the given dilations in each corresponding dimension.</summary>
    /// <param name="dilations">The dilations to use.</param>
    member a.dilate(dilations:seq<int>) =
        let dilations = dilations |> Array.ofSeq
        Shape.checkCanDilate a.dim dilations
        let inline fRaw(a:RawTensor) = a.DilateT(dilations)
        let inline fTensor(a:Tensor) = a.dilate(dilations)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.dilate(dilations)
        let inline dfRev(a) = DilateT(a, dilations)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Reverse the dilation of the tensor in using the given dilations in each corresponding dimension.</summary>
    /// <param name="dilations">The dilations to use.</param>
    member a.undilate(dilations:seq<int>) =
        let dilations = dilations |> Array.ofSeq
        let inline fRaw(a:RawTensor) = a.UndilateT(dilations)
        let inline fTensor(a:Tensor) = a.undilate(dilations)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.undilate(dilations)
        let inline dfRev(a) = UndilateT(a, dilations)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Repeat elements of a tensor</summary>
    /// <param name="dim">The dimension along which to repeat values.</param>
    /// <param name="times">The number of repetitions for each element.</param>
    member a.repeat(dim:int, times:int) =
        // Note: the repeat op was used in the days before broadcasting was implemented
        // Most of its uses are now covered by broadcast and expand. But the operation
        // is well defined and correct so we can keep it.
        Shape.checkCanRepeat a.shape dim
        let newShape = a.shape |> Array.copy
        newShape[dim] <- times
        let mutable ret = a.zerosLike(newShape)
        let location = Array.create a.dim 0
        for i=0 to times-1 do
            location[dim] <- i
            ret <- ret.addSlice(location, a)
        ret

    /// <summary>Gathers values along an axis specified by dim.</summary>
    /// <param name="dim">The axis along which to index.</param>
    /// <param name="indices">The the indices of elements to gather.</param>
    member a.gather(dim:int, indices:Tensor) =
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        Shape.checkCanGather a.shape dim indices.shape indices.dtype
        let inline fRaw(a:RawTensor) = a.GatherT(dim, indices.primalRaw)
        let inline fTensor(a:Tensor) = a.gather(dim, indices)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.gather(dim, indices)
        let inline dfRev(a) = GatherT(a, dim, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Scatter values along an axis specified by dim.</summary>
    /// <param name="dim">The axis along which to index.</param>
    /// <param name="indices">The the indices of elements to gather.</param>
    /// <param name="destinationShape">The destination shape.</param>
    member a.scatter(dim:int, indices:Tensor, destinationShape:seq<int>) =
        let destinationShape = destinationShape|>Shape.create
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        Shape.checkCanScatter a.shape dim indices.shape indices.dtype destinationShape
        let inline fRaw(a:RawTensor) = a.ScatterT(dim, indices.primalRaw, destinationShape)
        let inline fTensor(a:Tensor) = a.scatter(dim, indices, destinationShape)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.scatter(dim, indices, destinationShape)
        let inline dfRev(a) = ScatterT(a, dim, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.</summary>
    /// <remarks>
    ///   The returned tensor shares the same data and must have the same number of elements, but may have a different size. 
    ///   For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension,
    ///   or only span across original dimensions \(d, d+1, \dots, d+kd,d+1,…,d+k\) that satisfy the following contiguity-like condition that
    ///   \(\forall i = d, \dots, d+k-1∀i=d,…,d+k−1 ,\) \[\text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]\]
    /// </remarks>
    /// <param name="shape">The desired shape of returned tensor.</param>
    member a.view(shape:seq<int>) =
        let shape = shape |> Shape.create |> Shape.complete a.nelement  // Handles -1 semantics
        if a.shape = shape then a // Do nothing if the shapes are the same
        else
        Shape.checkCanView a.shape shape
        let inline fRaw(a:RawTensor) = a.ViewT(shape)
        let inline fTensor(a:Tensor) = a.view(shape)
        let inline dfFwd(ap,ad:Tensor,fp) = ad.view(shape)
        let inline dfRev(a) = ViewT(a, a.shape)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a new tensor with the same data as the object tensor but of a different shape.</summary>
    /// <remarks>
    ///   The returned tensor shares the same data and must have the same number of elements, but may have a different size. 
    ///   For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension,
    ///   or only span across original dimensions \(d, d+1, \dots, d+kd,d+1,…,d+k\) that satisfy the following contiguity-like condition that
    ///   \(\forall i = d, \dots, d+k-1∀i=d,…,d+k−1 ,\) \[\text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]\]
    /// </remarks>
    /// <param name="shape">the desired shape</param>
    member t.view(shape:int) = t.view([|shape|])

    /// <summary>View this tensor as the same size as other.</summary>
    /// <remarks>The returned tensor shares the same data and must have the same number of elements, but may have a different size. For a tensor to be viewed, the new view size must be compatible with its original size.
    ///   The returned tensor shares the same data and must have the same number of elements, but may have a different size. 
    ///   For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension,
    ///   or only span across original dimensions \(d, d+1, \dots, d+kd,d+1,…,d+k\) that satisfy the following contiguity-like condition that
    ///   \(\forall i = d, \dots, d+k-1∀i=d,…,d+k−1 ,\) \[\text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]\]
    /// </remarks>
    /// <param name="other">The result tensor has the same size as other.</param>
    member a.viewAs(other:Tensor) = a.view(other.shape)

    /// <summary>Flattens a contiguous range of dims in a tensor.</summary>
    /// <param name="startDim">The first dim to flatten.</param>
    /// <param name="endDim">The last dim to flatten.</param>
    member a.flatten(?startDim:int, ?endDim:int) =
        if a.dim < 2 then 
            a
        else
            let startDim = defaultArg startDim 0
            let endDim = defaultArg endDim (a.dim - 1)
            Shape.checkCanFlatten a.shape startDim endDim
            a.view(a.shape |> Shape.flatten startDim endDim)

    /// <summary>Unflattens a tensor dimension by expanding it to the given shape.</summary>
    /// <param name="dim">The dimension to unflatten.</param>
    /// <param name="unflattenedShape">New shape of the unflattened dimenension.</param>
    member a.unflatten(dim:int, unflattenedShape:seq<int>) =
        let dim = Shape.completeDim a.dim dim
        if Shape.nelement (unflattenedShape |> Array.ofSeq) <> a.shape[dim] then failwithf "Expecting unflattenedShape (%A) to have the same number of elements with tensor's shape (%A) at given dim (%A)" unflattenedShape a.shape dim
        let newShape = a.shape |> Array.removeAt dim |> Array.insertManyAt dim unflattenedShape
        a.view(newShape)

    member internal a.clampWithMask(?low:scalar, ?high:scalar) =
        let lowTensor, highTensor = 
            match low, high with
            | Some l, Some h -> a.like(l), a.like(h)
            | Some l, None   -> a.like(l), a.like(System.Double.PositiveInfinity) // Having PositiveInfinity as upper limit is critical here, using a.max() does not work for some edge cases
            | None,   Some h -> a.like(System.Double.NegativeInfinity), a.like(h) // Having NegativeInfinity as lower limit is critical here, using a.min() does not work for some edge cases
            | None, None     -> failwithf "Expecting at least one of low, high"
        let mask() = // one-zero mask where the clamped values are zero and the rest are one
            let ll = lowTensor.expand(a.shape)
            let hh = highTensor.expand(a.shape)
            1 - (a.lt(ll) + a.gt(hh)).cast(a.dtype)
        match a with
        | TensorC(ap)          -> let result, mask = ap.ClampT(lowTensor.primalRaw, highTensor.primalRaw), mask() in TensorC(result), mask
        | TensorF(ap,ad,at)    -> let result, mask = ap.clampWithMask(?low=low, ?high=high) in TensorF(result, ad * mask, at), mask
        | TensorR(ap,_,_,_,at) -> let result, mask = ap.clampWithMask(?low=low, ?high=high) in TensorR(result, ref (a.zerosLike([0])), ClampT(a, mask), ref 0u, at), mask

    /// <summary>Clamp all elements in input into the range [ low..high] and return a resulting tensor</summary>
    /// <param name="low">The lower-bound of the range to be clamped to.</param>
    /// <param name="high">The upper-bound of the range to be clamped to.</param>
    member a.clamp(?low:scalar, ?high:scalar) = a.clampWithMask(?low=low, ?high=high) |> fst

    /// <summary>Returns a new tensor with the signs of the elements of input.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    member a.sign() =
        let inline fRaw(a:RawTensor) = a.SignT()
        let inline fTensor(a:Tensor) = a.sign()
        let inline dfFwd(ap,ad,fp:Tensor) = fp.zerosLike()
        let inline dfRev(a) = SignT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)
    // static member Sign(a:Tensor) = a.sign() // not supported because FSharp.Core sign operator returns int

    /// <summary>Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    member a.floor() =
        let inline fRaw(a:RawTensor) = a.FloorT()
        let inline fTensor(a:Tensor) = a.floor()
        let inline dfFwd(ap,ad,fp:Tensor) = fp.zerosLike()
        let inline dfRev(a) = FloorT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>floor</c>.</summary>
    static member Floor(a:Tensor) = a.floor() // needed for FSharp.Core floor operator overload

    /// <summary>Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    member a.ceil() =
        let inline fRaw(a:RawTensor) = a.CeilT()
        let inline fTensor(a:Tensor) = a.ceil()
        let inline dfFwd(ap,ad,fp:Tensor) = fp.zerosLike()
        let inline dfRev(a) = CeilT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>ceil</c>.</summary>
    static member Ceiling(a:Tensor) = a.ceil() // needed for FSharp.Core ceil operator overload

    /// <summary>Returns a new tensor with each of the elements of input rounded to the closest integer.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    member a.round() =
        let inline fRaw(a:RawTensor) = a.RoundT()
        let inline fTensor(a:Tensor) = a.round()
        let inline dfFwd(ap,ad,fp:Tensor) = fp.zerosLike()
        let inline dfRev(a) = RoundT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>round</c>.</summary>
    static member Round(a:Tensor) = a.round() // needed for FSharp.Core round operator overload

    /// <summary>Computes the element-wise absolute value of the given input tensor.</summary>
    member a.abs() =
        let inline fRaw(a:RawTensor) = a.AbsT()
        let inline fTensor(a:Tensor) = a.abs()
        let inline dfFwd(ap:Tensor,ad,fp) = ad * ap.sign()
        let inline dfRev(a) = AbsT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>abs</c>.</summary>
    static member Abs(a:Tensor) : Tensor = a.abs() // needed for FSharp.Core abs operator overload

    /// <summary>Applies the rectified linear unit function element-wise.</summary>
    member a.relu() =
        let inline fRaw(a:RawTensor) = a.ReluT()
        let inline fTensor(a:Tensor) = a.relu()
        let inline dfFwd(ap:Tensor,ad:Tensor,fp) = let sap = ap.sign() in ad * sap.abs() * (sap + 1.) / 2.
        let inline dfRev(a) = ReluT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Applies the leaky rectified linear unit function element-wise</summary>
    /// <remarks>\[\text{leakyRelu}(x) = \max(0, x) + \text{negativeSlope} * \min(0, x)\]</remarks>
    /// <param name="negativeSlope">Controls the angle of the negative slope. Default: 0.01.</param>
    member a.leakyRelu(?negativeSlope:float) =
        let negativeSlope = defaultArg negativeSlope 0.01
        let zeros = a.zerosLike() in zeros.max(a) + negativeSlope * zeros.min(a)

    /// <summary>Applies the sigmoid element-wise function</summary>
    /// <remarks>\[\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}\]</remarks>
    member a.sigmoid() =
        let inline fRaw(a:RawTensor) = a.SigmoidT()
        let inline fTensor(a:Tensor) = a.sigmoid()
        let inline dfFwd(ap,ad,fp:Tensor) = ad * fp * (1. - fp)
        let inline dfRev(a) = SigmoidT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Applies the exp function element-wise.</summary>
    member a.exp() =
        let inline fRaw(a:RawTensor) = a.ExpT()
        let inline fTensor(a:Tensor) = a.exp()
        let inline dfFwd(ap,ad,fp) = ad * fp
        let inline dfRev(a) = ExpT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>exp</c>.</summary>
    static member Exp(a:Tensor) = a.exp() // needed for FSharp.Core exp operator overload

    /// <summary>Returns a new tensor with the natural logarithm of the elements of input.</summary>
    /// <remarks> \[y_{i} = \log_{e} (x_{i})\]</remarks>
    member a.log() =
        let inline fRaw(a:RawTensor) = a.LogT()
        let inline fTensor(a:Tensor) = a.log()
        let inline dfFwd(ap,ad,fp) = ad / ap
        let inline dfRev(a) = LogT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>log</c>.</summary>
    static member Log(a:Tensor) = a.log() // needed for FSharp.Core log operator overload

    /// <summary>Returns the logarithm of the tensor after clamping the tensor so that all its elements are greater than epsilon. This is to avoid a -inf result for elements equal to zero.</summary>
    member a.safelog(?epsilon:float) =
        let epsilon = defaultArg epsilon 1e-12
        a.clamp(low=epsilon).log()

    /// <summary>Applies the softplus function element-wise.</summary>
    /// <remarks>\[\text{softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))\]</remarks>
    member a.softplus() =
        let inline fRaw(a:RawTensor) = a.SoftplusT()
        let inline fTensor(a:Tensor) = a.softplus()
        let inline dfFwd(ap:Tensor,ad,fp) = ad / (1. + ap.neg().exp())
        let inline dfRev(a) = SoftplusT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Returns a new tensor with the logarithm to the base 10 of the elements of input.</summary>
    /// <remarks>\[y_{i} = \log_{10} (x_{i})\]</remarks>
    member a.log10() =
        let inline fRaw(a:RawTensor) = a.Log10T()
        let inline fTensor(a:Tensor) = a.log10()
        let inline dfFwd(ap:Tensor,ad,fp) = ad / (ap * log10Val)
        let inline dfRev(a) = Log10T(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>log10</c>.</summary>
    static member Log10(a:Tensor) = a.log10() // needed for FSharp.Core log10 operator overload

    /// <summary>Returns a new tensor with the square-root of the elements of input.</summary>
    member a.sqrt() =
        let inline fRaw(a:RawTensor) = a.SqrtT()
        let inline fTensor(a:Tensor) = a.sqrt()
        let inline dfFwd(ap,ad,fp:Tensor) = ad / (2. * fp)
        let inline dfRev(a) = SqrtT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>sqrt</c>.</summary>
    static member Sqrt(a:Tensor) = a.sqrt() // needed for FSharp.Core sqrt operator overload

    /// <summary>Returns a new tensor with the sine of the elements of input</summary>
    member a.sin() =
        let inline fRaw(a:RawTensor) = a.SinT()
        let inline fTensor(a:Tensor) = a.sin()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = ad * ap.cos()
        let inline dfRev(a) = SinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>sin</c>.</summary>
    static member Sin(a:Tensor) = a.sin() // needed for FSharp.Core sin operator overload

    /// <summary>Returns a new tensor with the cosine of the elements of input</summary>
    member a.cos() =
        let inline fRaw(a:RawTensor) = a.CosT()
        let inline fTensor(a:Tensor) = a.cos()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = -ad * ap.sin()
        let inline dfRev(a) = CosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>cos</c>.</summary>
    static member Cos(a:Tensor) = a.cos() // needed for FSharp.Core cos operator overload

    /// <summary>Returns a new tensor with the tangent of the elements of input</summary>
    member a.tan() =
        let inline fRaw(a:RawTensor) = a.TanT()
        let inline fTensor(a:Tensor) = a.tan()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = let cosap = ap.cos() in ad / (cosap * cosap)
        let inline dfRev(a) = TanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>tan</c>.</summary>
    static member Tan(a:Tensor) = a.tan() // needed for FSharp.Core tan operator overload

    /// <summary>Returns a new tensor with the hyperbolic sine of the elements of input.</summary>
    member a.sinh() =
        let inline fRaw(a:RawTensor) = a.SinhT()
        let inline fTensor(a:Tensor) = a.sinh()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = ad * ap.cosh()
        let inline dfRev(a) = SinhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>sinh</c>.</summary>
    static member Sinh(a:Tensor) = a.sinh() // needed for FSharp.Core sinh operator overload

    /// <summary>Returns a new tensor with the hyperbolic cosine of the elements of input.</summary>
    member a.cosh() =
        let inline fRaw(a:RawTensor) = a.CoshT()
        let inline fTensor(a:Tensor) = a.cosh()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = ad * ap.sinh()
        let inline dfRev(a) = CoshT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>cosh</c>.</summary>
    static member Cosh(t:Tensor) = t.cosh() // needed for FSharp.Core cosh operator overload

    /// <summary>Returns a new tensor with the hyperbolic tangent of the elements of input.</summary>
    member a.tanh() =
        let inline fRaw(a:RawTensor) = a.TanhT()
        let inline fTensor(a:Tensor) = a.tanh()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = let coshap = ap.cosh() in ad / (coshap * coshap)
        let inline dfRev(a) = TanhT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>tanh</c>.</summary>
    static member Tanh(t:Tensor) = t.tanh() // needed for FSharp.Core tanh operator overload

    /// <summary>Returns a new tensor with the arcsine of the elements of input.</summary>
    member a.asin() =
        let inline fRaw(a:RawTensor) = a.AsinT()
        let inline fTensor(a:Tensor) = a.asin()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = ad / (1. - ap*ap).sqrt()
        let inline dfRev(a) = AsinT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>asin</c>.</summary>
    static member Asin(t:Tensor) = t.asin() // needed for FSharp.Core asin operator overload

    /// <summary>Returns a new tensor with the arccosine of the elements of input.</summary>
    member a.acos() =
        let inline fRaw(a:RawTensor) = a.AcosT()
        let inline fTensor(a:Tensor) = a.acos()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = -ad / (1. - ap*ap).sqrt()
        let inline dfRev(a) = AcosT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>acos</c>.</summary>
    static member Acos(t:Tensor) = t.acos() // needed for FSharp.Core acos operator overload

    /// <summary>Returns a new tensor with the arctangent of the elements of input.</summary>
    member a.atan() =
        let inline fRaw(a:RawTensor) = a.AtanT()
        let inline fTensor(a:Tensor) = a.atan()
        let inline dfFwd(ap:Tensor,ad,fp:Tensor) = ad / (1. + ap*ap)
        let inline dfRev(a) = AtanT(a)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>A method to enable the use of the F# function <c>atan</c>.</summary>
    static member Atan(t:Tensor) = t.atan() // needed for FSharp.Core atan operator overload

    /// <summary>Add the given tensor as a slice at the given location.</summary>
    member a.addSlice(location:seq<int>, b:Tensor) =
        let location = location |> Seq.toArray
        Shape.checkCanAddSlice a.shape location b.shape
        if a.shape = b.shape && location |> Array.forall ((=) 0) then a + b // No need to do the slice addition below
        else
        let inline fRaw(a:RawTensor,b) = a.AddTTSlice(location, b)
        let inline fTensor(a:Tensor,b) = a.addSlice(location, b)
        let inline dfFwdTT(ap,ad:Tensor,bp:Tensor,bd:Tensor,fp) = ad.addSlice(location, bd)
        let inline dfFwdTC(ap,ad,fp) = ad
        let inline dfFwdCT(bp,bd,fp:Tensor) = fp.zerosLike().addSlice(location, bd)
        let inline dfRevTT(a,b) = AddTTSlice(a,location,b)
        let inline dfRevTC(a,b) = AddTTConstSlice(a)
        let inline dfRevCT(a,b) = AddTConstTSlice(location,b)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)

    /// <summary>Applies a softmax function.</summary>
    /// <remarks>Softmax is defined as: \text{softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}.</remarks>
    /// <param name="dim">A dimension along which softmax will be computed.</param>
    member a.softmax(dim:int) =
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        let e = (a - a.max().noDiff()).exp()
        let esum = e.sum(dim, keepDim=true)
        e / esum

    /// <summary>Applies a softmax followed by a logarithm.</summary>
    /// <param name="dim">A dimension along which softmax will be computed.</param>
    member a.logsoftmax(dim:int) =
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        a - a.logsumexp(dim, keepDim=true)

    /// <summary>Applies a logsumexp.</summary>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    member a.logsumexp(dim:int, ?keepDim:bool) =
        let dim = Shape.completeDim a.dim dim  // Handles -1 semantics
        let keepDim = defaultArg keepDim false
        let amax = a.max().noDiff()
        let e = (a - amax).exp()
        let res = amax + e.sum(dim).add(System.Single.Epsilon).log()
        if keepDim then res.unsqueeze(dim) else res

    /// <summary>Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input and the target.</summary>
    /// <param name="target">The target tensor.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    member input.mseLoss(target:Tensor, ?reduction:string) = 
        if input.shape <> target.shape then failwithf "Expecting input.shape (%A) and target.shape (%A) to be the same" input.shape target.shape
        let reduction = defaultArg reduction "mean"
        if not (reduction = "none" || reduction = "mean" || reduction = "sum") then failwithf "Expecting reduction (%A) to be one of (none, mean, sum)" reduction
        let z = input - target
        let l = z * z
        if reduction = "none" then
            l
        elif reduction = "mean" then
            l.mean()
        else // reduction = "sum"
            l.sum()

    /// <summary>Creates a criterion that measures the Binary Cross Entropy between the target and the output</summary>
    /// <param name="target">The target tensor.</param>
    /// <param name="weight">A manual rescaling weight given to the loss of each batch element.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    member input.bceLoss(target:Tensor, ?weight:Tensor, ?reduction:string) =
        if input.shape <> target.shape then failwithf "Expecting input shape (%A) and target shape (%A) to be the same" input.shape target.shape
        if float (input.max()) > 1. || float (input.min()) < 0. then failwithf "Expecting input values to be between 0 and 1, received %A %.20f %A.20f" input (float (input.max())) (float (input.min()))
        if float (target.max()) > 1. || float (target.min()) < 0. then failwithf "Expecting target values to be between 0 and 1, received %A" target
        if input.dim < 1 then let ret:Tensor = input.view(-1).bceLoss(target.view(-1), ?weight=weight, ?reduction=reduction) in if ret.dim = 0 then ret else ret[0]
        else
        let n = input.shape[0]
        let weight = defaultArg weight (input.onesLike(shape=[|n|]))
        if weight.shape[0] <> n then failwithf "Expecting weight to be a vector of size %A, but received %A" n weight.shape[0]
        let reduction = defaultArg reduction "mean"
        if not (reduction = "none" || reduction = "mean" || reduction = "sum") then failwithf "Expecting reduction (%A) to be one of (none, mean, sum)" reduction
        let epsilon = 1e-12
        let clampLog = -100
        let l = -weight.unsqueezeAs(input)*(target * input.safelog(epsilon).clamp(low=clampLog) + (1.-target) * (1.-input).safelog(epsilon).clamp(low=clampLog))
        if reduction = "none" then
            l
        elif reduction = "mean" then
            l.mean()
        else // reduction = "sum"
            l.sum()

    /// <summary>This criterion combines logsoftmax and nllLoss in a single function</summary>
    /// <param name="target">The target tensor.</param>
    /// <param name="weight">A optional manual rescaling weight given to the loss of each batch element.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    member input.crossEntropyLoss(target:Tensor, ?weight:Tensor, ?reduction:string) =
        input.logsoftmax(dim=1).nllLoss(target, ?weight=weight, ?reduction=reduction)

    /// <summary>The negative log likelihood loss.</summary>
    /// <param name="target">The target tensor.</param>
    /// <param name="weight">A optional manual rescaling weight given to the loss of each batch element.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    member input.nllLoss(target:Tensor, ?weight:Tensor, ?reduction:string) =
        let n, classes, d = 
            if input.dim < 2 
                then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
            elif input.dim = 2 then
                let n, c = input.shape[0], input.shape[1]
                if target.shape <> [|n|] then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
                n, c, [||]
            else
                let n, c, d = input.shape[0], input.shape[1], input.shape[2..]
                if target.shape[0] <> n then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
                if d <> target.shape[1..] then failwithf "Expecting either: input with shape (N,C) and target with shape (N); or input with shape (N,C,d1,d2,...,dk) and target with shape (N,d1,d2,...,dk). Received input.shape %A and target.shape %A" input.shape target.shape
                n, c, d
        let target = target.int()
        let weightSpecified, weight = 
            match weight with
            | Some w -> 
                if w.dim <> 1 || w.shape[0] <> classes then failwithf "Expecting weight with shape (C). Received weight.shape %A" w.shape
                let vv = Array.create input.dim 1
                vv[1] <- classes
                true, w.view(vv).expandAs(input).gather(1, target.unsqueeze(1)).squeeze(1)
            | None -> false, input.zeroLike()
        let reduction = defaultArg reduction "mean"
        if not (reduction = "none" || reduction = "mean" || reduction = "sum") then failwithf "Expecting reduction (%A) to be one of (none, mean, sum)" reduction
        let mutable l = input.gather(1, target.unsqueeze(1)).squeeze(1).neg()
        if weightSpecified then
            l <- l * weight
        if reduction = "none" then
            l
        elif reduction = "mean" then
            if weightSpecified then l.sum()/weight.sum() else l.mean()
        else // reduction = "sum"
            l.sum()

    /// <summary>Add zero padding to each side of a tensor</summary>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    member a.pad(paddings:seq<int>) =
        let paddings = paddings |> Array.ofSeq
        Shape.checkCanPad a.shape paddings
        if paddings |> Array.sum = 0 then
            a
        else
            let shape = Array.copy a.shape
            for i in 0..shape.Length-1 do
                shape[i] <- shape[i] + paddings[i] * 2
            let ret = a.zerosLike(shape)
            ret.addSlice(paddings, a)

    /// <summary>Applies a 1D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    member a.maxpool1di(kernelSize:int, ?stride:int, ?padding:int) =
        let stride = defaultArg stride kernelSize
        let padding = defaultArg padding 0
        Shape.checkCanMaxpool1d a.dtype a.shape kernelSize stride padding  |> ignore
        match a with
        | TensorC(ap)          -> let result, indices = ap.MaxPool1D(kernelSize, stride, padding) in TensorC(result), TensorC(indices)
        | TensorF(ap,ad,at)    -> let result, indices = ap.maxpool1di(kernelSize, stride, padding) in TensorF(result, ad.gather(dim=2, indices=indices), at), indices
        | TensorR(ap,_,_,_,at) -> let result, indices = ap.maxpool1di(kernelSize, stride, padding) in TensorR(result, ref (a.zerosLike([0])), MaxPool1DT(a, indices, kernelSize), ref 0u, at), indices

    /// <summary>Applies a 1D max pooling over an input signal composed of several input planes.</summary>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    member a.maxpool1d(kernelSize:int, ?stride:int, ?padding:int) = a.maxpool1di(kernelSize, ?stride=stride, ?padding=padding) |> fst

    /// <summary>Computes a partial inverse of maxpool1di</summary>
    /// <param name="indices">The indices selected by maxpool1di.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="outputSize">The targeted output size.</param>
    member a.maxunpool1d(indices:Tensor, kernelSize:int, ?stride:int, ?padding:int, ?outputSize:seq<int>) =
        let stride = defaultArg stride kernelSize
        let padding = defaultArg padding 0
        let outputSize = 
            match outputSize with
            | Some o -> let o = o |> Array.ofSeq in if o.Length <> 3 then failwithf "Expecting outputSize to be 3-dimensional" else o
            | None -> 
                let inputSize = a.shape[2]
                [|indices.shape[0]; indices.shape[1]; ((inputSize-1) * stride - 2*padding + kernelSize)|]
        Shape.checkCanMaxunpool1d a.dtype a.shape indices.dtype indices.shape outputSize |> ignore
        let inline fRaw(a:RawTensor) = a.MaxUnpool1D(indices.primalRaw, outputSize)
        let inline fTensor(a:Tensor) = a.maxunpool1d(indices, kernelSize, stride=stride, padding=padding, outputSize=outputSize)
        let inline dfFwd(ap:Tensor,ad:Tensor,fp:Tensor) = ad.maxunpool1d(indices, kernelSize, stride=stride, padding=padding, outputSize=outputSize)
        let inline dfRev(a) = MaxUnpool1DT(a, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Applies a 2D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    member a.maxpool2di(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        let kernelSizes, strides, paddings = Shape.resolve2dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
        Shape.checkCanMaxpool2d a.dtype a.shape kernelSizes strides paddings  |> ignore
        match a with
        | TensorC(ap)          -> let result, indices = ap.MaxPool2D(kernelSizes, strides, paddings) in TensorC(result), TensorC(indices)
        | TensorF(ap,ad,at)    -> let result, indices = ap.maxpool2di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorF(result, ad.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(indices), at), indices
        | TensorR(ap,_,_,_,at) -> let result, indices = ap.maxpool2di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorR(result, ref (a.zerosLike([0])), MaxPool2DT(a, indices, kernelSizes), ref 0u, at), indices

    /// <summary>Applies a 2D max pooling over an input signal composed of several input planes.</summary>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    member a.maxpool2d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool2di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

    /// <summary>Computes a partial inverse of maxpool2di</summary>
    /// <param name="indices">The indices selected by maxpool2di.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    /// <param name="outputSize">The targeted output size.</param>
    member a.maxunpool2d(indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) =
        let kernelSizes, strides, paddings = Shape.resolve2dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
        let outputSize = 
            match outputSize with
            | Some o -> let o = o |> Array.ofSeq in if o.Length <> 4 then failwithf "Expecting outputSize to be 4-dimensional" else o
            | None -> 
                let inputHeight = a.shape[2]
                let inputWidth = a.shape[3]
                [|indices.shape[0]; indices.shape[1]; ((inputHeight-1) * strides[0] - 2*paddings[0] + kernelSizes[0]); ((inputWidth-1) * strides[1] - 2*paddings[1] + kernelSizes[1])|]
        Shape.checkCanMaxunpool2d a.dtype a.shape indices.dtype indices.shape outputSize |> ignore
        let inline fRaw(a:RawTensor) = a.MaxUnpool2D(indices.primalRaw, outputSize)
        let inline fTensor(a:Tensor) = a.maxunpool2d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let inline dfFwd(ap:Tensor,ad:Tensor,fp:Tensor) = ad.maxunpool2d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let inline dfRev(a) = MaxUnpool2DT(a, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Applies a 3D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    member a.maxpool3di(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        let kernelSizes, strides, paddings = Shape.resolve3dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
        Shape.checkCanMaxpool3d a.dtype a.shape kernelSizes strides paddings |> ignore
        match a with
        | TensorC(ap)          -> let result, indices = ap.MaxPool3D(kernelSizes, strides, paddings) in TensorC(result), TensorC(indices)
        | TensorF(ap,ad,at)    -> let result, indices = ap.maxpool3di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorF(result, ad.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(indices), at), indices
        | TensorR(ap,_,_,_,at) -> let result, indices = ap.maxpool3di(kernelSizes=kernelSizes, strides=strides, paddings=paddings) in TensorR(result, ref (a.zerosLike([0])), MaxPool3DT(a, indices, kernelSizes), ref 0u, at), indices

    /// <summary>Applies a 3D max pooling over an input signal composed of several input planes.</summary>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    member a.maxpool3d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool3di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

    /// <summary>Computes a partial inverse of maxpool3di</summary>
    /// <param name="indices">The indices selected by maxpool3di.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    /// <param name="outputSize">The targeted output size.</param>
    member a.maxunpool3d(indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) =
        let kernelSizes, strides, paddings = Shape.resolve3dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings
        let outputSize = 
            match outputSize with
            | Some o -> let o = o |> Array.ofSeq in if o.Length <> 5 then failwithf "Expecting outputSize to be 5-dimensional" else o
            | None -> 
                let inputDepth = a.shape[2]
                let inputHeight = a.shape[3]
                let inputWidth = a.shape[4]
                [|indices.shape[0]; indices.shape[1]; ((inputDepth-1) * strides[0] - 2*paddings[0] + kernelSizes[0]); ((inputHeight-1) * strides[1] - 2*paddings[1] + kernelSizes[1]); ((inputWidth-1) * strides[2] - 2*paddings[2] + kernelSizes[2])|]
        Shape.checkCanMaxunpool3d a.dtype a.shape indices.dtype indices.shape outputSize |> ignore
        let inline fRaw(a:RawTensor) = a.MaxUnpool3D(indices.primalRaw, outputSize)
        let inline fTensor(a:Tensor) = a.maxunpool3d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let inline dfFwd(ap:Tensor,ad:Tensor,fp:Tensor) = ad.maxunpool3d(indices, kernelSizes=kernelSizes, strides=strides, paddings=paddings, outputSize=outputSize)
        let inline dfRev(a) = MaxUnpool3DT(a, indices)
        Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)

    /// <summary>Applies a 1D convolution over an input signal composed of several input planes</summary>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit paddings on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    member a.conv1d(filters:Tensor, ?stride:int, ?padding:int, ?dilation:int) =
        let b = filters
        let stride = defaultArg stride 1
        let padding = defaultArg padding 0
        let dilation = defaultArg dilation 1
        Shape.checkCanConv1d a.deviceType b.deviceType a.dtype b.dtype a.shape b.shape stride padding dilation |> ignore
        let mutable b = b
        if dilation > 1 then
            b <- b.dilate([|1;1;dilation|])
        let inline fRaw(a:RawTensor,b) = a.Conv1D(b, stride, padding)
        let inline fTensor(a:Tensor,b) = a.conv1d(b, stride, padding)
        let inline dfFwdTT(ap:Tensor,ad:Tensor,bp:Tensor,bd:Tensor,fp) = ad.conv1d(bp, stride, padding) + ap.conv1d(bd, stride, padding)
        let inline dfFwdTC(ap,ad:Tensor,fp) = ad.conv1d(b, stride, padding)
        let inline dfFwdCT(bp,bd,fp) = a.conv1d(bd, stride, padding)
        let inline dfRevTT(a,b) = Conv1DTT(a,b, stride, padding)
        let inline dfRevTC(a,b) = Conv1DTTConst(a,b, stride, padding)
        let inline dfRevCT(a,b) = Conv1DTConstT(a,b, stride, padding)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)

    // a: input, NxCxI (batchSize x inputChannels x inputLength)
    // b: filters, KxCxF (outputChannels x inputChannels x kernelLength)
    // t: output, NxKxL (batchSize x outputChannels x outputLength)
    static member internal conv1dReverseDiff(a: Tensor, b:Tensor, fderivative:Tensor, aConst:bool, bConst:bool, stride:int, padding:int) =
        let a = if aConst then a else a.primal
        let b = if bConst then b else b.primal
        let batchSize = fderivative.shape[0]
        let outputChannels = fderivative.shape[1]
        // let outputLength = fderivative.shape[2]
        let inputChannels = a.shape[1]
        let inputLength = a.shape[2]
        let kernelLength = b.shape[2]
        let mutable fderivative = fderivative
        if stride > 1 then
            fderivative <- fderivative.dilate([|1;1;stride|])
        let mutable aderivative = a.zeroLike()
        let mutable bderivative = b.zeroLike()
        if not aConst then
            // propagate to a
            let bFlipped = b.flip([|2|])
            let mutable ad = fderivative.conv1d(bFlipped.transpose(0, 1), padding=kernelLength-1)
            if padding > 0 then
                let adBounds = array2D [[0; batchSize-1; 0]; [0; inputChannels-1; 0]; [padding; padding + inputLength - 1; 0]]
                ad <- ad.GetSlice(adBounds)
                ad <- ad.view([|batchSize; inputChannels; inputLength|])
            aderivative <- a.zerosLike().addSlice([|0; 0; 0|], ad)
        if not bConst then
            // propagate to b
            let aa = a.transpose(0, 1)
            let fd = fderivative.transpose(0, 1)
            let bd = aa.conv1d(fd, padding=padding).transpose(0, 1)
            let bdBounds = array2D [[0;outputChannels-1;0]; [0;inputChannels-1;0]; [0;kernelLength-1;0]]
            bderivative <- bd.GetSlice(bdBounds)
        aderivative, bderivative
    
    /// <summary>Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
    member a.convTranspose1d(filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int) =
        let b = filters
        let stride = defaultArg stride 1
        let padding = defaultArg padding 0
        let dilation = defaultArg dilation 1
        let outputPadding = defaultArg outputPadding 0

        let _, _, _, _, _, outputShape =
            Shape.checkCanConvTranspose1d a.deviceType b.deviceType a.dtype b.dtype a.shape b.shape stride padding dilation outputPadding
        let mutable b = b
        if dilation > 1 then
            b <- b.dilate([|1; 1; dilation|])
        let fderivative = a
        let a = a.zerosLike(outputShape)
        // Use convolution reverse mode to implement transposed convolution
        let (aderivative:Tensor), _ = Tensor.conv1dReverseDiff(a, b, fderivative, aConst=false, bConst=true, stride=stride, padding=padding)
        aderivative

    /// <summary>Applies a 2D convolution over an input signal composed of several input planes</summary>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    member a.conv2d(filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>) =
        let b = filters
        let strides, paddings, dilations = Shape.resolve2dConvSizes stride strides padding paddings dilation dilations
        Shape.checkCanConv2d a.deviceType b.deviceType a.dtype b.dtype a.shape b.shape strides paddings dilations |> ignore
        let mutable b = b
        if dilations[0] > 1 || dilations[1] > 1 then
            b <- b.dilate([|1; 1; dilations[0]; dilations[1]|])
        let inline fRaw(a:RawTensor,b) = a.Conv2D(b, strides, paddings)
        let inline fTensor(a:Tensor,b) = a.conv2d(b, strides=strides, paddings=paddings)
        let inline dfFwdTT(ap:Tensor,ad:Tensor,bp,bd,fp) = ad.conv2d(bp, strides=strides, paddings=paddings) + ap.conv2d(bd, strides=strides, paddings=paddings)
        let inline dfFwdTC(ap,ad:Tensor,fp) = ad.conv2d(b, strides=strides, paddings=paddings)
        let inline dfFwdCT(bp,bd,fp) = a.conv2d(bd, strides=strides, paddings=paddings)
        let inline dfRevTT(a,b) = Conv2DTT(a,b, strides, paddings)
        let inline dfRevTC(a,b) = Conv2DTTConst(a,b, strides, paddings)
        let inline dfRevCT(a,b) = Conv2DTConstT(a,b, strides, paddings)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)

    // a: input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth)
    // b: filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth)
    // t: output, NxKxLxM (batchSize x outputChannels x outputHeight x outputWidth)
    static member internal conv2dReverseDiff(a: Tensor, b:Tensor, fderivative:Tensor, aConst:bool, bConst:bool, strides:int[], paddings:int[]) =
        let a = if aConst then a else a.primal
        let b = if bConst then b else b.primal
        let batchSize = fderivative.shape[0]
        let outputChannels = fderivative.shape[1]
        // let outputHeight = fderivative.shape[2]
        // let outputWidth = fderivative.shape[3]
        let inputChannels = a.shape[1]
        let inputHeight = a.shape[2]
        let inputWidth = a.shape[3]
        let kernelHeight = b.shape[2]
        let kernelWidth = b.shape[3]
        let mutable fderivative = fderivative
        if strides[0] > 1 || strides[1] > 1 then
            fderivative <- fderivative.dilate([|1;1;strides[0];strides[1]|])
        let mutable aderivative = a.zeroLike()
        let mutable bderivative = b.zeroLike()
        if not aConst then
            // propagate to a
            let bFlipped = b.flip([|2;3|])
            let mutable ad = fderivative.conv2d(bFlipped.transpose(0, 1), paddings=[|kernelHeight-1; kernelWidth-1|])
            if paddings[0] > 0 || paddings[1] > 0 then
                let adBounds = array2D [[0; batchSize-1; 0]; 
                                       [0; inputChannels-1; 0]; 
                                       [paddings[0]; paddings[0] + inputHeight - 1; 0]; 
                                       [paddings[1]; paddings[1] + inputWidth - 1; 0]]
                ad <- ad.GetSlice(adBounds)
                ad <- ad.view([|batchSize; inputChannels; inputHeight; inputWidth|])
            aderivative <- a.zerosLike().addSlice([|0; 0; 0; 0|], ad)
        if not bConst then
            // propagate to b
            let aa = a.transpose(0, 1)
            let fd = fderivative.transpose(0, 1)
            let bd = aa.conv2d(fd, paddings=paddings).transpose(0, 1)
            let bdBounds = array2D [[0;outputChannels-1;0]; [0;inputChannels-1;0]; [0;kernelHeight-1;0]; [0;kernelWidth-1;0]]
            bderivative <- bd.GetSlice(bdBounds)
        aderivative, bderivative
    
    /// <summary>Applies a 2D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
    /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
    member a.convTranspose2d(filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) =
        let b = filters
        let strides, paddings, dilations = Shape.resolve2dConvSizes stride strides padding paddings dilation dilations
        let outputPaddings = Shape.resolve2dConvOutputPadding outputPadding outputPaddings
        let _, _, _, _, outputShape =
            Shape.checkCanConvTranspose2d a.deviceType b.deviceType a.dtype b.dtype a.shape b.shape strides paddings dilations outputPaddings
        let mutable b = b
        if dilations[0] > 1 || dilations[1] > 1 then
            b <- b.dilate([|1; 1; dilations[0]; dilations[1]|])
        let fderivative = a
        let a = a.zerosLike(outputShape)
        // Use convolution reverse mode to implement transposed convolution
        let (aderivative:Tensor), _ = Tensor.conv2dReverseDiff(a, b, fderivative, aConst=false, bConst=true, strides=strides, paddings=paddings)
        aderivative

    /// <summary>Applies a 3D convolution over an input signal composed of several input planes</summary>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    member a.conv3d(filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>) =
        let b = filters
        let strides, paddings, dilations = Shape.resolve3dConvSizes stride strides padding paddings dilation dilations
        Shape.checkCanConv3d a.deviceType b.deviceType a.dtype b.dtype a.shape b.shape strides paddings dilations |> ignore
        let mutable b = b
        if dilations[0] > 1 || dilations[1] > 1 || dilations[2] > 1 then
            b <- b.dilate([|1; 1; dilations[0]; dilations[1]; dilations[2]|])
        let inline fRaw(a:RawTensor,b) = a.Conv3D(b, strides, paddings)
        let inline fTensor(a:Tensor,b) = a.conv3d(b, strides=strides, paddings=paddings)
        let inline dfFwdTT(ap:Tensor,ad:Tensor,bp,bd,fp) = ad.conv3d(bp, strides=strides, paddings=paddings) + ap.conv3d(bd, strides=strides, paddings=paddings)
        let inline dfFwdTC(ap,ad:Tensor,fp) = ad.conv3d(b, strides=strides, paddings=paddings)
        let inline dfFwdCT(bp,bd,fp) = a.conv3d(bd, strides=strides, paddings=paddings)
        let inline dfRevTT(a,b) = Conv3DTT(a,b, strides, paddings)
        let inline dfRevTC(a,b) = Conv3DTTConst(a,b, strides, paddings)
        let inline dfRevCT(a,b) = Conv3DTConstT(a,b, strides, paddings)
        Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)

    // a: input, NxCxDxHxW (batchSize x inputChannels x inputDepth x inputHeight x inputWidth)
    // b: filters, KxCxExFxG (outputChannels x inputChannels x kernelDepth x kernelHeight x kernelWidth)
    // t: output, NxKxLxMxN (batchSize x outputChannels x outputDepth x outputHeight x outputWidth)
    static member internal conv3dReverseDiff(a: Tensor, b:Tensor, fderivative:Tensor, aConst:bool, bConst:bool, strides:int[], paddings:int[]) =
        let a = if aConst then a else a.primal
        let b = if bConst then b else b.primal
        let batchSize = fderivative.shape[0]
        let outputChannels = fderivative.shape[1]
        // let outputDepth = fderivative.shape[2]
        // let outputHeight = fderivative.shape[3]
        // let outputWidth = fderivative.shape[4]
        let inputChannels = a.shape[1]
        let inputDepth = a.shape[2]
        let inputHeight = a.shape[3]
        let inputWidth = a.shape[4]
        let kernelDepth = b.shape[2]
        let kernelHeight = b.shape[3]
        let kernelWidth = b.shape[4]
        let mutable fderivative = fderivative
        if strides[0] > 1 || strides[1] > 1 || strides[2] > 1 then
            fderivative <- fderivative.dilate([|1;1;strides[0];strides[1];strides[2]|])
        let mutable aderivative = a.zeroLike()
        let mutable bderivative = b.zeroLike()
        if not aConst then
            // propagate to a
            let bFlipped = b.flip([|2;3;4|])
            let mutable ad = fderivative.conv3d(bFlipped.transpose(0, 1), paddings=[|kernelDepth-1; kernelHeight-1; kernelWidth-1|])
            if paddings[0] > 0 || paddings[1] > 0 || paddings[2] > 0 then
                let adBounds = array2D [[0; batchSize-1; 0]; 
                                       [0; inputChannels-1; 0]; 
                                       [paddings[0]; paddings[0] + inputDepth - 1; 0]; 
                                       [paddings[1]; paddings[1] + inputHeight - 1; 0];
                                       [paddings[2]; paddings[2] + inputWidth - 1; 0]]
                ad <- ad.GetSlice(adBounds)
                ad <- ad.view([|batchSize; inputChannels; inputDepth; inputHeight; inputWidth|])
            aderivative <- a.zerosLike().addSlice([|0; 0; 0; 0; 0|], ad)
        if not bConst then
            // propagate to b
            let aa = a.transpose(0, 1)
            let fd = fderivative.transpose(0, 1)
            let bd = aa.conv3d(fd, paddings=paddings).transpose(0, 1)
            let bdBounds = array2D [[0;outputChannels-1;0]; [0;inputChannels-1;0]; [0;kernelDepth-1;0]; [0;kernelHeight-1;0]; [0;kernelWidth-1;0]]
            bderivative <- bd.GetSlice(bdBounds)                
        aderivative, bderivative

    /// <summary>Applies a 3D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
    /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
    member a.convTranspose3d(filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) =
        let b = filters
        let strides, paddings, dilations = Shape.resolve3dConvSizes stride strides padding paddings dilation dilations
        let outputPaddings = Shape.resolve3dConvOutputPadding outputPadding outputPaddings
        let _, _, _, _, outputShape =
            Shape.checkCanConvTranspose3d a.deviceType b.deviceType a.dtype b.dtype a.shape b.shape strides paddings dilations outputPaddings
        let mutable b = b
        if dilations[0] > 1 || dilations[1] > 1 || dilations[2] > 1 then
            b <- b.dilate([|1; 1; dilations[0]; dilations[1]; dilations[2]|])
        let fderivative = a
        let a = a.zerosLike(outputShape)
        // Use convolution reverse mode to implement transposed convolution
        let (aderivative:Tensor), _ = Tensor.conv3dReverseDiff(a, b, fderivative, aConst=false, bConst=true, strides=strides, paddings=paddings)
        aderivative

    /// <summary>Propagate the reverse-mode derivative backwards in the computation graph, starting from this tensor.</summary>
    /// <param name="value">The derivative value to propagate backwards. Should have the same shape with this tensor.</param>
    /// <param name="zeroDerivatives">Indicates whether any existing derivatives in the computation graph (for example from a previous reverse propagation that was executed) should be zeroed or not before starting this propagation. Default: true</param>
    member t.reverse(?value:Tensor, ?zeroDerivatives:bool) =
        let value = defaultArg value (t.onesLike())
        let zeroDerivatives = defaultArg zeroDerivatives true
        if value.shape <> t.shape then 
            printfn "%A" t
            printfn "%A" value
            failwithf "Reverse propagation: Expecting t.shape (%A) and value.shape (%A) to be the same" t.shape value.shape
        t.reverseReset(zeroDerivatives)
        t.reversePush(value)

    /// <summary>See <c>reverse</c></summary>
    member inline t.backward(value) = t.reverse(value)

    /// <summary>Reset the reverse mode computation graph associated with the given output tensor.</summary>
    // TODO: other designs without this reverseReset function are possible, but they introduce more complications in the handling of fanout counters and multiple reverse passes of the same graph
    member t.reverseReset(zeroDerivatives:bool) =
        let rec reset (ts: Tensor list) =
            match ts with
            | [] -> ()
            | t :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    if zeroDerivatives then t.derivative <- t.zeroLike()
                    elif t.derivative.shape = [|0|] then t.derivative <- t.zeroLike()
                    t.fanout <- t.fanout + 1u
                    if t.fanout = 1u then
                        match o with
                        | AddTT(a,b) -> reset (a::b::tt)
                        | AddTTConst(a) -> reset (a::tt)
                        | AddTT0(a,b) -> reset (a::b::tt)
                        | AddTT0Const(a) -> reset (a::tt)
                        | SubTT(a,b) -> reset (a::b::tt)
                        | SubTTConst(a) -> reset (a::tt)
                        | SubTConstT(b) -> reset (b::tt)
                        | SubTT0(a,b) -> reset (a::b::tt)
                        | SubTT0Const(a) -> reset (a::tt)
                        | SubT0ConstT(b) -> reset (b::tt)
                        | MulTT(a,b) -> reset (a::b::tt)
                        | MulTTConst(a,_) -> reset (a::tt)
                        | MulTT0(a,b) -> reset (a::b::tt)
                        | MulTConstT0(_,b) -> reset (b::tt)
                        | MulTT0Const(a,_) -> reset (a::tt)
                        | DivTT(a,b) -> reset (a::b::tt)
                        | DivTTConst(a,_) -> reset (a::tt)
                        | DivTConstT(_,b) -> reset (b::tt)
                        | DivT0T(a,b) -> reset (a::b::tt)
                        | DivT0ConstT(_,b) -> reset (b::tt)
                        | DivTT0(a,b) -> reset (a::b::tt)
                        | DivTT0Const(a,_) -> reset (a::tt)
                        | PowTT(a,b) -> reset (a::b::tt)
                        | PowTTConst(a,_) -> reset (a::tt)
                        | PowTConstT(_,b) -> reset (b::tt)
                        | PowT0ConstT(_,b) -> reset (b::tt)
                        | PowTT0Const(a,_) -> reset (a::tt)
                        | MatMulTT(a,b) -> reset (a::b::tt)
                        | MatMulTTConst(a,_) -> reset (a::tt)
                        | MatMulTConstT(_,b) -> reset (b::tt)
                        | MaxPool1DT(a,_,_) -> reset (a::tt)
                        | MaxPool2DT(a,_,_) -> reset (a::tt)
                        | MaxPool3DT(a,_,_) -> reset (a::tt)
                        | MaxUnpool1DT(a,_) -> reset (a::tt)
                        | MaxUnpool2DT(a,_) -> reset (a::tt)
                        | MaxUnpool3DT(a,_) -> reset (a::tt)
                        | Conv1DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv1DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv1DTConstT(_,b,_,_) -> reset (b::tt)
                        | Conv2DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv2DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv2DTConstT(_,b,_,_) -> reset (b::tt)
                        | Conv3DTT(a,b,_,_) -> reset (a::b::tt)
                        | Conv3DTTConst(a,_,_,_) -> reset (a::tt)
                        | Conv3DTConstT(_,b,_,_) -> reset (b::tt)
                        | NegT(a) -> reset (a::tt)
                        | SumT(a) -> reset (a::tt)
                        | SumTDim(a,_) -> reset (a::tt)
                        | ExpandT(a) -> reset (a::tt)
                        | StackTs(a,_) -> reset (List.append (a |> List.ofSeq) tt)
                        | UnstackT(a,_,_) -> reset (a::tt)
                        | CatTs(a,_) -> reset (List.append (a |> List.ofSeq) tt)
                        | SplitT(a,_,_,_) -> reset (a::tt)
                        | GatherT(a,_,_) -> reset (a::tt)
                        | ScatterT(a,_,_) -> reset (a::tt)
                        | PermuteT(a,_) -> reset (a::tt)
                        | TransposeT(a,_,_) -> reset (a::tt)
                        | TransposeT2(a) -> reset (a::tt)
                        | SqueezeT(a) -> reset (a::tt)
                        | UnsqueezeT(a) -> reset (a::tt)
                        | FlipT(a,_) -> reset (a::tt)
                        | DilateT(a,_) -> reset (a::tt)
                        | UndilateT(a,_) -> reset (a::tt)
                        | ViewT(a,_) -> reset (a::tt)
                        | ClampT(a,_) -> reset (a::tt)
                        | SliceT(a,_) -> reset (a::tt)
                        | AddTTSlice(a,_,b) -> reset (a::b::tt)
                        | AddTTConstSlice(a) -> reset (a::tt)
                        | AddTConstTSlice(_, b) -> reset (b::tt)
                        | SignT(a) -> reset (a::tt)
                        | FloorT(a) -> reset (a::tt)
                        | CeilT(a) -> reset (a::tt)
                        | RoundT(a) -> reset (a::tt)
                        | AbsT(a) -> reset (a::tt)
                        | ReluT(a) -> reset (a::tt)
                        | SoftplusT(a) -> reset (a::tt)
                        | SigmoidT(a) -> reset (a::tt)
                        | ExpT(a) -> reset (a::tt)
                        | LogT(a) -> reset (a::tt)
                        | Log10T(a) -> reset (a::tt)
                        | SqrtT(a) -> reset (a::tt)
                        | SinT(a) -> reset (a::tt)
                        | CosT(a) -> reset (a::tt)
                        | TanT(a) -> reset (a::tt)
                        | SinhT(a) -> reset (a::tt)
                        | CoshT(a) -> reset (a::tt)
                        | TanhT(a) -> reset (a::tt)
                        | AsinT(a) -> reset (a::tt)
                        | AcosT(a) -> reset (a::tt)
                        | AtanT(a) -> reset (a::tt)
                        | NewT -> reset tt
                        | OpUnaryT(a,_,_) -> reset (a::tt)
                        | OpBinaryTT(a,b,_,_) -> reset (a::b::tt)
                        | OpBinaryTC(a,_,_,_) -> reset (a::tt)
                        | OpBinaryCT(_,b,_,_) -> reset (b::tt)
                    else reset tt
                | _ -> reset tt
        reset [t]

    /// <summary>Push the given value as part of the reverse-mode computation at the given output tensor.</summary>
    /// <param name="value">The value to apply.</param>
    member t.reversePush(value:Tensor) =
        let check (v:Tensor,t:Tensor) = 
            // Check that either:
            // 1. shape of backpropagated adjoint matches shape of primal of node to which it is being propagated
            // 2. the backpropagated adjoint is zero, indicating that the derivative accumulation was already performed by the code that called check (this behavior is for efficiency reasons, eliminating a zerosLike call for several ops involving sliced tensors)
            // assert (v.shape = t.primal.shape || (v.dim = 0 && float(v) = 0.))
            if v.shape <> t.primal.shape then
                if not (v.dim = 0 && float(v) = 0.) then
                    failwithf "Cannot reverse push value with shape %A to tensor with shape %A" v.shape t.shape
            // The following is good for debugging NaN cases during gradient descent, but probably shouldn't be enabled by default. This is about where we would like the user to discover a NaN case (during differentiation or after differentiation).
            // assert not (v.hasinfnan())
            (v,t)

        let rec push (ts:(Tensor*Tensor) list) =
            match ts with
            | [] -> ()
            | (v, t) :: tt ->
                match t with
                | TensorR(_,_,o,_,_) ->
                    // if t.derivative.hasnan() || t.derivative.hasinf() then failwithf "t.derivative has nan, inf, or -inf\n%A\n%A" t.derivative t.derivative.shape
                    // if v.hasnan() || v.hasinf() then failwithf "v has nan, inf, or -inf\n%A\n%A\n%s" v v.shape (snd (t.ancestors()))
                    t.derivative <- t.derivative + v
                    t.fanout <- t.fanout - 1u
                    if t.fanout = 0u then
                        let td = t.derivative
                        match o with
                        | AddTT(a,b) -> push (check(td, a) :: check(td, b) :: tt)
                        | AddTTConst(a) -> push (check(td, a) :: tt)
                        | AddTT0(a,b) -> push (check(td, a) :: check(td.sum(), b) :: tt)
                        | AddTT0Const(a) -> push (check(td, a) :: tt)
                        | SubTT(a,b) -> push (check(td, a) :: check(-td, b) :: tt)
                        | SubTTConst(a) -> push (check(td, a) :: tt)
                        | SubTConstT(b) -> push (check(-td, b) :: tt)
                        | SubTT0(a,b) -> push (check(td, a) :: check(-td.sum(), b) :: tt)
                        | SubTT0Const(a) -> push (check(td, a) :: tt)
                        | SubT0ConstT(b) -> push (check(-td, b) :: tt)
                        | MulTT(a,b) -> push (check(td * b.primal, a) :: check(td * a.primal, b) :: tt)
                        | MulTTConst(a,b) -> push (check(td * b, a) :: tt)
                        | MulTT0(a,b) -> push (check(td * b.primal, a) :: check((td * a.primal).sum(), b) :: tt)
                        | MulTConstT0(a,b) -> push (check((td * a).sum(), b) :: tt)
                        | MulTT0Const(a,b) -> push (check(td * b, a) :: tt)
                        | DivTT(a,b) -> push (check(td / b.primal, a) :: check((td * (-a.primal / (b.primal * b.primal))), b) :: tt)
                        | DivTTConst(a,b) -> push (check(td / b, a) :: tt)
                        | DivTConstT(a,b) -> push (check((td * (-a / (b.primal * b.primal))), b) :: tt)
                        | DivT0T(a,b) -> push (check((td / b.primal).sum(), a) :: check((td * (-a.primal / (b.primal * b.primal))), b) :: tt)
                        | DivT0ConstT(a,b) -> push (check((td * (a.neg() / (b.primal * b.primal))), b) :: tt)
                        | DivTT0(a,b) -> push (check(td / b.primal, a) :: check((td * (-a.primal / (b.primal * b.primal))).sum(), b) :: tt)
                        | DivTT0Const(a,b) -> push (check(td / b, a) :: tt)
                        | PowTT(a,b) -> push (check(td * (a.primal ** (b.primal - 1.)) * b.primal, a) :: check(td * (a.primal ** b.primal) * log a.primal, b) :: tt)
                        | PowTTConst(a,b) -> push (check(td * (a.primal ** (b - 1.)) * b, a) :: tt)
                        | PowTConstT(a,b) -> push (check(td * (a ** b.primal) * log a, b) :: tt)
                        | PowT0ConstT(a,b) -> push (check(td * (Tensor.Pow(a, b.primal)) * a.log(), b) :: tt)
                        | PowTT0Const(a,b) -> push (check(td * (a.primal ** (b.sub(1.))) * b, a) :: tt)
                        | MatMulTT(a,b) -> push (check(td.matmul(b.primal.transpose()), a) :: check(a.primal.transpose(0,1).matmul(td), b) :: tt)
                        | MatMulTTConst(a,b) -> push (check(td.matmul(b.transpose()), a) :: tt)
                        | MatMulTConstT(a,b) -> push (check(a.transpose().matmul(td), b) :: tt)
                        | MaxPool1DT(a, indices, kernelSize) -> push (check(td.maxunpool1d(indices, kernelSize=kernelSize, outputSize=a.shape), a) :: tt)
                        | MaxPool2DT(a, indices, kernelSizes) -> push (check(td.maxunpool2d(indices, kernelSizes=kernelSizes, outputSize=a.shape), a) :: tt)
                        | MaxPool3DT(a, indices, kernelSizes) -> push (check(td.maxunpool3d(indices, kernelSizes=kernelSizes, outputSize=a.shape), a) :: tt)
                        | MaxUnpool1DT(a, indices) -> push (check(td.gather(dim=2, indices=indices), a) :: tt)
                        | MaxUnpool2DT(a, indices) -> push (check(td.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(a), a) :: tt)
                        | MaxUnpool3DT(a, indices) -> push (check(td.flatten(startDim=2).gather(dim=2, indices=indices.flatten(startDim=2)).viewAs(a), a) :: tt)
                        | Conv1DTT(a,b,stride,padding) -> 
                            let aderivative, bderivative = Tensor.conv1dReverseDiff(a, b, td, false, false, stride, padding)
                            push (check(aderivative, a) :: check(bderivative, b) :: tt)
                        | Conv1DTTConst(a,b,stride,padding) ->
                            let aderivative, _ = Tensor.conv1dReverseDiff(a, b, td, false, true, stride, padding)
                            push (check(aderivative, a) :: tt)                        
                        | Conv1DTConstT(a,b,stride,padding) ->
                            let _, bderivative = Tensor.conv1dReverseDiff(a, b, td, true, false, stride, padding)
                            push (check(bderivative, b) :: tt)                        
                        | Conv2DTT(a,b,stride,padding) -> 
                            let aderivative, bderivative = Tensor.conv2dReverseDiff(a, b, td, false, false, stride, padding)
                            push (check(aderivative, a) :: check(bderivative, b) :: tt)
                        | Conv2DTTConst(a,b,stride,padding) ->
                            let aderivative, _ = Tensor.conv2dReverseDiff(a, b, td, false, true, stride, padding)
                            push (check(aderivative, a) :: tt)
                        | Conv2DTConstT(a,b,stride,padding) ->
                            let _, bderivative = Tensor.conv2dReverseDiff(a, b, td, true, false, stride, padding)
                            push (check(bderivative, b) :: tt)
                        | Conv3DTT(a,b,stride,padding) -> 
                            let aderivative, bderivative = Tensor.conv3dReverseDiff(a, b, td, false, false, stride, padding)
                            push (check(aderivative, a) :: check(bderivative, b) :: tt)
                        | Conv3DTTConst(a,b,stride,padding) ->
                            let aderivative, _ = Tensor.conv3dReverseDiff(a, b, td, false, true, stride, padding)
                            push (check(aderivative, a) :: tt)
                        | Conv3DTConstT(a,b,stride,padding) ->
                            let _, bderivative = Tensor.conv3dReverseDiff(a, b, td, true, false, stride, padding)
                            push (check(bderivative, b) :: tt)
                        | NegT(a) -> push (check(-td, a) :: tt)
                        | SumT(a) -> push (check(td.expand(a.shape), a) :: tt)
                        | SumTDim(a, dim) -> 
                            let s = Array.copy a.shape
                            s[dim] <- 1
                            push (check(td.view(s).expand(a.shape), a) :: tt)
                        | ExpandT(a) -> push (check(td.sumToSize(a.shape), a) :: tt)
                        | StackTs(a,dim) ->
                            push (List.append (Array.zip (td.unstack(dim)) a |> Array.map check |> Array.toList) tt)
                        | UnstackT(a,dim,i) -> 
                            if a.derivative.dim = 0 then a.derivative <- a.derivative.expandAs(a)
                            a.derivative <- a.derivative.addSlice(Array.init a.dim (fun j -> if j=dim then i else 0), td.unsqueeze(dim))
                            push (check(a.zeroLike(), a) :: tt)
                        | CatTs(a, dim) ->
                            let sizes = a |> Array.map (fun x -> x.shape[dim])
                            push (List.append (Array.zip (td.split(sizes, dim=dim)) a |> Array.map check |> Array.toList) tt)
                        | SplitT(a,sizes,dim,i) -> 
                            if a.derivative.dim = 0 then a.derivative <- a.derivative.expandAs(a)
                            let locs = (0,sizes) ||> Array.scan (+)
                            a.derivative <- a.derivative.addSlice(Array.init a.dim (fun j -> if j=dim then locs[i] else 0), td)
                            push (check(a.zeroLike(), a) :: tt)
                        | GatherT(a,dim,indices) -> push (check(td.scatter(dim, indices, a.shape), a) :: tt)
                        | ScatterT(a,dim,indices) -> push (check(td.gather(dim, indices), a) :: tt)
                        | PermuteT(a, inversePermutation) -> push (check(td.permute(inversePermutation), a) :: tt)
                        | TransposeT(a, dim0, dim1) -> push (check(td.transpose(dim0, dim1), a) :: tt)
                        | TransposeT2(a) -> push (check(td.transpose(), a) :: tt)
                        | SqueezeT(a) -> push (check(td.viewAs(a), a) :: tt)
                        | UnsqueezeT(a) -> push (check(td.viewAs(a), a) :: tt)
                        | FlipT(a, dims) -> push (check(td.flip(dims), a) :: tt)
                        | DilateT(a, dilations) -> push (check(td.undilate(dilations), a) :: tt)
                        | UndilateT(a, dilations) -> push (check(td.dilate(dilations), a) :: tt)
                        | ViewT(a,aShape) -> push (check((td.view(aShape)), a) :: tt)
                        | ClampT(a, mask) -> push (check(td * mask, a) :: tt)
                        | SliceT(a,bounds) -> 
                            // TODO: a.zerosLike() below is to handle non-scalar TensorRs with a scalar derivative Tensor(0.) (representing the initialization before accumulation). This is correct but can be changed to eliminate the extra op.
                            if a.derivative.dim = 0 then a.derivative <- a.derivative.expandAs(a)
                            a.derivative <- a.derivative.addSlice(boundsToLocation bounds, td.view(boundsToShape bounds))
                            push (check(a.zeroLike(), a) :: tt)
                        | AddTTSlice(a,location,b) -> 
                            push (check(td, a) :: check(td.GetSlice(Shape.locationToBounds b.shape location), b):: tt)
                        | AddTTConstSlice(a) -> push (check(td, a) :: tt)
                        | AddTConstTSlice(location, b) -> push (check(td.GetSlice(Shape.locationToBounds b.shape location), b):: tt)
                        | SignT(a) -> push (check(a.zerosLike(), a) :: tt)
                        | FloorT(a) -> push (check(a.zerosLike(), a) :: tt)
                        | CeilT(a) -> push (check(a.zerosLike(), a) :: tt)
                        | RoundT(a) -> push (check(a.zerosLike(), a) :: tt)
                        | AbsT(a) -> push (check(td * a.primal.sign(), a) :: tt)
                        | ReluT(a) -> let sap = a.primal.sign() in push (check(td * (sap.abs()) * (sap + 1.) / 2., a) :: tt)
                        | SoftplusT(a) -> push (check(td / (1. + a.primal.neg().exp()), a) :: tt)
                        | SigmoidT(a) -> push (check(td * t.primal * (1. - t.primal), a) :: tt)
                        | ExpT(a) -> push (check(td * t.primal, a) :: tt)
                        | LogT(a) -> push (check(td / a.primal, a) :: tt)
                        | Log10T(a) -> push (check(td / (a.primal * log10Val), a) :: tt)
                        | SqrtT(a) -> push (check(td / (2. * t.primal), a) :: tt)
                        | SinT(a) -> push (check(td * (a.primal.cos()), a) :: tt)
                        | CosT(a) -> push (check(-td * (a.primal.sin()), a) :: tt)
                        | TanT(a) -> let cosap = a.primal.cos() in push (check(td / (cosap * cosap), a) :: tt)
                        | SinhT(a) -> push (check(td * (a.primal.cosh()), a) :: tt)
                        | CoshT(a) -> push (check(td * (a.primal.sinh()), a) :: tt)
                        | TanhT(a) -> let coshap = a.primal.cosh() in push (check(td / (coshap * coshap), a) :: tt)
                        | AsinT(a) -> push (check(td / Tensor.Sqrt(1. - a.primal*a.primal), a) :: tt)
                        | AcosT(a) -> push (check(-td / Tensor.Sqrt(1. - a.primal*a.primal), a) :: tt)
                        | AtanT(a) -> push (check(td / (1. + a.primal*a.primal), a) :: tt)
                        | NewT -> push tt
                        | OpUnaryT(a, rev, _) -> push (check(rev(a.primal, t.primal, td), a) :: tt)
                        | OpBinaryTT(a, b, rev, _) -> let ad, bd = rev(a.primal, b.primal, t.primal, td) in push (check(ad, a) :: check(bd, b) :: tt)
                        | OpBinaryTC(a, b, rev, _) -> let ad = rev(a.primal, b, t.primal, td) in push (check(ad, a) :: tt)
                        | OpBinaryCT(a, b, rev, _) -> let bd = rev(a, b.primal, t.primal, td) in push (check(bd, b) :: tt)
                    else push tt
                | _ -> push tt
        push [(value, t)]

and TensorOp =
    | AddTT of Tensor * Tensor
    | AddTTConst of Tensor
    | AddTT0 of Tensor * Tensor
    | AddTT0Const of Tensor
    
    | SubTT of Tensor * Tensor
    | SubTTConst of Tensor
    | SubTConstT of Tensor
    | SubTT0 of Tensor * Tensor
    | SubTT0Const of Tensor
    | SubT0ConstT of Tensor

    | MulTT of Tensor * Tensor
    | MulTTConst of Tensor * Tensor
    | MulTT0 of Tensor * Tensor
    | MulTT0Const of Tensor * scalar
    | MulTConstT0 of Tensor * Tensor

    | DivTT of Tensor * Tensor
    | DivTTConst of Tensor * Tensor
    | DivTConstT of Tensor * Tensor
    | DivT0T of Tensor * Tensor
    | DivT0ConstT of scalar * Tensor
    | DivTT0 of Tensor * Tensor
    | DivTT0Const of Tensor * scalar

    | PowTT of Tensor * Tensor
    | PowTTConst of Tensor * Tensor
    | PowTConstT of Tensor * Tensor
    | PowT0ConstT of scalar * Tensor
    | PowTT0Const of Tensor * scalar

    | MatMulTT of Tensor * Tensor
    | MatMulTTConst of Tensor * Tensor
    | MatMulTConstT of Tensor * Tensor

    | MaxPool1DT of Tensor * Tensor * int
    | MaxUnpool1DT of Tensor * Tensor

    | MaxPool2DT of Tensor * Tensor * int[]
    | MaxUnpool2DT of Tensor * Tensor

    | MaxPool3DT of Tensor * Tensor * int[]
    | MaxUnpool3DT of Tensor * Tensor

    | Conv1DTT of Tensor * Tensor * int * int
    | Conv1DTTConst of Tensor * Tensor * int * int
    | Conv1DTConstT of Tensor * Tensor * int * int

    | Conv2DTT of Tensor * Tensor * int[] * int[]
    | Conv2DTTConst of Tensor * Tensor * int[] * int[]
    | Conv2DTConstT of Tensor * Tensor * int[] * int[]

    | Conv3DTT of Tensor * Tensor * int[] * int[]
    | Conv3DTTConst of Tensor * Tensor * int[] * int[]
    | Conv3DTConstT of Tensor * Tensor * int[] * int[]

    | AddTTSlice of Tensor * int[] * Tensor
    | AddTTConstSlice of Tensor
    | AddTConstTSlice of int[] * Tensor

    | NegT of Tensor
    | SumT of Tensor
    | SumTDim of Tensor * int
    | ExpandT of Tensor
    | StackTs of Tensor[] * dim:int
    | UnstackT of Tensor * dim:int * i:int
    | CatTs of Tensor[] * dim:int
    | SplitT of Tensor * int[] * dim:int * i:int
    | SliceT of Tensor * int[,]
    | GatherT of Tensor * int * Tensor
    | ScatterT of Tensor * int * Tensor
    | PermuteT of Tensor * inversePermutation: int[]
    | TransposeT of Tensor * int * int
    | TransposeT2 of Tensor
    | SqueezeT of Tensor
    | UnsqueezeT of Tensor
    | FlipT of Tensor * int[]
    | DilateT of Tensor * int[]
    | UndilateT of Tensor * int[]
    | ViewT of Tensor * int[]
    | ClampT of Tensor * Tensor
    | SignT of Tensor
    | FloorT of Tensor
    | CeilT of Tensor
    | RoundT of Tensor
    | AbsT of Tensor
    | ReluT of Tensor
    | SoftplusT of Tensor
    | SigmoidT of Tensor
    | ExpT of Tensor
    | LogT of Tensor
    | Log10T of Tensor
    | SqrtT of Tensor
    | SinT of Tensor
    | CosT of Tensor
    | TanT of Tensor
    | SinhT of Tensor
    | CoshT of Tensor
    | TanhT of Tensor
    | AsinT of Tensor
    | AcosT of Tensor
    | AtanT of Tensor
    | NewT
    | OpUnaryT of Tensor*(Tensor*Tensor*Tensor->Tensor)*string
    | OpBinaryTT of Tensor*Tensor*(Tensor*Tensor*Tensor*Tensor->Tensor*Tensor)*string
    | OpBinaryTC of Tensor*Tensor*(Tensor*Tensor*Tensor*Tensor->Tensor)*string
    | OpBinaryCT of Tensor*Tensor*(Tensor*Tensor*Tensor*Tensor->Tensor)*string

    override op.ToString() =
        match op with
        | OpUnaryT(_,_,s) -> s
        | OpBinaryTT(_,_,_,s) -> s
        | OpBinaryTC(_,_,_,s) -> s
        | OpBinaryCT(_,_,_,s) -> s
        | NewT -> "NewT" // Needed because op.GetType().Name does not give "NewT" for this case and gives "TensorOp"
        | _ -> op.GetType().Name

/// <summary>Defines a new op implementing a unary function and its derivatives. Instances of this class are used with the <see cref="M:DiffSharp.Tensor.Op(DiffSharp.UnaryOp)"/> method to define a new differentiable tensor function that supports forward, reverse, and nested differentiation.</summary>
/// <remarks>
/// <para>This type represents the most generic definition of a new op representing a unary function, allowing the specification of: (1) the <see cref="T:DiffSharp.Backends.RawTensor"/> operation, (2) the derivative propagation rule for the forward differentiation mode and (3) the derivative propagation rule for the reverse differentiation mode.</para>
/// <para>In general, if you are implementing a simple elementwise op, you should prefer using the <see cref="T:DiffSharp.UnaryOpElementwise"/> type, which is much simpler to use.</para>
/// </remarks>
/// <example>
/// <code>
/// { new UnaryOp("transpose") with
///     member _.fRaw(a) = a.TransposeT2()
///     member _.ad_dfda(a,ad,f) = ad.transpose()
///     member _.fd_dfda(a,f,fd) = fd.transpose()
/// }
/// </code>
/// </example>
[<AbstractClass>]
type UnaryOp(name:string) =

    /// Name of the op.
    member _.name = name

    /// <summary>RawTensor operation \( f(a) \) performing the op.</summary>
    /// <param name="a">The argument \( a \).</param>
    /// <returns>The function's value \( f(a) \).</returns>
    abstract fRaw: a:RawTensor->RawTensor

    /// <summary>Derivative propagation rule for forward differentiation mode. This represents the derivative of \( f(a) \) with respect a value \( x \) earlier in the computation graph than the function's argument \( a \). In other words, it computes \( \frac{\partial f(a)}{\partial x} = \frac{\partial a}{\partial x} \frac{\partial f(a)}{\partial a} \).</summary>
    /// <param name="a">The argument \( a \).</param>
    /// <param name="ad">The argument's derivative \( \frac{\partial a}{\partial x} \).</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <returns>The tensor corresponding to \( \frac{\partial f(a)}{\partial x} = \frac{\partial a}{\partial x} \frac{\partial f(a)}{\partial a} \).</returns>
    abstract ad_dfda: a:Tensor*ad:Tensor*f:Tensor->Tensor

    /// <summary>Derivative propagation rule for reverse differentiation mode. This represents the derivative of a value \( y \), which comes later in the computation graph than the function's value \( f(a) \), with respect to the function's argument \( a \). In other words, it computes \( \frac{\partial y}{\partial a} = \frac{\partial y}{\partial f(a)} \frac{\partial f(a)}{\partial a} \).</summary>
    /// <param name="a">The argument \( a \).</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <param name="fd">The derivative with respect to the function's output \( \frac{\partial y}{\partial f(a)} \).</param>
    /// <returns>The tensor corresponding to \( \frac{\partial y}{\partial a} = \frac{\partial y}{\partial f(a)} \frac{\partial f(a)}{\partial a} \).</returns>
    abstract fd_dfda: a:Tensor*f:Tensor*fd:Tensor->Tensor


/// <summary>Defines a new op implementing an elementwise unary function and its derivatives. Instances of this class are used with the <see cref="M:DiffSharp.Tensor.Op(DiffSharp.UnaryOp)"/> method to define a new differentiable tensor function that supports forward, reverse, and nested differentiation.</summary>
/// <remarks>
/// <para>This type is specialized to elementwise ops. It requires the user to specify only (1) the <see cref="T:DiffSharp.Backends.RawTensor"/> operation and (2) the derivative of the function with respect to its argument. The corresponding derivative propagation rules for the forward and reverse differentiation modes are automatically generated.</para>
/// <para>If you are implementing a complex op that is not elementwise, you can use the generic type <see cref="T:DiffSharp.UnaryOp"/>, which allows you to define the full derivative propagation rules.</para>
/// </remarks>
/// <example>
/// <code>
/// { new UnaryOpElementwise("cos") with
///     member _.fRaw(a) = a.CosT()
///     member _.dfda(a,f) = -a.sin()
/// }
///
/// { new UnaryOpElementwise("exp") with
///     member _.fRaw(a) = a.ExpT()
///     member _.dfda(a,f) = f
/// }
///
/// { new UnaryOpElementwise("log") with
///     member _.fRaw(a) = a.LogT()
///     member _.dfda(a,f) = 1/a
/// }
/// </code>
/// </example>
[<AbstractClass>]
type UnaryOpElementwise(name) =
    inherit UnaryOp(name)
    /// <summary>Derivative of the function with respect to its argument, \( \frac{\partial f(a)}{\partial a} \).</summary>
    /// <param name="a">The argument \( a \)</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <returns>The tensor corresponding to \( \frac{\partial f(a)}{\partial a} \).</returns>
    abstract dfda: a:Tensor*f:Tensor->Tensor

    override op.ad_dfda(a,ad,f) = ad*op.dfda(a,f)
    override op.fd_dfda(a,f,fd) = fd*op.dfda(a,f)


/// <summary>Defines a new op implementing a binary function and its derivatives. Instances of this class are used with the <see cref="M:DiffSharp.Tensor.Op(DiffSharp.BinaryOp)"/> method to define a new differentiable tensor function that supports forward, reverse, and nested differentiation.</summary>
/// <remarks>
/// <para>This type represents the most generic definition of a new op representing a binary function, allowing the specification of: (1) the <see cref="T:DiffSharp.Backends.RawTensor"/> operation, (2) the derivative propagation rule for the forward differentiation mode and (3) the derivative propagation rule for the reverse differentiation mode.</para>
/// <para>In general, if you are implementing a simple elementwise op, you should prefer using the <see cref="T:DiffSharp.BinaryOpElementwise"/> type, which is much simpler to use.</para>
/// </remarks>
/// <example>
/// <code>
/// { new BinaryOp("matmul") with
///     member _.fRaw(a,b) = a.MatMulTT(b)
///     member _.ad_dfda(a,ad,b,f) = ad.matmul(b)
///     member _.bd_dfdb(a,b,bd,f) = a.matmul(bd)
///     member _.fd_dfda(a,b,f,fd) = fd.matmul(b.transpose())
///     member _.fd_dfdb(a,b,f,fd) = a.transposeExt().matmul(fd)
/// }
/// </code>
/// </example>
[<AbstractClass>]
type BinaryOp(name:string) =
    /// Name of the op.
    member _.name = name
    /// <summary>RawTensor operation \( f(a, b) \) performing the op.</summary>
    /// <param name="a">The first argument \( a \).</param>
    /// <param name="b">The second argument \( b \).</param>
    /// <returns>The function's value \( f(a, b) \).</returns>
    abstract fRaw: a:RawTensor*b:RawTensor->RawTensor

    /// <summary>Derivative propagation rule for forward differentiation mode for the partial derivative with respect to the first argument of the function. This represents the contribution of the function's first argument \( a \) to the derivative of \( f(a, b) \) with respect a value \( x \) earlier in the computation graph than the function's arguments. In other words, it computes the first term in the right-hand side of the equation \( \frac{\partial f(a, b)}{\partial x} = \frac{\partial a}{\partial x} \frac{\partial f(a, b)}{\partial a} + \frac{\partial b}{\partial x} \frac{\partial f(a, b)}{\partial b} \).</summary>
    /// <param name="a">The first argument \( a \).</param>
    /// <param name="ad">The first argument's derivative \( \frac{\partial a}{\partial x} \).</param>
    /// <param name="b">The second argument \( b \).</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a, b) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <returns>The tensor corresponding to \( \frac{\partial a}{\partial x} \frac{\partial f(a, b)}{\partial a} \).</returns>
    abstract ad_dfda: a:Tensor*ad:Tensor*b:Tensor*f:Tensor->Tensor

    /// <summary>Derivative propagation rule for forward differentiation mode for the partial derivative with respect to the second argument of the function. This represents the contribution of the function's second argument \( b \) to the derivative of \( f(a, b) \) with respect a value \( x \) earlier in the computation graph than the function's arguments. In other words, it computes the second term in the right-hand side of the equation \( \frac{\partial f(a, b)}{\partial x} = \frac{\partial a}{\partial x} \frac{\partial f(a, b)}{\partial a} + \frac{\partial b}{\partial x} \frac{\partial f(a, b)}{\partial b} \).</summary>
    /// <param name="a">The first argument \( a \).</param>
    /// <param name="b">The second argument \( b \).</param>
    /// <param name="bd">The second argument's derivative \( \frac{\partial b}{\partial x} \).</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a, b) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <returns>The tensor corresponding to \( \frac{\partial b}{\partial x} \frac{\partial f(a, b)}{\partial b} \).</returns>
    abstract bd_dfdb: a:Tensor*b:Tensor*bd:Tensor*f:Tensor->Tensor

    /// <summary>Derivative propagation rule for reverse differentiation mode for the partial derivative with respect to the first argument of the function. This represents the derivative of a value \( y \), which comes later in the computation graph than the function's value \( f(a, b) \), with respect to the function's first argument \( a \). In other words, it computes \( \frac{\partial y}{\partial a} = \frac{\partial y}{\partial f(a, b)} \frac{\partial f(a, b)}{\partial a} \).</summary>
    /// <param name="a">The first argument \( a \).</param>
    /// <param name="b">The second argument \( b \).</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a, b) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <param name="fd">The derivative with respect to the function's output \( \frac{\partial y}{\partial f(a, b)} \).</param>
    /// <returns>The tensor corresponding to \( \frac{\partial y}{\partial a} = \frac{\partial y}{\partial f(a, b)} \frac{\partial f(a, b)}{\partial a} \).</returns>
    abstract fd_dfda: a:Tensor*b:Tensor*f:Tensor*fd:Tensor->Tensor

    /// <summary>Derivative propagation rule for reverse differentiation mode for the partial derivative with respect to the second argument of the function. This represents the derivative of a value \( y \), which comes later in the computation graph than the function's value \( f(a, b) \), with respect to the function's second argument \( b \). In other words, it computes \( \frac{\partial y}{\partial b} = \frac{\partial y}{\partial f(a, b)} \frac{\partial f(a, b)}{\partial b} \).</summary>
    /// <param name="a">The first argument \( a \).</param>
    /// <param name="b">The second argument \( b \).</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a, b) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <param name="fd">The derivative with respect to the function's output \( \frac{\partial y}{\partial f(a, b)} \).</param>
    /// <returns>The tensor corresponding to \( \frac{\partial y}{\partial b} = \frac{\partial y}{\partial f(a, b)} \frac{\partial f(a, b)}{\partial b} \).</returns>
    abstract fd_dfdb: a:Tensor*b:Tensor*f:Tensor*fd:Tensor->Tensor


/// <summary>Defines a new op implementing an elementwise binary function and its derivatives. Instances of this class are used with the <see cref="M:DiffSharp.Tensor.Op(DiffSharp.BinaryOp)"/> method to define a new differentiable tensor function that supports forward, reverse, and nested differentiation.</summary>
/// <remarks>
/// This type is specialized to elementwise ops. It requires the user to specify only (1) the <see cref="T:DiffSharp.Backends.RawTensor"/> operation and (2) the derivative of the function with respect to each argument. The corresponding derivative propagation rules for the forward and reverse differentiation modes are automatically generated.
/// <para>If you are implementing a complex op that is not elementwise, you can use the generic type <see cref="T:DiffSharp.BinaryOp"/>, which allows you to define the full derivative propagation rules.</para>
/// </remarks>
/// <example>
/// <code>
/// { new BinaryOpElementwise("pow") with
///     member _.fRaw(a,b) = a.PowTT(b)
///     member _.dfda(a,b,f) = b * f / a
///     member _.dfdb(a,b,f) = f * a.log()
/// }
/// 
/// { new BinaryOpElementwise("mul") with
///     member _.fRaw(a,b) = a.MulTT(b)
///     member _.dfda(a,b,f) = b
///     member _.dfdb(a,b,f) = a
/// }
/// </code>
/// </example>
[<AbstractClass>]
type BinaryOpElementwise(name) =
    inherit BinaryOp(name)
    /// <summary>Derivative of the function with respect to its first argument, \( \frac{\partial f(a, b)}{\partial a} \).</summary>
    /// <param name="a">The first argument \( a \)</param>
    /// <param name="b">The second argument \( b \)</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a, b) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <returns>The tensor corresponding to \( \frac{\partial f(a, b)}{\partial a} \).</returns>
    abstract dfda: a:Tensor*b:Tensor*f:Tensor->Tensor

    /// <summary>Derivative of the function with respect to its second argument, \( \frac{\partial f(a, b)}{\partial b} \).</summary>
    /// <param name="a">The first argument \( a \)</param>
    /// <param name="b">The second argument \( b \)</param>
    /// <param name="f">The function's pre-computed primal evaluation result \( f(a, b) \), which can be one of the terms involved in the derivative computation (e.g., the derivative of the exponential function) and be used without the need to recompute it.</param>
    /// <returns>The tensor corresponding to \( \frac{\partial f(a, b)}{\partial b} \).</returns>
    abstract dfdb: a:Tensor*b:Tensor*f:Tensor->Tensor

    override op.ad_dfda(a,ad,b,f) = ad*op.dfda(a,b,f)
    override op.bd_dfdb(a,b,bd,f) = bd*op.dfdb(a,b,f)
    override op.fd_dfda(a,b,f,fd) = fd*op.dfda(a,b,f)
    override op.fd_dfdb(a,b,f,fd) = fd*op.dfdb(a,b,f)


type Tensor with
    /// <summary>Allows the definition of a new unary tensor op.</summary>
    /// <param name="ext">The definition of the new op.</param>
    /// <returns>The new op.</returns>
    static member Op(ext: UnaryOp) =
        fun a ->
            let fRaw = ext.fRaw
            let fTensor = Tensor.Op ext
            let dfFwd(ap,ad,fp) = ext.ad_dfda(ap,ad,fp) // ad*ext.dfda(ap,fp)
            let dfRev(a) = OpUnaryT(a, (fun (ap,fp,fd) -> ext.fd_dfda(ap,fp,fd)), ext.name) // fd*ext.dfda(ap,fp)
            Tensor.OpUnary(a, fRaw, fTensor, dfFwd, dfRev)
    
    /// <summary>Allows the definition of a new binary tensor op.</summary>
    /// <param name="ext">The definition of the new op.</param>
    /// <returns>The new op.</returns>
    static member Op(ext: BinaryOp) =
        fun (a, b) ->
            let fRaw = ext.fRaw
            let fTensor = Tensor.Op ext
            let dfFwdTT(ap,ad,bp,bd,fp) = ext.ad_dfda(ap,ad,bp,fp) + ext.bd_dfdb(ap,bp,bd,fp)
            let dfFwdTC(ap,ad,fp) = ext.ad_dfda(ap,ad,b,fp)
            let dfFwdCT(bp,bd,fp) = ext.bd_dfdb(a,bp,bd,fp)
            let dfRevTT(a,b) = OpBinaryTT(a, b, (fun (ap,bp,fp,fd) -> (ext.fd_dfda(ap,bp,fp,fd)), (ext.fd_dfdb(ap,bp,fp,fd))), ext.name+"TT")
            let dfRevTC(a,b) = OpBinaryTC(a, b, (fun (ap,b,fp,fd) -> (ext.fd_dfda(ap,b,fp,fd))), ext.name+"TC")
            let dfRevCT(a,b) = OpBinaryCT(a, b, (fun (a,bp,fp,fd) -> (ext.fd_dfdb(a,bp,fp,fd))), ext.name+"CT")
            Tensor.OpBinary(a, b, fRaw, fTensor, dfFwdTT, dfFwdTC, dfFwdCT, dfRevTT, dfRevTC, dfRevCT)
