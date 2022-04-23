// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace rec DiffSharp.Backends

open System
open DiffSharp
open DiffSharp.Util

/// <summary>
///   Represents the static functionality for tensors implemented by a DiffSharp backend.
/// </summary>
///
/// <namespacedoc>
///   <summary>Contains types and functionality related to backend implementations for DiffSharp.</summary>
/// </namespacedoc>
[<AbstractClass>]
type BackendTensorStatics() = 
    // cache for most recently accessed backend
    static let hook = BackendFunctionality<BackendTensorStatics>()

    /// Sets the seed for the default random number generator of the backend
    abstract Seed: seed:int -> unit

    /// Gets the scalar 0 tensor for the given device
    abstract Zero: dtype: Dtype * device: Device -> RawTensor

    /// Gets a tensor filled with arbitrary values for the given shape and device
    abstract Empty: shape:Shape * dtype: Dtype * device: Device -> RawTensor

    /// Gets a tensor filled with zeros for the given shape and device
    abstract Zeros: shape:Shape * dtype: Dtype * device: Device -> RawTensor

    /// Gets the scalar 1 tensor for the given device
    abstract One: dtype: Dtype * device: Device -> RawTensor

    /// Gets a tensor filled with ones for the given shape and device
    abstract Ones: shape:Shape * dtype: Dtype * device: Device -> RawTensor

    /// Gets a tensor filled with the given value for the given shape and device
    abstract Full: shape:Shape * value: scalar * dtype: Dtype * device: Device -> RawTensor

    /// Gets a tensor filled with random values for the given shape and device
    abstract Random: shape:Shape * dtype: Dtype * device: Device -> RawTensor

    /// Gets a tensor filled with random values from the normal distribution for the given shape and device
    abstract RandomNormal: shape:Shape * dtype: Dtype * device: Device -> RawTensor

    /// Gets a tensor filled with random integers from the given range for the given shape and device
    abstract RandomInt: shape:Shape * low:int * high:int * dtype: Dtype * device: Device -> RawTensor

    /// Gets the devices supported by this backend
    abstract GetDevices: ?deviceType: DeviceType -> Device list

    /// Indicates if a device type is supported by this backend
    abstract IsDeviceTypeAvailable: deviceType: DeviceType -> bool
    
    /// Seed all backends with the given random seed, or a new seed based on the current time
    /// if no seed is specified.
    static member Seed(?seed:int) =
        let seed = defaultArg seed (int DateTime.Now.Ticks)
        Random.Seed(seed) // Do not remove. util.Random seed would be set by the Reference backend if it's currently loaded. However we still need to keep this here to ensure util.Random seed is set (it may be used in code other than the Reference backend).
        for KeyValue(_, backend) in hook.Backends do
            backend.Seed(seed)

    /// Create a tensor of appropriate dtype from a scalar or array of appropriate values.
    /// A backend type is delivered consistent with in-memory data - a type for dtype Int32 gets int32 data etc.
    abstract CreateFromFlatArray: data: System.Array * shape: Shape * dtype: Dtype * device: Device -> RawTensor

    /// Get the backend implementation for the given tensor element type and backend.
    static member Get(?backend: Backend) =
        hook.Get(?backend=backend)

/// <summary>
///   Represents a raw (i.e. non-differentiable immutable) tensor implemented by a DiffSharp backend.
/// </summary>
///
/// <remarks>
///  Each backend will provide one of more .NET implementations of this type, which may in turn
///  wrap handles to native implementations.
/// </remarks>
[<AbstractClass>]
type RawTensor() =

    /// Gets the shape of the tensor
    abstract Shape: Shape

    /// Gets the dimensionality of the tensor
    abstract Dim: int

    /// Gets the number of elements in the tensor
    // TODO: int32 might not be enough for very large tensors
    abstract Nelement: int

    /// Gets the element storage type for the tensor
    abstract Dtype: Dtype

    /// Gets the device for the tensor
    abstract Device: Device

    /// Gets the device type for the tensor
    abstract DeviceType: DeviceType

    /// Gets the backend for the tensor
    abstract Backend: Backend

    /// Gets a handle to the underlying representation of the the tensor. For example, if the Torch
    /// backend is used this will be the corresponding TorchSharp TorchTensor.
    abstract Handle: obj

    override t.ToString() = t.Print()
    
    /// Gets a tensor containing arbitrary values for the given shape and configuration
    static member Empty(shape:Shape, ?dtype, ?device, ?backend) = 
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.Empty(shape, dtype, device)

    /// Gets the scalar zero tensor for the given configuration
    static member Zero(?dtype, ?device, ?backend) = 
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.Zero(dtype, device)

    /// Gets the zero tensor for the given shape and configuration
    static member Zeros(shape:Shape, ?dtype, ?device, ?backend) = 
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.Zeros(shape, dtype, device)

    /// Gets the scalar 1 tensor for the given configuration
    static member One(?dtype, ?device, ?backend) = 
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.One(dtype, device)

    /// Gets a tensor filled with 1 values for the given shape and configuration
    static member Ones(shape:Shape, ?dtype, ?device, ?backend) =
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.Ones(shape, dtype, device)

    /// Gets a tensor filled with the given value for the given shape and configuration
    static member Full(shape:Shape, value, ?dtype, ?device, ?backend) =
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.Full(shape, value, dtype, device)

    /// Gets a tensor filled with random values for the given shape and configuration
    static member Random(shape:Shape, ?dtype, ?device, ?backend) =
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.Random(shape, dtype, device)

    /// Gets a tensor filled with random values from the normal distribution for the given shape and configuration
    static member RandomNormal(shape:Shape, ?dtype, ?device, ?backend) =
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.RandomNormal(shape, dtype, device)

    /// Gets a tensor filled with random integer values from the given range for the given shape and configuration
    static member RandomInt(shape:Shape, low, high, ?dtype, ?device, ?backend) =
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device.Default
        statics.RandomInt(shape, low, high, dtype, device)

    /// <summary>
    ///   Gets a tensor filled with values drawn from the given .NET object.
    /// </summary>
    ///
    /// <remarks>
    ///  The value may be a scalar, an array, or an array of tupled objects. If the <c>dtype</c> is not specified
    ///  then it is inferred from the .NET type of the object.
    /// </remarks>
    static member Create(values: obj, ?dtype, ?device, ?backend) =
        // We deliver consistent in-memory data to the backend - a dtype Int32 gets int32 etc.
        let data, shape, dtype2 =
            match dtype with 
            | Some Dtype.Int64 ->
                let a,s = DataConverter.dataOfValuesForInt64 values
                (a :> Array), s, Dtype.Int64
            | Some Dtype.Int32 ->
                let a,s = DataConverter.dataOfValuesForInt32 values
                (a :> Array), s, Dtype.Int32
            | Some Dtype.Int16 ->
                let a,s = DataConverter.dataOfValuesForInt16 values
                (a :> Array), s, Dtype.Int16
            | Some Dtype.Int8 ->
                let a,s = DataConverter.dataOfValuesForInt8 values
                (a :> Array), s, Dtype.Int8
            | Some Dtype.Byte ->
                let a,s = DataConverter.dataOfValuesForByte values
                (a :> Array), s, Dtype.Byte
            | Some Dtype.Bool ->
                let a,s = DataConverter.dataOfValuesForBool values
                (a :> Array), s, Dtype.Bool
            | Some Dtype.Float64 ->
                let a,s = DataConverter.dataOfValuesForFloat64 values
                (a :> Array), s, Dtype.Float64
            | Some Dtype.Float32 ->
                let a,s = DataConverter.dataOfValuesForFloat32 values
                (a :> Array), s, Dtype.Float32
            | Some Dtype.Float16 ->
                let a,s = DataConverter.dataOfValuesForFloat32 values
                (a :> Array), s, Dtype.Float16
            | Some Dtype.BFloat16 ->
                let a,s = DataConverter.dataOfValuesForFloat32 values
                (a :> Array), s, Dtype.BFloat16
            // If no dtype is given, use a dtype inferred from the given data. This is consistent with PyTorch's behavior.
            | None ->
                match values |> DataConverter.tryFlatArrayAndShape<float32> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Float32)
                | _ ->
                // Exception: If data is double and no dtype is given by the user, prefer a Float32 tensor
                match values |> DataConverter.tryFlatArrayAndShape<double> with
                | Some (values, shape) -> ((values |> Array.map float32 :> Array), shape, Dtype.Float32)
                | _ ->
                match values |> DataConverter.tryFlatArrayAndShape<int64> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Int64)
                | _ ->
                match values |> DataConverter.tryFlatArrayAndShape<int32> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Int32)
                | _ ->
                match values |> DataConverter.tryFlatArrayAndShape<int16> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Int16)
                | _ ->
                match values |> DataConverter.tryFlatArrayAndShape<bool> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Bool)
                | _ ->
                match values |> DataConverter.tryFlatArrayAndShape<byte> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Byte)
                | _ ->
                match values |> DataConverter.tryFlatArrayAndShape<int8> with
                | Some (values, shape) -> ((values :> Array), shape, Dtype.Int8)
                | _ ->
                failwithf "Cannot create tensor from data: %A" values

        let statics = BackendTensorStatics.Get(?backend=backend)
        let device = defaultArg device Device.Default

        statics.CreateFromFlatArray(data, shape, dtype2, device)

    static member CreateFromFlatArray(values: Array, shape:Shape, ?dtype, ?device, ?backend) =
        let statics = BackendTensorStatics.Get(?backend=backend)
        let dtype = defaultArg dtype Dtype.Default
        let device = defaultArg device Device .Default
        statics.CreateFromFlatArray(values, shape, dtype, device)

    /// Gets a tensor filled with values drawn from the given .NET object for the
    /// given configuration settings, defaulting to the configuration settings of the object tensor.
    member t.CreateLike(values: obj, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Create(values, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a tensor filled with arbitrary values for the given shape and configuration settings,
    /// defaulting to the configuration settings of the object tensor
    member t.EmptyLike(shape: Shape, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Empty(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a zero tensor for the given configuration settings, defaulting to the configuration settings of the object tensor
    member t.ZeroLike(?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Zero(dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a tensor filled with zero values for the given shape and configuration settings,
    /// defaulting to the configuration settings of the object tensor
    member t.ZerosLike(shape: Shape, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Zeros(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a scalar one tensor for the given configuration settings, defaulting to the configuration settings of the object tensor
    member t.OneLike(?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.One(dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a tensor filled with one values for the given shape and configuration settings,
    /// defaulting to the configuration settings of the object tensor
    member t.OnesLike(shape: Shape, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Ones(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a tensor filled with the given scalar value for the given shape and configuration settings,
    /// defaulting to the configuration settings of the object tensor
    member t.FullLike(shape: Shape, value: scalar, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Full(shape, value, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a tensor filled with random values for the given shape and configuration settings,
    /// defaulting to the configuration settings of the object tensor
    member t.RandomLike(shape: Shape, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.Random(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a tensor filled with random values from a normal distribution for the given shape and configuration settings,
    /// defaulting to the configuration settings of the object tensor
    member t.RandomNormalLike(shape: Shape, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.RandomNormal(shape=shape, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Gets a tensor filled with random integer values from the given range for the given shape and configuration settings,
    /// defaulting to the configuration settings of the object tensor
    member t.RandomIntLike(shape: Shape, low:int, high:int, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        RawTensor.RandomInt(shape=shape, low=low, high=high, dtype=defaultArg dtype t.Dtype, device=defaultArg device t.Device, backend=defaultArg backend t.Backend)

    /// Clone the underlying storage of the tensor.
    abstract Clone: unit -> RawTensor

    /// Expand the shape of the tensor.
    abstract Expand: newShape: Shape -> RawTensor

    /// Stack the given tensors along the given dimension
    abstract StackTs: tensors: RawTensor[] * dim:int -> RawTensor

    /// Unstack the given tensors along the given dimension
    abstract UnstackT: dim:int -> RawTensor[]

    /// Concatenate the given tensors along the given dimension
    abstract CatTs: tensors: RawTensor[] * dim: int -> RawTensor

    /// Split the given tensors along the given dimensions
    abstract SplitT: sizes: int[] * dim: int -> RawTensor[]

    /// <summary> Get a slice of the given tensor.</summary>
    ///
    /// <param name="fullBounds">
    ///  The indexes are an Nx3 array.  The first row is the start bounds, the second row is
    ///  the end bounds, the third is 1/0 indicating dimension removal.
    /// </param>
    abstract GetSlice: fullBounds: int[,] -> RawTensor

    /// Gets a .NET object representing the value of the tensor at the given indexes
    abstract GetItem: [<ParamArray>] indexes: int[] -> scalar

    /// Gets a .NET object representing the value of a scalar tensor 
    abstract ToScalar: unit -> scalar

    /// <summary>Get a .NET object for all the values in the tensor.</summary>
    ///
    /// <remarks>The runtime type of the returned object is either a .NET scalar
    /// or array corresponding to the shape and element type of the tensor.</remarks>
    abstract ToValues: unit -> obj

    /// Compare two tensors for equality
    abstract Equals: t2: RawTensor -> bool

    /// Returns a tensor where the elements have each been cast to the given tensor element storage type.
    abstract Cast: dtype: Dtype -> RawTensor

    /// Returns a tensor moved to the given device.
    abstract MoveTo: device: Device -> RawTensor

    /// Returns a hash of the contents of the tensor. This operation may cause the
    /// tensor to be moved to the CPU, and its entire contents iterated.
    abstract ComputeHash: unit -> int

    /// Indicates if the two tensors have the same shape and element type, and all corresponding values
    /// are equal up to the given tolerances.
    abstract AllClose: t2: RawTensor * relativeTolerance: float * absoluteTolerance: float -> bool

    /// Returns a tensor with values constrained by the corresponding elements in the low/high tensors.
    abstract ClampT: low: RawTensor * high: RawTensor -> RawTensor

    /// Returns a tensor selecting the given indices from the given dimension and stacking those in the order specified.
    abstract GatherT: dim: int * indices: RawTensor -> RawTensor

    /// Returns a tensor with given destination shape where values are copied from the current tensor to locations specified by the dimension and indices.
    abstract ScatterT: dim: int * indices: RawTensor * destinationShape: Shape -> RawTensor

    /// Returns a boolean tensor comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract LtTT: t2: RawTensor -> RawTensor

    /// Returns a boolean tensor comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract GtTT: t2: RawTensor -> RawTensor

    /// Returns a boolean tensor comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract LeTT: t2: RawTensor -> RawTensor

    /// Returns a boolean tensor comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract GeTT: t2: RawTensor -> RawTensor

    /// Returns a boolean tensor comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract EqTT: t2: RawTensor -> RawTensor

    /// Returns a boolean tensor comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract NeqTT: t2: RawTensor -> RawTensor

    /// Returns a boolean tensor where each element indicates if the corresponding element in the tensor is an infinity value
    abstract IsInfT: unit -> RawTensor

    /// Returns a boolean tensor where each element indicates if the corresponding element in the tensor is a NaN value
    abstract IsNaNT: unit -> RawTensor

    /// Gets a tensor containing values and indexes of a maximum value of the tensor reducing along the given dimension
    abstract MaxReduceT: dim: int * keepdim: bool -> RawTensor * RawTensor

    /// Gets the index of a maximum value of the tensor 
    abstract MaxIndexT: unit -> int[]

    /// Gets a tensor containing values and indexes of a minimum value of the tensor reducing along the given dimension
    abstract MinReduceT: dim: int * keepdim: bool -> RawTensor * RawTensor

    /// Gets the index of a minimum value of the tensor
    abstract MinIndexT: unit -> int[]

    /// Returns the element-wise addition of the two tensors
    abstract AddTT: RawTensor * ?alpha: scalar -> RawTensor

    /// Returns the element-wise addition of a tensor and a scalar
    abstract AddTT0: b: scalar * ?alpha: scalar -> RawTensor

    /// Adds a slice of <c>t2</c> at the given location to the tensor
    abstract AddTTSlice: location: int[] * t2: RawTensor -> RawTensor

    /// Returns the element-wise subtraction of two tensors
    abstract SubTT: t2: RawTensor -> RawTensor

    /// Returns the element-wise subtraction of the scalar and a tensor, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract SubFromT0T: t1: scalar -> RawTensor

    /// Returns the element-wise subtraction of the tensor and a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract SubTT0: t2: scalar -> RawTensor

    /// Returns the element-wise multiplication of two tensors
    abstract MulTT: t2: RawTensor -> RawTensor

    /// Returns the element-wise multiplication of a tensor and a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract MulTT0: t2: scalar -> RawTensor

    /// Returns the element-wise division of two tensors
    abstract DivTT: t2: RawTensor -> RawTensor

    /// Returns the element-wise division of a scalar by a tensor, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract DivFromT0T: t1: scalar -> RawTensor

    /// Returns the element-wise division of a tensor by a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract DivTT0: t2: scalar -> RawTensor

    /// Returns the element-wise exponentiation of two tensors
    abstract PowTT: t2: RawTensor -> RawTensor

    /// Returns the element-wise exponentiation of a scalar and a tensor, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract PowFromT0T: t1: scalar -> RawTensor

    /// Returns the element-wise exponentiation of a tensor and a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract PowTT0: t2: scalar -> RawTensor

    /// Returns the matrix multiplication of two tensors
    abstract MatMulTT: t2: RawTensor -> RawTensor

    /// Returns the batched matrix multiplication of two tensors
    abstract BMMTT: t2: RawTensor -> RawTensor

    /// Returns the 1D maxpool of a tensor and its chosen maximum indices
    abstract MaxPool1D: kernelSize: int * stride: int * padding: int -> RawTensor * RawTensor

    /// Returns the 2D maxpool of a tensor and its chosen maximum indices
    abstract MaxPool2D: kernelSize: int[] * strides: int[] * padding: int[] -> RawTensor * RawTensor

    /// Returns the 3D maxpool of a tensor and its chosen maximum indices
    abstract MaxPool3D: kernelSize: int[] * strides: int[] * padding: int[] -> RawTensor * RawTensor

    /// Returns the 1D maxunpool of a tensor using the given indices for locations of maximums
    abstract MaxUnpool1D: indices: RawTensor * outputSize: int[] -> RawTensor

    /// Returns the 2D maxunpool of a tensor using the given indices for locations of maximums
    abstract MaxUnpool2D: indices: RawTensor * outputSize: int[] -> RawTensor

    /// Returns the 3D maxunpool of a tensor using the given indices for locations of maximums
    abstract MaxUnpool3D: indices: RawTensor * outputSize: int[] -> RawTensor

    /// Returns the 1D avgpool of a tensor 
    abstract AvgPool1D: kernelSize: int * stride: int * padding: int (* * ceil_mode: bool * count_include_pad: bool *) -> RawTensor

    /// Returns the 2D avgpool of a tensor 
    abstract AvgPool2D: kernelSize: int[] * stride: int[] * padding: int[] (* * ceil_mode: bool * count_include_pad: bool *) -> RawTensor

    /// Returns the 2D avgpool of a tensor 
    abstract AvgPool3D: kernelSize: int[] * stride: int[] * padding: int[] (* * ceil_mode: bool * count_include_pad: bool *) -> RawTensor

    /// <summary>Returns the reverse mode of a 1D avgpool of a tensor, apportioning each part of the adjoint equally to each corresponding input</summary>
    /// <remarks>The originalInput parameter is only used for shape information</remarks>
    abstract AvgPoolReverse1D: originalInput: RawTensor * kernelSize: int * stride: int * padding: int (* * ceil_mode: bool * count_include_pad: bool *) -> RawTensor

    /// <summary>Returns the reverse mode of a 2D avgpool of a tensor, apportioning each part of the adjoint equally to each corresponding input</summary>
    /// <remarks>The originalInput parameter is only used for shape information</remarks>
    abstract AvgPoolReverse2D: originalInput: RawTensor * kernelSize: int[] * stride: int[] * padding: int[] (* * ceil_mode: bool * count_include_pad: bool *) -> RawTensor

    /// <summary>Returns the reverse mode of a 3D avgpool of a tensor, apportioning each part of the adjoint equally to each corresponding input</summary>
    /// <remarks>The originalInput parameter is only used for shape information</remarks>
    abstract AvgPoolReverse3D: originalInput: RawTensor * kernelSize: int[] * stride: int[] * padding: int[] (* * ceil_mode: bool * count_include_pad: bool *) -> RawTensor

    /// Returns the 1D convolution of the tensor
    abstract Conv1D: kernel: RawTensor * stride: int * padding: int -> RawTensor

    /// Returns the 2D convolution of the tensor
    abstract Conv2D: kernel: RawTensor * strides: int[] * padding: int[] -> RawTensor

    /// Returns the 3D convolution of the tensor
    abstract Conv3D: kernel: RawTensor * strides: int[] * padding: int[] -> RawTensor

    /// Returns a view of the original tensor with its dimensions permuted
    abstract PermuteT: permutation: int[] -> RawTensor

    /// Returns the element-wise negation of the tensor
    abstract NegT: unit -> RawTensor

    /// Returns the scalar tensor for the summation of all elements in the tensor 
    abstract SumT: ?resultType: Dtype -> RawTensor

    /// Returns the tensor representing the summation of the tensor along the given dimension
    abstract SumTDim: dim: int * ?resultType: Dtype -> RawTensor

    /// Returns the transpose of the tensor between the given dimensions
    abstract TransposeT: dim0: int * dim1: int -> RawTensor

    /// Returns the transpose of a 2D tensor
    abstract TransposeT2: unit -> RawTensor

    /// Returns the inverse of a single square matrix (2d tensor) or a batch of square matrices (3d tensor)
    abstract InverseT: unit -> RawTensor

    /// Returns the determinant of a square matrix
    abstract DetT: unit -> RawTensor

    /// Returns the solution of single a square system of linear equations with a unique solution or a batch of several such systems
    abstract SolveTT: RawTensor -> RawTensor
    
    /// Returns the tensor with the same values and the given dimension removed. The given dimension must be of size 1.
    abstract SqueezeT: dim: int -> RawTensor

    /// Returns the tensor with the same values and a dimension of size 1 inserted before the given dimension.
    abstract UnsqueezeT: dim: int -> RawTensor

    /// Returns the flip of the tensor along the given dimensions 
    abstract FlipT: dims: int[] -> RawTensor

    /// Returns the dilation of the tensor using the given dilations parameters
    abstract DilateT: dilations: int[] -> RawTensor

    /// Returns the reverse of the dilation of the tensor using the given dilations parameters
    abstract UndilateT: dilations: int[] -> RawTensor

    /// Returns the tensor with the same values viewed as a different shape
    abstract ViewT: shape: Shape -> RawTensor

    /// Returns the element-wise sign of the tensor
    abstract SignT: unit -> RawTensor

    /// Returns the element-wise integer floor of the tensor
    abstract FloorT: unit -> RawTensor

    /// Returns the element-wise integer ceiling of the tensor
    abstract CeilT: unit -> RawTensor

    /// Returns the element-wise rounding of the tensor
    abstract RoundT: unit -> RawTensor

    /// Returns the element-wise absolute value of the tensor
    abstract AbsT: unit -> RawTensor

    /// Returns the element-wise ReLU of the tensor
    abstract ReluT: unit -> RawTensor

    /// Returns the element-wise softplus of the tensor
    abstract SoftplusT: unit -> RawTensor

    /// Returns the element-wise sigmoid of the tensor
    abstract SigmoidT: unit -> RawTensor

    /// Returns the element-wise natural exponentiation of the tensor
    abstract ExpT: unit -> RawTensor

    /// Returns the element-wise natural logarithm of the tensor
    abstract LogT: unit -> RawTensor

    /// Returns the element-wise base10 logarithm of the tensor
    abstract Log10T: unit -> RawTensor

    /// Returns the element-wise square root of the tensor
    abstract SqrtT: unit -> RawTensor

    /// Returns the element-wise sine of the tensor
    abstract SinT: unit -> RawTensor

    /// Returns the element-wise cosine of the tensor
    abstract CosT: unit -> RawTensor

    /// Returns the element-wise tangent of the tensor
    abstract TanT: unit -> RawTensor

    /// Returns the element-wise sinh of the tensor
    abstract SinhT: unit -> RawTensor

    /// Returns the element-wise cosh of the tensor
    abstract CoshT: unit -> RawTensor

    /// Returns the element-wise tanh of the tensor
    abstract TanhT: unit -> RawTensor

    /// Returns the element-wise asin of the tensor
    abstract AsinT: unit -> RawTensor

    /// Returns the element-wise cos of the tensor
    abstract AcosT: unit -> RawTensor

    /// Returns the element-wise atan of the tensor
    abstract AtanT: unit -> RawTensor

    default t.IsInfT() =
        match t.Dtype with 
        | Dtype.IntegralOrBool -> t.FullLike(t.Shape, false, dtype=Dtype.Bool)
        | _ -> t.AbsT().EqTT(t.FullLike(t.Shape,System.Single.PositiveInfinity))

    default t.IsNaNT() =
        match t.Dtype with 
        | Dtype.IntegralOrBool -> t.FullLike(t.Shape, false, dtype=Dtype.Bool)
        | _ -> t.NeqTT(t)

    member t.Print(?postfix: string) =
        // TODO: this code is not ideal and can be reimplemented to be cleaner and more efficient
        let postfix = defaultArg postfix ""
        if t.Nelement = 0 then sprintf "tensor([])%s" postfix
        else
        let threshold = Printer.Default.threshold
        let edgeItems = Printer.Default.edgeItems
        let precision = Printer.Default.precision

        let vmin = t.GetItem(t.MinIndexT()).toDouble()
        let vmax = t.GetItem(t.MaxIndexT()).toDouble()
        let absMax = max (abs vmin) (abs vmax)
        let precisionStr = (String.replicate precision "0")
        let floatMaxStrLen1 = System.String.Format("{0:G"+precision.ToString()+"}", absMax).Length
        let floatMaxStrLen2 = System.String.Format("{0:0."+precisionStr+"}", absMax).Length
        let floatFormat1 = "{0,"+floatMaxStrLen1.ToString()+":G"+precision.ToString()+"}"
        let floatFormat2 = "{0,"+floatMaxStrLen2.ToString()+":0."+precisionStr+"}"
        let floatFormat3 = "{0,"+floatMaxStrLen2.ToString()+": 0."+precisionStr+";-0."+precisionStr+"}"
        let floatNoDecimals = t.Dtype.IsFloatingPoint && (let tt = t.Cast(Dtype.Float64) in tt.CeilT().Equals(tt))
        let floatNonNegative = t.Dtype.IsFloatingPoint && (let tt = t.Cast(Dtype.Float64) in tt.AbsT().Equals(tt))
        let printFloat (v:float) =
            if absMax >= 1.e8 || floatNoDecimals then
                let p = System.String.Format(floatFormat1, v)
                if p.Contains(".") || p.Contains("e") || p.Contains("E") || p.Contains("NaN") || p.Contains("Inf") || p.Contains("∞") then p else p + "."
            elif floatNonNegative then
                System.String.Format(floatFormat2, v)
            else
                System.String.Format(floatFormat3, v)

        let intMaxStrLen = System.String.Format("{0:D}", int64 (if vmin < 0. then -absMax else absMax)).Length
        let intFormat = "{0,"+intMaxStrLen.ToString()+":D}"
        let printInt (v:int64) =
            System.String.Format(intFormat, v)

        let printVal (x:scalar) = 
            match x.GetTypeCode() with 
            | TypeCode.Single -> printFloat (x.toDouble())
            | TypeCode.Double -> printFloat (x.toDouble())
            | TypeCode.Int32 -> printInt (x.toInt64())
            | TypeCode.Int64 -> printInt (x.toInt64())
            | TypeCode.Byte -> printInt (x.toInt64())
            | TypeCode.SByte -> printInt (x.toInt64())
            | TypeCode.Int16 -> printInt (x.toInt64())
            | TypeCode.Boolean -> if (x.toBool()) then " true" else "false"
            | _ -> printFloat (x.toDouble()) // Handles Float16, BFloat16

        let sb = System.Text.StringBuilder()
        sb.Append("tensor(") |> ignore
        match t.Dim with
        | 0 ->
            sb.Append(printVal (t.ToScalar())) |> ignore
        | _ ->
            let rec print (shape:Shape) externalCoords = 
                if shape.Length = 1 then
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    if (shape[0] >= threshold) && (edgeItems*2 < shape[0]) then
                        for i=0 to edgeItems-1 do
                            let globalCoords = Array.append externalCoords [|i|]
                            sb.Append(prefix) |> ignore
                            sb.Append(printVal (t.GetItem(globalCoords))) |> ignore
                            prefix <- ", "
                        sb.Append(", ...") |> ignore
                        for i=shape[0]-edgeItems to shape[0]-1 do
                            let globalCoords = Array.append externalCoords [|i|]
                            sb.Append(prefix) |> ignore
                            sb.Append(printVal (t.GetItem(globalCoords))) |> ignore
                            // prefix <- ", "
                    else
                        for i=0 to shape[0]-1 do
                            let globalCoords = Array.append externalCoords [|i|]
                            sb.Append(prefix) |> ignore
                            sb.Append(printVal (t.GetItem(globalCoords))) |> ignore
                            prefix <- ", "
                    sb.Append("]") |> ignore
                else
                    sb.Append("[") |> ignore
                    let mutable prefix = ""
                    let prefix2 = sprintf ",%s%s" (String.replicate (max 1 (shape.Length-1)) "\n       ") (String.replicate (externalCoords.Length+1) " ")
                    if (shape[0] >= threshold) && (edgeItems*2 < shape[0]) then
                        for i=0 to edgeItems-1 do
                            sb.Append(prefix) |> ignore
                            print shape[1..] (Array.append externalCoords [|i|])
                            prefix <- prefix2
                        sb.Append(prefix) |> ignore
                        sb.Append("...") |> ignore
                        for i=shape[0]-edgeItems to shape[0]-1 do
                            sb.Append(prefix) |> ignore
                            print shape[1..] (Array.append externalCoords [|i|])
                            // prefix <- prefix2
                    else
                        for i=0 to shape[0]-1 do
                            sb.Append(prefix) |> ignore
                            print shape[1..] (Array.append externalCoords [|i|])
                            prefix <- prefix2
                    sb.Append("]") |> ignore
            print t.Shape [||]
        if t.Dtype <> Dtype.Default then
            sb.Append ",dtype=" |> ignore
            sb.Append (t.Dtype.ToString()) |> ignore
        if t.Device <> Device.Default then
            sb.Append ",device=" |> ignore
            sb.Append (t.Device.ToString()) |> ignore
        if t.Backend <> Backend.Default then
            sb.Append ",backend=" |> ignore
            sb.Append (t.Backend.ToString()) |> ignore
        sb.Append(")") |> ignore
        sb.Append(postfix) |> ignore
        sb.ToString()

    override x.Equals(yobj: obj) = 
        match yobj with
        | :? RawTensor as y -> x.Equals(y)
        | _ -> false

    override x.GetHashCode() = x.ComputeHash()

    interface System.IComparable with 
        member x.CompareTo(yobj) =
            match yobj with
            | :? RawTensor as y -> Unchecked.compare (x.ToScalar()) (y.ToScalar())
            | _ -> failwithf "Cannot compare RawTensor with object of type %A" (yobj.GetType())

    default t.GetItem(indexes) =
        let t0 = t.GetSlice(Array2D.init indexes.Length 3 (fun i j -> if j = 0 || j = 1 then indexes[i] else 1))
        t0.ToScalar()

    /// Returns a .NET object for the value of a scalar tensor
    override t.ToScalar() =
        match t.Nelement with
        | 1 -> t.ViewT([||]).ToValues() :?> scalar
        | _ -> failwithf "Only one element tensors can be converted to scalars. This tensor has shape %A." t.Shape

    /// Returns a .NET array object for the values of a non-scalar tensor
    member t.ToArray() =
        match t.Dim with
        | 0 -> failwithf "Cannot convert scalar tensor to array"
        | _ ->
            match t.ToValues() with 
            | :? System.Array as a -> a
            | _ -> failwithf "ToValues() should return an array but returned type %A" (t.GetType())

    /// A backdoor to switch this tensor to be usable as a mutable tensor. You should have a unique handle to
    /// this tensor for the entire time it is being used as a mutable tensor.
    abstract SetMutable: unit -> unit

    abstract IsMutable: bool

    /// Modifies the tensor by with values constrained by the corresponding elements in the low/high tensors.
    abstract ClampInPlace: low: RawTensor * high: RawTensor -> unit

    /// Modifies the tensor by comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract LtInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract GtInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract LeInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract GeInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract EqInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by comparing each element pairwise with the corresponding element in <c>t2</c>
    abstract NeqInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by the element-wise addition of the two tensors
    abstract AddInPlace: RawTensor * ?alpha: scalar -> unit

    /// Modifies the tensor by the element-wise addition of two scalars
    abstract AddScalarInPlace: b: scalar -> unit

    /// Adds a slice of <c>t2</c> at the given location to the tensor
    abstract AddSliceInPlace: location: int[] * t2: RawTensor -> unit

    /// Modifies the tensor by the element-wise subtraction of two tensors
    abstract SubInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by the element-wise subtraction of the tensor and a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract SubScalarInPlace: b: scalar -> unit

    /// Modifies the tensor by the element-wise multiplication of two tensors
    abstract MulInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by the element-wise multiplication of a tensor and a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract MulScalarInPlace: b: scalar -> unit

    /// Modifies the tensor by the element-wise division of two tensors
    abstract DivInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by the element-wise division of a tensor by a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract DivScalarInPlace: t2: scalar  -> unit

    /// Modifies the tensor by the element-wise exponentiation of two tensors
    abstract PowInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by the element-wise exponentiation of a tensor and a scalar, where the scalar is logically
    /// broadcast to the same shape as the tensor
    abstract PowScalarInPlace: t2: scalar -> unit

    /// Modifies the tensor by the matrix multiplication of two tensors
    abstract MatMulInPlace: t2: RawTensor -> unit

    /// Modifies the tensor by the element-wise negation of the tensor
    abstract NegInPlace: unit -> unit

    /// Modifies the tensor by the element-wise sign of the tensor
    abstract SignInPlace: unit -> unit

    /// Modifies the tensor by the element-wise integer floor of the tensor
    abstract FloorInPlace: unit -> unit

    /// Modifies the tensor by the element-wise integer ceiling of the tensor
    abstract CeilInPlace: unit -> unit

    /// Modifies the tensor by the element-wise rounding of the tensor
    abstract RoundInPlace: unit -> unit

    /// Modifies the tensor by the element-wise absolute value of the tensor
    abstract AbsInPlace: unit -> unit

    /// Modifies the tensor by the element-wise ReLU of the tensor
    abstract ReluInPlace: unit -> unit

    /// Modifies the tensor by the element-wise softplus of the tensor
    abstract SoftplusInPlace: unit -> unit

    /// Modifies the tensor by the element-wise sigmoid of the tensor
    abstract SigmoidInPlace: unit -> unit

    /// Modifies the tensor by the element-wise natural exponentiation of the tensor
    abstract ExpInPlace: unit -> unit

    /// Modifies the tensor by the element-wise natural logarithm of the tensor
    abstract LogInPlace: unit -> unit

    /// Modifies the tensor by the element-wise base10 logarithm of the tensor
    abstract Log10InPlace: unit -> unit

    /// Modifies the tensor by the element-wise square root of the tensor
    abstract SqrtInPlace: unit -> unit

    /// Modifies the tensor by the element-wise sine of the tensor
    abstract SinInPlace: unit -> unit

    /// Modifies the tensor by the element-wise cosine of the tensor
    abstract CosInPlace: unit -> unit

    /// Modifies the tensor by the element-wise tangent of the tensor
    abstract TanInPlace: unit -> unit

    /// Modifies the tensor by the element-wise sinh of the tensor
    abstract SinhInPlace: unit -> unit

    /// Modifies the tensor by the element-wise cosh of the tensor
    abstract CoshInPlace: unit -> unit

    /// Modifies the tensor by the element-wise tanh of the tensor
    abstract TanhInPlace: unit -> unit

    /// Modifies the tensor by the element-wise asin of the tensor
    abstract AsinInPlace: unit -> unit

    /// Modifies the tensor by the element-wise cos of the tensor
    abstract AcosInPlace: unit -> unit

    /// Modifies the tensor by the element-wise atan of the tensor
    abstract AtanInPlace: unit -> unit

    /// Modifies the tensor by setting all values to one
    abstract OnesInPlace: unit -> unit

    /// Modifies the tensor by setting all values to zero
    abstract ZerosInPlace: unit -> unit

    /// Modifies the tensor by setting it to random values taken from a uniform distribution in [0, 1).
    abstract RandomInPlace: unit -> unit

    /// Modifies the tensor by setting all values taken from a normal distribution with mean 0 and variance 1.
    abstract RandomNormalInPlace: unit -> unit

    /// Gets a tensor filled with random integers from the given range 
    abstract RandomIntInPlace: low:int * high:int -> unit

