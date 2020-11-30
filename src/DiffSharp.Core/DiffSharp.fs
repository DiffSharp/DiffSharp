// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp

open DiffSharp.Backends
open DiffSharp.Util

/// Tensor operations
type dsharp =

    /// <summary>
    /// Creates a new tensor from the given data, using the given element type and configuration.
    /// </summary>
    /// 
    /// <example><code>
    ///    let t1 = dsharp.tensor [ 1 .. 10 ]
    ///    let t2 = dsharp.tensor [ [ 1.0; 3.0; 4.0 ];
    ///                             [ 1.02; 3.04; 4.01 ] ]
    /// </code></example>
    /// 
    /// <remarks>
    ///  The data is converted from arrays, sequences, lists and tuples of primitive values to a tensor whose shape is inferred from the data.
    /// </remarks>
    /// <param name="value">The .NET object used to form the initial values for the tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    /// <remarks>The fastest creation technique is a one dimensional array matching the desired dtype. Then use 'view' to reshape.</remarks>
    static member tensor(value:obj, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        Tensor.create(value=value, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Seeds all backends with the given random seed, or a new seed based on the current time if no seed is specified.</summary>
    static member seed(?seed:int) = BackendTensorStatics.Seed(?seed=seed)

    /// <summary>Indicates if an object is a tensor</summary>
    static member isTensor(value:obj) = value :? Tensor

    /// <summary>Saves the tensor to the given file using a bespoke binary format.</summary>
    /// <remarks>
    ///   The binary format records the elements, backend, element type and shape. It does not record the device.
    ///   The format used may change from version to version of DiffSharp.
    /// </remarks>
    static member save(tensor:Tensor, fileName) = tensor.save(fileName)

    /// <summary>Loads the tensor from the given file using the given element type and configuration.</summary>
    ///
    /// <param name="fileName">The file from which to load the tensor.</param>
    /// <param name="dtype">The element type of the resulting tensor. Defaults to the element type of the saved tensor.</param>
    /// <param name="device">The device of the resulting tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The device of the resulting tensor. Default: if None, uses Backend.Default.</param>
    ///
    /// <remarks>
    ///    The backend at the time of saving the tensor must be available when the tensor is reloaded.
    ///    The tensor is first loaded into that backend and then moved. As a result, intermediate tensors may be created
    ///    in the process of reloading.
    /// </remarks>
    static member load(fileName, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        Tensor.load(fileName, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given shape, element type and configuration</summary>
    /// <param name="shape">The desired shape of returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member empty(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Empty(shape|>Seq.toArrayQuick, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given length, element type and configuration</summary>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member empty(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Empty([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Get the scalar zero tensor for the given configuration</summary>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member zero(?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Zero(?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration</summary>
    /// <param name="shape">The desired shape of returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member zeros(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Zeros(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with '0' values for the given length, element type and configuration</summary>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member zeros(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Zeros([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Get the scalar '1' tensor for the given configuration</summary>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member one(?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.One(?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration</summary>
    /// <param name="shape">The desired shape of returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member ones(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Ones(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor of the given length filled with '1' values for the given element type and configuration</summary>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member ones(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Ones([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration</summary>
    /// <param name="shape">The desired shape of returned tensor.</param>
    /// <param name="value">The scalar used to form the initial values for the tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member full(shape:seq<int>, value:scalar, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Full(shape|>Shape.create, value, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor of the given length filled with <paramref name="value" />, for the given element type and configuration</summary>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="value">The scalar giving the the initial values for the tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member full(length:int, value:scalar, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).fullLike(value, [|length|])

    /// <summary>Returns a new scalar tensor with the value <paramref name="value" />, for the given element type and configuration</summary>
    /// <param name="value">The scalar giving the the initial values for the tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member scalar(value:scalar, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        dsharp.full(Shape.scalar, value, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>
    /// Returns a 1-D tensor of size \(\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil\)
    /// with values from the interval [start, end) taken with common difference step beginning from start.
    /// </summary>
    /// 
    /// <remarks>
    ///  Non-integer steps may be subject to floating point rounding errors when comparing against end.
    /// </remarks>
    /// <param name="endVal">The ending value for the set of points.</param>
    /// <param name="startVal">The starting value for the set of points. Default: 0.</param>
    /// <param name="step">The gap between each pair of adjacent points. Default: 1.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member arange(endVal:float, ?startVal:float, ?step:float, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).arangeLike(endVal=endVal, ?startVal=startVal, ?step=step)

    /// <summary>
    /// Returns a 1-D tensor of size \(\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil\)
    /// with values from the interval [start, end) taken with common difference step beginning from start.
    /// </summary>
    /// <param name="endVal">The ending value for the set of points.</param>
    /// <param name="startVal">The starting value for the set of points. Default: 0.</param>
    /// <param name="step">The gap between each pair of adjacent points. Default: 1.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member arange(endVal:int, ?startVal:int, ?step:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).arangeLike(endVal=endVal, ?startVal=startVal, ?step=step)

    /// <summary>Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.</summary>
    /// <param name="rows">The number of rows</param>
    /// <param name="cols">The number of columns with default being n</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member eye(rows:int, ?cols:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        Tensor.eye(rows=rows, ?cols=cols, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a one-hot tensor, with one location set to 1, and all others 0.</summary>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="hot">The location to set to 1.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member onehot(length:int, hot:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = 
        dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).onehotLike(length, hot)

    /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)</summary>
    /// <param name="shape">The desired shape of returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member rand(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Random(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1)</summary>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member rand(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.Random([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).</summary>
    /// <param name="shape">The desired shape of returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member randn(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.RandomNormal(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).</summary>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member randn(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.RandomNormal([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).</summary>
    /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
    /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
    /// <param name="shape">The desired shape of returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member randint(low:int, high:int, shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.RandomInt(shape|>Shape.create, low, high, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).</summary>
    /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
    /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member randint(low:int, high:int, length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        TensorC(RawTensor.RandomInt([|length|], low, high, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a tensor where each row contains numSamples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.</summary>
    /// <param name="probs">The input tensor containing probabilities.</param>
    /// <param name="numSamples">The number of samples to draw.</param>
    /// <param name="normalize">Indicates where the probabilities should first be normalized by their sum.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member multinomial(probs:Tensor, numSamples:int, ?normalize:bool, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        probs.multinomial(numSamples, ?normalize=normalize, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Draws binary random numbers (0 or 1) from a Bernoulli distribution</summary>
    /// <param name="probs">The input tensor of probability values for the Bernoulli distribution.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member bernoulli(probs:Tensor, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        probs.bernoulli(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="p">The probability of an element to be zeroed. Default: 0.5.</param>
    static member dropout(input:Tensor, ?p:double) = input.dropout(?p=p)

    /// <summary>Randomly zero out entire channels (a channel is a 2D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 2D tensor \text{input}[i, j]input[i,j] ). Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="p">The probability of an element to be zeroed. Default: 0.5.</param>
    static member dropout2d(input:Tensor, ?p:double) = input.dropout2d(?p=p)

    /// <summary>Randomly zero out entire channels (a channel is a 3D feature map, e.g., the jj -th channel of the ii -th sample in the batched input is a 3D tensor \text{input}[i, j]input[i,j] ). Each channel will be zeroed out independently on every forward call with probability p using samples from a Bernoulli distribution.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="p">The probability of an element to be zeroed. Default: 0.5.</param>
    static member dropout3d(input:Tensor, ?p:double) = input.dropout3d(?p=p)

    /// <summary>Returns a new tensor filled with '0' values with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member zerosLike(input:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) =
        input.zerosLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a new tensor filled with '1' values with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member onesLike(input:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) =
        input.onesLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a new tensor filled with the given scalar value with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="value">The scalar giving the the initial values for the tensor.</param>
    /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member fullLike(input:Tensor, value:scalar, ?shape:seq<int>, ?dtype, ?device, ?backend) =
        input.fullLike(value, ?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>
    /// A version of dsharp.arange with characteristics based on the input tensor.
    /// </summary>
    /// 
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="endVal">The ending value for the set of points.</param>
    /// <param name="startVal">The starting value for the set of points. Default: 0.</param>
    /// <param name="step">The gap between each pair of adjacent points. Default: 1.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member arangeLike(input:Tensor, endVal:float, ?startVal:float, ?step:float, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        input.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>
    /// A version of dsharp.arange with characteristics based on the input tensor.
    /// </summary>
    /// 
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="endVal">The ending value for the set of points.</param>
    /// <param name="startVal">The starting value for the set of points. Default: 0.</param>
    /// <param name="step">The gap between each pair of adjacent points. Default: 1.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member arangeLike(input:Tensor, endVal:int, ?startVal:int, ?step:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
        input.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>
    /// A version of dsharp.onehot with characteristics based on the input tensor.
    /// </summary>
    /// 
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="length">The length of the returned tensor.</param>
    /// <param name="hot">The location to set to 1.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member onehotLike(input:Tensor, length:int, hot:int, ?dtype, ?device, ?backend) =
        input.onehotLike(length, hot, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1) with characteristics based on the input tensor</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member randLike(input:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) =
            input.randLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution) with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member randnLike(input:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) =
        input.randnLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive) with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
    /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
    /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member randintLike(input:Tensor, low:int, high:int, ?shape:seq<int>, ?dtype, ?device, ?backend) =
        input.randintLike(low=low, high=high, ?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns the '0' scalar tensor with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member zeroLike(input:Tensor, ?dtype, ?device, ?backend) =
        input.zeroLike(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns the '0' scalar tensor with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member oneLike(input:Tensor, ?dtype, ?device, ?backend) =
        input.oneLike(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns the total number of elements in the input tensor.</summary>
    /// <param name="input">The input tensor.</param>
    static member nelement(input:Tensor) = input.nelement

    /// <summary>Returns a new tensor based on the given .NET value with characteristics based on the input tensor.</summary>
    /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
    /// <param name="value">The .NET object giving the the initial values for the tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member like(input:Tensor, value:obj, ?dtype, ?device, ?backend) =
        input.like(value, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a new tensor with the same characteristics and storage cloned.</summary>
    /// <param name="input">The input tensor.</param>
    static member clone(input:Tensor) = input.clone()

    /// <summary>Returns a boolean tensor for the element-wise less-than comparison of the elements in the two tensors.</summary>
    /// <remarks>The shapes of input and other don’t need to match, but they must be broadcastable.</remarks>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member lt(a:Tensor, b:Tensor) = a.lt(b)

    /// <summary>Returns a boolean tensor for the element-wise greater-than comparison of the elements in the two tensors.</summary>
    /// <remarks>The shapes of input and other don’t need to match, but they must be broadcastable.</remarks>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member gt(a:Tensor, b:Tensor) = a.gt(b)

    /// <summary>Return a boolean tensor for the element-wise less-than-or-equal comparison of the elements in the two tensors.</summary>
    /// <remarks>The shapes of input and other don’t need to match, but they must be broadcastable.</remarks>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member le(a:Tensor, b:Tensor) = a.le(b)

    /// <summary>Returns a boolean tensor for the element-wise greater-than-or-equal comparison of the elements in the two tensors.</summary>
    /// <remarks>The shapes of input and other don’t need to match, but they must be broadcastable.</remarks>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member ge(a:Tensor, b:Tensor) = a.ge(b)

    /// <summary>Returns a boolean tensor where each element indicates if the corresponding element in the input tensor is an infinity value.</summary>
    /// <param name="input">The input tensor.</param>
    static member isinf(input:Tensor) = input.isinf()

    /// <summary>Returns a boolean tensor where each element indicates if the corresponding element in the input tensor is a NaN (not-a-number) value.</summary>
    /// <param name="input">The input tensor.</param>
    static member isnan(input:Tensor) = input.isnan()

    /// <summary>Returns a boolean indicating if any element of the tensor is infinite.</summary>
    /// <param name="input">The input tensor.</param>
    static member hasinf(input:Tensor) = input.hasinf()

    /// <summary>Returns a boolean indicating if any element of the tensor is a not-a-number (NaN) value.</summary>
    /// <param name="input">The input tensor.</param>
    static member hasnan(input:Tensor) = input.hasnan()

    /// <summary>Returns the indices of the maximum value of all elements in the input tensor.</summary>
    /// <param name="input">The input tensor.</param>
    static member argmax(input:Tensor) = input.argmax()

    /// <summary>Returns the indices of the minimum value of all elements in the input tensor.</summary>
    /// <param name="input">The input tensor.</param>
    static member argmin(input:Tensor) = input.argmin()

    /// <summary>Returns the maximum value of all elements in the input tensor</summary>
    /// <param name="input">The input tensor.</param>
    static member max(input:Tensor) = input.max()

    /// <summary>Returns the minimum value of all elements in the input tensor</summary>
/// <param name="input">The input tensor.</param>
    static member min(input:Tensor) = input.min()

    /// <summary>Each element of the tensor input is compared with the corresponding element of the tensor other and an element-wise maximum is taken.</summary>
    /// <remarks>The shapes of input and other don’t need to match, but they must be broadcastable.</remarks>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member max(a:Tensor, b:Tensor) = a.max(b)

    /// <summary>Each element of the tensor input is compared with the corresponding element of the tensor other and an element-wise minimum is taken.</summary>
    /// <remarks>The shapes of input and other don’t need to match, but they must be broadcastable.</remarks>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member min(a:Tensor, b:Tensor) = a.min(b)

    /// <summary>Clamp all elements in input into the range [ low..high] and return a resulting tensor</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="low">The lower-bound of the range to be clamped to.</param>
    /// <param name="high">The upper-bound of the range to be clamped to.</param>
    static member clamp(input:Tensor, ?low:scalar, ?high:scalar) = input.clamp(?low=low, ?high=high)

    /// <summary>Normalizes a vector so all the values are between zero and one (min-max scaling to 0..1).</summary>
    /// <param name="input">The input tensor.</param>
    static member normalize(input:Tensor) = input.normalize()

    /// <summary>Returns the tensor after standardization (z-score normalization)</summary>
    /// <param name="input">The input tensor.</param>
    static member standardize(input:Tensor) = input.standardize()

    /// <summary>
    ///  Returns a tensor with the diagonal elements with respect to <c>dim1</c> and <c>dim2</c>.
    ///  The argument offset controls which diagonal to consider.
    /// </summary>
    /// <param name="input">The input tensor. Must be at least 2-dimensional.</param>
    /// <param name="offset">Which diagonal to consider. Default: 0.</param>
    /// <param name="dim1">The first dimension with respect to which to take diagonal. Default: 0..</param>
    /// <param name="dim2">The second dimension with respect to which to take diagonal. Default: 1.</param>
    static member diagonal(input:Tensor, ?offset:int, ?dim1:int, ?dim2:int) =
        input.diagonal(?offset=offset, ?dim1=dim1, ?dim2=dim2)

    /// <summary>Returns the sum of the elements of the diagonal of the input 2-D matrix</summary>
    /// <param name="input">The input tensor.</param>
    static member trace(input:Tensor) = input.trace()

    /// <summary>Returns a new view of the input tensor with singleton dimensions expanded to a larger size</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="shape">The desired shape of returned tensor.</param>
    static member expand(input:Tensor, shape:seq<int>) = input.expand(shape)

    /// <summary>Expand the input tensor to the same size as other tensor</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="other">The result tensor has the same size as other.</param>
    static member expandAs(input:Tensor, other:Tensor) = input.expandAs(other)

    /// <summary>Concatenates sequence of tensors along a new dimension</summary>
    /// <remarks>All tensors need to be of the same size.</remarks>
    /// <param name="tensors">The sequence of tensors to concatenate.</param>
    /// <param name="dim">The dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive).</param>
    static member stack(tensors:seq<Tensor>, ?dim:int) = Tensor.stack(tensors, ?dim=dim)

    /// <summary>Removes a tensor dimension</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension to remove.</param>
    static member unstack(input:Tensor, ?dim:int) = input.unstack(?dim=dim)

    /// <summary>Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.</summary>
    /// <param name="tensors">The sequence of tensors to concatenate.</param>
    /// <param name="dim">The the dimension over which the tensors are concatenated.</param>
    static member cat(tensors:seq<Tensor>, ?dim:int) = Tensor.cat(tensors, ?dim=dim)

    /// <summary>Splits the tensor into chunks. The tensor will be split into sizes.Length chunks each with a corresponding size in the given dimension.</summary>
    /// <param name="input">The tensor to split.</param>
    /// <param name="sizes">The size of a single chunk or list of sizes for each chunk.</param>
    /// <param name="dim">The dimension along which to split the tensor.</param>
    static member split(input:Tensor, sizes:seq<int>, ?dim:int) = input.split(sizes, ?dim=dim)

    /// <summary>Return the element-wise addition of the two tensors.</summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member add(a:Tensor, b:Tensor) = a.add(b)

    /// <summary>Return the element-wise subtraction of the two tensors.</summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member sub(a:Tensor, b:Tensor) = a.sub(b)

    /// <summary>Return the element-wise multiplication of the two tensors.</summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member mul(a:Tensor, b:Tensor) = a.mul(b)

    /// <summary>Return the element-wise division of the two tensors.</summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member div(a:Tensor, b:Tensor) = a.div(b)

    /// <summary>Return the element-wise exponentiation of the two tensors.</summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member pow(a:Tensor, b:Tensor) = a.pow(b)

    /// <summary>Matrix product of two tensors.</summary>
    static member matmul(a:Tensor, b:Tensor) = a.matmul(b)

    /// <summary>Computes the dot product (inner product) of two tensors.</summary>
    /// <param name="a">The first tensor.</param>
    /// <param name="b">The second tensor.</param>
    static member dot(a:Tensor, b:Tensor) = a.dot(b)

    /// <summary>Return the element-wise negation of the input tensor.</summary>
    /// <param name="input">The input tensor.</param>
    static member neg(input:Tensor) = input.neg()

    /// <summary>Returns the sum of all elements in the input tensor</summary>
    /// <param name="input">The input tensor.</param>
    static member sum(input:Tensor) = input.sum()

    /// <summary>Returns the sum of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.</summary>
    /// <remarks>
    ///  If keepdim is true, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    static member sum(input:Tensor, dim:int, ?keepDim:bool) = input.sum(dim, ?keepDim=keepDim)

    /// <summary>Returns the mean value of all elements in the input tensor.</summary>
    /// <param name="input">The input tensor.</param>
    static member mean(input:Tensor) = input.mean()

    /// <summary>Returns the mean value of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.</summary>
    /// <remarks>
    ///  If keepdim is true, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    static member mean(input:Tensor, dim:int, ?keepDim:bool) = input.mean(dim, ?keepDim=keepDim)

    /// <summary>Returns the variance of all elements in the input tensor.</summary>
    /// <remarks>
    ///  If unbiased is False, then the variance will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    static member variance(input:Tensor, ?unbiased:bool) = input.variance(?unbiased=unbiased)

    /// <summary>Returns the variance of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.</summary>
    /// <remarks>
    ///  If keepdim is true, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
    ///  If unbiased is False, then the variance will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    static member variance(input:Tensor, dim:int, ?keepDim:bool, ?unbiased:bool) = input.variance(dim, ?keepDim=keepDim, ?unbiased=unbiased)

    /// <summary>Returns the standard deviation of all elements in the input tensor.</summary>
    /// <remarks>
    ///  If unbiased is False, then the standard deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    static member stddev(input:Tensor, ?unbiased:bool) = input.stddev(?unbiased=unbiased)

    /// <summary>Returns the standard deviation of each row of the input tensor in the given dimension dim. If dim is a list of dimensions, reduce over all of them.</summary>
    /// <remarks>
    ///  If keepdim is true, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1. Otherwise, dim is squeezed, resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
    ///  If unbiased is False, then the standard deviation will be calculated via the biased estimator. Otherwise, Bessel’s correction will be used.
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    /// <param name="unbiased">Whether to use the unbiased estimation or not.</param>
    static member stddev(input:Tensor, dim:int, ?keepDim:bool, ?unbiased:bool) = input.stddev(dim, ?keepDim=keepDim, ?unbiased=unbiased)

    /// <summary>Gathers values along an axis specified by dim.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The axis along which to index.</param>
    /// <param name="indices">The the indices of elements to gather.</param>
    static member gather(input:Tensor, dim:int, indices:Tensor) = input.gather(dim, indices)

    /// <summary>Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim0">The first dimension to be transposed.</param>
    /// <param name="dim1">The second dimension to be transposed.</param>
    static member transpose(input:Tensor, dim0:int, dim1:int) = input.transpose(dim0, dim1)

    /// <summary>Returns a tensor that is a transposed version of input with dimensions 0 and 1 swapped.</summary>
    /// <param name="input">The input tensor.</param>
    static member transpose(input:Tensor) = input.transpose()

    /// <summary>Returns a tensor with all the dimensions of input of size 1 removed.</summary>
    /// <remarks>If the tensor has a batch dimension of size 1, then squeeze(input) will also remove the batch dimension, which can lead to unexpected errors.</remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">If given, the input will be squeezed only in this dimension.</param>
    static member squeeze(input:Tensor, ?dim:int) = input.squeeze(?dim=dim)

    /// <summary>Returns a new tensor with a dimension of size one inserted at the specified position</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The index at which to insert the singleton dimension.</param>
    static member unsqueeze(input:Tensor, dim:int) = input.unsqueeze(dim)

    /// <summary>Reverse the order of a n-D tensor along given axis in dims</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dims">The axis to flip on.</param>
    static member flip(input:Tensor, dims:seq<int>) = input.flip(dims)

    /// <summary>Dilate the tensor in using the given dilations in each corresponding dimension.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dilations">The dilations to use.</param>
    static member dilate(input:Tensor, dilations:seq<int>) = input.dilate(dilations)

    /// <summary>Reverse the dilation of the tensor in using the given dilations in each corresponding dimension.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dilations">The dilations to use.</param>
    static member undilate(input:Tensor, dilations:seq<int>) = input.undilate(dilations)

    /// <summary>Repeat elements of a tensor</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension along which to repeat values.</param>
    /// <param name="times">The number of repetitions for each element.</param>
    static member repeat(input:Tensor, dim:int, times:int) = input.repeat(dim, times)

    /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.</summary>
    /// <remarks>The returned tensor shares the same data and must have the same number of elements, but may have a different size. For a tensor to be viewed, the new view size must be compatible with its original size.
    ///   The returned tensor shares the same data and must have the same number of elements, but may have a different size. 
    ///   For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension,
    ///   or only span across original dimensions \(d, d+1, \dots, d+kd,d+1,…,d+k\) that satisfy the following contiguity-like condition that
    ///   \(\forall i = d, \dots, d+k-1∀i=d,…,d+k−1 ,\) \[\text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]\]
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="shape">The desired shape of returned tensor.</param>
    static member view(input:Tensor, shape:seq<int>) = input.view(shape)

    /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.</summary>
    /// <remarks>The returned tensor shares the same data and must have the same number of elements, but may have a different size. For a tensor to be viewed, the new view size must be compatible with its original size.
    ///   The returned tensor shares the same data and must have the same number of elements, but may have a different size. 
    ///   For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension,
    ///   or only span across original dimensions \(d, d+1, \dots, d+kd,d+1,…,d+k\) that satisfy the following contiguity-like condition that
    ///   \(\forall i = d, \dots, d+k-1∀i=d,…,d+k−1 ,\) \[\text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]\]
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="shape">The desired shape of returned tensor.</param>
    static member view(input:Tensor, shape:int) = input.view(shape)

    /// <summary>View this tensor as the same size as other.</summary>
    /// <remarks>The returned tensor shares the same data and must have the same number of elements, but may have a different size. For a tensor to be viewed, the new view size must be compatible with its original size.
    ///   The returned tensor shares the same data and must have the same number of elements, but may have a different size. 
    ///   For a tensor to be viewed, the new view size must be compatible with its original size and stride, i.e., each new view dimension must either be a subspace of an original dimension,
    ///   or only span across original dimensions \(d, d+1, \dots, d+kd,d+1,…,d+k\) that satisfy the following contiguity-like condition that
    ///   \(\forall i = d, \dots, d+k-1∀i=d,…,d+k−1 ,\) \[\text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]\]
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="other">The result tensor has the same size as other.</param>
    static member viewAs(input:Tensor, other:Tensor) = input.viewAs(other)

    /// <summary>Flattens a contiguous range of dims in a tensor.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="startDim">The first dim to flatten.</param>
    /// <param name="endDim">The last dim to flatten.</param>
    static member flatten(input:Tensor, ?startDim:int, ?endDim:int) = input.flatten(?startDim=startDim, ?endDim=endDim)

    /// <summary>Returns a new tensor with the signs of the elements of input.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    /// <param name="input">The input tensor.</param>
    static member sign(input:Tensor) = input.sign()

    /// <summary>Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    /// <param name="input">The input tensor.</param>
    static member floor(input:Tensor) = input.floor()

    /// <summary>Returns a new tensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    /// <param name="input">The input tensor.</param>
    static member ceil(input:Tensor) = input.ceil()

    /// <summary>Returns a new tensor with each of the elements of input rounded to the closest integer.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    /// <param name="input">The input tensor.</param>
    static member round(input:Tensor) = input.round()

    /// <summary>Computes the element-wise absolute value of the given input tensor.</summary>
    /// <remarks>The tensor will have the same element type as the input tensor.</remarks>
    static member abs(input:Tensor) = input.abs()

    /// <summary>Applies the rectified linear unit function element-wise.</summary>
    /// <param name="input">The input tensor.</param>
    static member relu(input:Tensor) = input.relu()

    /// <summary>Applies the leaky rectified linear unit function element-wise</summary>
    /// <remarks>\[\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)\]</remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="negativeSlope">Controls the angle of the negative slope. Default: 0.01.</param>
    static member leakyRelu(input:Tensor, ?negativeSlope:float) = input.leakyRelu(?negativeSlope=negativeSlope)

    /// <summary>Applies the sigmoid element-wise function</summary>
    /// <remarks>\[\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}\]</remarks>
    /// <param name="input">The input tensor.</param>
    static member sigmoid(input:Tensor) = input.sigmoid()

    /// <summary>Applies the softplus function element-wise.</summary>
    /// <remarks>\[\text{Softplus}(x) = \frac{1}{\beta} * \log(1 + \exp(\beta * x))\]</remarks>
    /// <param name="input">The input tensor.</param>
    static member softplus(input:Tensor) = input.softplus()

    /// <summary>Applies the exp function element-wise.</summary>
    /// <param name="input">The input tensor.</param>
    static member exp(input:Tensor) = input.exp()

    /// <summary>Returns a new tensor with the natural logarithm of the elements of input.</summary>
    /// <remarks> \[y_{i} = \log_{e} (x_{i})\]</remarks>
    /// <param name="input">The input tensor.</param>
    static member log(input:Tensor) = input.log()

    /// <summary>Returns a new tensor with the logarithm to the base 10 of the elements of input.</summary>
    /// <remarks>\[y_{i} = \log_{10} (x_{i})\]</remarks>
    /// <param name="input">The input tensor.</param>
    static member log10(input:Tensor) = input.log10()

    /// <summary>Returns a new tensor with the square-root of the elements of input.</summary>
    /// <param name="input">The input tensor.</param>
    static member sqrt(input:Tensor) = input.sqrt()

    /// <summary>Returns a new tensor with the sine of the elements of input</summary>
    /// <param name="input">The input tensor.</param>
    static member sin(input:Tensor) = input.sin()

    /// <summary>Returns a new tensor with the cosine of the elements of input</summary>
    /// <param name="input">The input tensor.</param>
    static member cos(input:Tensor) = input.cos()

    /// <summary>Returns a new tensor with the tangent of the elements of input</summary>
    /// <param name="input">The input tensor.</param>
    static member tan(input:Tensor) = input.tan()

    /// <summary>Returns a new tensor with the hyperbolic sine of the elements of input.</summary>
    static member sinh(input:Tensor) = input.sinh()

    /// <summary>Returns a new tensor with the hyperbolic cosine of the elements of input.</summary>
    static member cosh(input:Tensor) = input.cosh()

    /// <summary>Returns a new tensor with the hyperbolic tangent of the elements of input.</summary>
    /// <param name="input">The input tensor.</param>
    static member tanh(input:Tensor) = input.tanh()

    /// <summary>Returns a new tensor with the arcsine of the elements of input.</summary>
    /// <param name="input">The input tensor.</param>
    static member asin(input:Tensor) = input.asin()

    /// <summary>Returns a new tensor with the arccosine of the elements of input.</summary>
    /// <param name="input">The input tensor.</param>
    static member acos(input:Tensor) = input.acos()
    
    /// <summary>Returns a new tensor with the arctangent of the elements of input.</summary>
    /// <param name="input">The input tensor.</param>
    static member atan(input:Tensor) = input.atan()

    /// <summary>Applies a softmax function.</summary>
    /// <remarks>Softmax is defined as: \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}.</remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">A dimension along which softmax will be computed.</param>
    static member softmax(input:Tensor, dim:int) = input.softmax(dim)

    /// <summary>Applies a softmax followed by a logarithm.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">A dimension along which softmax will be computed.</param>
    static member logsoftmax(input:Tensor, dim:int) = input.logsoftmax(dim)

    /// <summary>Applies a logsumexp followed by a logarithm.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="dim">The dimension to reduce.</param>
    /// <param name="keepDim">Whether the output tensor has dim retained or not.</param>
    static member logsumexp(input:Tensor, dim:int, ?keepDim:bool) = input.logsumexp(dim, ?keepDim=keepDim)

    /// <summary>Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input and the target.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    static member mseLoss(input:Tensor, target:Tensor, ?reduction:string) =
        input.mseLoss(target, ?reduction=reduction)

    /// <summary>Creates a criterion that measures the Binary Cross Entropy between the target and the output</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor.</param>
    /// <param name="weight">A manual rescaling weight given to the loss of each batch element.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    static member bceLoss(input:Tensor, target:Tensor, ?weight:Tensor, ?reduction:string) =
        input.bceLoss(target, ?weight=weight, ?reduction=reduction)

    /// <summary>The negative log likelihood loss.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor.</param>
    /// <param name="weight">A optional manual rescaling weight given to the loss of each batch element.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    static member nllLoss(input:Tensor, target:Tensor, ?weight:Tensor, ?reduction:string) =
        input.nllLoss(target, ?weight=weight, ?reduction=reduction)

    /// <summary>This criterion combines logsoftmax and nllLoss in a single function</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor.</param>
    /// <param name="weight">A optional manual rescaling weight given to the loss of each batch element.</param>
    /// <param name="reduction">Optionally specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. 'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of elements in the output, 'sum': the output will be summed. Note: size_average and reduce are in the process of being deprecated, and in the meantime, specifying either of those two args will override reduction. Default: 'mean'.</param>
    static member crossEntropyLoss(input:Tensor, target:Tensor, ?weight:Tensor, ?reduction:string) =
        input.crossEntropyLoss(target, ?weight=weight, ?reduction=reduction)

    /// <summary>Applies a 1D max pooling over an input signal composed of several input planes.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    static member maxpool1d(input:Tensor, kernelSize:int, ?stride:int, ?padding:int) =
        input.maxpool1d(kernelSize, ?stride=stride, ?padding=padding)

    /// <summary>Applies a 1D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    static member maxpool1di(input:Tensor, kernelSize:int, ?stride:int, ?padding:int) =
        input.maxpool1di(kernelSize, ?stride=stride, ?padding=padding)

    /// <summary>Applies a 2D max pooling over an input signal composed of several input planes.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    static member maxpool2d(input:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        input.maxpool2d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>Applies a 2D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    static member maxpool2di(input:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        input.maxpool2di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>Applies a 3D max pooling over an input signal composed of several input planes.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    static member maxpool3d(input:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        input.maxpool3d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>Applies a 3D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    static member maxpool3di(input:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) =
        input.maxpool3di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>Computes a partial inverse of maxpool1di</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="indices">The indices selected by maxpool1di.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="outputSize">The targeted output size.</param>
    static member maxunpool1d(input:Tensor, indices:Tensor, kernelSize:int, ?stride:int, ?padding:int, ?outputSize:seq<int>) =
        input.maxunpool1d(indices, kernelSize, ?stride=stride, ?padding=padding, ?outputSize=outputSize)

    /// <summary>Computes a partial inverse of maxpool2di</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="indices">The indices selected by maxpool2di.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    /// <param name="outputSize">The targeted output size.</param>
    static member maxunpool2d(input:Tensor, indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) =
        input.maxunpool2d(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, ?outputSize=outputSize)

    /// <summary>Computes a partial inverse of maxpool3di</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="indices">The indices selected by maxpool3di.</param>
    /// <param name="kernelSize">The size of the window to take a max over.</param>
    /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
    /// <param name="padding">The implicit zero padding to be added on both sides.</param>
    /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
    /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
    /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
    /// <param name="outputSize">The targeted output size.</param>
    static member maxunpool3d(input:Tensor, indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) =
        input.maxunpool3d(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, ?outputSize=outputSize)

    /// <summary>Applies a 1D convolution over an input signal composed of several input planes</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit paddings on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    static member conv1d(input:Tensor, filters:Tensor, ?stride:int, ?padding:int, ?dilation:int) =
        input.conv1d(filters, ?stride=stride, ?padding=padding, ?dilation=dilation)

    /// <summary>Applies a 2D convolution over an input signal composed of several input planes</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    static member conv2d(input:Tensor, filters:Tensor, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>) =
        input.conv2d(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

    /// <summary>Applies a 3D convolution over an input signal composed of several input planes</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    static member conv3d(input:Tensor, filters:Tensor, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>) =
        input.conv3d(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

    /// <summary>Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
    static member convTranspose1d(input:Tensor, filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int) =
        input.convTranspose1d(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

    /// <summary>Applies a 2D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
    /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
    static member convTranspose2d(input:Tensor, filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) =
        input.convTranspose2d(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

    /// <summary>Applies a 3D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="filters">The filters.</param>
    /// <param name="stride">The stride of the convolving kernel.</param>
    /// <param name="padding">The implicit padding on both sides of the input.</param>
    /// <param name="dilation">The spacing between kernel elements.</param>
    /// <param name="strides">The strides of the convolving kernel.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    /// <param name="dilations">The spacings between kernel elements.</param>
    /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
    /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
    static member convTranspose3d(input:Tensor, filters:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) =
        input.convTranspose3d(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

    /// <summary>Add zero padding to each side of a tensor</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
    static member pad(input:Tensor, paddings:seq<int>) = input.pad(paddings)

    /// <summary>Convert tensor to an image tensor with shape Channels x Height x Width</summary>
    /// <remarks>If the input tensor has 4 dimensions, then make a single image grid.</remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="pixelMin">The minimum pixel value.</param>
    /// <param name="pixelMax">The maximum pixel value.</param>
    /// <param name="normalize">If True, shift the image to the range (0, 1), by the min and max values specified by range.</param>
    /// <param name="gridCols">Number of columns of images in the grid.</param>
    static member toImage(input:Tensor, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int) =
        input.toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)

    /// <summary>Convert tensor to a grayscale image tensor and return a string representation approximating grayscale values</summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="pixelMin">The minimum pixel value.</param>
    /// <param name="pixelMax">The maximum pixel value.</param>
    /// <param name="normalize">If True, shift the image to the range (0, 1), by the min and max values specified by range.</param>
    /// <param name="gridCols">Number of columns of images in the grid.</param>
    /// <param name="asciiPalette">The ASCII pallette to use.</param>
    static member toImageString(input:Tensor, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int, ?asciiPalette:string) =
        input.toImageString(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols, ?asciiPalette=asciiPalette)

    /// <summary>Load an image file as a tensor.</summary>
    /// <param name="fileName">The file name of the image to load.</param>
    /// <param name="normalize">If True, shift the image to the range (0, 1).</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    static member loadImage(fileName:string, ?normalize:bool, ?dtype: Dtype, ?device: Device, ?backend: Backend) =
        Tensor.loadImage(fileName=fileName, ?normalize=normalize, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Save a given Tensor into an image file.</summary>
    /// <remarks>If the input tensor has 4 dimensions, then make a single image grid.</remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="fileName">The name of the file to save to.</param>
    /// <param name="pixelMin">The minimum pixel value.</param>
    /// <param name="pixelMax">The maximum pixel value.</param>
    /// <param name="normalize">If True, shift the image to the range (0, 1), by the min and max values specified by range.</param>
    /// <param name="gridCols">Number of columns of images in the grid.</param>
    static member saveImage(input:Tensor, fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int) =
        input.saveImage(fileName=fileName, ?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)

    /// <summary>Convert the tensor to one with the given element type.</summary>
    /// <remarks>If the element type is unchanged the input tensor will be returned.</remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor.</param>
    static member cast(input:Tensor, dtype:Dtype) = input.cast(dtype)

    /// <summary>Move the tensor to a difference device, backend and/or change its element type.</summary>
    /// <remarks>If the characteristics are unchanged the input tensor will be returned.</remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
    static member move(input:Tensor, ?dtype, ?device, ?backend) =
        input.move(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Configure the default element type, device and/or backend.</summary>
    /// <param name="dtype">The new default element type.</param>
    /// <param name="device">The new default device.</param>
    /// <param name="backend">The new default backend.</param>
    static member config(?dtype: Dtype, ?device: Device, ?backend: Backend) = 
        dtype |> Option.iter (fun d -> Dtype.Default <- d)
        device |> Option.iter (fun d -> Device.Default <- d)
        backend |> Option.iter (fun d -> Backend.Default <- d)
        dsharp.tensor(0.) |> ignore // We need this to ensure the backend assemblies are loaded and backend is ready to set the random seed immediately after config

    /// <summary>Return the current default element type, device and backend.</summary>
    static member config() = Dtype.Default, Device.Default, Backend.Default

    /// <summary>Configure the default element type, device and/or backend.</summary>
    /// <param name="configuration">A tuple of the new default element type, default device and default backend.</param>
    static member config(configuration: (Dtype * Device * Backend)) =
        let (dtype,device,backend) = configuration
        dsharp.config(dtype, device, backend)

    /// <summary>Return the list of available devices.</summary>
    /// <param name="deviceType">If given, only return devices for this device type.</param>
    /// <param name="backend">Return information for this backend. Defaults to Backend.Default.</param>
    static member devices(?deviceType, ?backend) = BackendTensorStatics.Get(?backend=backend).GetDevices(?deviceType=deviceType)

    /// <summary>Indicates if the given device type is supported.</summary>
    /// <param name="deviceType">The requested device type.</param>
    /// <param name="backend">Return information for this backend. Defaults to Backend.Default.</param>
    static member isDeviceTypeSupported(deviceType, ?backend) = BackendTensorStatics.Get(?backend=backend).IsDeviceTypeSupported(deviceType)

    /// <summary>Indicates if CUDA is supported.</summary>
    /// <param name="backend">Return information for this backend. Defaults to Backend.Default.</param>
    static member isCudaSupported(?backend) = BackendTensorStatics.Get(?backend=backend).IsDeviceTypeSupported(DeviceType.CUDA)


// Differentiable methods mirroring F# collection modules
// TODO: implement more differentiable higher-order functions and corresponding unit tests for their derivatives
type dsharp with

    /// <summary>Create a new 1D tensor using the given initializer for each element.</summary>
    /// <param name="count">The length of the tensor.</param>
    /// <param name="initializer">The function used to initialize each element.</param>
    static member init (count:int) (initializer:int->'a) = Array.init count initializer |> dsharp.tensor

    /// <summary>Create a new 2D tensor using the given initializer for each element.</summary>
    /// <param name="length1">The length of the tensor in the first dimension.</param>
    /// <param name="length2">The length of the tensor in the second dimension.</param>
    /// <param name="initializer">The function used to initialize each element.</param>
    static member init2d (length1:int) (length2:int) (initializer:int->int->'a) = Array2D.init length1 length2 initializer |> dsharp.tensor

    /// <summary>Create a new 3D tensor using the given initializer for each element.</summary>
    /// <param name="length1">The length of the tensor in the 1st dimension.</param>
    /// <param name="length2">The length of the tensor in the 2nd dimension.</param>
    /// <param name="length3">The length of the tensor in the 3rd dimension.</param>
    /// <param name="initializer">The function used to initialize each element.</param>
    static member init3d (length1:int) (length2:int) (length3:int) (initializer:int->int->int->'a) = Array3D.init length1 length2 length3 initializer |> dsharp.tensor

    /// <summary>Create a new 4D tensor using the given initializer for each element.</summary>
    /// <param name="length1">The length of the tensor in the 1st dimension.</param>
    /// <param name="length2">The length of the tensor in the 2nd dimension.</param>
    /// <param name="length3">The length of the tensor in the 3rd dimension.</param>
    /// <param name="length4">The length of the tensor in the 4th dimension.</param>
    /// <param name="initializer">The function used to initialize each element.</param>
    static member init4d (length1:int) (length2:int) (length3:int) (length4:int) (initializer:int->int->int->int->'a) = Array4D.init length1 length2 length3 length4 initializer |> dsharp.tensor

    /// <summary>Create a new 1D tensor using the given value for each element.</summary>
    /// <param name="count">The number of elements in the tensor.</param>
    /// <param name="value">The initial value for each element of the tensor.</param>
    static member create (count:int) (value:'a) = Array.create count value |> dsharp.tensor

    /// <summary>Create a new 1D tensor using '0' as value for each element.</summary>
    /// <param name="count">The number of elements in the tensor.</param>
    static member zeroCreate (count:int) = Array.zeroCreate count |> dsharp.tensor

    /// <summary>Produce a new tensor by mapping a function over all elements of the input tensor.</summary>
    /// <param name="mapping">The function is passed the index of each element. The function to apply to each element of the tensor.</param>
    /// <param name="tensor">The input tensor.</param>
    static member mapi (mapping:int[]->Tensor->Tensor) (tensor:Tensor) = // Differentiable map
        let tflat = tensor.view(-1)
        let items = Array.init (tflat.nelement) (fun i -> mapping (flatIndexToIndex tensor.shape i) tflat.[i])
        dsharp.stack(items).view(tensor.shape)

    /// <summary>Produce a new tensor by mapping a function over all corresponding elements of two input tensors.</summary>
    /// <remarks>The function is passed the index of each element. The shapes of the two tensors must be identical.</remarks>
    /// <param name="mapping">The function to apply to each element of the tensor.</param>
    /// <param name="tensor1">The first input tensor.</param>
    /// <param name="tensor2">The second input tensor.</param>
    static member mapi2 (mapping:int[]->Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) =  // Differentiable map2
        if tensor1.shape <> tensor2.shape then failwithf "Expecting tensor1.shape (%A) and tensor2.shape (%A) to be the same" tensor1.shape tensor2.shape
        let tflat1 = tensor1.view(-1)
        let tflat2 = tensor2.view(-1)
        let items = Array.init (tflat1.nelement) (fun i -> mapping (flatIndexToIndex tensor1.shape i) tflat1.[i] tflat2.[i])
        dsharp.stack(items).view(tensor1.shape)

    /// <summary>Produce a new tensor by mapping a function over all corresponding elements of three input tensors.</summary>
    /// <remarks>The function is passed the index of each element. The shapes of the three tensors must be identical.</remarks>
    /// <param name="mapping">The function to apply to each element of the tensor.</param>
    /// <param name="tensor1">The first input tensor.</param>
    /// <param name="tensor2">The second input tensor.</param>
    /// <param name="tensor3">The third input tensor.</param>
    static member mapi3 (mapping:int[]->Tensor->Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) (tensor3:Tensor) =  // Differentiable map3
        if (tensor1.shape <> tensor2.shape) || (tensor2.shape <> tensor3.shape) then failwithf "Expecting tensor1.shape (%A), tensor2.shape (%A), tensor3.shape (%A) to be the same" tensor1.shape tensor2.shape tensor3.shape
        let tflat1 = tensor1.view(-1)
        let tflat2 = tensor2.view(-1)
        let tflat3 = tensor3.view(-1)
        let items = Array.init (tflat1.nelement) (fun i -> mapping (flatIndexToIndex tensor1.shape i) tflat1.[i] tflat2.[i] tflat3.[i])
        dsharp.stack(items).view(tensor1.shape)

    /// <summary>Produce a new tensor by mapping a function over all elements of the input tensor.</summary>
    /// <param name="mapping">The function to apply to each element of the tensor.</param>
    /// <param name="tensor">The input tensor.</param>
    static member map (mapping:Tensor->Tensor) (tensor:Tensor) = tensor |> dsharp.mapi (fun _ v -> mapping v)

    /// <summary>Produce a new tensor by mapping a function over all corresponding elements of two input tensors.</summary>
    /// <remarks>The shapes of the two tensors must be identical.</remarks>
    /// <param name="mapping">The function to apply to each element of the tensor.</param>
    /// <param name="tensor1">The first input tensor.</param>
    /// <param name="tensor2">The second input tensor.</param>
    static member map2 (mapping:Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) = dsharp.mapi2 (fun _ v1 v2 -> mapping v1 v2) tensor1 tensor2

    /// <summary>Produce a new tensor by mapping a function over all corresponding elements of three input tensors.</summary>
    /// <remarks>The shapes of the three tensors must be identical.</remarks>
    /// <param name="mapping">The function to apply to each element of the tensor.</param>
    /// <param name="tensor1">The first input tensor.</param>
    /// <param name="tensor2">The second input tensor.</param>
    /// <param name="tensor3">The third input tensor.</param>
    static member map3 (mapping:Tensor->Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) (tensor3:Tensor) = dsharp.mapi3 (fun _ v1 v2 v3 -> mapping v1 v2 v3) tensor1 tensor2 tensor3


// Functional automatic differentiation API
type dsharp with

    /// <summary>Increase the global nesting level for automatic differentiation.</summary>
    static member nest() = GlobalNestingLevel.Next() |> ignore

    /// <summary>Set the global nesting level for automatic differentiation.</summary>
    /// <param name="level">The new nesting level.</param>
    static member nest(level) = GlobalNestingLevel.Set(level)

    /// <summary>Get the global nesting level for automatic differentiation.</summary>
    static member nestLevel() = GlobalNestingLevel.Current

    /// <summary>Reset the global nesting level for automatic differentiation to zero.</summary>
    static member nestReset() = GlobalNestingLevel.Reset()

    /// <summary>Get the primal value of the tensor.</summary>
    static member primal (tensor:Tensor) = tensor.primal

    /// <summary>Get the derivative value of the tensor.</summary>
    static member derivative (tensor:Tensor) = tensor.derivative

    /// <summary>Get the primal and derivative values of the tensor.</summary>
    static member primalDerivative (tensor:Tensor) = tensor.primal, tensor.derivative

    /// <summary>Produce a new tensor suitable for calculating the forward-mode derivative at the given level tag.</summary>
    /// <param name="tag">The level tag.</param>
    /// <param name="derivative">The derivative of the input.</param>
    /// <param name="tensor">The input.</param>
    static member forwardDiff (tag:uint32) (derivative:Tensor) (tensor:Tensor) = tensor.forwardDiff(derivative, tag)

    /// <summary>Produce a new tensor suitable for calculating the reverse-mode derivative at the given level tag.</summary>
    /// <param name="tag">The level tag.</param>
    /// <param name="tensor">The output tensor.</param>
    static member reverseDiff (tag:uint32) (tensor:Tensor) = tensor.reverseDiff(tag)

    /// <summary>Reset the reverse mode computation associated with the given output tensor.</summary>
    /// <param name="tensor">The output tensor.</param>
    static member reverseReset (tensor:Tensor) = tensor.reverseReset(true)

    /// <summary>Push the given value as part of the reverse-mode computation at the given output tensor.</summary>
    /// <param name="value">The value to apply.</param>
    /// <param name="tensor">The output tensor.</param>
    static member reversePush (value:Tensor) (tensor:Tensor) = tensor.reversePush(value)

    /// <summary>Compute the reverse-mode derivative at the given output tensor.</summary>
    /// <param name="value">The value to apply.</param>
    /// <param name="tensor">The output tensor.</param>
    static member reverse (value:Tensor) (tensor:Tensor) = tensor.reverse(value)

    /// <summary>TBD</summary>
    static member evalForwardDiff f x v =
        x |> dsharp.forwardDiff (GlobalNestingLevel.Next()) v |> f |> dsharp.primalDerivative

    /// <summary>TBD</summary>
    static member evalReverseDiff f x =
        let x = x |> dsharp.reverseDiff (GlobalNestingLevel.Next())
        let fx = f x
        let r = fun v -> fx |> dsharp.reverse v; x.derivative
        fx.primal, r

    /// <summary>TBD</summary>
    static member evalForwardDiffs (f:Tensor->Tensor) x (v:Tensor[]) =
        let n = v.Length
        if n = 0 then [|f x|]
        else
            let mutable x = x
            for i in 0..n-1 do
                x <- x |> dsharp.forwardDiff (GlobalNestingLevel.Next()) v.[i]
            let mutable fx = f x
            [|for _ in 0..n-1 do
                let d = fx.derivativeDeep
                fx <- fx.primal
                d
                |] |> Array.rev |> Array.append [|fx|]

    /// <summary>TBD</summary>
    /// <param name="f">TBD</param>
    /// <param name="x">TBD</param>
    /// <param name="v">TBD</param>
    /// <remarks>The <c>x</c> and <c>v</c> tensors should have the same number of elements.</remarks>
    static member fjacobianv f (x:Tensor) (v:Tensor) = 
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let fx, d = dsharp.evalForwardDiff f x v
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, d

    /// <summary>TBD</summary>
    static member jacobianv f x v = dsharp.fjacobianv f x v |> snd

    /// <summary>TBD</summary>
    /// <param name="f">TBD</param>
    /// <param name="x">TBD</param>
    /// <param name="v">TBD</param>
    /// <remarks>The <c>x</c> and <c>v</c> tensors should have the same number of elements.</remarks>
    static member fgradv f (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let fx, d = dsharp.evalForwardDiff f x v
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, d

    /// <summary>TBD</summary>
    static member gradv f x v = dsharp.fgradv f x v |> snd

    /// <summary>TBD</summary>
    static member fdiff f (x:Tensor) =
        let fx, d = dsharp.evalForwardDiff f x (x.onesLike())
        if x.dim <> 0 then failwithf "f must be a function of a scalar, encountered f:%A->%A" x.shape fx.shape
        fx, d

    /// <summary>TBD</summary>
    static member diff f x = dsharp.fdiff f x |> snd

    /// <summary>TBD</summary>
    static member ffdiffn (n:int) (f:Tensor->Tensor) (x:Tensor) =
        if n < 0 then failwith "Differentiation order n must be >= 0"
        if x.dim <> 0 then failwithf "f must be a function of a scalar"
        dsharp.evalForwardDiffs f x (Array.create n (x.onesLike()))

    /// <summary>TBD</summary>
    static member fdiffn n f x = let a = dsharp.ffdiffn n f x in a |> Array.head, a |> Array.last

    /// <summary>TBD</summary>
    static member diffn n f x = dsharp.fdiffn n f x |> snd

    /// <summary>TBD</summary>
    static member fdiff2 f x = dsharp.fdiffn 2 f x

    /// <summary>TBD</summary>
    static member diff2 f x = dsharp.diffn 2 f x

    /// <summary>Original value and transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`</summary>
    /// <param name="f">vector-to-vector function</param>
    /// <param name="x">Point at which the function <c>f</c> will be evaluated, it must have a single dimension.</param>
    /// <param name="v">Vector</param>
    static member fjacobianTv f x (v:Tensor) =
        let fx, r = dsharp.evalReverseDiff f x
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        if fx.nelement <> v.nelement then failwithf "(f x) and v must have the same number of elements"
        fx, r v

    /// <summary>Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`</summary>
    /// <param name="f">vector-to-vector function</param>
    /// <param name="x">Point at which the function <c>f</c> will be evaluated, it must have a single dimension.</param>
    /// <param name="v">Vector</param>
    static member jacobianTv f x v = dsharp.fjacobianTv f x v |> snd

    /// <summary>TBD</summary>
    static member fjacobian (f:Tensor->Tensor) x =
        let fx, r = dsharp.evalReverseDiff f x
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        if x.nelement > fx.nelement then
            fx, dsharp.stack(Array.init fx.nelement (fun i -> r (x.onehotLike(fx.nelement, i))), 0)
        else
            fx, dsharp.stack(Array.init x.nelement (fun j -> dsharp.jacobianv f x (x.onehotLike(x.nelement, j))), 1)

    /// <summary>TBD</summary>
    static member jacobian f x = dsharp.fjacobian f x |> snd

    /// <summary>TBD</summary>
    static member fgrad f (x:Tensor) =
        if x.dim = 0 then 
            dsharp.fdiff f x
        else
            let fx, r = dsharp.evalReverseDiff f x
            if x.dim > 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector or scalar, encountered f:%A->%A" x.shape fx.shape
            fx, r (fx.onesLike())

    /// <summary>TBD</summary>
    static member grad f x = dsharp.fgrad f x |> snd

    /// <summary>TBD</summary>
    static member fgradhessianv f (x:Tensor) (v:Tensor) =
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let x = x |> dsharp.reverseDiff (GlobalNestingLevel.Next())
        let fx, gv = dsharp.fgradv f x v
        gv.reverse()
        fx.primal, gv.primal, x.derivative

    /// <summary>TBD</summary>
    static member gradhessianv f x v = let _, gv, hv = dsharp.fgradhessianv f x v in gv, hv

    /// <summary>TBD</summary>
    static member fhessianv f x v = let fx, _, hv = dsharp.fgradhessianv f x v in fx, hv

    /// <summary>TBD</summary>
    static member hessianv f x v = dsharp.fhessianv f x v |> snd

    /// <summary>TBD</summary>
    static member fgradhessian (f:Tensor->Tensor) (x:Tensor) =
        let mutable fx = dsharp.zero()
        let gvs, hvs = Array.init x.nelement (fun j -> let ffxx, gv, hv = dsharp.fgradhessianv f x (x.onehotLike(x.nelement, j)) in fx <- ffxx; gv, hv) |> Array.unzip
        let h = dsharp.stack(hvs, 1)
        let g = dsharp.stack(gvs)
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, g, h

    /// <summary>TBD</summary>
    static member gradhessian f x = let _, g, h = dsharp.fgradhessian f x in g, h

    /// <summary>TBD</summary>
    static member fhessian (f:Tensor->Tensor) (x:Tensor) =
        let mutable fx = dsharp.zero()
        let h = dsharp.stack(Array.init x.nelement (fun j -> let ffxx, hv = dsharp.fhessianv f x (x.onehotLike(x.nelement, j)) in fx <- ffxx; hv), 1)
        if x.dim <> 1 || fx.dim <> 0 then failwithf "f must be a scalar-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, h

    /// <summary>TBD</summary>
    static member hessian f x = dsharp.fhessian f x |> snd

    /// <summary>TBD</summary>
    static member flaplacian f x =
        let fx, h = dsharp.fhessian f x
        fx, h.trace()

    /// <summary>TBD</summary>
    static member laplacian f x = dsharp.flaplacian f x |> snd

    /// <summary>TBD</summary>
    static member fcurl f x =
        let fx, j = dsharp.fjacobian f x
        if j.shape <> [|3; 3|] then failwithf "f must be a function with a three-by-three Jacobian"
        fx, dsharp.stack([j.[2, 1] - j.[1, 2]; j.[0, 2] - j.[2, 0]; j.[1, 0] - j.[0, 1]])

    /// <summary>TBD</summary>
    static member curl f x = dsharp.fcurl f x |> snd

    /// <summary>TBD</summary>
    static member fdivergence f x =
        let fx, j = dsharp.fjacobian f x
        if j.shape.[0] <> j.shape.[1] then failwithf "f must have a square Jacobian"
        fx, j.trace()

    /// <summary>TBD</summary>
    static member divergence f x = dsharp.fdivergence f x |> snd

    /// <summary>TBD</summary>
    static member fcurldivergence f x =
        let fx, j = dsharp.fjacobian f x
        if j.shape <> [|3; 3|] then failwithf "f must be a function with a three-by-three Jacobian"
        fx, dsharp.stack([j.[2, 1] - j.[1, 2]; j.[0, 2] - j.[2, 0]; j.[1, 0] - j.[0, 1]]), j.trace()

    /// <summary>TBD</summary>
    static member curldivergence f x = let _, c, d = dsharp.fcurldivergence f x in c, d
