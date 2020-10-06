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
    static member tensor(value:obj, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor.create(value=value, ?dtype=dtype, ?device=device, ?backend=backend)

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
    /// <param name="device">The device of the resulting tensor. Defaults to the current default device.</param>
    /// <param name="backend">The device of the resulting tensor. Defaults to the current default backend.</param>
    ///
    /// <remarks>
    ///    The backend at the time of saving the tensor must be available when the tensor is reloaded.
    ///    The tensor is first loaded into that backend and then moved. As a result, intermediate tensors may be created
    ///    in the process of reloading.
    /// </remarks>
    static member load(fileName, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor.load(fileName, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given shape, element type and configuration</summary>
    static member empty(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Empty(shape|>Seq.toArrayQuick, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given length, element type and configuration</summary>
    static member empty(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Empty([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Get the scalar zero tensor for the given configuration</summary>
    static member zero(?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Zero(?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration</summary>
    static member zeros(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Zeros(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with '0' values for the given length, element type and configuration</summary>
    static member zeros(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Zeros([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Get the scalar '1' tensor for the given configuration</summary>
    static member one(?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.One(?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration</summary>
    static member ones(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Ones(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor of the given length filled with '1' values for the given element type and configuration</summary>
    static member ones(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Ones([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration</summary>
    static member full(shape:seq<int>, value:obj, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Full(shape|>Shape.create, value, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>Returns a new tensor of the given length filled with <paramref name="value" />, for the given element type and configuration</summary>
    static member full(length:int, value:scalar, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).fullLike(value, [|length|])

    /// <summary>
    /// Returns a 1-D tensor of size \(\left\lceil \frac{\text{end} - \text{start}}{\text{step}} \right\rceil\)
    /// with values from the interval [start, end) taken with common difference step beginning from start.
    /// </summary>
    /// 
    /// <remarks>
    ///  Non-integer steps may be subject to floating point rounding errors when comparing against end.
    /// </remarks>
    static member arange(endVal:float, ?startVal:float, ?step:float, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).arangeLike(endVal=endVal, ?startVal=startVal, ?step=step)

    /// <summary>TBD</summary>
    static member arange(endVal:int, ?startVal:int, ?step:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).arangeLike(endVal=endVal, ?startVal=startVal, ?step=step)

    /// <summary>TBD</summary>
    static member eye(rows:int, ?cols:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor.eye(rows=rows, ?cols=cols, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member onehot(length:int, hot:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).onehotLike(length, hot)

    /// <summary>TBD</summary>
    static member rand(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Random(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>TBD</summary>
    static member rand(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.Random([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>TBD</summary>
    static member randn(shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.RandomNormal(shape|>Shape.create, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>TBD</summary>
    static member randn(length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.RandomNormal([|length|], ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>TBD</summary>
    static member randint(low:int, high:int, shape:seq<int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.RandomInt(shape|>Shape.create, low, high, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>TBD</summary>
    static member randint(low:int, high:int, length:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor(RawTensor.RandomInt([|length|], low, high, ?dtype=dtype, ?device=device, ?backend=backend))

    /// <summary>TBD</summary>
    static member multinomial(probs:Tensor, numSamples:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.multinomial(numSamples, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member bernoulli(probs:Tensor, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.bernoulli(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member dropout(a:Tensor, ?p:double) = a.dropout(?p=p)

    /// <summary>TBD</summary>
    static member dropout2d(a:Tensor, ?p:double) = a.dropout2d(?p=p)

    /// <summary>TBD</summary>
    static member dropout3d(a:Tensor, ?p:double) = a.dropout3d(?p=p)

    /// <summary>TBD</summary>
    static member zerosLike(a:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) = a.zerosLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member onesLike(a:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) = a.onesLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member fullLike(a:Tensor, value:scalar, ?shape:seq<int>, ?dtype, ?device, ?backend) = a.fullLike(value, ?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member arangeLike(a:Tensor, endVal:float, ?startVal:float, ?step:float, ?dtype:Dtype, ?device:Device, ?backend:Backend) = a.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member arangeLike(a:Tensor, endVal:int, ?startVal:int, ?step:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = a.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member onehotLike(a:Tensor, length:int, hot:int, ?dtype, ?device, ?backend) = a.onehotLike(length, hot, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member randLike(a:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) = a.randLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member randnLike(a:Tensor, ?shape:seq<int>, ?dtype, ?device, ?backend) = a.randnLike(?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member randintLike(a:Tensor, low:int, high:int, ?shape:seq<int>, ?dtype, ?device, ?backend) = a.randintLike(low=low, high=high, ?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member zeroLike(a:Tensor, ?dtype, ?device, ?backend) = a.zeroLike(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member oneLike(a:Tensor, ?dtype, ?device, ?backend) = a.oneLike(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member nelement(a:Tensor) = a.nelement

    /// <summary>TBD</summary>
    static member like(a:Tensor, value:obj, ?dtype, ?device, ?backend) = a.like(value, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member clone(a:Tensor) = a.clone()

    /// <summary>TBD</summary>
    static member lt(a:Tensor, b:Tensor) = a.lt(b)

    /// <summary>TBD</summary>
    static member gt(a:Tensor, b:Tensor) = a.gt(b)

    /// <summary>TBD</summary>
    static member le(a:Tensor, b:Tensor) = a.le(b)

    /// <summary>TBD</summary>
    static member ge(a:Tensor, b:Tensor) = a.ge(b)

    /// <summary>TBD</summary>
    static member isinf(a:Tensor) = a.isinf()

    /// <summary>TBD</summary>
    static member isnan(a:Tensor) = a.isnan()

    /// <summary>TBD</summary>
    static member hasinf(a:Tensor) = a.hasinf()

    /// <summary>TBD</summary>
    static member hasnan(a:Tensor) = a.hasnan()

    /// <summary>TBD</summary>
    static member argmax(a:Tensor) = a.argmax()

    /// <summary>TBD</summary>
    static member argmin(a:Tensor) = a.argmin()

    /// <summary>TBD</summary>
    static member max(a:Tensor) = a.max()

    /// <summary>TBD</summary>
    static member min(a:Tensor) = a.min()

    /// <summary>TBD</summary>
    static member max(a:Tensor, b:Tensor) = a.max(b)

    /// <summary>TBD</summary>
    static member min(a:Tensor, b:Tensor) = a.min(b)

    /// <summary>TBD</summary>
    static member clamp(a:Tensor, ?low:scalar, ?high:scalar) = a.clamp(?low=low, ?high=high)

    /// <summary>TBD</summary>
    static member normalize(a:Tensor) = a.normalize()

    /// <summary>TBD</summary>
    static member standardize(a:Tensor) = a.standardize()

    /// <summary>TBD</summary>
    static member diagonal(a:Tensor, ?offset:int, ?dim1:int, ?dim2:int) = a.diagonal(?offset=offset, ?dim1=dim1, ?dim2=dim2)

    /// <summary>TBD</summary>
    static member trace(a:Tensor) = a.trace()

    /// <summary>TBD</summary>
    static member expand(a:Tensor, shape:seq<int>) = a.expand(shape)

    /// <summary>TBD</summary>
    static member expandAs(a:Tensor, b:Tensor) = a.expandAs(b)

    /// <summary>TBD</summary>
    static member stack(tensors:seq<Tensor>, ?dim:int) = Tensor.stack(tensors, ?dim=dim)

    /// <summary>TBD</summary>
    static member unstack(a:Tensor, ?dim:int) = a.unstack(?dim=dim)

    /// <summary>TBD</summary>
    static member cat(tensors:seq<Tensor>, ?dim:int) = Tensor.cat(tensors, ?dim=dim)

    /// <summary>TBD</summary>
    static member split(a:Tensor, sizes:seq<int>, ?dim:int) = a.split(sizes, ?dim=dim)

    /// <summary>TBD</summary>
    static member add(a:Tensor, b:Tensor) = a.add(b)

    /// <summary>TBD</summary>
    static member sub(a:Tensor, b:Tensor) = a.sub(b)

    /// <summary>TBD</summary>
    static member mul(a:Tensor, b:Tensor) = a.mul(b)

    /// <summary>TBD</summary>
    static member div(a:Tensor, b:Tensor) = a.div(b)

    /// <summary>TBD</summary>
    static member pow(a:Tensor, b:Tensor) = a.pow(b)

    /// <summary>TBD</summary>
    static member matmul(a:Tensor, b:Tensor) = a.matmul(b)

    /// <summary>TBD</summary>
    static member dot(a:Tensor, b:Tensor) = a.dot(b)

    /// <summary>TBD</summary>
    static member neg(a:Tensor) = a.neg()

    /// <summary>TBD</summary>
    static member sum(a:Tensor) = a.sum()

    /// <summary>TBD</summary>
    static member sum(a:Tensor, dim:int, ?keepDim:bool) = a.sum(dim, ?keepDim=keepDim)

    /// <summary>TBD</summary>
    static member mean(a:Tensor) = a.mean()

    /// <summary>TBD</summary>
    static member mean(a:Tensor, dim:int, ?keepDim:bool) = a.mean(dim, ?keepDim=keepDim)

    /// <summary>TBD</summary>
    static member variance(a:Tensor, ?unbiased:bool) = a.variance(?unbiased=unbiased)

    /// <summary>TBD</summary>
    static member variance(a:Tensor, dim:int, ?keepDim:bool, ?unbiased:bool) = a.variance(dim, ?keepDim=keepDim, ?unbiased=unbiased)

    /// <summary>TBD</summary>
    static member stddev(a:Tensor, ?unbiased:bool) = a.stddev(?unbiased=unbiased)

    /// <summary>TBD</summary>
    static member stddev(a:Tensor, dim:int, ?keepDim:bool, ?unbiased:bool) = a.stddev(dim, ?keepDim=keepDim, ?unbiased=unbiased)

    /// <summary>TBD</summary>
    static member gather(a:Tensor, dim:int, indices:Tensor) = a.gather(dim, indices)

    /// <summary>TBD</summary>
    static member transpose(a:Tensor, dim0:int, dim1:int) = a.transpose(dim0, dim1)

    /// <summary>TBD</summary>
    static member transpose(a:Tensor) = a.transpose()

    /// <summary>TBD</summary>
    static member squeeze(a:Tensor, ?dim:int) = a.squeeze(?dim=dim)

    /// <summary>TBD</summary>
    static member unsqueeze(a:Tensor, dim:int) = a.unsqueeze(dim)

    /// <summary>TBD</summary>
    static member flip(a:Tensor, dims:seq<int>) = a.flip(dims)

    /// <summary>TBD</summary>
    static member dilate(a:Tensor, dilations:seq<int>) = a.dilate(dilations)

    /// <summary>TBD</summary>
    static member undilate(a:Tensor, dilations:seq<int>) = a.undilate(dilations)

    /// <summary>TBD</summary>
    static member repeat(a:Tensor, dim:int, times:int) = a.repeat(dim, times)

    /// <summary>TBD</summary>
    static member view(a:Tensor, shape:seq<int>) = a.view(shape)

    /// <summary>TBD</summary>
    static member view(a:Tensor, shape:int) = a.view(shape)

    /// <summary>TBD</summary>
    static member viewAs(a:Tensor, b:Tensor) = a.viewAs(b)

    /// <summary>TBD</summary>
    static member flatten(a:Tensor, ?startDim:int, ?endDim:int) = a.flatten(?startDim=startDim, ?endDim=endDim)

    /// <summary>TBD</summary>
    static member sign(a:Tensor) = a.sign()

    /// <summary>TBD</summary>
    static member floor(a:Tensor) = a.floor()

    /// <summary>TBD</summary>
    static member ceil(a:Tensor) = a.ceil()

    /// <summary>TBD</summary>
    static member round(a:Tensor) = a.round()

    /// <summary>TBD</summary>
    static member abs(a:Tensor) = a.abs()

    /// <summary>TBD</summary>
    static member relu(a:Tensor) = a.relu()

    /// <summary>TBD</summary>
    static member leakyRelu(a:Tensor, ?negativeSlope:float) = a.leakyRelu(?negativeSlope=negativeSlope)

    /// <summary>TBD</summary>
    static member sigmoid(a:Tensor) = a.sigmoid()

    /// <summary>TBD</summary>
    static member softplus(a:Tensor) = a.softplus()

    /// <summary>TBD</summary>
    static member exp(a:Tensor) = a.exp()

    /// <summary>TBD</summary>
    static member log(a:Tensor) = a.log()

    /// <summary>TBD</summary>
    static member log10(a:Tensor) = a.log10()

    /// <summary>TBD</summary>
    static member sqrt(a:Tensor) = a.sqrt()

    /// <summary>TBD</summary>
    static member sin(a:Tensor) = a.sin()

    /// <summary>TBD</summary>
    static member cos(a:Tensor) = a.cos()

    /// <summary>TBD</summary>
    static member tan(a:Tensor) = a.tan()

    /// <summary>TBD</summary>
    static member sinh(a:Tensor) = a.sinh()

    /// <summary>TBD</summary>
    static member cosh(a:Tensor) = a.cosh()

    /// <summary>TBD</summary>
    static member tanh(a:Tensor) = a.tanh()

    /// <summary>TBD</summary>
    static member asin(a:Tensor) = a.asin()

    /// <summary>TBD</summary>
    static member acos(a:Tensor) = a.acos()
    
    /// <summary>TBD</summary>
    static member atan(a:Tensor) = a.atan()

    /// <summary>TBD</summary>

    /// <summary>TBD</summary>
    static member softmax(a:Tensor, dim:int) = a.softmax(dim)

    /// <summary>TBD</summary>
    static member logsoftmax(a:Tensor, dim:int) = a.logsoftmax(dim)

    /// <summary>TBD</summary>
    static member logsumexp(a:Tensor, dim:int, ?keepDim:bool) = a.logsumexp(dim, ?keepDim=keepDim)

    /// <summary>TBD</summary>
    static member mseLoss(input:Tensor, target:Tensor, ?reduction:string) = input.mseLoss(target, ?reduction=reduction)

    /// <summary>TBD</summary>
    static member bceLoss(input:Tensor, target:Tensor, ?weight:Tensor, ?reduction:string) = input.bceLoss(target, ?weight=weight, ?reduction=reduction)

    /// <summary>TBD</summary>
    static member nllLoss(input:Tensor, target:Tensor, ?weight:Tensor, ?reduction:string) = input.nllLoss(target, ?weight=weight, ?reduction=reduction)

    /// <summary>TBD</summary>
    static member crossEntropyLoss(input:Tensor, target:Tensor, ?weight:Tensor, ?reduction:string) = input.crossEntropyLoss(target, ?weight=weight, ?reduction=reduction)

    /// <summary>TBD</summary>
    static member maxpool1d(a:Tensor, kernelSize:int, ?stride:int, ?padding:int) = a.maxpool1d(kernelSize, ?stride=stride, ?padding=padding)

    /// <summary>TBD</summary>
    static member maxpool1di(a:Tensor, kernelSize:int, ?stride:int, ?padding:int) = a.maxpool1di(kernelSize, ?stride=stride, ?padding=padding)

    /// <summary>TBD</summary>
    static member maxpool2d(a:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool2d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>TBD</summary>
    static member maxpool2di(a:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool2di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>TBD</summary>
    static member maxpool3d(a:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool3d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>TBD</summary>
    static member maxpool3di(a:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = a.maxpool3di(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>TBD</summary>
    static member maxunpool1d(a:Tensor, indices:Tensor, kernelSize:int, ?stride:int, ?padding:int, ?outputSize:seq<int>) = a.maxunpool1d(indices, kernelSize, ?stride=stride, ?padding=padding, ?outputSize=outputSize)

    /// <summary>TBD</summary>
    static member maxunpool2d(a:Tensor, indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) = a.maxunpool2d(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, ?outputSize=outputSize)

    /// <summary>TBD</summary>
    static member maxunpool3d(a:Tensor, indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) = a.maxunpool3d(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, ?outputSize=outputSize)

    /// <summary>TBD</summary>
    static member conv1d(a:Tensor, b:Tensor, ?stride:int, ?padding:int, ?dilation:int) = a.conv1d(b, ?stride=stride, ?padding=padding, ?dilation=dilation)

    /// <summary>TBD</summary>
    static member conv2d(a:Tensor, b:Tensor, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>) = a.conv2d(b, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

    /// <summary>TBD</summary>
    static member conv3d(a:Tensor, b:Tensor, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>) = a.conv3d(b, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

    /// <summary>TBD</summary>
    static member convTranspose1d(a:Tensor, b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int) = a.convTranspose1d(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

    /// <summary>TBD</summary>
    static member convTranspose2d(a:Tensor, b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) = a.convTranspose2d(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

    /// <summary>TBD</summary>
    static member convTranspose3d(a:Tensor, b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) = a.convTranspose3d(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

    /// <summary>TBD</summary>
    static member pad(a:Tensor, paddings:seq<int>) = a.pad(paddings)

    /// <summary>TBD</summary>
    static member toImage(a:Tensor, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int) = a.toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)

    /// <summary>TBD</summary>
    static member toImageString(a:Tensor, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int, ?asciiPalette:string) = a.toImageString(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols, ?asciiPalette=asciiPalette)

    /// <summary>TBD</summary>
    static member loadImage(fileName:string, ?normalize:bool, ?dtype: Dtype, ?device: Device, ?backend: Backend) = Tensor.loadImage(fileName=fileName, ?normalize=normalize, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member saveImage(a:Tensor, fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?gridCols:int) = a.saveImage(fileName=fileName, ?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?gridCols=gridCols)

    /// <summary>TBD</summary>
    static member cast(a:Tensor, dtype:Dtype) = a.cast(dtype)

    /// <summary>TBD</summary>
    static member move(a:Tensor, ?dtype, ?device, ?backend) = a.move(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member config(?dtype: Dtype, ?device: Device, ?backend: Backend) = 
        dtype |> Option.iter (fun d -> Dtype.Default <- d)
        device |> Option.iter (fun d -> Device.Default <- d)
        backend |> Option.iter (fun d -> Backend.Default <- d)
        dsharp.tensor(0.) |> ignore // We need this to ensure the backend assemblies are loaded and backend is ready to set the random seed immediately after config

    /// <summary>TBD</summary>
    static member config() = Dtype.Default, Device.Default, Backend.Default

    /// <summary>TBD</summary>
    static member config((dtype,device,backend)) = dsharp.config(dtype, device, backend)

    /// <summary>TBD</summary>
    static member devices(?deviceType, ?backend) = BackendTensorStatics.Get(?backend=backend).GetDevices(?deviceType=deviceType)

    /// <summary>TBD</summary>
    static member isDeviceTypeSupported(deviceType, ?backend) = BackendTensorStatics.Get(?backend=backend).IsDeviceTypeSupported(deviceType)

    /// <summary>TBD</summary>
    static member isCudaSupported(?backend) = BackendTensorStatics.Get(?backend=backend).IsDeviceTypeSupported(DeviceType.CUDA)


// Differentiable methods mirroring F# collection modules
// TODO: implement more differentiable higher-order functions and corresponding unit tests for their derivatives
type dsharp with

    /// <summary>TBD</summary>
    static member init (count:int) (initializer:int->'a) = Array.init count initializer |> dsharp.tensor

    /// <summary>TBD</summary>
    static member init2d (length1:int) (length2:int) (initializer:int->int->'a) = Array2D.init length1 length2 initializer |> dsharp.tensor

    /// <summary>TBD</summary>
    static member init3d (length1:int) (length2:int) (length3:int) (initializer:int->int->int->'a) = Array3D.init length1 length2 length3 initializer |> dsharp.tensor

    /// <summary>TBD</summary>
    static member init4d (length1:int) (length2:int) (length3:int) (length4:int) (initializer:int->int->int->int->'a) = Array4D.init length1 length2 length3 length4 initializer |> dsharp.tensor

    /// <summary>TBD</summary>
    static member create (count:int) (value:'a) = Array.create count value |> dsharp.tensor

    /// <summary>TBD</summary>
    static member zeroCreate (count:int) = Array.zeroCreate count |> dsharp.tensor

    /// <summary>TBD</summary>
    static member mapi (mapping:int[]->Tensor->Tensor) (tensor:Tensor) = // Differentiable map
        let tflat = tensor.view(-1)
        let items = Array.init (tflat.nelement) (fun i -> mapping (flatIndexToIndex tensor.shape i) tflat.[i])
        dsharp.stack(items).view(tensor.shape)

    /// <summary>TBD</summary>
    static member mapi2 (mapping:int[]->Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) =  // Differentiable map2
        if tensor1.shape <> tensor2.shape then failwithf "Expecting tensor1.shape (%A) and tensor2.shape (%A) to be the same" tensor1.shape tensor2.shape
        let tflat1 = tensor1.view(-1)
        let tflat2 = tensor2.view(-1)
        let items = Array.init (tflat1.nelement) (fun i -> mapping (flatIndexToIndex tensor1.shape i) tflat1.[i] tflat2.[i])
        dsharp.stack(items).view(tensor1.shape)

    /// <summary>TBD</summary>
    static member mapi3 (mapping:int[]->Tensor->Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) (tensor3:Tensor) =  // Differentiable map3
        if (tensor1.shape <> tensor2.shape) || (tensor2.shape <> tensor3.shape) then failwithf "Expecting tensor1.shape (%A), tensor2.shape (%A), tensor3.shape (%A) to be the same" tensor1.shape tensor2.shape tensor3.shape
        let tflat1 = tensor1.view(-1)
        let tflat2 = tensor2.view(-1)
        let tflat3 = tensor3.view(-1)
        let items = Array.init (tflat1.nelement) (fun i -> mapping (flatIndexToIndex tensor1.shape i) tflat1.[i] tflat2.[i] tflat3.[i])
        dsharp.stack(items).view(tensor1.shape)

    /// <summary>TBD</summary>
    static member map (mapping:Tensor->Tensor) (tensor:Tensor) = tensor |> dsharp.mapi (fun _ v -> mapping v)

    /// <summary>TBD</summary>
    static member map2 (mapping:Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) = dsharp.mapi2 (fun _ v1 v2 -> mapping v1 v2) tensor1 tensor2

    /// <summary>TBD</summary>
    static member map3 (mapping:Tensor->Tensor->Tensor->Tensor) (tensor1:Tensor) (tensor2:Tensor) (tensor3:Tensor) = dsharp.mapi3 (fun _ v1 v2 v3 -> mapping v1 v2 v3) tensor1 tensor2 tensor3


// Functional automatic differentiation API
type dsharp with

    /// <summary>TBD</summary>
    static member nest() = GlobalNestingLevel.Next() |> ignore

    /// <summary>TBD</summary>
    static member nest(level) = GlobalNestingLevel.Set(level)

    /// <summary>TBD</summary>
    static member nestLevel() = GlobalNestingLevel.Current

    /// <summary>TBD</summary>
    static member nestReset() = GlobalNestingLevel.Reset()

    /// <summary>TBD</summary>
    static member primal (tensor:Tensor) = tensor.primal

    /// <summary>TBD</summary>
    static member derivative (tensor:Tensor) = tensor.derivative

    /// <summary>TBD</summary>
    static member primalDerivative (tensor:Tensor) = tensor.primal, tensor.derivative

    /// <summary>TBD</summary>
    static member forwardDiff (tag:uint32) (derivative:Tensor) (tensor:Tensor) = tensor.forwardDiff(derivative, tag)

    /// <summary>TBD</summary>
    static member reverseDiff (tag:uint32) (tensor:Tensor) = tensor.reverseDiff(tag)

    /// <summary>TBD</summary>
    static member reverseReset (tensor:Tensor) = tensor.reverseReset(true)

    /// <summary>TBD</summary>
    static member reversePush (value:Tensor) (tensor:Tensor) = tensor.reversePush(value)

    /// <summary>TBD</summary>
    static member reverse (value:Tensor) (tensor:Tensor) = tensor.reverse(value)

    /// <summary>TBD</summary>
    static member evalForwardDiff f x v = x |> dsharp.forwardDiff (GlobalNestingLevel.Next()) v |> f |> dsharp.primalDerivative

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
    static member fjacobianv f (x:Tensor) (v:Tensor) = 
        if x.nelement <> v.nelement then failwithf "x and v must have the same number of elements"
        let fx, d = dsharp.evalForwardDiff f x v
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        fx, d

    /// <summary>TBD</summary>
    static member jacobianv f x v = dsharp.fjacobianv f x v |> snd

    /// <summary>TBD</summary>
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

    /// <summary>TBD</summary>
    static member fjacobianTv f x (v:Tensor) =
        let fx, r = dsharp.evalReverseDiff f x
        if x.dim <> 1 || fx.dim <> 1 then failwithf "f must be a vector-valued function of a vector, encountered f:%A->%A" x.shape fx.shape
        if fx.nelement <> v.nelement then failwithf "(f x) and v must have the same number of elements"
        fx, r v

    /// <summary>TBD</summary>
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
