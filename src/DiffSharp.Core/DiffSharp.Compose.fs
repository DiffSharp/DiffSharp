module DiffSharp.Compose

// Pipelined operations for composing Tensor -> Tensor functions
// The rule for binary operations like add, sub, mul, etc. is simple:
// in the returned function, the functions argument is always taken as the first operand of the binary operation
// For example:
// static member add(b:Tensor) = fun (a:Tensor) -> a.add(b)
// static member sub(b:Tensor) = fun (a:Tensor) -> a.sub(b)

type dsharp with
    /// <summary>Returns a tensor where each row contains <paramref name="numSamples"/> indices sampled from the multinomial probability distribution located in the corresponding row of tensor input.</summary>
    /// <param name="numSamples">Number of samples to draw</param>
    /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
    /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
    /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
    /// <remarks>
    /// Indices are ordered from left to right according to when each was sampled (first samples are placed in first column).
    /// 
    /// If input is a vector, out is a vector of size num_samples.
    /// 
    /// If input is a matrix with m rows, the result is an matrix of shape (m × numSamples)
    /// </remarks>
    static member multinomial(numSamples:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = fun (probs:Tensor) -> probs.multinomial(numSamples, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member bernoulli(?dtype:Dtype, ?device:Device, ?backend:Backend) = fun (probs:Tensor) -> probs.bernoulli(?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member dropout(?p:double) = fun (a:Tensor) -> a.dropout(?p=p)

    /// <summary>TBD</summary>
    static member dropout2d(?p:double) = fun (a:Tensor) -> a.dropout2d(?p=p)

    /// <summary>TBD</summary>
    static member dropout3d(?p:double) = fun (a:Tensor) -> a.dropout3d(?p=p)

    /// <summary>TBD</summary>
    static member zerosLike(shape:seq<int>, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.zerosLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member onesLike(shape:seq<int>, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.onesLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member fullLike(value:scalar, ?shape, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.fullLike(value, ?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member arangeLike(endVal:float, ?startVal:float, ?step:float, ?dtype:Dtype, ?device:Device, ?backend:Backend) = fun (a:Tensor) -> a.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member arangeLike(endVal:int, ?startVal:int, ?step:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = fun (a:Tensor) -> a.arangeLike(endVal=endVal, ?startVal=startVal, ?step=step, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member onehotLike(length:int, hot:int, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.onehotLike(length, hot, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member randLike(shape:seq<int>, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.randLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member randnLike(shape:seq<int>, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.randnLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member randintLike(low:int, high:int, ?shape:seq<int>, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.randintLike(low=low, high=high, ?shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member like(value:obj, ?dtype, ?device, ?backend) = fun (a:Tensor) -> a.like(value, ?dtype=dtype, ?device=device, ?backend=backend)

    /// <summary>TBD</summary>
    static member lt(b:Tensor) = fun (a:Tensor) -> a.lt(b)

    /// <summary>TBD</summary>
    static member gt(b:Tensor) = fun (a:Tensor) -> a.gt(b)

    /// <summary>TBD</summary>
    static member le(b:Tensor) = fun (a:Tensor) -> a.le(b)

    /// <summary>TBD</summary>
    static member ge(b:Tensor) = fun (a:Tensor) -> a.ge(b)

    /// <summary>TBD</summary>
    static member clamp(?low:scalar, ?high:scalar) = fun (a:Tensor) -> a.clamp(?low=low, ?high=high)

    /// <summary>TBD</summary>
    static member diagonal(offset:int, ?dim1:int, ?dim2:int) = fun (a:Tensor) -> a.diagonal(offset=offset, ?dim1=dim1, ?dim2=dim2)

    /// <summary>TBD</summary>
    static member expand(shape:seq<int>) = fun (a:Tensor) -> a.expand(shape)

    /// <summary>TBD</summary>
    static member expandAs(b:Tensor) = fun (a:Tensor) -> a.expandAs(b)

    /// <summary>TBD</summary>
    static member stack(dim:int) = fun (tensors:seq<Tensor>) -> Tensor.stack(tensors, dim=dim)

    /// <summary>TBD</summary>
    static member unstack(dim:int) = fun (a:Tensor) -> a.unstack(dim=dim)

    /// <summary>TBD</summary>
    static member cat(dim:int) = fun (tensors:seq<Tensor>) -> Tensor.cat(tensors, dim=dim)

    /// <summary>TBD</summary>
    static member split(sizes:seq<int>, ?dim:int) = fun (a:Tensor) -> a.split(sizes, ?dim=dim)

    /// <summary>TBD</summary>
    static member add(b:Tensor) = fun (a:Tensor) -> a.add(b)

    /// <summary>TBD</summary>
    static member sub(b:Tensor) = fun (a:Tensor) -> a.sub(b)

    /// <summary>TBD</summary>
    static member mul(b:Tensor) = fun (a:Tensor) -> a.mul(b)

    /// <summary>TBD</summary>
    static member div(b:Tensor) = fun (a:Tensor) -> a.div(b)

    /// <summary>TBD</summary>
    static member pow(b:Tensor) = fun (a:Tensor) -> a.pow(b)

    /// <summary>TBD</summary>
    static member matmul(b:Tensor) = fun (a:Tensor) -> a.matmul(b)

    /// <summary>TBD</summary>
    static member dot(b:Tensor) = fun (a:Tensor) -> a.dot(b)

    /// <summary>TBD</summary>
    static member sum(dim:int, ?keepDim:bool) = fun (a:Tensor) -> a.sum(dim, ?keepDim=keepDim)

    /// <summary>TBD</summary>
    static member mean(dim:int, ?keepDim:bool) = fun (a:Tensor) -> a.mean(dim, ?keepDim=keepDim)

    /// <summary>TBD</summary>
    static member variance(dim:int, ?keepDim:bool, ?unbiased:bool) = fun (a:Tensor) -> a.variance(dim, ?keepDim=keepDim, ?unbiased=unbiased)

    /// <summary>TBD</summary>
    static member stddev(dim:int, ?keepDim:bool, ?unbiased:bool) = fun (a:Tensor) -> a.stddev(dim, ?keepDim=keepDim, ?unbiased=unbiased)

    /// <summary>TBD</summary>
    static member gather(dim:int, indices:Tensor) = fun (a:Tensor) -> a.gather(dim, indices)

    /// <summary>TBD</summary>
    static member transpose(dim0:int, dim1:int) = fun (a:Tensor) -> a.transpose(dim0, dim1)

    /// <summary>TBD</summary>
    static member squeeze(dim:int) = fun (a:Tensor) -> a.squeeze(dim=dim)

    /// <summary>TBD</summary>
    static member unsqueeze(dim:int) = fun (a:Tensor) -> a.unsqueeze(dim)

    /// <summary>TBD</summary>
    static member flip(dims:seq<int>) = fun (a:Tensor) -> a.flip(dims)

    /// <summary>TBD</summary>
    static member dilate(dilations:seq<int>) = fun (a:Tensor) -> a.dilate(dilations)

    /// <summary>TBD</summary>
    static member undilate(dilations:seq<int>) = fun (a:Tensor) -> a.undilate(dilations)

    /// <summary>TBD</summary>
    static member repeat(dim:int, times:int) = fun (a:Tensor) -> a.repeat(dim, times)

    /// <summary>TBD</summary>
    static member view(shape:seq<int>) = fun (a:Tensor) -> a.view(shape)

    /// <summary>TBD</summary>
    static member view(shape:int) = fun (a:Tensor) -> a.view(shape)

    /// <summary>TBD</summary>
    static member viewAs(b:Tensor) = fun (a:Tensor) -> a.viewAs(b)

    /// <summary>TBD</summary>
    static member flatten(startDim:int, ?endDim:int) = fun (a:Tensor) -> a.flatten(startDim=startDim, ?endDim=endDim)

    /// <summary>TBD</summary>
    static member leakyRelu(negativeSlope:float) = fun (a:Tensor) -> a.leakyRelu(negativeSlope=negativeSlope)

    /// <summary>TBD</summary>
    static member softmax(dim:int) = fun (a:Tensor) -> a.softmax(dim)

    /// <summary>TBD</summary>
    static member logsoftmax(dim:int) = fun (a:Tensor) -> a.logsoftmax(dim)

    /// <summary>TBD</summary>
    static member logsumexp(dim:int, ?keepDim:bool) = fun (a:Tensor) -> a.logsumexp(dim, ?keepDim=keepDim)

    /// <summary>TBD</summary>
    static member mseLoss(target:Tensor) = fun (input:Tensor) -> input.mseLoss(target)

    /// <summary>TBD</summary>
    static member bceLoss(target:Tensor) = fun (input:Tensor) -> input.bceLoss(target)

    /// <summary>TBD</summary>
    static member nllLoss(target:Tensor) = fun (input:Tensor) -> input.nllLoss(target)

    /// <summary>TBD</summary>
    static member crossEntropyLoss(target:Tensor) = fun (input:Tensor) -> input.crossEntropyLoss(target)

    /// <summary>TBD</summary>
    static member maxpool1d(kernelSize:int, ?stride:int, ?padding:int) = fun (a:Tensor) -> a.maxpool1d(kernelSize, ?stride=stride, ?padding=padding)

    /// <summary>TBD</summary>
    static member maxpool2d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = fun (a:Tensor) -> a.maxpool2d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>TBD</summary>
    static member maxpool3d(?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>) = fun (a:Tensor) -> a.maxpool3d(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

    /// <summary>TBD</summary>
    static member maxunpool1d(indices:Tensor, kernelSize:int, ?stride:int, ?padding:int, ?outputSize:seq<int>) = fun (a:Tensor) -> a.maxunpool1d(indices, kernelSize, ?stride=stride, ?padding=padding, ?outputSize=outputSize)

    /// <summary>TBD</summary>
    static member maxunpool2d(indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) = fun (a:Tensor) -> a.maxunpool2d(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, ?outputSize=outputSize)

    /// <summary>TBD</summary>
    static member maxunpool3d(indices:Tensor, ?kernelSize:int, ?stride:int, ?padding:int, ?kernelSizes:seq<int>, ?strides:seq<int>, ?paddings:seq<int>, ?outputSize:seq<int>) = fun (a:Tensor) -> a.maxunpool3d(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, ?outputSize=outputSize)

    /// <summary>TBD</summary>
    static member conv1d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int) = fun (a:Tensor) -> a.conv1d(b, ?stride=stride, ?padding=padding, ?dilation=dilation)

    /// <summary>TBD</summary>
    static member conv2d(b:Tensor, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>) = fun (a:Tensor) -> a.conv2d(b, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

    /// <summary>TBD</summary>
    static member conv3d(b:Tensor, ?stride:int, ?strides:seq<int>, ?padding:int, ?paddings:seq<int>, ?dilation:int, ?dilations:seq<int>) = fun (a:Tensor) -> a.conv3d(b, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

    /// <summary>TBD</summary>
    static member convTranspose1d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int) = fun (a:Tensor) -> a.convTranspose1d(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

    /// <summary>TBD</summary>
    static member convTranspose2d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) = fun (a:Tensor) -> a.convTranspose2d(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

    /// <summary>TBD</summary>
    static member convTranspose3d(b:Tensor, ?stride:int, ?padding:int, ?dilation:int, ?outputPadding:int, ?strides:seq<int>, ?paddings:seq<int>, ?dilations:seq<int>, ?outputPaddings:seq<int>) = fun (a:Tensor) -> a.convTranspose3d(b, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

    /// <summary>TBD</summary>
    static member pad(paddings:seq<int>) = fun (a:Tensor) -> a.pad(paddings)

    /// <summary>TBD</summary>
    static member toImage(?pixelMin:double, ?pixelMax:double, ?normalize:bool) = fun (a:Tensor) -> a.toImage(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize)

    /// <summary>TBD</summary>
    static member toImageString(?pixelMin:double, ?pixelMax:double, ?normalize:bool, ?asciiPalette:string) = fun (a:Tensor) -> a.toImageString(?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize, ?asciiPalette=asciiPalette)

    /// <summary>TBD</summary>
    static member saveImage(fileName:string, ?pixelMin:double, ?pixelMax:double, ?normalize:bool) = fun (a:Tensor) -> a.saveImage(fileName=fileName, ?pixelMin=pixelMin, ?pixelMax=pixelMax, ?normalize=normalize)

    /// <summary>TBD</summary>
    static member cast(dtype:Dtype) = fun (a:Tensor) -> a.cast(dtype)

    /// <summary>TBD</summary>
    static member move(?dtype, ?device, ?backend) = fun (a:Tensor) -> a.move(?dtype=dtype, ?device=device, ?backend=backend)
