namespace DiffSharp.ShapeChecking

open DiffSharp
open DiffSharp.Backends

// Note: this file exposes internal functionality from 'Tensor' as public only
// if you open DiffSharp.ShapeChecking.

[<AutoOpen>]
/// Augments the dsharp and Tensor APIs with inputs accepting potentially-symbolic shapes (Shape) and lengths/indicies (Int)
module ShapedInferenceAutoOpens =

    type Tensor with

        /// <summary>Returns a new view of the object tensor with singleton dimensions expanded to a larger size.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="newShape">The requested shape.</param>
        member a.expand(newShape:Shape) =
            a.expandx(newShape)

        /// <summary>Returns a new view of the object tensor with singleton dimensions expanded to a larger size.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="newShape">The requested shape.</param>
        member a.expand(newShape:seq<Int>) =
            a.expandx(Shape newShape)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        member a.view(shape:Shape) =
            a.viewx(shape)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        member a.view(shape:seq<Int>) =
            a.viewx(Shape shape)

        /// <summary>Dilate the tensor in using the given dilations in each corresponding dimension.</summary>
        /// <param name="dilations">The dilations to use.</param>
        member a.dilate(dilations:seq<Int>) =
            a.dilatex(dilations)

        /// <summary>Reverse the dilation of the tensor in using the given dilations in each corresponding dimension.</summary>
        /// <param name="dilations">The dilations to use.</param>
        member a.undilate(dilations:seq<Int>) =
            a.undilatex(dilations)

        /// <summary>Repeat elements of a tensor</summary>
        /// <param name="dim">The dimension along which to repeat values.</param>
        /// <param name="times">The number of repetitions for each element.</param>
        member a.repeat(dim:int, times:Int) =
            a.repeatx(dim, times)

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        member a.maxpool1d(kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding) |> fst

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        member a.maxpool1di(kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding)

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool2d(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool2di(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool3d(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        member a.maxpool3di(?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Computes a partial inverse of maxpool1di.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="indices">The indices selected by maxpool1di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        member a.maxunpool1d(indices:Tensor, outputSize:seq<Int>, kernelSize:Int, ?stride:Int, ?padding:Int) =
            a.maxunpool1dx(indices, kernelSize, ?stride=stride, ?padding=padding, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool2di.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="indices">The indices selected by maxpool2di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        member a.maxunpool2d(indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxunpool2dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool3di.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="indices">The indices selected by maxpool3di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        member a.maxunpool3d(indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            a.maxunpool3dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Applies a 1D convolution over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit paddings on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        member a.conv1d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int) =
            a.conv1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation)

        /// <summary>Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        member a.convTranspose1d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int) =
            a.convTranspose1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

        /// <summary>Applies a 2D convolution over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        member a.conv2d(filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:Int, ?dilations:seq<Int>) =
            a.conv2dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 2D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
        member a.convTranspose2d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?outputPaddings:seq<Int>) = 
            a.convTranspose2dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        /// <summary>Applies a 3D convolution over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        member a.conv3d(filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:Int, ?dilations:seq<Int>) =
            a.conv3dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 3D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        /// <param name="outputPaddings">The additional sizes added to one side of each dimension in the output shape.</param>
        member a.convTranspose3d(filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?outputPaddings:seq<Int>) =
            a.convTranspose3dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        /// <summary>Add zero padding to each side of each dimension of a tensor. 
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        member a.pad(paddings:seq<Int>) = a.padx(paddings)

        /// <summary>Returns a new tensor filled with '0' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.zerosLike(shape:Shape, ?dtype, ?device, ?backend) = 
            a.zerosLikex(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '0' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.zerosLike(shape:Int[], ?dtype, ?device, ?backend) = 
            a.zerosLikex(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '1' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.onesLike(shape:Shape, ?dtype, ?device, ?backend) = 
            a.onesLikex(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '1' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.onesLike(shape:Int[], ?dtype, ?device, ?backend) = 
            a.onesLikex(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with the given scalar value with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="value">The scalar giving the the initial values for the tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.fullLike(value:scalar, shape:Shape, ?dtype, ?device, ?backend) = 
            a.fullLikex(value, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with the given scalar value with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="value">The scalar giving the the initial values for the tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.fullLike(value:scalar, shape:Int[], ?dtype, ?device, ?backend) = 
            a.fullLikex(value, shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1) with characteristics based on the input tensor
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.randLike(shape:Shape, ?dtype, ?device, ?backend) = 
            a.randLikex(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1) with characteristics based on the input tensor
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.randLike(shape:Int[], ?dtype, ?device, ?backend) = 
            a.randLikex(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.randnLike(shape:Shape, ?dtype, ?device, ?backend) = 
            a.randnLikex(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.randnLike(shape:Int[], ?dtype, ?device, ?backend) = 
            a.randnLikex(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.randintLike(low:int, high:int, shape: Shape, ?dtype, ?device, ?backend) = 
            a.randintLikex(low, high, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.randintLike(low:int, high:int, shape:Int[], ?dtype, ?device, ?backend) = 
            a.randintLikex(low, high, shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>
        /// A version of dsharp.onehot with characteristics based on the input tensor. 
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// 
        /// <param name="length">The length of the returned tensor.</param>
        /// <param name="hot">The location to set to 1.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        member a.onehotLike(length:Int, hot:Int, ?dtype, ?device, ?backend) =
            a.onehotLikex(length, hot, ?dtype=dtype, ?device=device, ?backend=backend)

    type dsharp with

        /// <summary>Returns a new view of the input tensor with singleton dimensions expanded to a larger size.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        static member expand(input:Tensor, shape:seq<Int>) = input.expand(shape)

        /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member empty(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Empty(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member empty(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Empty(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new uninitialized tensor filled with arbitrary values for the given length, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="length">The length of the returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member empty(length:Int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Empty(Shape [| length |], ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member zeros(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Zeros(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member zeros(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Zeros(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '0' values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="length">The length of the returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member zeros(length:Int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Zeros(Shape [| length |], ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member ones(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Ones(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member ones(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Ones(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with '1' values for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="length">The length of the returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member ones(length:Int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Ones(Shape [| length |], ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="value">The .NET object used to form the initial values for the tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member full(shape:Shape, value:scalar, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Full(shape, value, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="value">The .NET object used to form the initial values for the tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member full(shape:seq<Int>, value:scalar, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Full(Shape shape, value, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a new tensor filled with the scalar <paramref name="value" />, for the given shape, element type and configuration.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="length">The length of the returned tensor.</param>
        /// <param name="value">The .NET object used to form the initial values for the tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member full(length:Int, value:scalar, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Full(Shape [| length |], value, ?dtype=dtype, ?device=device, ?backend=backend))

        // /// <summary>TBD</summary>
        // static member eye(rows:int, ?cols:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = Tensor.eye(rows=rows, ?cols=cols, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a one-hot tensor, with one location set to 1, and all others 0.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="length">The length of the returned tensor.</param>
        /// <param name="hot">The location to set to 1.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member onehot(length:Int, hot:Int, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            dsharp.zero(?dtype=dtype, ?device=device, ?backend=backend).onehotLikex(length, hot)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member rand(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Random(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member rand(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.Random(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randn(shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.RandomNormal(shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution).
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randn(shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.RandomNormal(Shape shape, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randint(low:int, high:int, shape:Shape, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.RandomInt(shape, low, high, ?dtype=dtype, ?device=device, ?backend=backend))

        /// <summary>Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, uses Dtype.Default.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, uses Device.Default.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, uses Backend.Default.</param>
        static member randint(low:int, high:int, shape:seq<Int>, ?dtype:Dtype, ?device:Device, ?backend:Backend) =
            TensorC(RawTensor.RandomInt(Shape shape, low, high, ?dtype=dtype, ?device=device, ?backend=backend))

        // /// <summary>TBD</summary>
        // static member multinomial(probs:Tensor, numSamples:int, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.multinomial(numSamples, ?dtype=dtype, ?device=device, ?backend=backend)

        // /// <summary>TBD</summary>
        // static member bernoulli(probs:Tensor, ?dtype:Dtype, ?device:Device, ?backend:Backend) = probs.bernoulli(?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '0' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member zerosLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            input.zerosLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '0' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member zerosLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.zerosLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '1' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member onesLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            input.onesLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with '1' values with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member onesLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.onesLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with the given scalar value with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="value">The scalar giving the the initial values for the tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member fullLike(input:Tensor, value:scalar, shape:Shape, ?dtype, ?device, ?backend) =
            input.fullLike(value, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a new tensor filled with the given scalar value with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="value">The scalar giving the the initial values for the tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member fullLike(input:Tensor, value:scalar, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.fullLike(value, shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

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
        static member onehotLike(input:Tensor, length:Int, hot:Int, ?dtype, ?device, ?backend) =
            input.onehotLikex(length, hot, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1) with characteristics based on the input tensor
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            input.randLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1) with characteristics based on the input tensor
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.randLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randnLike(input:Tensor, shape:Shape, ?dtype, ?device, ?backend) =
            input.randnLike(shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor filled with random numbers from a normal distribution with mean 0 and variance 1 (also called the standard normal distribution) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randnLike(input:Tensor, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.randnLike(shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randintLike(input:Tensor, low:int, high:int, shape:Shape, ?dtype, ?device, ?backend) =
            input.randintLike(low=low, high=high, shape=shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly between low (inclusive) and high (exclusive) with characteristics based on the input tensor.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The shape and characteristics of input will determine those of the output tensor.</param>
        /// <param name="low">Lowest integer to be drawn from the distribution. Default: 0..</param>
        /// <param name="high">One above the highest integer to be drawn from the distribution.</param>
        /// <param name="shape">The desired shape of returned tensor. Default: If None, the shape of the input tensor is used.</param>
        /// <param name="dtype">The desired element type of returned tensor. Default: if None, the element type of the input tensor is used.</param>
        /// <param name="device">The desired device of returned tensor. Default: if None, the device of the input tensor is used.</param>
        /// <param name="backend">The desired backend of returned tensor. Default: if None, the backend of the input tensor is used.</param>
        static member randintLike(input:Tensor, low:int, high:int, shape:seq<Int>, ?dtype, ?device, ?backend) =
            input.randintLike(low=low, high=high, shape=Shape shape, ?dtype=dtype, ?device=device, ?backend=backend)

        /// <summary>Add zero padding to each side of each dimension of a tensor</summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        static member pad(input:Tensor, paddings:seq<Int>) =
            input.pad(paddings)

        /// <summary>Returns the full shape information about the tensor, returning potentially symbolic shape information (Shape and Int).
        /// </summary>
        static member fullshape(input:Tensor) = input.shapex

        /// <summary>Dilate the tensor in using the given dilations in each corresponding dimension.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dilations">The dilations to use.</param>
        static member dilate(input:Tensor, dilations:seq<Int>) = input.dilate(dilations)

        /// <summary>Reverse the dilation of the tensor in using the given dilations in each corresponding dimension.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dilations">The dilations to use.</param>
        static member undilate(input:Tensor, dilations:seq<Int>) = input.undilate(dilations)

        /// <summary>Repeat elements of a tensor. 
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to repeat values.</param>
        /// <param name="times">The number of repetitions for each element.</param>
        static member repeat(input:Tensor, dim:int, times:Int) = input.repeat(dim, times)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        static member view(input:Tensor, shape:Shape) = input.viewx(shape)

        /// <summary>Returns a new tensor with the same data as the self tensor but of a different shape.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The desired shape of returned tensor.</param>
        static member view(input:Tensor, shape:seq<Int>) = input.viewx(Shape shape)

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        static member maxpool1d(input:Tensor, kernelSize:Int, ?stride:Int, ?padding:Int) =
            input.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding) |> fst

        /// <summary>Applies a 1D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        static member maxpool1di(input:Tensor, kernelSize:Int, ?stride:Int, ?padding:Int) =
            input.maxpool1dix(kernelSize, ?stride=stride, ?padding=padding)

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool2d(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 2D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool2di(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool2dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool3d(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings) |> fst

        /// <summary>Applies a 3D max pooling over an input signal composed of several input planes, returning the max indices along with the outputs.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSize.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        static member maxpool3di(input:Tensor, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxpool3dix(?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings)

        /// <summary>Computes a partial inverse of maxpool1di. 
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="indices">The indices selected by maxpool1di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        static member maxunpool1d(input:Tensor, indices:Tensor, outputSize:seq<Int>, kernelSize:Int, ?stride:Int, ?padding:Int) =
            input.maxunpool1dx(indices, kernelSize, ?stride=stride, ?padding=padding, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool2di.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="indices">The indices selected by maxpool2di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        static member maxunpool2d(input:Tensor, indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxunpool2dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Computes a partial inverse of maxpool3di. 
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="indices">The indices selected by maxpool3di.</param>
        /// <param name="kernelSize">The size of the window to take a max over.</param>
        /// <param name="stride">The stride of the window. Default value is kernelSize.</param>
        /// <param name="padding">The implicit zero padding to be added on both sides.</param>
        /// <param name="kernelSizes">The sizes of the window to take a max over.</param>
        /// <param name="strides">The strides of the window. Default value is kernelSizes.</param>
        /// <param name="paddings">The implicit zero paddings to be added on corresponding sides.</param>
        /// <param name="outputSize">The targeted output size.</param>
        static member maxunpool3d(input:Tensor, indices:Tensor, outputSize:seq<Int>, ?kernelSize:Int, ?stride:Int, ?padding:Int, ?kernelSizes:seq<Int>, ?strides:seq<Int>, ?paddings:seq<Int>) =
            input.maxunpool3dx(indices, ?kernelSize=kernelSize, ?stride=stride, ?padding=padding, ?kernelSizes=kernelSizes, ?strides=strides, ?paddings=paddings, outputSize=outputSize)

        /// <summary>Applies a 1D convolution over an input signal composed of several input planes. 
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit paddings on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        static member conv1d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int) =
            input.conv1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation)

        /// <summary>Applies a 2D convolution over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        static member conv2d(input:Tensor, filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:Int, ?dilations:seq<Int>) =
            input.conv2dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 3D convolution over an input signal composed of several input planes.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on corresponding sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="strides">The strides of the convolving kernel.</param>
        /// <param name="paddings">The implicit paddings on corresponding sides of the input.</param>
        /// <param name="dilations">The spacings between kernel elements.</param>
        static member conv3d(input:Tensor, filters:Tensor, ?stride:Int, ?strides:seq<Int>, ?padding:Int, ?paddings:seq<Int>, ?dilation:Int, ?dilations:seq<Int>) =
            input.conv3dx(filters, ?stride=stride, ?strides=strides, ?padding=padding, ?paddings=paddings, ?dilation=dilation, ?dilations=dilations)

        /// <summary>Applies a 1D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="filters">The filters.</param>
        /// <param name="stride">The stride of the convolving kernel.</param>
        /// <param name="padding">The implicit padding on both sides of the input.</param>
        /// <param name="dilation">The spacing between kernel elements.</param>
        /// <param name="outputPadding">The additional size added to one side of each dimension in the output shape.</param>
        static member convTranspose1d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int) =
            input.convTranspose1dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding)

        /// <summary>Applies a 2D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
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
        static member convTranspose2d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?outputPaddings:seq<Int>) = 
            input.convTranspose2dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

        /// <summary>Applies a 3D transposed convolution operator over an input signal composed of several input planes, sometimes also called 'deconvolution'.
        ///   This overload acceps potentially symbolic shape information (Shape and Int).
        /// </summary>
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
        static member convTranspose3d(input:Tensor, filters:Tensor, ?stride:Int, ?padding:Int, ?dilation:Int, ?outputPadding:Int, ?strides:seq<Int>, ?paddings:seq<Int>, ?dilations:seq<Int>, ?outputPaddings:seq<Int>) =
            input.convTranspose3dx(filters, ?stride=stride, ?padding=padding, ?dilation=dilation, ?outputPadding=outputPadding, ?strides=strides, ?paddings=paddings, ?dilations=dilations, ?outputPaddings=outputPaddings)

