// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.
namespace DiffSharp

open DiffSharp.Util

/// Represents the shape of a tensor.
type Shape = int[]

/// Contains functions and values related to tensor shapes.
module rec Shape =

    /// Gets the total number of elements in the shape.
    let nelement (shape: Shape) =
        if shape.Length = 0 then 1
        else Array.reduce (*) shape

    /// The shape for a scalar value.
    let scalar : Shape = [| |]

    /// Indicates if one shape contains another.
    let contains (bigShape:Shape) (smallShape: Shape) =
        if bigShape.Length <> smallShape.Length then failwithf "Expecting bigShape (%A) and smallShape (%A) to have the same number of dimensions" bigShape.Length smallShape.Length
        Array.map2 (<=) smallShape bigShape |> Array.forall id

    /// Checks if the given shapes are appropriate for a stack operation and returns information related to the resulting shape.
    let checkCanStack (shapes:Shape[]) (dim: int) =
        if not (Seq.allEqual shapes) then failwithf "Cannot stack tensors with different shapes: %A" shapes
        let n = shapes.Length
        if n = 0 then failwithf "Expecting a non-empty sequence of tensors"
        let shape = shapes[0]
        if dim < 0 || dim > shape.Length then failwithf "Expecting 0 <= dim (%A) <= %A" dim shape.Length
        if dim < 0 || dim > n then failwithf "Expecting 0 <= dim (%A) <= %A" dim n
        let shape1 = shape[0..dim-1]
        let shape2 = shape[dim..]
        let outputShape = [| yield! shape1; yield n; yield! shape2 |]
        n, shape1, shape2, outputShape

    /// Checks if the given shapes are appropriate for a GetSlice operation and returns information related to the resulting shape.
    let checkCanGetSlice (shape: Shape) (fullBounds: int[,]) =
        if Array2D.length1 fullBounds <> shape.Length then failwithf "Expecting %i-by-3 fullBounds" shape.Length
        let outputShape =
            [|for i=0 to (fullBounds.GetLength(0) - 1) do
                let len = fullBounds[i,1] - fullBounds[i,0] + 1
                if fullBounds[i, 2] = 1 then
                    if len > 1 then yield len // if len=1 then squeeze this dimension
                else
                    yield len|]
        outputShape

    /// Checks if the given index is valid in the context of the given shape.
    let checkCanIndex (shape: int[]) (index: int[]) =
        if shape.Length <> index.Length then failwithf "Expecting shape (%A) and index (%A) to have the same length" shape index
        let valid = Array.forall2 (fun s i -> (i < s) && (i >= 0)) shape index
        if not valid then failwithf "index (%A) is not valid for shape (%A)" index shape

    /// Computes the shape that results from a dilation operation.
    let dilated (shape: Shape) (dilations: int[]) =
        Array.map2 (fun n d -> n + (n - 1) * (d - 1)) shape dilations

    /// Checks if the given shapes are appropriate for a concatenation operation and returns information related to the resulting shape.
    let checkCanCat (shapes: Shape[]) (dim: int) =
        let n = shapes.Length
        if n = 0 then invalidArg "tensors" "Expecting at least one tensor"
        let shape = shapes[0]
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape1 = shape[0..dim-1]
        let shape3 = shape[dim+1..]
        if shapes |> Array.exists (fun shapeOther -> shapeOther[0..dim-1] <> shape1 || shapeOther[dim+1..] <> shape3) then
            invalidArg "tensors" "Expecting tensors with similar shapes"
        let m2 = shapes |> Array.sumBy (fun shape -> shape[dim])
        let outputShape = [| yield! shape1; yield m2; yield! shape3 |]
        n, shape1, m2, shape3, outputShape

    /// Checks if the given shapes are appropriate for a split operation and returns information related to the resulting shape.
    let checkCanSplit (shape: Shape) (sizes: int[]) (dim: int) =
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        if Array.sum sizes <> shape[dim] then invalidArg "sizes" "the sum of sizes must equal the relevant dimension"
        let shape1 = shape[0..dim-1]
        let shape2 = shape[dim+1..]
        let outputShapes = sizes |> Array.map (fun sz -> [| yield! shape1; yield sz; yield! shape2 |])
        outputShapes

    /// Checks if the given shapes are appropriate for an unstack operation and returns information related to the resulting shape.
    let checkCanUnstack (shape: Shape) (dim: int) =
        if shape.Length < 1 then failwith "Cannot unstack scalar Tensor (dim < 1)"
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape1 = shape[0..dim-1]
        let shape2 = shape[dim+1..]
        let outputShape = Array.append shape1 shape2
        shape1, shape2, outputShape

    /// Checks if the given shapes are appropriate for a transpose operation and returns information related to the resulting shape.
    let computeTranspose2d (shape: Shape) =
        let nrows = shape[0]
        let ncols = shape[1]
        let outputShape = [| ncols; nrows |]
        outputShape

    /// Checks if the two device types are equal.
    let checkDeviceTypes (deviceType1: DeviceType) (deviceType2: DeviceType) =
        if deviceType1 <> deviceType2 then failwithf "Expecting input device types %A and %A to be the same" deviceType1 deviceType2

    /// Checks if the two tensor element types are equal.
    let checkDtypes (dtype1: Dtype) (dtype2: Dtype) =
        if dtype1 <> dtype2 then failwithf "Expecting input tensor types %A and %A to be the same" dtype1 dtype2

    /// Check if the tensor element type is appropriate for a convolution operation.
    let private checkConvDType op (dtype: Dtype) =
        match dtype with
        | Dtype.Bool -> opNotSupported op dtype
        | _ -> ()

    /// Checks if the given shapes are appropriate for a convolution operation and returns information related to the resulting shape.
    let checkCanConv1d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1:Shape) (shape2:Shape) (stride: int) (padding: int) (dilation: int) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "conv1d" dtype1
        if shape1.Length <> 3 || shape2.Length <> 3 then failwithf "Expecting two 3d tensors t1, t2 where t1 is input (NxCxI: batchSize x inputChannels x inputLength) and t2 is filters (KxCxF: outputChannels x inputChannels x kernelLength), received tensors with shapes %A, %A" shape1 shape2
        if padding < 0 then failwithf "Expecting padding (%A) >= 0" padding
        if stride < 1 then failwithf "Expecting stride (%A) >= 1" stride
        if dilation < 1 then failwithf "Expecting dilation (%A) >=1" dilation
        let batchSize = shape1[0]
        let inputChannels = shape1[1]
        let inputLength = shape1[2]
        let outputChannels = shape2[0]
        let filtersChannels = shape2[1]
        let kernelLength = shape2[2]
        let inputLengthAfterPadding = inputLength + 2*padding
        if shape2[1] <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        if kernelLength > inputLengthAfterPadding then failwithf "Expecting kernelLength (%A) <= inputLengthAfterPadding (%A)" kernelLength inputLengthAfterPadding
        let outputSize = int (floor (float (inputLengthAfterPadding - kernelLength)/(float stride))) + 1
        let outputShape = [|batchSize; outputChannels; outputSize|]
        batchSize, inputChannels, kernelLength, outputChannels, outputSize, outputShape

    /// Checks if the given shapes are appropriate for a convolution operation and returns information related to the resulting shape.
    let checkCanConv2d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (strides: int[]) (paddings: int[]) (dilations: int[]) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "conv2d" dtype1
        if shape1.Length <> 4 || shape2.Length <> 4 then failwithf "Expecting two 4d tensors t1, t2 where t1 is input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth) and t2 is filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth), received tensors with shapes %A, %A" shape1 shape2
        if strides.Length <> 2 then failwithf "Expecting strides (%A) to be a two-dimensional array" strides
        if paddings.Length <> 2 then failwithf "Expecting paddings (%A) to be a two-dimensional array" paddings
        if dilations.Length <> 2 then failwithf "Expecting dilations (%A) to be a two-dimensional array" dilations
        if paddings[0] < 0 || paddings[1] < 0 then failwithf "Expecting all paddings (%A) >= 0" paddings
        if strides[0] < 1 || strides[1] < 1 then failwithf "Expecting all strides (%A) >= 1" strides
        if dilations[0] < 1 || dilations[1] < 1 then failwithf "Expecting all dilations (%A) >= 1" dilations
        let batchSize = shape1[0]
        let inputChannels = shape1[1]
        let inputHeight = shape1[2]
        let inputWidth = shape1[3]
        let outputChannels = shape2[0]
        let filtersChannels = shape2[1]
        let kernelHeight = shape2[2]
        let kernelWidth = shape2[3]
        let inputHeightAfterPadding = inputHeight + 2*paddings[0]
        let inputWidthAfterPadding = inputWidth + 2*paddings[1]
        if filtersChannels <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        if kernelHeight > inputHeightAfterPadding then failwithf "Expecting kernelHeight (%A) <= inputHeightAfterPadding (%A)" kernelHeight inputHeightAfterPadding
        if kernelWidth > inputWidthAfterPadding then failwithf "Expecting kernelWidth (%A) <= inputWidthAfterPadding (%A)" kernelWidth inputWidthAfterPadding
        let outputHeight = int (floor (float (inputHeightAfterPadding - kernelHeight)/(float strides[0]))) + 1
        let outputWidth = int (floor (float (inputWidthAfterPadding - kernelWidth)/(float strides[1]))) + 1
        let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
        batchSize, inputChannels, (kernelHeight, kernelWidth), (outputChannels, outputHeight, outputWidth), outputShape

    /// Checks if the given shapes are appropriate for a convolution operation and returns information related to the resulting shape.
    let checkCanConv3d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (strides: int[]) (paddings: int[]) (dilations: int[]) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "conv3d" dtype1
        if shape1.Length <> 5 || shape2.Length <> 5 then failwithf "Expecting two 4d tensors t1, t2 where t1 is input, NxCxDxHxW (batchSize x inputChannels x inputDepth x inputHeight x inputWidth) and t2 is filters, KxCxExFxG (outputChannels x inputChannels x kernelDepth x kernelHeight x kernelWidth), received tensors with shapes %A, %A" shape1 shape2
        if strides.Length <> 3 then failwithf "Expecting strides (%A) to be a length-three array" strides
        if paddings.Length <> 3 then failwithf "Expecting paddings (%A) to be a length-three array" paddings
        if dilations.Length <> 3 then failwithf "Expecting dilations (%A) to be a length-three array" dilations
        if paddings[0] < 0 || paddings[1] < 0 || paddings[2] < 0 then failwithf "Expecting all paddings (%A) >= 0" paddings
        if strides[0] < 1 || strides[1] < 1 || strides[2] < 1 then failwithf "Expecting all strides (%A) >= 1" strides
        if dilations[0] < 1 || dilations[1] < 1 || dilations[2] < 1 then failwithf "Expecting all dilations (%A) >= 1" dilations
        let batchSize = shape1[0]
        let inputChannels = shape1[1]
        let inputDepth = shape1[2]
        let inputHeight = shape1[3]
        let inputWidth = shape1[4]
        let outputChannels = shape2[0]
        let filtersChannels = shape2[1]
        let kernelDepth = shape2[2]
        let kernelHeight = shape2[3]
        let kernelWidth = shape2[4]
        let inputDepthAfterPadding = inputDepth + 2*paddings[0]
        let inputHeightAfterPadding = inputHeight + 2*paddings[1]
        let inputWidthAfterPadding = inputWidth + 2*paddings[2]
        if filtersChannels <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        if kernelDepth > inputDepthAfterPadding then failwithf "Expecting kernelDepth (%A) <= inputDepthAfterPadding (%A)" kernelDepth inputDepthAfterPadding
        if kernelHeight > inputHeightAfterPadding then failwithf "Expecting kernelHeight (%A) <= inputHeightAfterPadding (%A)" kernelHeight inputHeightAfterPadding
        if kernelWidth > inputWidthAfterPadding then failwithf "Expecting kernelWidth (%A) <= inputWidthAfterPadding (%A)" kernelWidth inputWidthAfterPadding
        let outputDepth = int (floor (float (inputDepthAfterPadding - kernelDepth)/(float strides[0]))) + 1
        let outputHeight = int (floor (float (inputHeightAfterPadding - kernelHeight)/(float strides[1]))) + 1
        let outputWidth = int (floor (float (inputWidthAfterPadding - kernelWidth)/(float strides[2]))) + 1
        let outputShape = [|batchSize; outputChannels; outputDepth; outputHeight; outputWidth|]
        batchSize, inputChannels, (kernelDepth, kernelHeight, kernelWidth), (outputChannels, outputDepth, outputHeight, outputWidth), outputShape

    /// Checks if the given shapes are appropriate for a transposed convolution operation and returns information related to the resulting shape.
    let checkCanConvTranspose1d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (stride: int) (padding: int) (dilation: int) (outputPadding: int) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "convTranspose1d" dtype1
        if shape1.Length <> 3 || shape2.Length <> 3 then failwithf "Expecting two 3d tensors t1, t2 where t1 is input (NxCxI: batchSize x inputChannels x inputLength) and t2 is filters (KxCxF: outputChannels x inputChannels x kernelLength), received tensors with shapes %A, %A" shape1 shape2
        if padding < 0 then failwithf "Expecting padding (%A) >= 0" padding
        if stride < 1 then failwithf "Expecting stride (%A) >= 1" stride
        if dilation < 1 then failwithf "Expecting dilation (%A) >=1" dilation
        if outputPadding < 0 then failwithf "Expecting outputPadding (%A) >= 0" outputPadding
        let batchSize = shape1[0]
        let inputChannels = shape1[1]
        let inputLength = shape1[2]
        let outputChannels = shape2[1]
        let filtersChannels = shape2[0]
        let kernelLength = shape2[2]
        let kernelShape = [|kernelLength|]
        let kernelShapeAfterDilation = dilated kernelShape [|dilation|]
        let kernelLength = kernelShapeAfterDilation[0]
        if filtersChannels <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        let outputSize = stride * (inputLength - 1) + kernelLength - 2 * padding + outputPadding
        let outputShape = [|batchSize; outputChannels; outputSize|]
        batchSize, inputChannels, kernelLength, outputChannels, outputSize, outputShape

    /// Checks if the given shapes are appropriate for a transposed convolution operation and returns information related to the resulting shape.
    let checkCanConvTranspose2d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (strides: int[]) (paddings: int[]) (dilations: int[]) (outputPaddings: int[]) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "convTranspose2d" dtype1
        if shape1.Length <> 4 || shape2.Length <> 4 then failwithf "Expecting two 4d tensors t1, t2 where t1 is input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth) and t2 is filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth), received tensors with shapes %A, %A" shape1 shape2
        if strides.Length <> 2 then failwithf "Expecting strides (%A) to be a length-two array" strides
        if paddings.Length <> 2 then failwithf "Expecting paddings (%A) to be a length-two array" paddings
        if dilations.Length <> 2 then failwithf "Expecting dilations (%A) to be a length-two array" dilations
        if outputPaddings.Length <> 2 then failwithf "Expecting outputPaddings (%A) to be a length-two array" outputPaddings
        if paddings[0] < 0 || paddings[1] < 0 then failwithf "Expecting all paddings (%A) >= 0" paddings
        if strides[0] < 1 || strides[1] < 1 then failwithf "Expecting all strides (%A) >= 1" strides
        if dilations[0] < 1 || dilations[1] < 1 then failwithf "Expecting all dilations (%A) >= 1" dilations
        if outputPaddings[0] < 0 || outputPaddings[1] < 0 then failwithf "Expecting all outputPaddings (%A) >= 0" outputPaddings
        let batchSize = shape1[0]
        let inputChannels = shape1[1]
        let inputHeight = shape1[2]
        let inputWidth = shape1[3]
        let outputChannels = shape2[1]
        let filtersChannels = shape2[0]
        let kernelHeight = shape2[2]
        let kernelWidth = shape2[3]
        let kernelShape = [|kernelHeight; kernelWidth|]
        let kernelShapeAfterDilation = dilated kernelShape dilations
        let kernelHeight = kernelShapeAfterDilation[0]
        let kernelWidth = kernelShapeAfterDilation[1]
        if filtersChannels <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        let outputHeight = strides[0] * (inputHeight - 1) + kernelHeight - 2 * paddings[0] + outputPaddings[0]
        let outputWidth = strides[1] * (inputWidth - 1) + kernelWidth - 2 * paddings[1] + outputPaddings[1]
        let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
        batchSize, inputChannels, (kernelHeight, kernelWidth), (outputChannels, outputHeight, outputWidth), outputShape

    /// Checks if the given shapes are appropriate for a transposed convolution operation and returns information related to the resulting shape.
    let checkCanConvTranspose3d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (strides: int[]) (paddings: int[]) (dilations: int[]) (outputPaddings: int[]) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "convTranspose3d" dtype1
        if shape1.Length <> 5 || shape2.Length <> 5 then failwithf "Expecting two 4d tensors t1, t2 where t1 is input, NxCxDxHxW (batchSize x inputChannels x inputDepth x inputHeight x inputWidth) and t2 is filters, KxCxExFxG (outputChannels x inputChannels x kernelDepth x kernelHeight x kernelWidth), received tensors with shapes %A, %A" shape1 shape2
        if strides.Length <> 3 then failwithf "Expecting strides (%A) to be a length-three array" strides
        if paddings.Length <> 3 then failwithf "Expecting paddings (%A) to be a length-three array" paddings
        if dilations.Length <> 3 then failwithf "Expecting dilations (%A) to be a length-three array" dilations
        if outputPaddings.Length <> 3 then failwithf "Expecting outputPaddings (%A) to be a length-three array" outputPaddings
        if paddings[0] < 0 || paddings[1] < 0 || paddings[2] < 0 then failwithf "Expecting all paddings (%A) >= 0" paddings
        if strides[0] < 1 || strides[1] < 1 || strides[2] < 1 then failwithf "Expecting all strides (%A) >= 1" strides
        if dilations[0] < 1 || dilations[1] < 1 || dilations[2] < 1 then failwithf "Expecting all dilations (%A) >= 1" dilations
        if outputPaddings[0] < 0 || outputPaddings[1] < 0 || outputPaddings[2] < 0 then failwithf "Expecting all outputPaddings (%A) >= 0" outputPaddings
        let batchSize = shape1[0]
        let inputChannels = shape1[1]
        let inputDepth = shape1[2]
        let inputHeight = shape1[3]
        let inputWidth = shape1[4]
        let outputChannels = shape2[1]
        let filtersChannels = shape2[0]
        let kernelDepth = shape2[2]
        let kernelHeight = shape2[3]
        let kernelWidth = shape2[4]
        let kernelShape = [|kernelDepth; kernelHeight; kernelWidth|]
        let kernelShapeAfterDilation = dilated kernelShape dilations
        let kernelDepth = kernelShapeAfterDilation[0]
        let kernelHeight = kernelShapeAfterDilation[1]
        let kernelWidth = kernelShapeAfterDilation[2]
        if filtersChannels <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        let outputDepth = strides[0] * (inputDepth - 1) + kernelDepth - 2 * paddings[0] + outputPaddings[0]
        let outputHeight = strides[1] * (inputHeight - 1) + kernelHeight - 2 * paddings[1] + outputPaddings[1]
        let outputWidth = strides[2] * (inputWidth - 1) + kernelWidth - 2 * paddings[2] + outputPaddings[2]
        let outputShape = [|batchSize; outputChannels; outputDepth; outputHeight; outputWidth|]
        batchSize, inputChannels, (kernelDepth, kernelHeight, kernelWidth), (outputChannels, outputDepth, outputHeight, outputWidth), outputShape

    /// Checks if the given shapes are appropriate for a maxpool operation and returns information related to the resulting shape.
    let checkCanMaxOrAvgpool1d nm (dtype: Dtype) (shape: Shape) (kernelSize: int) (stride: int) (padding: int) =
        match dtype with
        | Dtype.Bool | Dtype.Integral -> opNotSupported nm dtype
        | _ ->
        if shape.Length <> 3 then failwithf "Expecting a 3d tensor (NxCxL: batchSize x inputChannels x inputLength), received tensor with shape %A" shape
        if kernelSize < 1 then failwithf "Expecting kernelSize (%A) >= 1" kernelSize
        if padding < 0 then failwithf "Expecting padding (%A) >= 0" padding
        if padding > kernelSize/2 then failwithf "Expecting padding (%A) < kernelSize (%A) / 2" padding kernelSize
        if stride < 1 then failwithf "Expecting stride (%A) >= 1" stride
        let batchSize = shape[0]
        let channels = shape[1]
        let inputSize = shape[2]
        let inputLengthAfterPadding = inputSize + 2*padding
        if kernelSize > inputLengthAfterPadding then failwithf "Expecting kernelSize (%A) <= inputLengthAfterPadding (%A)" kernelSize inputLengthAfterPadding
        let outputSize = int (floor (float (inputLengthAfterPadding - kernelSize)/(float stride))) + 1
        let outputShape = [|batchSize; channels; outputSize|]
        batchSize, channels, inputSize, outputSize, outputShape

    /// Checks if the given shapes are appropriate for a maxpool operation and returns information related to the resulting shape.
    let checkCanMaxpool1d dtype shape kernelSize stride padding =
        checkCanMaxOrAvgpool1d "maxpool1d" dtype shape kernelSize stride padding

    /// Checks if the given shapes are appropriate for an avgpool operation and returns information related to the resulting shape.
    let checkCanAvgpool1d dtype shape kernelSize stride padding =
        checkCanMaxOrAvgpool1d "maxpool1d" dtype shape kernelSize stride padding

    /// Checks if the given shapes are appropriate for a maxpool operation and returns information related to the resulting shape.
    let checkCanMaxOrAvgpool2d nm (dtype: Dtype) (shape: Shape) (kernelSize: int[]) (strides: int[]) (paddings: int[]) =
        match dtype with
        | Dtype.Bool | Dtype.Integral -> opNotSupported nm dtype
        | _ ->
        if shape.Length <> 4 then failwithf "Expecting a 4d tensor (NxCxHxW: batchSize x inputChannels x inputHeight x inputWidth), received tensor with shape %A" shape
        if kernelSize[0] < 1 || kernelSize[1] < 1 then failwithf "Expecting all kernelSizes (%A) >= 1" kernelSize
        if paddings[0] < 0 || paddings[1] < 0 then failwithf "Expecting all paddings (%A) >= 0" paddings
        if paddings[0] > kernelSize[0]/2 || paddings[1] > kernelSize[1]/2 then failwithf "Expecting all paddings (%A) < kernelSizes (%A) / 2" paddings kernelSize
        if strides[0] < 1 || strides[1] < 1 then failwithf "Expecting all strides (%A) >= 1" strides
        let batchSize = shape[0]
        let channels = shape[1]
        let inputHeight = shape[2]
        let inputWidth = shape[3]
        let kernelHeight = kernelSize[0]
        let kernelWidth = kernelSize[1]
        let inputHeightAfterPadding = inputHeight + 2*paddings[0]
        let inputWidthAfterPadding = inputWidth + 2*paddings[1]
        if kernelSize[0] > inputHeightAfterPadding then failwithf "Expecting kernelSize[0] (%A) <= inputHeightAfterPadding (%A)" kernelSize[0] inputHeightAfterPadding
        if kernelSize[1] > inputWidthAfterPadding then failwithf "Expecting kernelSize[1] (%A) <= inputWidthAfterPadding (%A)" kernelSize[1] inputWidthAfterPadding
        let outputHeight = int (floor (float (inputHeightAfterPadding - kernelHeight)/(float strides[0]))) + 1
        let outputWidth = int (floor (float (inputWidthAfterPadding - kernelWidth)/(float strides[1]))) + 1
        let outputShape = [|batchSize; channels; outputHeight; outputWidth|]
        (batchSize, channels, (inputHeight, inputWidth), (kernelHeight, kernelWidth), (outputHeight, outputWidth), outputShape)

    /// Checks if the given shapes are appropriate for a maxpool operation and returns information related to the resulting shape.
    let checkCanMaxpool2d dtype shape kernelSize strides paddings =
        checkCanMaxOrAvgpool2d "maxpool2d" dtype shape kernelSize strides paddings

    /// Checks if the given shapes are appropriate for an avgpool operation and returns information related to the resulting shape.
    let checkCanAvgpool2d dtype shape kernelSize strides paddings =
        checkCanMaxOrAvgpool2d "avgpool2d" dtype shape kernelSize strides paddings

    /// Checks if the given shapes are appropriate for a maxpool operation and returns information related to the resulting shape.
    let checkCanMaxOrAvgpool3d nm (dtype: Dtype) (shape: Shape) (kernelSize: int[]) (strides: int[]) (paddings: int[]) =
        match dtype with
        | Dtype.Bool | Dtype.Integral -> opNotSupported nm dtype
        | _ ->
        if shape.Length <> 5 then failwithf "Expecting a 5d tensor (NxCxDxHxW: batchSize x inputChannels x inputDepth x inputHeight x inputWidth), received tensor with shape %A" shape
        if kernelSize[0] < 1 || kernelSize[1] < 1 || kernelSize[2] < 1 then failwithf "Expecting all kernelSizes (%A) >= 1" kernelSize
        if paddings[0] < 0 || paddings[1] < 0 || paddings[2] < 0 then failwithf "Expecting all paddings (%A) >= 0" paddings
        if paddings[0] > kernelSize[0]/2 || paddings[1] > kernelSize[1]/2 || paddings[2] > kernelSize[2]/2 then failwithf "Expecting all paddings (%A) < kernelSizes (%A) / 2" paddings kernelSize
        if strides[0] < 1 || strides[1] < 1 || strides[2] < 1 then failwithf "Expecting all strides (%A) >= 1" strides
        let batchSize = shape[0]
        let channels = shape[1]
        let inputDepth = shape[2]
        let inputHeight = shape[3]
        let inputWidth = shape[4]
        let kernelDepth = kernelSize[0]
        let kernelHeight = kernelSize[1]
        let kernelWidth = kernelSize[2]
        let inputDepthAfterPadding = inputDepth + 2*paddings[0]
        let inputHeightAfterPadding = inputHeight + 2*paddings[1]
        let inputWidthAfterPadding = inputWidth + 2*paddings[2]
        if kernelSize[0] > inputDepthAfterPadding then failwithf "Expecting kernelSize[0] (%A) <= inputDepthAfterPadding (%A)" kernelSize[0] inputDepthAfterPadding
        if kernelSize[1] > inputHeightAfterPadding then failwithf "Expecting kernelSize[1] (%A) <= inputHeightAfterPadding (%A)" kernelSize[1] inputHeightAfterPadding
        if kernelSize[2] > inputWidthAfterPadding then failwithf "Expecting kernelSize[1] (%A) <= inputWidthAfterPadding (%A)" kernelSize[1] inputWidthAfterPadding
        let outputDepth = int (floor (float (inputDepthAfterPadding - kernelDepth)/(float strides[0]))) + 1
        let outputHeight = int (floor (float (inputHeightAfterPadding - kernelHeight)/(float strides[1]))) + 1
        let outputWidth = int (floor (float (inputWidthAfterPadding - kernelWidth)/(float strides[2]))) + 1
        let outputShape = [|batchSize; channels; outputDepth; outputHeight; outputWidth|]
        (batchSize, channels, (inputDepth, inputHeight, inputWidth), (kernelDepth, kernelHeight, kernelWidth), (outputDepth, outputHeight, outputWidth), outputShape)

    /// Checks if the given shapes are appropriate for a maxpool operation and returns information related to the resulting shape.
    let checkCanMaxpool3d dtype shape kernelSize strides paddings =
        checkCanMaxOrAvgpool3d "maxpool3d" dtype shape kernelSize strides paddings

    /// Checks if the given shapes are appropriate for an avgpool operation and returns information related to the resulting shape.
    let checkCanAvgpool3d dtype shape kernelSize strides paddings =
        checkCanMaxOrAvgpool3d "avgpool3d" dtype shape kernelSize strides paddings

    /// Checks if the given shapes are appropriate for a maxunpool operation and returns information related to the resulting shape.
    let checkCanMaxunpool1d (dtype: Dtype) (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        match dtype with
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxunpool2d" dtype
        | _ ->
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 3 then failwithf "Expecting outputSize (%A) to be 3-dimensional" outputSize
        let batchSize = shape[0]
        let channels = shape[1]
        let inputSize = shape[2]
        if outputSize[0] <> indicesShape[0] || outputSize[1] <> indicesShape[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize[2]|]
        batchSize, channels, inputSize, outputShape

    /// Checks if the given shapes are appropriate for a maxunpool operation and returns information related to the resulting shape.
    let checkCanMaxunpool2d (dtype: Dtype) (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        match dtype with
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxunpool2d" dtype
        | _ ->
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 4 then failwithf "Expecting outputSize (%A) to be 4-dimensional" outputSize
        let batchSize = shape[0]
        let channels = shape[1]
        let inputHeight = shape[2]
        let inputWidth = shape[3]
        if outputSize[0] <> indicesShape[0] || outputSize[1] <> indicesShape[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize[2]; outputSize[3]|]
        batchSize, channels, (inputHeight, inputWidth), outputShape

    /// Checks if the given shapes are appropriate for a maxunpool operation and returns information related to the resulting shape.
    let checkCanMaxunpool3d (dtype: Dtype) (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        match dtype with
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxunpool2d" dtype
        | _ ->
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 5 then failwithf "Expecting outputSize (%A) to be 5-dimensional" outputSize
        let batchSize = shape[0]
        let channels = shape[1]
        let inputDepth = shape[2]
        let inputHeight = shape[3]
        let inputWidth = shape[4]
        if outputSize[0] <> indicesShape[0] || outputSize[1] <> indicesShape[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize[2]; outputSize[3]; outputSize[4]|]
        batchSize, channels, (inputDepth, inputHeight, inputWidth), outputShape

    /// Indicates if one shape can expand into another through the addition of broadcast dimensions.
    let canExpand (oldShape: Shape) (newShape: Shape) =
        newShape.Length >= oldShape.Length &&
        let trim = newShape.Length - oldShape.Length
        newShape[..trim-1] |> Array.forall (fun m -> m >= 1)
            && (oldShape,newShape[trim..]) ||> Array.forall2 (fun n m -> n = 1 || n = m)

    /// Checks if one shape can expand into another through the addition of broadcast dimensions.
    let checkCanExpand (oldShape: Shape) (newShape: Shape) =
        let isOK = canExpand oldShape newShape
        if not isOK then failwithf "can't expand from shape %A to %A - each dimension must either be equal or expand from 1" oldShape newShape

    /// Checks if the given shape is appropriate for a transpose operation and returns information related to the resulting shape.
    let checkCanTranspose (shape: Shape) (dim0: int) (dim1: int) =
        if dim0 < 0 || dim0 >= shape.Length then failwithf "Expecting 0 <= dim0 (%A) < shape.Length (%A)" dim0 shape.Length
        if dim1 < 0 || dim1 >= shape.Length then failwithf "Expecting 0 <= dim1 (%A) < shape.Length (%A)" dim1 shape.Length

    /// Checks if the given shape is appropriate for a transpose operation.
    let checkCanTranspose2d (dim: int) =
        if dim <> 2 then failwith "Expecting dim=2 when no specific dimensions are given to transpose. Consider using general transpose(dim0, dim1)."

    /// Checks if the given shape is appropriate for a transpose operation.
    let checkCanInvert (shape: Shape) =
        let dim = shape.Length
        if not (dim = 2 || dim = 3) then failwith "Expecting 2d tensor (a square matrix) or a 3d tensor (a batch of square matrices)."
        if dim = 2 then if shape[0] <> shape[1] then failwith "Expecting a square matrix"
        if dim = 3 then if shape[1] <> shape[2] then failwith "Expecting square matrices"
    
    /// Checks if the given shape is appropriate for a determinant operation.
    let checkCanDet (shape: Shape) =
        let dim = shape.Length
        if not (dim = 2 || dim = 3) then failwith "Expecting 2d tensor (a square matrix) or a 3d tensor (a batch of square matrices)."
        if dim = 2 then if shape[0] <> shape[1] then failwith "Expecting a square matrix"
        if dim = 3 then if shape[1] <> shape[2] then failwith "Expecting square matrices"
    
    /// Checks if the given shapes are appropriate for a linear solve operation, and returns the resulting shape of the solution
    let checkCanSolve (shapeA: Shape) (shapeB: Shape) =
        let dimA = shapeA.Length
        let dimB = shapeB.Length
        let newShape =
            if dimA = 2 then
                let n = shapeA[0]
                if n <> shapeA[1] then failwithf "Expecting A to be a square matrix, received A with shape %A." shapeA
                if n <> shapeB[0] then failwithf "Expecting A and B to have the same number of rows (1st dimension), received A and B with shapes %A and %A." shapeA shapeB
                if dimB = 1 then
                    // k = 1
                    [|n|]
                elif dimB = 2 then
                    let k = shapeB[1]
                    [|n; k|]
                else
                    failwithf "Expecting B to be a 1d or 2d tensor, received B with shape %A." shapeB
            elif dimA = 3 then 
                let batchSize = shapeA[0]
                if batchSize <> shapeB[0] then failwithf "Expecting A and B to have the same number of batch items (1st dimension), received A and B with shapes %A and %A." shapeA shapeB
                let n = shapeA[1]
                if n <> shapeA[2] then failwithf "Expecting A to be a batch of square matrices, received A with shape %A." shapeA
                if n <> shapeB[1] then failwithf "Expecting the matrices in batches A and B to have the same number of rows items (2nd dimension), received A and B with shapes %A and %A." shapeA shapeB
                if dimB = 2 then
                    // k = 1
                    [|batchSize; n|]
                elif dimB = 3 then
                    let k = shapeB[2]
                    [|batchSize; n; k|]
                else
                    failwithf "Expecting B to be a 2d tensor (batch of vectors) or 3d tensor (a batch of matrices), received B with shape %A." shapeB
            else
                failwithf "Expecting A to be a 2d tensor (a square matrix) or a 3d tensor (a batch of square matrices), received A with shape %A." shapeA
        newShape

    /// Checks if the given shape is appropriate for a permute operation and returns information related to the resulting shape.
    let checkCanPermute (shape: Shape) (permutation: int[]) =
        if shape.Length <> permutation.Length then failwithf "Expecting tensor's shape (%A) and permutation (%A) to have the same dims" shape permutation
        if Seq.hasDuplicates permutation then failwithf "Expecting permutation (%A) to have no duplicate values" permutation
        let inversePermutation = Array.permute (fun i -> permutation[i]) [| 0.. shape.Length-1 |]
        let newShape = Array.permute (fun i -> inversePermutation[i]) shape
        inversePermutation, newShape

    /// Checks if the given shape is appropriate for a flip operation.
    let checkCanFlip (dim: int) (dims: int[]) =
        if dims.Length > dim then failwithf "Expecting dims (list of dimension indices to flip) of length less than Tensor's dimensions, received %A, %A" dims.Length dim
        if Seq.hasDuplicates dims then failwithf "Expecting dims (list of dimension indices to flip) without repetition, received %A" dims
        if (Array.max dims) >= dim then failwithf "Expecting dims (list of dimension indices to flip) where all indices are less than the tensor dimension, received %A, %A" dims dim

    /// Checks if the given shape is appropriate for a repeat operation.
    let checkCanRepeat (shape: Shape) (dim: int) =
        if shape[dim] <> 1 then failwithf "Expecting Tensor's shape (%A) at dim (%A) to be 1" shape dim

    /// Checks if the given shape is appropriate for a dilate operation.
    let checkCanDilate (dim: int) (dilations: int[]) =
        if dilations.Length <> dim then failwithf "Expecting dilations (dilation to use in each dimension) of same length with Tensor's dimensions, received %A, %A" dilations.Length dim
        if (Array.min dilations) < 1 then failwithf "Expecting dilations (dilation to use in each dimension) >= 1 where 1 represents no dilation, received %A" dilations

    /// Checks if the given shape is appropriate for a gather operation.
    let checkCanGather (shape: Shape) (dim: int) (indicesShape: Shape) (indicesDtype:Dtype) =
        if shape.Length <> indicesShape.Length then failwithf "Expecting tensor (%A) and indices (%A) to have the same number of dimensions" shape indicesShape
        if dim < 0 || dim > shape.Length-1 then failwithf "Expecting 0<= dim (%A) < tensor dim (%A)" dim shape.Length
        if indicesShape[dim] < 1 then failwithf "Expecting indices shape at dim %A (%A) >= 1" dim indicesShape[dim]
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32

    /// Checks if the given shape is appropriate for a scatter operation.
    let checkCanScatter (shape: Shape) (dim: int) (indicesShape: Shape) (indicesDtype:Dtype) (destinationShape: Shape)=
        if shape.Length <> indicesShape.Length then failwithf "Expecting tensor (%A) and indices (%A) to have the same number of dimensions" shape indicesShape
        if shape.Length <> destinationShape.Length then failwithf "Expecting tensor (%A) and destination (%A) to have the same number of dimensions" shape destinationShape
        if not (contains shape indicesShape) then failwithf "Expecting tensor shape (%A) to contain indices shape (%A)" shape indicesShape
        if dim < 0 || dim > shape.Length-1 then failwithf "Expecting 0<= dim (%A) < tensor dim (%A)" dim shape.Length
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32

    /// Checks if the given shape is appropriate for a view operation.
    let checkCanView (shape1: Shape) (shape2: Shape) =
        if nelement shape1 <> nelement shape2 then failwithf "Cannot view Tensor of shape %A as shape %A" shape1 shape2

    /// Checks if the given shape is appropriate for a flatten operation.
    let checkCanFlatten (shape: Shape) (startDim: int) (endDim: int) =
        if startDim < 0 || startDim >= shape.Length then failwithf "Expecting 0 <= startDim (%A) < %A" startDim shape.Length
        if endDim < 0 || endDim >= shape.Length then failwithf "Expecting 0 <= endDim (%A) < %A" endDim shape.Length
        if endDim <= startDim then failwithf "Expecting startDim (%A) < endDim (%A)" startDim endDim

    /// Checks if the given shape is appropriate for an addSlice operation.
    let checkCanAddSlice (shape1: Shape) (location: int[]) (shape2: Shape) =
        if not (contains shape1 shape2) then failwithf "Expecting shape1 to contain shape2, received %A, %A" shape1 shape2
        if location.Length <> shape1.Length then failwithf "Expecting location of the same length as shape1, received %A, %A" (location.Length) shape1

    /// Checks if the given shapes are appropriate for a matmul operation.
    let checkCanMatmul (shape1: Shape) (shape2: Shape) =
        if shape1.Length < 2 || shape2.Length < 2 then failwithf "Expecting tensors to have at least two dimensions, received tensors with shapes %A, %A" shape1 shape2
        let aBatchPart, aMatrixPart = Array.splitAt (shape1.Length-2) shape1
        let bBatchPart, bMatrixPart = Array.splitAt (shape2.Length-2) shape2
        if aMatrixPart[1] <> bMatrixPart[0] then failwithf "Cannot matrix multiply tensors with shapes %A, %A - mismatch in matrix dimension" shape1 shape2
        (aBatchPart, aMatrixPart), (bBatchPart, bMatrixPart)

    /// Checks if the given shapes are appropriate for a batched matrix multiplication operation.
    let checkCanBMM (shape1: Shape) (shape2: Shape) =
        if shape1.Length <> 3 || shape2.Length <> 3 then failwithf "Expecting two 3d tensors, received tensors with shapes %A, %A" shape1 shape2
        if shape1[0] <> shape2[0] then failwithf "Cannot batch matrix multiply tensors with shapes %A, %A - mismatch in batch dimension" shape1 shape2
        let batchSize = shape1[0]
        if shape1[2] <> shape2[1] then failwithf "Cannot batch matrix multiply tensors with shapes %A, %A - mismatch in matrix dimension" shape1 shape2
        let outputShape = [|batchSize; shape1[1]; shape2[2]|]
        outputShape

    /// Checks if the given shape is appropriate for a dot product operation.
    let checkCanDot (shape1: Shape) (shape2: Shape) =
        if shape1.Length <> 1 || shape2.Length <> 1 then failwithf "Expecting two vectors (1d Tensors), received tensors with shapes %A, %A" shape1 shape2
        if shape1[0] <> shape2[0] then failwithf "Cannot multiply vectors with different lengths %A, %A" shape1[0] shape2[0]

    /// Checks if the given shape is appropriate for a pad operation.
    let checkCanPad (shape: Shape) (paddings: int[]) =
        if shape.Length <> paddings.Length then failwithf "Expecting shape (%A) and paddings (%A) to have the same length" shape paddings
        if not (paddings |> Array.forall (fun p -> p >= 0)) then failwithf "Expecting all paddings (%A) >= 0" paddings

    /// Checks if the given shape is appropriate for a dropout operation.
    let checkCanDropout (p:double) =
        if p < 0. || p > 1. then failwithf "Expecting 0 <= p <= 1, but received %A" p

    /// Checks if the given shape is appropriate for a dropout2d operation.
    let checkCanDropout2d (shape: Shape) (p:double) =
        checkCanDropout p
        if shape.Length <> 4 then failwithf "Expecting shape (%A) to be 4-dimensional (NxCxHxW: batchSize, inputChannels, inputHeight, inputWidth)" shape

    /// Checks if the given shape is appropriate for a dropout3d operation.
    let checkCanDropout3d (shape: Shape) (p:double) =
        checkCanDropout p
        if shape.Length <> 5 then failwithf "Expecting shape (%A) to be 5-dimensional (NxCxDxHxW: batchSize, inputChannels, inputDepth, inputHeight, inputWidth)" shape

    /// Computes the shape that results from a squeeze operation.
    let squeeze (dim: int) (shape: Shape) =
        if dim = -1 then
            [|for s in shape do if s <> 1 then yield s|]
        elif shape[dim] = 1 then
            [|for i=0 to shape.Length - 1 do
                if i < dim then yield shape[i]
                elif i > dim then yield shape[i]|]
        else
            shape

    let checkCanMinMaxReduce (dim: int) (keepDim: bool) (shape: Shape) =
        if dim >= shape.Length || dim < 0 then failwithf "Expecting dim to be between 0 and %d" shape.Length
        let part1 = shape[..dim-1]
        let part2 = shape[dim+1..]
        [| yield! part1
           if keepDim then yield 1
           yield! part2 |] 

    /// Checks if the given shape is appropriate for an unsqueeze operation and returns the resulting shape.
    let checkCanUnsqueeze (dim: int) (shape: Shape) =
        if dim < 0 || dim > shape.Length then failwithf "Expecting dim in range [0, %A] but received %A" shape.Length dim
        [|for i=0 to shape.Length - 1 + 1 do
            if i < dim then yield shape[i]
            elif i = dim then yield 1
            else yield shape[i-1]|]

    /// Computes the shape that results from an unsqueezeAs operation.
    let unsqueezeAs (shape1: Shape) (shape2: Shape) =
        if shape1.Length > shape2.Length then failwithf "Expecting shape1.Length (%A) <= shape2.Length (%A)" shape1.Length shape2.Length
        let ones = Array.create (shape2.Length - shape1.Length) 1
        Array.append ones shape1

    /// Converts the given location to a three-element bounds array in the context of the given shape.
    let locationToBounds (shape: Shape) (location: int[]) =
        Array2D.init location.Length 3 (fun i j -> if j=0 then location[i] elif j=1 then location[i] + shape[i] - 1 else 0)

    /// Computes the shape that results from a flatten operation.
    let flatten (startDim: int) (endDim: int) (shape: Shape) =
        let shape = [|for i in 0..shape.Length-1 do if (i < startDim) || (i > endDim) then shape[i] else -1|]
        let mutable emitted = false
        [|for s in shape do if s <> -1 then s elif not emitted then emitted <- true; -1|]

    /// Finds the shape into which `shape1` and `shape2` can be expanded.
    let broadcast2 (shape1: Shape) (shape2: Shape) =
        if canExpand shape1 shape2 || canExpand shape2 shape1 then
            let n1 = shape1.Length
            let n2 = shape2.Length
            let mx = max n1 n2
            let mn = mx - min n1 n2
            Array.init mx (fun i ->
                if i < mn then (if n1 > n2 then shape1[i] else shape2[i])
                elif n1 > n2 then max shape1[i] shape2[i-mn]
                else max shape1[i-mn] shape2[i])
        else failwithf "shapes %A and %A are not related by broadcasting - each dimension must either be extra, equal, expand from 1" shape1 shape2

    /// Finds the shape into which all the shapes can be expanded.
    let broadcastShapes (shapes: Shape[]) = Array.reduce broadcast2 shapes

    // /// Computes the shape that results from a pairwise dilation operation.
    // let dilated2 (shape: Shape) (dilations: int[]) =
    //     Array.map2 (*) shape dilations

    /// Computes the shape that results from an undilation operation.
    let undilatedShape (shape: Shape) (dilations: int[]) =
        Array.map2 (fun n d -> (n + d - 1) / d) shape dilations

    /// Completes the given shape with respect to a tensor with the given number of elements.
    let complete (nelement: int) (shape: Shape) =
        if (shape |> Array.filter (fun x -> x < -1) |> Array.length) > 0 then failwithf "Invalid shape %A" shape
        let numUnspecified = shape |> Array.filter ((=) -1) |> Array.length
        if numUnspecified > 1 then
            failwithf "Cannot complete shape %A, expecting at most one unspecified dimension (-1)" shape
        elif numUnspecified = 0 then
            shape
        else
            let divisor = shape |> Array.filter ((<>) -1) |> Shape.nelement
            if nelement % divisor <> 0 then failwithf "Cannot complete shape %A to have %A elements" shape nelement
            let missing = nelement / divisor
            [|for d in shape do if d = -1 then yield missing else yield d|]

    /// Completes the given shape dimension with respect to a concrete dimension.
    let completeDim (dims:int) (dim:int) =
      if dim < -dims || dim >= dims then failwithf "Expecting dim (%A) to be within the range [%A, %A)" dim (-dims) dims
      if dim < 0 then dims+dim
      else dim

    /// Completes the given shape dimension with respect to a concrete dimension, for the unsqueeze operation.
    let completeDimUnsqueeze (dims:int) (dim:int) =
      if dim < (-1 - dims) || dim >= (dims + 1) then failwithf "Expecting dim (%A) to be within the range [%A, %A)" dim (-1 - dims) (dims + 1)
      if dim < 0 then dims + dim + 1
      else dim

    /// Completes the new shape for an expand operation based on the current shape of the tensor.
    let completeExpand (shape: Shape) (newShape: Shape) =
        let trim = newShape.Length - shape.Length
        newShape |> Array.mapi (fun i x -> if i>=trim && x = -1 then shape[i - trim] else x)

    let completeSliceBounds (shape: Shape) (bounds:int[,]) =
        let newBounds = Array2D.init (bounds.GetLength(0)) (bounds.GetLength(1)) 
                            (fun i j -> 
                                if j = 0 || j = 1 then completeDim shape[i] bounds[i, j]
                                else bounds[i, j])
        newBounds

    let inline create (xs: seq<int>) = Seq.toArrayQuick xs

    let resolve2dKernelSizes kernelSize kernelSizes = 
        match kernelSize, kernelSizes with
        | Some _ , Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
        | Some k, None -> [|k; k|]
        | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 2 then failwithf "Expecting kernelSizes to have length two" else k
        | _ -> [|1; 1|]

    let resolve3dKernelSizes kernelSize kernelSizes = 
        match kernelSize, kernelSizes with
        | Some _ , Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
        | Some k, None -> [|k; k; k|]
        | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 3 then failwithf "Expecting kernelSizes to have length three" else k
        | _ -> [|1; 1; 1|]

    let resolve2dConvSizes stride strides padding paddings dilation dilations =
        let strides = 
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 2 then failwithf "Expecting strides to be 2-dimensional" else s
            | _ -> [|1; 1|]
        let paddings = 
            match padding, paddings with
            | Some _ , Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 2 then failwithf "Expecting paddings to be 2-dimensional" else p
            | _ -> [|0; 0|]
        let dilations = 
            match dilation, dilations with
            | Some _ , Some _ -> failwithf "Expecting only one of dilation, dilations"
            | Some d, None -> [|d; d|]
            | None, Some d -> let d = d |> Array.ofSeq in if d.Length <> 2 then failwithf "Expecting dilations to be 2-dimensional" else d
            | _ -> [|1; 1|]
        strides, paddings, dilations

    let resolve3dConvSizes stride strides padding paddings dilation dilations =
        let strides = 
            match stride, strides with
            | Some _ , Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 3 then failwithf "Expecting strides to be 3-dimensional" else s
            | _ -> [|1; 1; 1|]
        let paddings = 
            match padding, paddings with
            | Some _ , Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 3 then failwithf "Expecting paddings to be 3-dimensional" else p
            | _ -> [|0; 0; 0|]
        let dilations = 
            match dilation, dilations with
            | Some _ , Some _ -> failwithf "Expecting only one of dilation, dilations"
            | Some d, None -> [|d; d; d|]
            | None, Some d -> let d = d |> Array.ofSeq in if d.Length <> 3 then failwithf "Expecting dilations to be 3-dimensional" else d
            | _ -> [|1; 1; 1|]
        strides, paddings, dilations

    let resolve2dConvOutputPadding outputPadding outputPaddings =
        match outputPadding, outputPaddings with
        | Some _ , Some _ -> failwithf "Expecting only one of outputPadding, outputPaddings"
        | Some p, None -> [|p; p|]
        | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 2 then failwithf "Expecting outputPaddings to be 2-dimensional" else p
        | _ -> [|0; 0|]

    let resolve3dConvOutputPadding outputPadding outputPaddings =
        match outputPadding, outputPaddings with
        | Some _ , Some _ -> failwithf "Expecting only one of outputPadding, outputPaddings"
        | Some p, None -> [|p; p; p|]
        | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 3 then failwithf "Expecting outputPaddings to be 3-dimensional" else p
        | _ -> [|0; 0; 0|]

    let resolve2dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings =
        let kernelSizes =
            match kernelSize, kernelSizes with
            | Some _, Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
            | Some k, None -> [|k; k|]
            | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 2 then failwithf "Expecting kernelSizes to be 2-dimensional" else k
            | _ -> failwithf "Expecting either kernelSize or kernelSizes"

        let strides =
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 2 then failwithf "Expecting strides to be 2-dimensional" else s
            | _ -> kernelSizes

        let paddings =
            match padding, paddings with
            | Some _, Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 2 then failwithf "Expecting paddings to be 2-dimensional" else p
            | _ -> [|0; 0|]
        kernelSizes, strides, paddings

    let resolve3dMaxPoolSizes kernelSize kernelSizes stride strides padding paddings =
        let kernelSizes =
            match kernelSize, kernelSizes with
            | Some _, Some _ -> failwithf "Expecting only one of kernelSize, kernelSizes"
            | Some k, None -> [|k; k; k|]
            | None, Some k -> let k = k |> Array.ofSeq in if k.Length <> 3 then failwithf "Expecting kernelSizes to be 3-dimensional" else k
            | _ -> failwithf "Expecting either kernelSize or kernelSizes"
        let strides =
            match stride, strides with
            | Some _, Some _ -> failwithf "Expecting only one of stride, strides"
            | Some s, None -> [|s; s; s|]
            | None, Some s -> let s = s |> Array.ofSeq in if s.Length <> 3 then failwithf "Expecting strides to be 3-dimensional" else s
            | _ -> kernelSizes
        let paddings =
            match padding, paddings with
            | Some _, Some _ -> failwithf "Expecting only one of padding, paddings"
            | Some p, None -> [|p; p; p|]
            | None, Some p -> let p = p |> Array.ofSeq in if p.Length <> 3 then failwithf "Expecting paddings to be 3-dimensional" else p
            | _ -> [|0; 0; 0|]
        kernelSizes, strides, paddings


[<AutoOpen>]
module ShapeAutoOpens =

    /// Gets the total number of elements in a shape.
    let shapeLength (shape: Shape) = Shape.nelement shape

    /// Checks if the full bounds is a scalar location
    let boundsIsScalar (bounds: int[,]) =
        let mutable res = true
        for i=0 to bounds.GetLength(0) - 1 do 
            res <- res && bounds[i,2] = 1
        res

    /// Converts the array of three-position bounds specifications to a location.
    let boundsToLocation (bounds: int[,]) =
        [|for i=0 to bounds.GetLength(0) - 1 do yield bounds[i, 0]|]

    /// Converts the array of three-position bounds specifications to a shape without squeezing out scalars
    let boundsToShape (bounds: int[,]) =
        [|for i=0 to bounds.GetLength(0) - 1 do 
             let len = bounds[i, 1] - bounds[i, 0] + 1
             yield len|]

    let shapeToFullBounds (shape: Shape) =
        Array2D.init (shape.Length) 3 (fun i j -> if j=0 then 0 elif j=1 then shape[i]-1 else 0)

    /// Mirrors the coordinates in the given dimensions in the context of the given shape.
    let mirrorCoordinates (coordinates: int[]) (shape: int[]) (mirrorDims: int[]) =
        if coordinates.Length <> shape.Length then failwithf "Expecting coordinates and shape of the same dimension, received %A, %A" coordinates.Length shape.Length
        let result = Array.copy coordinates
        for d=0 to coordinates.Length-1 do
            if mirrorDims |> Array.contains d then
                result[d] <- abs (coordinates[d] - shape[d] + 1)
        result

    /// Dilates the given coordinates.
    let dilatedCoordinates (coordinates: int[]) (dilations: int[]) =
        Array.map2 (*) coordinates dilations

    /// Converts the given index to a flat index in the context of the given shape.
    let indexToFlatIndex (shape: int[]) (index: int[]) =
        Shape.checkCanIndex shape index
        let mutable flatIndex = 0
        for i=0 to index.Length - 1 do
            let v = if i = index.Length - 1 then 1 else (Array.reduce (*) shape[i+1..])
            flatIndex <- flatIndex + index[i] * v
        flatIndex

    /// Converts the given flat index to an index in the context of the given shape.
    let flatIndexToIndex (shape: int[]) (flatIndex: int) =
        let dim = shape.Length
        let nelement = shapeLength shape
        let index = Array.create dim 0
        let mutable mul = nelement
        let mutable fi = flatIndex
        for i=dim downto 1 do
            mul <- mul / shape[dim-i]
            index[i-1] <- fi / mul
            fi <- fi - index[i-1] * mul
        index |> Array.rev
