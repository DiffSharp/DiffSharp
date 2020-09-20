namespace DiffSharp

open DiffSharp.Util

/// Represents the shape of a tensor.
type Shape = int[]

/// Contains functions and values related to tensor shapes.
module Shape =
        
    /// Get the number of dimensions in the shape.
    let length (shape: Shape) =
        if shape.Length = 0 then 1
        else Array.reduce (*) shape

    /// The shape for a scalar value.
    let scalar : Shape = [| |]

    /// Indicates if one shape contains another.
    let contains (bigShape:Shape) (smallShape: Shape) =
        if bigShape.Length <> smallShape.Length then failwithf "Expecting bigShape (%A) and smallShape (%A) to have the same number of dimensions" bigShape.Length smallShape.Length
        Array.map2 (<=) smallShape bigShape |> Array.forall id

    /// Check if the given shapes are appropriate for a stack operation and return information related to the resulting shape.
    let checkCanStack (shapes:Shape[]) (dim: int) =
        if not (Seq.allEqual shapes) then failwith "Cannot stack Tensors with different shapes"
        let n = shapes.Length
        if n = 0 then failwithf "Expecting a non-empty sequence of Tensors"
        let shape = shapes.[0]
        if dim < 0 || dim > shape.Length then invalidArg "dim" "invalid dimension"
        if dim < 0 || dim > n then invalidArg "dim" "invalid dimension"
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim..]
        let outputShape = [| yield! shape1; yield n; yield! shape2 |]
        n, shape1, shape2, outputShape

    /// Check if the given shapes are appropriate for a GetSlice operation and return information related to the resulting shape.
    let checkCanGetSlice (shape: Shape) (fullBounds: int[,]) =
        if Array2D.length1 fullBounds <> shape.Length then failwithf "Expecting %i-by-3 fullBounds" shape.Length
        let outputShape = 
            [|for i=0 to (fullBounds.GetLength(0) - 1) do
                let len = fullBounds.[i,1] - fullBounds.[i,0] + 1
                if fullBounds.[i, 2] = 1 then
                    if len > 1 then yield len // if len=1 then squeeze this dimension
                else
                    yield len|]
        outputShape

    /// Check if the given shapes are appropriate for a concatenation operation and return information related to the resulting shape.
    let checkCanCat (shapes: Shape[]) (dim: int) =
        let n = shapes.Length
        if n = 0 then invalidArg "tensors" "Expecting at least one tensor"
        let shape = shapes.[0]
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape1 = shape.[0..dim-1]
        let shape3 = shape.[dim+1..]
        if shapes |> Array.exists (fun shapeOther -> shapeOther.[0..dim-1] <> shape1 || shapeOther.[dim+1..] <> shape3) then
            invalidArg "tensors" "Expecting Tensors with similar shapes"
        let m2 = shapes |> Array.sumBy (fun shape -> shape.[dim])
        let outputShape = [| yield! shape1; yield m2; yield! shape3 |]
        n, shape1, m2, shape3, outputShape

    /// Check if the given shapes are appropriate for a split operation and return information related to the resulting shape.
    let checkCanSplit (shape: Shape) (sizes: int[]) (dim: int) =
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        if Array.sum sizes <> shape.[dim] then invalidArg "sizes" "the sum of sizes must equal the relevant dimension"
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim+1..]
        let outputShapes = sizes |> Array.map (fun sz -> [| yield! shape1; yield sz; yield! shape2 |])
        outputShapes

    /// Check if the given shapes are appropriate for an unstack operation and return information related to the resulting shape.
    let checkCanUnstack (shape: Shape) (dim: int) =
        if shape.Length < 1 then failwith "Cannot unstack scalar Tensor (dim < 1)"
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim+1..]
        let outputShape = Array.append shape1 shape2
        shape1, shape2, outputShape

    /// Check if the given shapes are appropriate for a transpose operation and return information related to the resulting shape.
    let computeTranspose2d (shape: Shape) =
        let nrows = shape.[0]
        let ncols = shape.[1]
        let outputShape = [| ncols; nrows |]
        outputShape

    /// Check the two device types are equal
    let checkDeviceTypes (deviceType1: DeviceType) (deviceType2: DeviceType) =
        if deviceType1 <> deviceType2 then failwithf "Expecting input device types %A and %A to be the same" deviceType1 deviceType2

    /// Check the two tensor element types are equal
    let checkDtypes (dtype1: Dtype) (dtype2: Dtype) =
        if dtype1 <> dtype2 then failwithf "Expecting input tensor types %A and %A to be the same" dtype1 dtype2

    /// Check the tensor element type is appropriate for a convolution operation
    let private checkConvDType op (dtype: Dtype) =
        match dtype with 
        | Dtype.Bool -> opNotSupported op dtype
        | _ -> ()

    /// Check if the given shapes are appropriate for a convolution operation and return information related to the resulting shape.
    let checkCanConv1d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1:Shape) (shape2:Shape) (stride: int) (padding: int) (dilation: int) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "conv1d" dtype1
        if shape1.Length <> 3 || shape2.Length <> 3 then failwithf "Expecting two 3d tensors t1, t2 where t1 is input (NxCxI: batchSize x inputChannels x inputLength) and t2 is filters (KxCxF: outputChannels x inputChannels x kernelLength), received Tensors with shapes %A, %A" shape1 shape2
        if padding < 0 then failwithf "Expecting padding (%A) >= 0" padding
        if stride < 1 then failwithf "Expecting stride (%A) >= 1" stride
        if dilation < 1 then failwithf "Expecting dilation (%A) >=1" dilation
        let batchSize = shape1.[0]
        let inputChannels = shape1.[1]
        let inputLength = shape1.[2]
        let outputChannels = shape2.[0]
        let filtersChannels = shape2.[1]
        let kernelLength = shape2.[2]
        let inputLengthAfterPadding = inputLength + 2*padding
        if shape2.[1] <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        if kernelLength > inputLengthAfterPadding then failwithf "Expecting kernelLength (%A) <= inputLengthAfterPadding (%A)" kernelLength inputLengthAfterPadding
        let outputSize = int (floor (float (inputLengthAfterPadding - kernelLength)/(float stride))) + 1
        let outputShape = [|batchSize; outputChannels; outputSize|]
        batchSize, inputChannels, kernelLength, outputChannels, outputSize, outputShape

    /// Check if the given shapes are appropriate for a convolution operation and return information related to the resulting shape.
    let checkCanConv2d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (stride: int[]) (padding: int[]) (dilation: int[]) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "conv2d" dtype1
        if shape1.Length <> 4 || shape2.Length <> 4 then failwithf "Expecting two 4d tensors t1, t2 where t1 is input, NxCxHxW (batchSize x inputChannels x inputHeight x inputWidth) and t2 is filters, KxCxFxG (outputChannels x inputChannels x kernelHeight x kernelWidth), received Tensors with shapes %A, %A" shape1 shape2
        if stride.Length <> 2 then failwithf "Expecting stride (%A) to be a two-dimensional array" stride
        if padding.Length <> 2 then failwithf "Expecting padding (%A) to be a two-dimensional array" padding
        if dilation.Length <> 2 then failwithf "Expecting dilation (%A) to be a two-dimensional array" dilation
        if padding.[0] < 0 || padding.[1] < 0 then failwithf "Expecting all paddings (%A) >= 0" padding
        if stride.[0] < 1 || stride.[1] < 1 then failwithf "Expecting all strides (%A) >= 1" stride
        if dilation.[0] < 1 || dilation.[1] < 1 then failwithf "Expecting all dilations (%A) >= 1" dilation
        let batchSize = shape1.[0]
        let inputChannels = shape1.[1]
        let inputHeight = shape1.[2]
        let inputWidth = shape1.[3]
        let outputChannels = shape2.[0]
        let filtersChannels = shape2.[1]
        let kernelHeight = shape2.[2]
        let kernelWidth = shape2.[3]
        let inputHeightAfterPadding = inputHeight + 2*padding.[0]
        let inputWidthAfterPadding = inputWidth + 2*padding.[1]
        if filtersChannels <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        if kernelHeight > inputHeightAfterPadding then failwithf "Expecting kernelHeight (%A) <= inputHeightAfterPadding (%A)" kernelHeight inputHeightAfterPadding
        if kernelWidth > inputWidthAfterPadding then failwithf "Expecting kernelWidth (%A) <= inputWidthAfterPadding (%A)" kernelWidth inputWidthAfterPadding
        let outputHeight = int (floor (float (inputHeightAfterPadding - kernelHeight)/(float stride.[0]))) + 1
        let outputWidth = int (floor (float (inputWidthAfterPadding - kernelWidth)/(float stride.[1]))) + 1
        let outputShape = [|batchSize; outputChannels; outputHeight; outputWidth|]
        batchSize, inputChannels, (kernelHeight, kernelWidth), (outputChannels, outputHeight, outputWidth), outputShape

    /// Check if the given shapes are appropriate for a convolution operation and return information related to the resulting shape.
    let checkCanConv3d (deviceType1: DeviceType) (deviceType2: DeviceType) (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (stride: int[]) (padding: int[]) (dilation: int[]) =
        checkDeviceTypes deviceType1 deviceType2
        checkDtypes dtype1 dtype2
        checkConvDType "conv3d" dtype1
        if shape1.Length <> 5 || shape2.Length <> 5 then failwithf "Expecting two 4d Tensors t1, t2 where t1 is input, NxCxDxHxW (batchSize x inputChannels x inputDepth x inputHeight x inputWidth) and t2 is filters, KxCxExFxG (outputChannels x inputChannels x kernelDepth x kernelHeight x kernelWidth), received Tensors with shapes %A, %A" shape1 shape2
        if stride.Length <> 3 then failwithf "Expecting stride (%A) to be a length-three array" stride
        if padding.Length <> 3 then failwithf "Expecting padding (%A) to be a length-three array" padding
        if dilation.Length <> 3 then failwithf "Expecting dilation (%A) to be a length-three array" dilation
        if padding.[0] < 0 || padding.[1] < 0 || padding.[2] < 0 then failwithf "Expecting all paddings (%A) >= 0" padding
        if stride.[0] < 1 || stride.[1] < 1 || stride.[2] < 1 then failwithf "Expecting all strides (%A) >= 1" stride
        if dilation.[0] < 1 || dilation.[1] < 1 || dilation.[2] < 1 then failwithf "Expecting all dilations (%A) >= 1" dilation
        let batchSize = shape1.[0]
        let inputChannels = shape1.[1]
        let inputDepth = shape1.[2]
        let inputHeight = shape1.[3]
        let inputWidth = shape1.[4]
        let outputChannels = shape2.[0]
        let filtersChannels = shape2.[1]
        let kernelDepth = shape2.[2]
        let kernelHeight = shape2.[3]
        let kernelWidth = shape2.[4]
        let inputDepthAfterPadding = inputDepth + 2*padding.[0]
        let inputHeightAfterPadding = inputHeight + 2*padding.[1]
        let inputWidthAfterPadding = inputWidth + 2*padding.[2]
        if filtersChannels <> inputChannels then failwithf "Input and filters have different number of channels: %A, %A" inputChannels filtersChannels
        if kernelDepth > inputDepthAfterPadding then failwithf "Expecting kernelDepth (%A) <= inputDepthAfterPadding (%A)" kernelDepth inputDepthAfterPadding
        if kernelHeight > inputHeightAfterPadding then failwithf "Expecting kernelHeight (%A) <= inputHeightAfterPadding (%A)" kernelHeight inputHeightAfterPadding
        if kernelWidth > inputWidthAfterPadding then failwithf "Expecting kernelWidth (%A) <= inputWidthAfterPadding (%A)" kernelWidth inputWidthAfterPadding
        let outputDepth = int (floor (float (inputDepthAfterPadding - kernelDepth)/(float stride.[0]))) + 1
        let outputHeight = int (floor (float (inputHeightAfterPadding - kernelHeight)/(float stride.[1]))) + 1
        let outputWidth = int (floor (float (inputWidthAfterPadding - kernelWidth)/(float stride.[2]))) + 1
        let outputShape = [|batchSize; outputChannels; outputDepth; outputHeight; outputWidth|]
        batchSize, inputChannels, (kernelDepth, kernelHeight, kernelWidth), (outputChannels, outputDepth, outputHeight, outputWidth), outputShape

    /// Check if the given shapes are appropriate for a maxpool operation and return information related to the resulting shape.
    let checkCanMaxpool1d (dtype: Dtype) (shape: Shape) (kernelSize: int) (stride: int) (padding: int) =
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxpool1d" dtype
        | _ ->
        if shape.Length <> 3 then failwithf "Expecting a 3d tensor (NxCxL: batchSize x inputChannels x inputLength), received tensor with shape %A" shape
        if kernelSize < 1 then failwithf "Expecting kernelSize (%A) >= 1" kernelSize
        if padding < 0 then failwithf "Expecting padding (%A) >= 0" padding
        if padding > kernelSize/2 then failwithf "Expecting padding (%A) < kernelSize (%A) / 2" padding kernelSize
        if stride < 1 then failwithf "Expecting stride (%A) >= 1" stride
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputSize = shape.[2]
        let inputLengthAfterPadding = inputSize + 2*padding
        if kernelSize > inputLengthAfterPadding then failwithf "Expecting kernelSize (%A) <= inputLengthAfterPadding (%A)" kernelSize inputLengthAfterPadding
        let outputSize = int (floor (float (inputSize + 2*padding - kernelSize)/(float stride))) + 1
        let outputShape = [|batchSize; channels; outputSize|]
        batchSize, channels, inputSize, outputSize, outputShape

    /// Check if the given shapes are appropriate for a maxpool operation and return information related to the resulting shape.
    let checkCanMaxpool2d (dtype: Dtype) (shape: Shape) (kernelSize: int[]) (stride: int[]) (padding: int[]) =
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxpool2d" dtype
        | _ ->
        if shape.Length <> 4 then failwithf "Expecting a 4d tensor (NxCxHxW: batchSize x inputChannels x inputHeight x inputWidth), received tensor with shape %A" shape
        if kernelSize.[0] < 1 || kernelSize.[1] < 1 then failwithf "Expecting all kernelSizes (%A) >= 1" kernelSize
        if padding.[0] < 0 || padding.[1] < 0 then failwithf "Expecting all paddings (%A) >= 0" padding
        if padding.[0] > kernelSize.[0]/2 || padding.[1] > kernelSize.[1]/2 then failwithf "Expecting all paddings (%A) < kernelSizes (%A) / 2" padding kernelSize
        if stride.[0] < 1 || stride.[1] < 1 then failwithf "Expecting all strides (%A) >= 1" stride
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputHeight = shape.[2]
        let inputWidth = shape.[3]
        let kernelHeight = kernelSize.[0]
        let kernelWidth = kernelSize.[1]
        let inputHeightAfterPadding = inputHeight + 2*padding.[0]
        let inputWidthAfterPadding = inputWidth + 2*padding.[1]
        if kernelSize.[0] > inputHeightAfterPadding then failwithf "Expecting kernelSize.[0] (%A) <= inputHeightAfterPadding (%A)" kernelSize.[0] inputHeightAfterPadding
        if kernelSize.[1] > inputWidthAfterPadding then failwithf "Expecting kernelSize.[1] (%A) <= inputWidthAfterPadding (%A)" kernelSize.[1] inputWidthAfterPadding
        let outputHeight = int (floor (float (inputHeight + 2*padding.[0] - kernelHeight)/(float stride.[0]))) + 1
        let outputWidth = int (floor (float (inputWidth + 2*padding.[1] - kernelWidth)/(float stride.[1]))) + 1
        let outputShape = [|batchSize; channels; outputHeight; outputWidth|]
        (batchSize, channels, (inputHeight, inputWidth), (kernelHeight, kernelWidth), (outputHeight, outputWidth), outputShape)

    /// Check if the given shapes are appropriate for a maxpool operation and return information related to the resulting shape.
    let checkCanMaxpool3d (dtype: Dtype) (shape: Shape) (kernelSize: int[]) (stride: int[]) (padding: int[]) =
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxpool3d" dtype
        | _ ->
        if shape.Length <> 5 then failwithf "Expecting a 5d tensor (NxCxDxHxW: batchSize x inputChannels x inputDepth x inputHeight x inputWidth), received tensor with shape %A" shape
        if kernelSize.[0] < 1 || kernelSize.[1] < 1 || kernelSize.[2] < 1 then failwithf "Expecting all kernelSizes (%A) >= 1" kernelSize
        if padding.[0] < 0 || padding.[1] < 0 || padding.[2] < 0 then failwithf "Expecting all paddings (%A) >= 0" padding
        if padding.[0] > kernelSize.[0]/2 || padding.[1] > kernelSize.[1]/2 || padding.[2] > kernelSize.[2]/2 then failwithf "Expecting all paddings (%A) < kernelSizes (%A) / 2" padding kernelSize
        if stride.[0] < 1 || stride.[1] < 1 || stride.[2] < 1 then failwithf "Expecting all strides (%A) >= 1" stride
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputDepth = shape.[2]
        let inputHeight = shape.[3]
        let inputWidth = shape.[4]
        let kernelDepth = kernelSize.[0]
        let kernelHeight = kernelSize.[1]
        let kernelWidth = kernelSize.[2]
        let inputDepthAfterPadding = inputDepth + 2*padding.[0]
        let inputHeightAfterPadding = inputHeight + 2*padding.[1]
        let inputWidthAfterPadding = inputWidth + 2*padding.[2]
        if kernelSize.[0] > inputDepthAfterPadding then failwithf "Expecting kernelSize.[0] (%A) <= inputDepthAfterPadding (%A)" kernelSize.[0] inputDepthAfterPadding
        if kernelSize.[1] > inputHeightAfterPadding then failwithf "Expecting kernelSize.[1] (%A) <= inputHeightAfterPadding (%A)" kernelSize.[1] inputHeightAfterPadding
        if kernelSize.[2] > inputWidthAfterPadding then failwithf "Expecting kernelSize.[1] (%A) <= inputWidthAfterPadding (%A)" kernelSize.[1] inputWidthAfterPadding
        let outputDepth = int (floor (float (inputDepth + 2*padding.[0] - kernelDepth)/(float stride.[0]))) + 1
        let outputHeight = int (floor (float (inputHeight + 2*padding.[1] - kernelHeight)/(float stride.[1]))) + 1
        let outputWidth = int (floor (float (inputWidth + 2*padding.[2] - kernelWidth)/(float stride.[2]))) + 1
        let outputShape = [|batchSize; channels; outputDepth; outputHeight; outputWidth|]
        (batchSize, channels, (inputDepth, inputHeight, inputWidth), (kernelDepth, kernelHeight, kernelWidth), (outputDepth, outputHeight, outputWidth), outputShape)

    /// Check if the given shapes are appropriate for a maxunpool operation and return information related to the resulting shape.
    let checkCanMaxunpool1d (dtype: Dtype) (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxunpool2d" dtype
        | _ ->
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 3 then failwithf "Expecting outputSize (%A) to be 3-dimensional" outputSize
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputSize = shape.[2]
        if outputSize.[0] <> indicesShape.[0] || outputSize.[1] <> indicesShape.[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize.[2]|]
        batchSize, channels, inputSize, outputShape

    /// Check if the given shapes are appropriate for a maxunpool operation and return information related to the resulting shape.
    let checkCanMaxunpool2d (dtype: Dtype) (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxunpool2d" dtype
        | _ ->
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 4 then failwithf "Expecting outputSize (%A) to be 4-dimensional" outputSize
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputHeight = shape.[2]
        let inputWidth = shape.[3]
        if outputSize.[0] <> indicesShape.[0] || outputSize.[1] <> indicesShape.[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize.[2]; outputSize.[3]|]
        batchSize, channels, (inputHeight, inputWidth), outputShape

    /// Check if the given shapes are appropriate for a maxunpool operation and return information related to the resulting shape.
    let checkCanMaxunpool3d (dtype: Dtype) (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        match dtype with 
        | Dtype.Bool | Dtype.Integral -> opNotSupported "maxunpool2d" dtype
        | _ ->
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 5 then failwithf "Expecting outputSize (%A) to be 5-dimensional" outputSize
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputDepth = shape.[2]
        let inputHeight = shape.[3]
        let inputWidth = shape.[4]
        if outputSize.[0] <> indicesShape.[0] || outputSize.[1] <> indicesShape.[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize.[2]; outputSize.[3]; outputSize.[4]|]
        batchSize, channels, (inputDepth, inputHeight, inputWidth), outputShape

    /// Indicates if one shape can expand into another through the addition of single or broadcast dimensions
    let canExpand (oldShape: Shape) (newShape: Shape) =
        newShape.Length >= oldShape.Length &&
        let trim = newShape.Length - oldShape.Length
        newShape.[..trim-1] |> Array.forall (fun m -> m >= 1)
            && (oldShape,newShape.[trim..]) ||> Array.forall2 (fun n m -> n = 1 || n = m)

    /// Check if one shape can expand into another through the addition of single or broadcast dimensions
    let checkCanExpand (oldShape: Shape) (newShape: Shape) =
        let isOK = canExpand oldShape newShape
        if not isOK then failwithf "can't expand from shape %A to %A - each dimension must either be equal or expand from 1" oldShape newShape

    /// Check if the given shape is appropriate for a transpose operation and return information related to the resulting shape.
    let checkCanTranspose (shape: Shape) (dim0: int) (dim1: int) =
        if dim0 < 0 || dim0 >= shape.Length then failwithf "Expecting 0 <= dim0 (%A) < shape.Length (%A)" dim0 shape.Length
        if dim1 < 0 || dim1 >= shape.Length then failwithf "Expecting 0 <= dim1 (%A) < shape.Length (%A)" dim1 shape.Length

    /// Check if the given shape is appropriate for a transpose operation.
    let checkCanTranspose2d (dim: int) =
        if dim <> 2 then failwith "Expecting dim=2 when no specific dimensions are given to transpose. Consider using general transpose(dim0, dim1)."

    /// Check if the given shape is appropriate for a flip operation.
    let checkCanFlip (dim: int) (dims: int[]) =
        if dims.Length > dim then failwithf "Expecting dims (list of dimension indices to flip) of length less than Tensor's dimensions, received %A, %A" dims.Length dim
        if Seq.hasDuplicates dims then failwithf "Expecting dims (list of dimension indices to flip) without repetition, received %A" dims
        if (Array.max dims) >= dim then failwithf "Expecting dims (list of dimension indices to flip) where all indices are less than the tensor dimension, received %A, %A" dims dim

    /// Check if the given shape is appropriate for a repeat operation.
    let checkCanRepeat (shape: Shape) (dim: int) =
        if shape.[dim] <> 1 then failwithf "Expecting Tensor's shape (%A) at dim (%A) to be 1" shape dim

    /// Check if the given shape is appropriate for a dilate operation.
    let checkCanDilate (dim: int) (dilations: int[]) =
        if dilations.Length <> dim then failwithf "Expecting dilations (dilation to use in each dimension) of same length with Tensor's dimensions, received %A, %A" dilations.Length dim
        if (Array.min dilations) < 1 then failwithf "Expecting dilations (dilation to use in each dimension) >= 1 where 1 represents no dilation, received %A" dilations

    /// Check if the given shape is appropriate for a gather operation.
    let checkCanGather (shape: Shape) (dim: int) (indicesShape: Shape) (indicesDtype:Dtype) =
        if shape.Length <> indicesShape.Length then failwithf "Expecting tensorShape (%A) and indicesShape (%A) to have the same number of dimensions" shape indicesShape
        if dim < 0 || dim > shape.Length-1 then failwithf "Expecting 0<= dim (%A) < tensorShape.Length (%A)" dim shape.Length
        if indicesShape.[dim] < 1 then failwithf "Expecting indicesShape.[dim] (%A) >= 1" indicesShape.[dim]
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32

    /// Check if the given shape is appropriate for a gather operation.
    let checkCanView (shape1: Shape) (shape2: Shape) =
        if length shape1 <> length shape2 then failwithf "Cannot view Tensor of shape %A as shape %A" shape1 shape2

    /// Check if the given shape is appropriate for a flatten operation.
    let checkCanFlatten (shape: Shape) (startDim: int) (endDim: int) =
        if startDim < 0 || startDim >= shape.Length then failwithf "Expecting 0 <= startDim (%A) < %A" startDim shape.Length
        if endDim < 0 || endDim >= shape.Length then failwithf "Expecting 0 <= endDim (%A) < %A" endDim shape.Length
        if endDim <= startDim then failwithf "Expecting startDim (%A) < endDim (%A)" startDim endDim

    /// Check if the given shape is appropriate for an addSlice operation.
    let checkCanAddSlice (shape1: Shape) (location: int[]) (shape2: Shape) =
        if not (contains shape1 shape2) then failwithf "Expecting shape1 to contain shape2, received %A, %A" shape1 shape2
        if location.Length <> shape1.Length then failwithf "Expecting location of the same length as shape1, received %A, %A" (location.Length) shape1

    /// Check if the given shape is appropriate for a matmul operation.
    let checkCanMatmul (shape1: Shape) (shape2: Shape) =
        if shape1.Length <> 2 || shape2.Length <> 2 then failwithf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" shape1 shape2
        if shape1.[1] <> shape2.[0] then failwithf "Cannot multiply Tensors with shapes %A, %A" shape1 shape2

    /// Check if the given shape is appropriate for a dot product operation.
    let checkCanDot (shape1: Shape) (shape2: Shape) =
        if shape1.Length <> 1 || shape2.Length <> 1 then failwithf "Expecting two vectors (1d Tensors), received Tensors with shapes %A, %A" shape1 shape2
        if shape1.[0] <> shape2.[0] then failwithf "Cannot multiply vectors with different lengths %A, %A" shape1.[0] shape2.[0]

    /// Check if the given shape is appropriate for a pad operation.
    let checkCanPad (shape: Shape) (paddings: int[]) =
        if shape.Length <> paddings.Length then failwithf "Expecting shape (%A) and paddings (%A) to have the same length" shape paddings
        if not (paddings |> Array.forall (fun p -> p >= 0)) then failwithf "Expecting all paddings (%A) >= 0" paddings

    /// Check if the given shape is appropriate for a dropout operation.
    let checkCanDropout (p:double) =
        if p < 0. || p > 1. then failwithf "Expecting 0 <= p <= 1, but received %A" p

    /// Check if the given shape is appropriate for a dropout2d operation.
    let checkCanDropout2d (shape: Shape) (p:double) =
        checkCanDropout p
        if shape.Length <> 4 then failwithf "Expecting shape (%A) to be 4-dimensional (NxCxHxW: batchSize, inputChannels, inputHeight, inputWidth)" shape

    /// Check if the given shape is appropriate for a dropout3d operation.
    let checkCanDropout3d (shape: Shape) (p:double) =
        checkCanDropout p
        if shape.Length <> 5 then failwithf "Expecting shape (%A) to be 5-dimensional (NxCxDxHxW: batchSize, inputChannels, inputDepth, inputHeight, inputWidth)" shape

    /// Compute the shape that results from a squeeze operation.
    let squeeze (dim: int) (shape: Shape) =
        if dim = -1 then
            [|for s in shape do if s <> 1 then yield s|]
        elif shape.[dim] = 1 then
            [|for i=0 to shape.Length - 1 do 
                if i < dim then yield shape.[i]
                elif i > dim then yield shape.[i]|]
        else
            shape

    /// Compute the shape that results from an unsqueeze operation.
    let checkCanUnsqueeze (dim: int) (shape: Shape) =
        if dim < 0 || dim > shape.Length then failwithf "Expecting dim in range [0, %A] but received %A" shape.Length dim
        [|for i=0 to shape.Length - 1 + 1 do 
            if i < dim then yield shape.[i]
            elif i = dim then yield 1
            else yield shape.[i-1]|]

    /// Compute the shape that results from an unsqueezeAs operation.
    let unsqueezeAs (shape1: Shape) (shape2: Shape) =
        if shape1.Length > shape2.Length then failwithf "Expecting shape1.Length (%A) <= shape2.Length (%A)" shape1.Length shape2.Length
        let ones = Array.create (shape2.Length - shape1.Length) 1
        Array.append ones shape1

    /// Convert the given location to a three-element bounds array in the context of the given shape.
    let locationToBounds (shape: Shape) (location: int[]) =
        Array2D.init location.Length 3 (fun i j -> if j=0 then location.[i] elif j=1 then location.[i] + shape.[i] - 1 else 1)

    /// Compute the shape that results from a flatten operation.
    let flatten (startDim: int) (endDim: int) (shape: Shape) =
        let shape = [|for i in 0..shape.Length-1 do if (i < startDim) || (i > endDim) then shape.[i] else -1|]
        let mutable emitted = false
        [|for s in shape do if s <> -1 then s elif not emitted then emitted <- true; -1|]

    /// Find the shape into which shape1 and shape2 can be expanded
    let broadcast2 (shape1: Shape) (shape2: Shape) =
        if canExpand shape1 shape2 || canExpand shape2 shape1 then 
            let n1 = shape1.Length
            let n2 = shape2.Length
            let mx = max n1 n2
            let mn = mx - min n1 n2
            Array.init mx (fun i -> 
                if i < mn then (if n1 > n2 then shape1.[i] else shape2.[i])
                elif n1 > n2 then max shape1.[i] shape2.[i-mn]
                else max shape1.[i-mn] shape2.[i])
        else failwithf "shapes %A and %A are not related by broadcasting - each dimension must either be extra, equal, expand from 1" shape1 shape2

    /// Find the shape into which all the shapes can be expanded
    let broadcastShapes (shapes: Shape[]) = Array.reduce broadcast2 shapes

    /// Compute the shape that results from a dilation operation.
    let dilated (shape: Shape) (dilations: int[]) =
        Array.map2 (fun n d -> n + (n - 1) * (d - 1)) shape dilations

    /// Compute the shape that results from a pairwise dilation operation.
    let dilated2 (shape: Shape) (dilations: int[]) =
        Array.map2 (*) shape dilations

    /// Compute the shape that results from an undilation operation.
    let undilatedShape (shape: Shape) (dilations: int[]) =
        Array.map2 (fun n d -> (n + d - 1) / d) shape dilations

    /// Complete the given shape with respect to a tensor with the given number of elements.
    let complete (nelement: int) (shape: Shape) =
        if (shape |> Array.filter (fun x -> x < -1) |> Array.length) > 0 then failwithf "Invalid shape %A" shape
        let numUnspecified = shape |> Array.filter ((=) -1) |> Array.length
        if numUnspecified > 1 then
            failwithf "Cannot complete shape %A, expecting at most one unspecified dimension (-1)" shape
        elif numUnspecified = 0 then 
            shape
        else
            let divisor = shape |> Array.filter ((<>) -1) |> length
            if nelement % divisor <> 0 then failwithf "Cannot complete shape %A to have %A elements" shape nelement
            let missing = nelement / divisor
            [|for d in shape do if d = -1 then yield missing else yield d|]

    /// Complete the given shape dimension with respect to a concrete dimension
    let completeDim (dims:int) (dim:int) =
      if dim < -dims || dim >= dims then failwithf "Invalid choice (%A) for dim (%A)" dim dims
      if dim < 0 then dims+dim
      else dim

    let completeExpand (shape: Shape) (newShape: Shape) =
        let trim = newShape.Length - shape.Length
        newShape |> Array.mapi (fun i x -> if i>=trim && x = -1 then shape.[i - trim] else x)


[<AutoOpen>]
module ShapeAutoOpens =

    /// Get the number of dimensions in a shape.
    let shapeLength (shape: Shape) = Shape.length shape

    /// Convert the array of three-position bounds specifications to a location.
    let boundsToLocation (bounds: int[,]) =
        [|for i=0 to bounds.GetLength(0) - 1 do yield bounds.[i, 0]|]

    /// Convert the array of three-position bounds specifications to a shape.
    let boundsToShape (bounds: int[,]) =
        [|for i=0 to bounds.GetLength(0) - 1 do yield bounds.[i, 1] - bounds.[i, 0] + 1|] 

    /// Mirror the coordinates in the given dimensions in the context of the given shape.
    let mirrorCoordinates (coordinates: int[]) (shape: Shape) (mirrorDims: int[]) =
        if coordinates.Length <> shape.Length then failwithf "Expecting coordinates and shape of the same dimension, received %A, %A" coordinates.Length shape.Length
        let result = Array.copy coordinates
        for d=0 to coordinates.Length-1 do
            if mirrorDims |> Array.contains d then
                result.[d] <- abs (coordinates.[d] - shape.[d] + 1)
        result

    /// Dilate the coordinates 
    let dilatedCoordinates (coordinates: int[]) (dilations: int[]) =
        Array.map2 (*) coordinates dilations

    /// Check if the given index is valid in the context of the given shape.
    let checkValidIndex (shape: Shape) (index: int[]) =
        if shape.Length <> index.Length then failwithf "Expecting shape (%A) and index (%A) to have the same length" shape index
        let valid = Array.forall2 (fun s i -> i < s) shape index
        if not valid then failwithf "index (%A) is not valid for shape (%A)" index shape

    /// Convert the given index to a flat index in the context of the given shape.
    let indexToFlatIndex (shape: Shape) (index: int[]) =
        checkValidIndex shape index
        let mutable flatIndex = 0
        for i=0 to index.Length - 1 do
            let v = if i = index.Length - 1 then 1 else (Array.reduce (*) shape.[i+1..])
            flatIndex <- flatIndex + index.[i] * v
        flatIndex

    /// Convert the given flat index to an index in the context of the given shape.
    let flatIndexToIndex (shape: Shape) (flatIndex: int) =
        let dim = shape.Length
        let nelement = shapeLength shape
        let index = Array.create dim 0
        let mutable mul = nelement
        let mutable fi = flatIndex
        for i=dim downto 1 do
            mul <- mul / shape.[dim-i]
            index.[i-1] <- fi / mul
            fi <- fi - index.[i-1] * mul
        index |> Array.rev
    
