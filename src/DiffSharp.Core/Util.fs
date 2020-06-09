module DiffSharp.Util

open System
open System.Net
open System.Collections
open System.Collections.Generic
open System.Diagnostics.CodeAnalysis
open System.IO
open System.IO.Compression
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary
open FSharp.Reflection

let logSqrt2Pi = log(sqrt(2. * Math.PI))
let log10Val = log 10.

type NestingLevel =
    val mutable Current:uint32
    new() = {Current = 0u}
    member t.Next() = t.Current <- t.Current + 1u; t.Current

type GlobalNestingLevel() =
    static let tagger = NestingLevel()
    static member Current = tagger.Current
    static member Next() = tagger.Next()
    static member Reset() = tagger.Current <- 0u
    static member Set(level) = tagger.Current <- level

[<ExcludeFromCodeCoverage>]
let inline cumulativeSum (a:_[]) = (Array.scan (+) LanguagePrimitives.GenericZero a).[1..]

type Random() =
    static let mutable rnd = System.Random()
    static member Seed(seed) = rnd <- System.Random(seed)
    static member Uniform() = rnd.NextDouble()
    static member Uniform(low, high) = low + (rnd.NextDouble() * (high-low))
    static member Normal() =
        let rec normal() = 
            let x, y = (rnd.NextDouble()) * 2.0 - 1.0, (rnd.NextDouble()) * 2.0 - 1.0
            let s = x * x + y * y
            if s > 1.0 then normal() else x * sqrt (-2.0 * (log s) / s)
        normal()
    static member Normal(mean, stddev) = mean + Random.Normal() * stddev
    static member Integer(low, high) = rnd.Next(low, high)
    static member ChoiceIndex(probs:float[]) =
        let probsSum = probs |> Array.sum
        let cumulativeProbs = probs |> Array.map (fun v -> v / probsSum) |> cumulativeSum
        let p = rnd.NextDouble()
        cumulativeProbs |> Array.findIndex (fun v -> v >= p)
    static member Choice(array:_[]) = array.[rnd.Next(array.Length)]
    static member Choice(array:_[], probs:float[]) = 
        if array.Length <> probs.Length then failwith "Expecting array and probs of same length"
        array.[Random.ChoiceIndex(probs)]
    static member Multinomial(probs:float[], numSamples:int) =
        Array.init numSamples (fun _ -> Random.ChoiceIndex(probs)) // Samples with replacement
    static member Multinomial(probs:float[,], numSamples:int) =
        Array2D.init (probs.GetLength(0)) numSamples (fun i _ -> Random.ChoiceIndex(probs.[i,*])) // Samples with replacement
    static member Bernoulli(prob:float) = if rnd.NextDouble() < prob then 1. else 0.
    static member Bernoulli() = Random.Bernoulli(0.5)
    static member Shuffle(array:_[]) =
        // Durstenfeld/Knuth shuffle
        let a = array |> Array.copy
        let mutable n = array.Length
        while n > 1 do
            n <- n - 1
            let i = rnd.Next(n+1)
            let temp = a.[i]
            a.[i] <- a.[n]
            a.[n] <- temp
        a

[<ExcludeFromCodeCoverage>]
let inline notNull value = not (obj.ReferenceEquals(value, null))

let duplicates l =
   l |> List.ofSeq
   |> List.groupBy id
   |> List.choose ( function
          | _, x::_::_ -> Some x
          | _ -> None )

let hasDuplicates l =
    (duplicates l) |> List.isEmpty |> not
        
let allEqual (items:seq<'a>) =
    let item0 = items |> Seq.head
    items |> Seq.forall ((=) item0)

type Shape = int[]

let shapeLength (shape: Shape) =
    if shape.Length = 0 then 1
    else Array.reduce (*) shape

module Shape =
    let scalar : Shape = [| |]

    let contains (bigShape:Shape) (smallShape: Shape) =
        if bigShape.Length <> smallShape.Length then failwithf "Expecting bigShape (%A) and smallShape (%A) to have the same number of dimensions" bigShape.Length smallShape.Length
        Array.map2 (<=) smallShape bigShape |> Array.forall id

    let checkCanStack (shapes:Shape[]) (dim: int) =
        if not (allEqual shapes) then failwith "Cannot stack Tensors with different shapes"
        let n = shapes.Length
        if n = 0 then failwithf "Expecting a non-empty sequence of Tensors"
        let shape = shapes.[0]
        if dim < 0 || dim > shape.Length then invalidArg "dim" "invalid dimension"
        if dim < 0 || dim > n then invalidArg "dim" "invalid dimension"
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim..]
        let outputShape = [| yield! shape1; yield n; yield! shape2 |]
        n, shape1, shape2, outputShape

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

    let checkCanSplit (shape: Shape) (sizes: int[]) (dim: int) =
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        if Array.sum sizes <> shape.[dim] then invalidArg "sizes" "the sum of sizes must equal the relevant dimension"
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim+1..]
        let outputShapes = sizes |> Array.map (fun sz -> [| yield! shape1; yield sz; yield! shape2 |])
        outputShapes

    let checkCanUnstack (shape: Shape) (dim: int) =
        if shape.Length < 1 then failwith "Cannot unstack scalar Tensor (dim < 1)"
        if dim < 0 || dim >= shape.Length then invalidArg "dim" "invalid dimension"
        let shape1 = shape.[0..dim-1]
        let shape2 = shape.[dim+1..]
        let outputShape = Array.append shape1 shape2
        shape1, shape2, outputShape

    let computeTranspose (shape: Shape) =
        let nrows = shape.[0]
        let ncols = shape.[1]
        let outputShape = [| ncols; nrows |]
        outputShape

    let checkCanConv1d (dtype1: Dtype) (dtype2: Dtype) (shape1:Shape) (shape2:Shape) (stride: int) (padding: int) (dilation: int) =
        if dtype1 <> dtype2 then failwithf "Expecting input type %A and weight type %A to be the same" dtype1 dtype2
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

    let checkCanConv2d (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (stride: int[]) (padding: int[]) (dilation: int[]) =
        if dtype1 <> dtype2 then failwithf "Expecting input type %A and weight type %A to be the same" dtype1 dtype2
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

    let checkCanConv3d (dtype1: Dtype) (dtype2: Dtype) (shape1: Shape) (shape2: Shape) (stride: int[]) (padding: int[]) (dilation: int[]) =
        if dtype1 <> dtype2 then failwithf "Expecting input type %A and weight type %A to be the same" dtype1 dtype2
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

    let checkCanMaxpool1d (shape: Shape) (kernelSize: int) (stride: int) (padding: int) =
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

    let checkCanMaxpool2d (shape: Shape) (kernelSize: int[]) (stride: int[]) (padding: int[]) =
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

    let checkCanMaxpool3d (shape: Shape) (kernelSize: int[]) (stride: int[]) (padding: int[]) =
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

    let checkCanMaxunpool1d (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 3 then failwithf "Expecting outputSize (%A) to be 3-dimensional" outputSize
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputSize = shape.[2]
        if outputSize.[0] <> indicesShape.[0] || outputSize.[1] <> indicesShape.[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize.[2]|]
        batchSize, channels, inputSize, outputShape

    let checkCanMaxunpool2d (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32
        if outputSize.Length <> 4 then failwithf "Expecting outputSize (%A) to be 4-dimensional" outputSize
        let batchSize = shape.[0]
        let channels = shape.[1]
        let inputHeight = shape.[2]
        let inputWidth = shape.[3]
        if outputSize.[0] <> indicesShape.[0] || outputSize.[1] <> indicesShape.[1] then failwithf "Expecting the first two elements of outputSize (%A) and indicesShape (%A) to be the same" outputSize indicesShape
        let outputShape = [|batchSize; channels; outputSize.[2]; outputSize.[3]|]
        batchSize, channels, (inputHeight, inputWidth), outputShape

    let checkCanMaxunpool3d (shape: Shape) (indicesDtype: Dtype) (indicesShape: Shape) (outputSize: int[]) =
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

    let canExpand (oldShape: Shape) (newShape: Shape) =
        newShape.Length >= oldShape.Length &&
        let trim = newShape.Length - oldShape.Length
        (oldShape,newShape.[trim..]) ||> Array.forall2 (fun n m -> n = 1 || n = m)

    let checkCanExpand (oldShape: Shape) (newShape: Shape) =
        let isOK = canExpand oldShape newShape
        if not isOK then failwithf "can't expand from shape %A to %A - each dimension must either be equal or expand from 1" oldShape newShape

    let checkCanTranspose (dim: int) =
        if dim <> 2 then failwith "Cannot transpose Tensor when dim=2"

    let checkCanFlip (dim: int) (dims: int[]) =
        if dims.Length > dim then failwithf "Expecting dims (list of dimension indices to flip) of length less than Tensor's dimensions, received %A, %A" dims.Length dim
        if hasDuplicates dims then failwithf "Expecting dims (list of dimension indices to flip) without repetition, received %A" dims
        if (Array.max dims) >= dim then failwithf "Expecting dims (list of dimension indices to flip) where all indices are less than the tensor dimension, received %A, %A" dims dim

    let checkCanRepeat (shape: Shape) (dim: int) =
        if shape.[dim] <> 1 then failwithf "Expecting Tensor's shape (%A) at dim (%A) to be 1" shape dim

    let checkCanDilate (dim: int) (dilations: int[]) =
        if dilations.Length <> dim then failwithf "Expecting dilations (dilation to use in each dimension) of same length with Tensor's dimensions, received %A, %A" dilations.Length dim
        if (Array.min dilations) < 1 then failwithf "Expecting dilations (dilation to use in each dimension) >= 1 where 1 represents no dilation, received %A" dilations

    let checkCanGather (shape: Shape) (dim: int) (indicesShape: Shape) (indicesDtype:Dtype) =
        if shape.Length <> indicesShape.Length then failwithf "Expecting tensorShape (%A) and indicesShape (%A) to have the same number of dimensions" shape indicesShape
        if dim < 0 || dim > shape.Length-1 then failwithf "Expecting 0<= dim (%A) < tensorShape.Length (%A)" dim shape.Length
        if indicesShape.[dim] < 1 then failwithf "Expecting indicesShape.[dim] (%A) >= 1" indicesShape.[dim]
        if indicesDtype <> Dtype.Int32 then failwithf "Expecting indices to have type %A" Dtype.Int32

    let checkCanView (shape1: Shape) (shape2: Shape) =
        if shapeLength shape1 <> shapeLength shape2 then failwithf "Cannot view Tensor of shape %A as shape %A" shape1 shape2

    let checkCanFlatten (shape: Shape) (startDim: int) (endDim: int) =
        if startDim < 0 || startDim >= shape.Length then failwithf "Expecting 0 <= startDim (%A) < %A" startDim shape.Length
        if endDim < 0 || endDim >= shape.Length then failwithf "Expecting 0 <= endDim (%A) < %A" endDim shape.Length
        if endDim <= startDim then failwithf "Expecting startDim (%A) < endDim (%A)" startDim endDim

    let checkCanAddSlice (shape1: Shape) (location: int[]) (shape2: Shape) =
        if not (contains shape1 shape2) then failwithf "Expecting shape1 to contain shape2, received %A, %A" shape1 shape2
        if location.Length <> shape1.Length then failwithf "Expecting location of the same length as shape1, received %A, %A" (location.Length) shape1

    let checkCanMatmul (shape1: Shape) (shape2: Shape) =
        if shape1.Length <> 2 || shape2.Length <> 2 then failwithf "Expecting two 2d Tensors, received Tensors with shapes %A, %A" shape1 shape2
        if shape1.[1] <> shape2.[0] then failwithf "Cannot multiply Tensors with shapes %A, %A" shape1 shape2

    let checkCanDot (shape1: Shape) (shape2: Shape) =
        if shape1.Length <> 1 || shape2.Length <> 1 then failwithf "Expecting two vectors (1d Tensors), received Tensors with shapes %A, %A" shape1 shape2
        if shape1.[0] <> shape2.[0] then failwithf "Cannot multiply vectors with different lengths %A, %A" shape1.[0] shape2.[0]

    let checkCanPad (shape: Shape) (paddings: int[]) =
        if shape.Length <> paddings.Length then failwithf "Expecting shape (%A) and paddings (%A) to have the same length" shape paddings
        if not (paddings |> Array.forall (fun p -> p >= 0)) then failwithf "Expecting all paddings (%A) >= 0" paddings

    let checkCanDropout (p:double) =
        if p < 0. || p > 1. then failwithf "Expecting 0 <= p <= 1, but received %A" p

    let checkCanDropout2d (shape: Shape) (p:double) =
        checkCanDropout p
        if shape.Length <> 4 then failwithf "Expecting shape (%A) to be 4-dimensional (NxCxHxW: batchSize, inputChannels, inputHeight, inputWidth)" shape

    let checkCanDropout3d (shape: Shape) (p:double) =
        checkCanDropout p
        if shape.Length <> 5 then failwithf "Expecting shape (%A) to be 5-dimensional (NxCxDxHxW: batchSize, inputChannels, inputDepth, inputHeight, inputWidth)" shape

    let squeeze (dim: int) (shape: Shape) =
        if dim = -1 then
            [|for s in shape do if s <> 1 then yield s|]
        elif shape.[dim] = 1 then
            [|for i=0 to shape.Length - 1 do 
                if i < dim then yield shape.[i]
                elif i > dim then yield shape.[i]|]
        else
            shape

    let checkCanUnsqueeze (dim: int) (shape: Shape) =
        if dim < 0 || dim > shape.Length then failwithf "Expecting dim in range [0, %A] but received %A" shape.Length dim
        [|for i=0 to shape.Length - 1 + 1 do 
            if i < dim then yield shape.[i]
            elif i = dim then yield 1
            else yield shape.[i-1]|]

    let unsqueezeAs (shape1: Shape) (shape2: Shape) =
        if shape1.Length > shape2.Length then failwithf "Expecting shape1.Length (%A) <= shape2.Length (%A)" shape1.Length shape2.Length
        let ones = Array.create (shape2.Length - shape1.Length) 1
        Array.append ones shape1

    let locationToBounds (shape: Shape) (location: int[]) =
        Array2D.init location.Length 3 (fun i j -> if j=0 then location.[i] elif j=1 then location.[i] + shape.[i] - 1 else 1)

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

    let dilated (shape: Shape) (dilations: int[]) =
        Array.map2 (fun n d -> n + (n - 1) * (d - 1)) shape dilations

    let dilated2 (shape: Shape) (dilations: int[]) =
        Array.map2 (*) shape dilations

    let undilatedShape (shape: Shape) (dilations: int[]) =
        Array.map2 (fun n d -> (n + d - 1) / d) shape dilations

    let complete (nelement: int) (shape: Shape) =
        if (shape |> Array.filter (fun x -> x < -1) |> Array.length) > 0 then failwithf "Invalid shape %A" shape
        let numUnspecified = shape |> Array.filter ((=) -1) |> Array.length
        if numUnspecified > 1 then
            failwithf "Cannot complete shape %A, expecting at most one unspecified dimension (-1)" shape
        elif numUnspecified = 0 then 
            shape
        else
            let divisor = shape |> Array.filter ((<>) -1) |> shapeLength
            if nelement % divisor <> 0 then failwithf "Cannot complete shape %A to have %A elements" shape nelement
            let missing = nelement / divisor
            [|for d in shape do if d = -1 then yield missing else yield d|]

module Array =
    [<ExcludeFromCodeCoverage>]
    let inline allClose (relativeTolerance:'T) (absoluteTolerance:'T) (array1:'T[]) (array2:'T[]) =
        let dim1 = array1.Length
        let dim2 = array2.Length
        if dim1 <> dim2 then false
        else (array1,array2) ||> Array.forall2 (fun a b -> abs(a-b) <= absoluteTolerance + relativeTolerance*abs(b)) 

let boundsToLocation (bounds: int[,]) =
    [|for i=0 to bounds.GetLength(0) - 1 do yield bounds.[i, 0]|]

let boundsToShape (bounds: int[,]) =
    [|for i=0 to bounds.GetLength(0) - 1 do yield bounds.[i, 1] - bounds.[i, 0] + 1|] 

let mirrorCoordinates (coordinates: int[]) (shape: Shape) (mirrorDims: int[]) =
    if coordinates.Length <> shape.Length then failwithf "Expecting coordinates and shape of the same dimension, received %A, %A" coordinates.Length shape.Length
    let result = Array.copy coordinates
    for d=0 to coordinates.Length-1 do
        if mirrorDims |> Array.contains d then
            result.[d] <- abs (coordinates.[d] - shape.[d] + 1)
    result

let dilatedCoordinates (coordinates: int[]) (dilations: int[]) =
    Array.map2 (*) coordinates dilations

let checkValidIndex (shape: Shape) (index: int[]) =
    if shape.Length <> index.Length then failwithf "Expecting shape (%A) and index (%A) to have the same length" shape index
    let valid = Array.forall2 (fun s i -> i < s) shape index
    if not valid then failwithf "index (%A) is not valid for shape (%A)" index shape

let indexToFlatIndex (shape: Shape) (index: int[]) =
    checkValidIndex shape index
    let mutable flatIndex = 0
    for i=0 to index.Length - 1 do
        let v = if i = index.Length - 1 then 1 else (Array.reduce (*) shape.[i+1..])
        flatIndex <- flatIndex + index.[i] * v
    flatIndex

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
    
/// Create a non-jagged 3D array from jagged data
let array3D data = 
    let data = data |> Array.ofSeq |> Array.map array2D
    let r1, r2, r3 = data.Length, data.[0].GetLength(0), data.[0].GetLength(1)
    for i in 0 .. r1-1 do 
        let q2 = data.[i].GetLength(0)
        let q3 = data.[i].GetLength(1)
        if q2 <> r2 || q3 <> r3 then 
            invalidArg "data" (sprintf "jagged input at position %d: first is _ x %d x %d, later is _ x _ x %d x %d" i r2 r3 q2 q3)
    Array3D.init r1 r2 r3 (fun i j k -> data.[i].[j,k])

/// Create a non-jagged 4D array from jagged data
let array4D data = 
    let data = data |> array2D |> Array2D.map array2D
    let r1,r2,r3,r4 = (data.GetLength(0), data.GetLength(1), data.[0,0].GetLength(0),data.[0,0].GetLength(1))
    for i in 0 .. r1-1 do 
      for j in 0 .. r2-1 do 
        let q3 = data.[i,j].GetLength(0)
        let q4 = data.[i,j].GetLength(1)
        if q3 <> r3 || q4 <> r4 then 
            invalidArg "data" (sprintf "jagged input at position (%d,%d): first is _ x _ x %d x %d, later is _ x _ x %d x %d" i j r2 r3 q3 q4)
    Array4D.init r1 r2 r3 r4 (fun i j k m -> data.[i,j].[k,m])

let arrayND (shape: Shape) f =
    match shape with 
    | [| |] -> f [| |] |> box
    | [| d0 |] -> Array.init d0 (fun i -> f [| i |]) |> box
    | [| d0; d1 |] -> Array2D.init d0 d1 (fun i1 i2 -> f [| i1; i2 |]) |> box
    | [| d0; d1; d2 |] -> Array3D.init d0 d1 d2 (fun i1 i2 i3 -> f [| i1; i2; i3 |]) |> box
    | [| d0; d1; d2; d3 |] -> Array4D.init d0 d1 d2 d3 (fun i1 i2 i3 i4 -> f [| i1; i2; i3; i4 |]) |> box
    | _ -> failwith "arrayND - nyi for dim > 4"

/// Get the elements of an arbitrary IEnumerble
let private seqElements (ie: obj) = 
    let e = (ie :?> IEnumerable).GetEnumerator()
    [| while e.MoveNext() do yield e.Current |]

/// Match an array type of arbitrary rank
let private (|ArrayTy|_|) (ty: Type) = 
    if ty.IsArray && ty.GetArrayRank() <= 4 then
        Some(ty.GetArrayRank(), ty.GetElementType())
    else 
       None

/// Match an tuple type
let private (|TupleTy|_|) (ty: Type) = 
    if FSharpType.IsTuple ty then 
        Some(FSharpType.GetTupleElements ty)
    else 
       None

let rec private  (|ListTy|_|) (ty: Type) = 
    if ty.IsGenericType && ty.GetGenericTypeDefinition().Equals(typedefof<list<int>>) then
       Some (ty.GetGenericArguments().[0])
    else   
        None

/// Match a 1D sequence type (seq<_>) or a subclass
let rec private  (|SeqTy|_|) (ty: Type) = 
    if ty.IsGenericType && ty.GetGenericTypeDefinition().Equals(typedefof<seq<int>>) then
       Some (ty.GetGenericArguments().[0])
    else   
        match ty.BaseType with 
        | null -> None 
        | _ -> 
            match ty.BaseType with 
            | SeqTy ety -> Some ety
            | _ -> 
                ty.GetInterfaces() |> Array.tryPick (|SeqTy|_|)

let rec formatType (ty: Type) = 
    match ty with 
    | ListTy ety -> sprintf "list<%s>" (formatType ety)
    | ArrayTy (_,ety) -> sprintf "%s[]" (formatType ety)
    | SeqTy ety -> sprintf "seq<%s>" (formatType ety)
    | TupleTy etys -> String.concat "*" (Array.map formatType etys)
    | ty when ty = typeof<int64> -> "int64"
    | ty when ty = typeof<int> -> "int"
    | ty when ty = typeof<double> -> "double"
    | ty when ty = typeof<float32> -> "float32"
    | _ -> ty.ToString()

let private (|SeqTupleTy|_|) (ty: Type) = 
    match ty with 
    | SeqTy (TupleTy etys) -> 
        match etys |> Array.tryFind (fun ety -> ety <> etys.[0]) with
        | None -> ()
        | Some ety2 -> failwithf "jagged input: unexpected mixed types in tuple being used as sequence notation, %s and %s" (formatType etys.[0]) (formatType ety2)
        Some (etys.[0])
    | _ -> None

let private (|TupleLeafTy|_|) (tgt: Type) (ty: Type) = 
    match ty with 
    | TupleTy etys when etys |> Array.forall (fun ety -> ety = tgt) -> Some ()
    | _ -> None

let private (|SeqTupleLeafTy|_|) (tgt: Type) (ty: Type) = 
    match ty with 
    | SeqTy (TupleLeafTy tgt) -> Some ()
    | _ -> None

let private flatArrayAndShape1D<'T> (v: 'T[]) =
    v, [|Array.length v|]

let private flatArrayAndShape2D<'T> (v: 'T[,]) =
    let n1 = Array2D.length1 v
    let n2 = Array2D.length2 v
    let arr =
        [|  for i=0 to n1-1 do
                for j=0 to n2-1 do
                   yield v.[i, j] |]
    arr, [| n1;n2|]

let private flatArrayAndShape3D<'T> (v: 'T[,,]) =
    let n1 = Array3D.length1 v
    let n2 = Array3D.length2 v
    let n3 = Array3D.length3 v
    let arr =
        [|  for i=0 to n1-1 do
                for j=0 to n2-1 do
                    for k=0 to n3-1 do
                        yield v.[i, j, k] |]
    arr, [| n1;n2;n3 |]

let private flatArrayAndShape4D<'T> (v: 'T[,,,]) =
    let n1 = Array4D.length1 v
    let n2 = Array4D.length2 v
    let n3 = Array4D.length3 v
    let n4 = Array4D.length4 v
    let arr =
        [|  for i=0 to n1-1 do
                for j=0 to n2-1 do
                    for k=0 to n3-1 do
                        for m=0 to n4-1 do
                            yield v.[i, j, k, m] |]
    arr, [| n1;n2;n3;n4 |]

let private seqTupleElements (els: obj) =
    match seqElements els with 
    | [| el |] -> FSharpValue.GetTupleFields(el) 
    | tup -> failwithf "unexpected multiple values in tuple list input: %A" (Array.toList tup)

let private arrayCast<'T> (els: obj[]) = els |> Array.map (fun v -> v :?> 'T)

let private (|SeqOrSeqTupleTy|_|) ty =
    match ty with 
    | SeqTupleTy ety -> Some (seqTupleElements, ety)
    | SeqTy ety -> Some (seqElements, ety)
    | _ -> None

let private (|SeqOrSeqTupleLeafTy|_|) tgt ty =
    match ty with 
    | SeqTupleLeafTy tgt -> Some (seqTupleElements)
    | SeqTy ety when ety = tgt -> Some (seqElements)
    | _ -> None

let rec tryFlatArrayAndShape<'T> (value:obj) : ('T[] * int[]) option =

    match value with
    | :? 'T as v -> Some ([|v|], [||])
    | :? ('T[]) as v -> Some (flatArrayAndShape1D v)
    | :? ('T[,]) as v -> Some (flatArrayAndShape2D<'T> v)
    | :? ('T[,,]) as v -> Some (flatArrayAndShape3D<'T> v)
    | :? ('T[,,,]) as v -> Some (flatArrayAndShape4D<'T> v)
    | :? seq<'T> as v -> Some (flatArrayAndShape1D (Seq.toArray v))
    | :? seq<seq<'T>> as v -> Some (flatArrayAndShape2D (array2D v))
    | :? seq<seq<seq<'T>>> as v -> Some (flatArrayAndShape3D (array3D v))
    | :? seq<seq<seq<seq<'T>>>> as v -> Some (flatArrayAndShape4D (array4D v))
    | _ -> 
    let vty = value.GetType()
    let tgt = (typeof<'T>)
    match vty with
    // list<int * int> -> dim 1
    | SeqTupleLeafTy tgt -> 
        let arr = value |> seqTupleElements |> arrayCast<'T>
        Some (arr, [| arr.Length |])
    // list<list<int * int>> etc. -> dim 2
    | SeqOrSeqTupleTy (fetcher, (SeqOrSeqTupleLeafTy tgt fetcher2)) -> 
        let els = value |> fetcher |> Array.map (fetcher2 >> arrayCast<'T>) |> array2D
        Some (flatArrayAndShape2D<'T> els)
    // ... -> dim 3
    | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleLeafTy tgt fetcher3)) -> 
        let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> arrayCast<'T>)) |> array3D
        Some (flatArrayAndShape3D<'T> els)
    // ... -> dim 4
    | SeqOrSeqTupleTy (fetcher1, SeqOrSeqTupleTy (fetcher2, SeqOrSeqTupleTy (fetcher3, SeqOrSeqTupleLeafTy tgt fetcher4))) -> 
        let els = value |> fetcher1 |> Array.map (fetcher2 >> Array.map (fetcher3 >> Array.map (fetcher4 >> arrayCast<'T>))) |> array4D
        Some (flatArrayAndShape4D<'T> els)
    | _ -> None

[<ExcludeFromCodeCoverage>]
let inline dataOfValues ofFloat32 ofFloat64 ofInt8 ofInt16 ofInt32 ofInt64 ofBool ofByte (value:obj) : (^T[] * int[]) = 
    match value |> tryFlatArrayAndShape<float32> with
    | Some (values, shape) -> (values |> Array.map ofFloat32, shape)
    | None -> 
    match value |> tryFlatArrayAndShape<double> with
    | Some (values, shape) -> (values |> Array.map ofFloat64, shape) 
    | None -> 
    match value |> tryFlatArrayAndShape<int32> with
    | Some (values, shape) -> (values |> Array.map ofInt32, shape) 
    | None -> 
    match value |> tryFlatArrayAndShape<int64> with
    | Some (values, shape) -> (values |> Array.map ofInt64, shape)
    | None -> 
    match value |> tryFlatArrayAndShape<int8>  with
    | Some (values, shape) -> (values |> Array.map ofInt8, shape)
    | None -> 
    match value |> tryFlatArrayAndShape<byte>  with
    | Some (values, shape) -> (values |> Array.map ofByte, shape)
    | None -> 
    match value |> tryFlatArrayAndShape<int16>  with
    | Some (values, shape) -> (values |> Array.map ofInt16, shape)
    | None -> 
    match value |> tryFlatArrayAndShape<bool> with
    | Some (values, shape) ->(values |> Array.map ofBool, shape) 
    | _ -> failwithf "Cannot convert value of type %A to RawTensorCPU" (value.GetType())

let dataOfValuesForFloat32 (value:obj) =
    dataOfValues float32 float32 float32 float32 float32 float32 (fun x -> if x then 1.0f else 0.0f) float32 value 

let dataOfValuesForFloat64 (value:obj) =
    dataOfValues double double double double double double (fun x -> if x then 1.0 else 0.0) double value 

let dataOfValuesForByte (value:obj) =
    dataOfValues byte byte byte byte byte byte (fun x -> if x then 1uy else 0uy) id value 

let dataOfValuesForInt8 (value:obj) =
    dataOfValues int8 int8 int8 int8 int8 int8 (fun x -> if x then 1y else 0y) int8 value 

let dataOfValuesForInt16 (value:obj) =
    dataOfValues int16 int16 int16 int16 int16 int16 (fun x -> if x then 1s else 0s) int16 value 

let dataOfValuesForInt32 (value:obj) =
    dataOfValues int32 int32 int32 int32 int32 int32 (fun x -> if x then 1 else 0) int32 value

let dataOfValuesForInt64 (value:obj) =
    dataOfValues int64 int64 int64 int64 int64 int64 (fun x -> if x then 1L else 0L) int64 value

let dataOfValuesForBool (value:obj) =
    dataOfValues (fun i -> abs i >= 1.0f) (fun i -> abs i >= 1.0) (fun i -> abs i > 0y) (fun i -> abs i > 0s) (fun i -> abs i > 0) (fun i -> abs i > 0L) id (fun i -> i > 0uy) value 

let toInt a =
    match box a with
    | :? float as a -> a |> int
    | :? float32 as a -> a |> int
    | :? int as a -> a
    | _ -> failwith "Cannot convert to int"

let maxIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.maxBy snd |> fst

let minIndex seq =  seq |> Seq.mapi (fun i x -> i, x) |> Seq.minBy snd |> fst

let memoize fn =
    let cache = new Dictionary<_,_>()
    fun x ->
        match cache.TryGetValue x with
        | true, v -> v
        | false, _ ->
            let v = fn x
            cache.Add(x,v)
            v

let getKeys (dictionary:Dictionary<string, 'a>) =
    let keys = Array.create dictionary.Count ""
    dictionary.Keys.CopyTo(keys, 0)
    keys

let download (url:string) (localFileName:string) =
    let wc = new WebClient()
    printfn "Downloading %A to %A" url localFileName
    wc.DownloadFile(url, localFileName)
    wc.Dispose()

let saveBinary (object:'a) (fileName:string) =
    let formatter = BinaryFormatter()
    let fs = new FileStream(fileName, FileMode.Create)
    let cs = new GZipStream(fs, CompressionMode.Compress)
    try
        formatter.Serialize(cs, object)
        cs.Flush()
        cs.Close()
        fs.Close()
    with
    | :? SerializationException as e -> failwithf "Cannot save to file. %A" e.Message

let loadBinary (fileName:string):'a =
    let formatter = BinaryFormatter()
    let fs = new FileStream(fileName, FileMode.Open)
    let cs = new GZipStream(fs, CompressionMode.Decompress)
    try
        let object = formatter.Deserialize(cs) :?> 'a
        cs.Close()
        fs.Close()
        object
    with
    | :? SerializationException as e -> failwithf "Cannot load from file. %A" e.Message

let shuffledIndices (length: int) =
    let indices = Array.init length id
    let indicesShuffled = Random.Shuffle(indices)
    fun (i: int) -> indicesShuffled.[i]

let indentNewLines (str:String) numSpaces =
    let mutable ret = ""
    let spaces = String.replicate numSpaces " "
    str |> Seq.toList |> List.iter (fun c -> 
                        if c = '\n' then 
                            ret <- ret + "\n" + spaces
                        else ret <- ret + string c)
    ret

let stringPad (s:string) (width:int) =
    if s.Length > width then s
    else String.replicate (width - s.Length) " " + s

let stringPadAs (s1:string) (s2:string) = stringPad s1 s2.Length

