// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace rec DiffSharp.Data

open DiffSharp
open DiffSharp.Compose
open DiffSharp.Util


/// <namespacedoc>
///   <summary>Contains datasets and components related to data loading.</summary>
/// </namespacedoc>
///
/// <summary>Represents a dataset.</summary>
[<AbstractClass>]
type Dataset() =
    abstract member length: int
    abstract member item: int -> Tensor * Tensor
    member d.loader(batchSize:int, ?shuffle:bool, ?dropLast:bool, ?device:Device, ?dtype:Dtype, ?backend:Backend, ?targetDevice:Device, ?targetDtype:Dtype, ?targetBackend:Backend) = DataLoader(d, batchSize=batchSize, ?shuffle=shuffle, ?dropLast=dropLast, ?device=device, ?dtype=dtype, ?backend=backend, ?targetDevice=targetDevice, ?targetDtype=targetDtype, ?targetBackend=targetBackend)
    override d.ToString() = sprintf "Dataset(%A)" d.length
    member d.Item
        with get(i:int) =
            d.item(i)
    member d.GetSlice(imin:int option, imax:int option) =
        let imin   = defaultArg imin 0
        let imax   = defaultArg imax d.length
        if imin >= imax then failwithf "Expecting imin (%A) < imax (%A)" imin imax
        DatasetSubset(d, [|imin..imax|])
    member d.filter(predicate:Tensor->Tensor->bool) =
        let indices = ResizeArray<int>()
        for i in 0..d.length-1 do
            let data, target = d.item(i)
            if predicate data target then
                indices.Add(i)
        if indices.Count = 0 then failwithf "Could not find any data items for which the predicate is true"
        DatasetSubset(d, indices.ToArray())


type DatasetSubset(dataset:Dataset, indices:int[]) =
    inherit Dataset()
    override d.length = indices.Length
    override d.item(i) = dataset.item(indices[i])


type DataLoader(dataset:Dataset, batchSize:int, ?shuffle:bool, ?dropLast:bool, ?device:Device, ?dtype:Dtype, ?backend:Backend, ?targetDevice:Device, ?targetDtype:Dtype, ?targetBackend:Backend) =
    let batchSize = min batchSize dataset.length
    let shuffle = defaultArg shuffle false
    let dropLast = defaultArg dropLast true
    let device = defaultArg device Device.Default
    let dtype = defaultArg dtype Dtype.Default
    let backend = defaultArg backend Backend.Default
    let targetDevice = defaultArg targetDevice device
    let targetDtype = defaultArg targetDtype dtype
    let targetBackend = defaultArg targetBackend backend
    let datalength = if dropLast then batchSize*(dataset.length/batchSize) else dataset.length
    member d.length = ((float datalength)/(float batchSize)) |> ceil |> int
    member d.epoch(?numBatches:int) =
        let numBatches = defaultArg numBatches d.length
        if numBatches < 1 || numBatches > d.length then failwithf "Expecting 1 <= numBatches (%A) <= %A" numBatches d.length
        let indexer = if shuffle then Random.shuffledIndices datalength else id
        let indices = Seq.init datalength id |> Seq.map indexer
        let batchIndices = indices |> Seq.chunkBySize batchSize
        let batches = batchIndices |> Seq.map (Array.map dataset.item >> Array.unzip)
        batches |> Seq.mapi (fun i (data, target) -> i, data |> dsharp.stack |> dsharp.move(device, dtype, backend), target |> dsharp.stack |> dsharp.move(targetDevice, targetDtype, targetBackend))
        |> Seq.truncate numBatches
    member d.batch(?batchSize:int) = 
        let _, data, target = d.epoch() |> Seq.head
        match batchSize with
        | Some(b) when b <= 0 -> failwithf "Expecting batchSize > 0"
        | Some(b) when b < data.shape[0]-> data[..b-1], target[..b-1]
        | _ -> data, target


type TensorDataset(data:Tensor, target:Tensor) =
    inherit Dataset()
    do if data.shape[0] <> target.shape[0] then failwith "Expecting data and target to have the same size in the first dimension"
    override d.length = data.shape[0]
    override d.item(i) = data[i], target[i]


type TextDataset(text:string, seqLength, ?chars) =
    inherit Dataset()
    // """0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ """
    let _chars = (defaultArg chars text) |> Seq.distinct |> Seq.toArray |> Array.sort
    let onehot = memoize (fun (length, hot) -> dsharp.onehot(length, hot, device=Device.CPU))
    let _charToIndex = memoize (fun c -> try Array.findIndex ((=) c) _chars with _ -> failwithf "Character %A not found in this TextDataset (chars: %A)" c _chars)
    let _indexToChar(index) = _chars[index]
    let textToIndices(text:string) = text |> Seq.map _charToIndex |> Seq.toArray
    let indicesToTensor(indices) = indices |> Array.map (fun i -> onehot(_chars.Length, i)) |> dsharp.stack
    let sequences = 
        if seqLength > text.Length then failwithf "Expecting text.Length (%A) >= seqLength (%A)" text.Length seqLength
        [|for i in 0..(text.Length - seqLength + 1)-1 do text.Substring(i, seqLength)|] |> Array.map textToIndices

    member d.indexToChar(i) = _indexToChar(i)
    member d.charToIndex(c) = _charToIndex(c)
    member d.textToTensor(text:string) = text |> textToIndices |> indicesToTensor
    member d.tensorToText(tensor:Tensor) =
        if tensor.dim <> 2 then failwithf "Expecting a 2d tensor with shape seqLen x features, received tensor with shape %A" tensor.shape 
        let t2text (tens:Tensor) = [|for i in 0..tens.shape[0]-1 do tens[i].argmax()[0]|] |> Array.map _indexToChar |> System.String |> string
        tensor |> t2text

    member d.chars = _chars
    member d.numChars = _chars.Length
    override d.length = sequences.Length
    override d.item(i) =
        let data = sequences[i] |> indicesToTensor
        let target = sequences[i] |> dsharp.tensor(dtype=Dtype.Default, device=Device.CPU)
        data, target

// More datasets (MNIST, CIFAR, etc.) are implemented in DiffSharp.Data project