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
    member d.loader(batchSize:int, ?shuffle:bool, ?dropLast:bool, ?dtype:Dtype, ?device:Device, ?backend:Backend, ?targetDtype:Dtype, ?targetDevice:Device, ?targetBackend:Backend) = DataLoader(d, batchSize=batchSize, ?shuffle=shuffle, ?dropLast=dropLast, ?dtype=dtype, ?device=device, ?backend=backend, ?targetDtype=targetDtype, ?targetDevice=targetDevice, ?targetBackend=targetBackend)
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
    override d.item(i) = dataset.item(indices.[i])


type DataLoader(dataset:Dataset, batchSize:int, ?shuffle:bool, ?dropLast:bool, ?dtype:Dtype, ?device:Device, ?backend:Backend, ?targetDtype:Dtype, ?targetDevice:Device, ?targetBackend:Backend) =
    let batchSize = min batchSize dataset.length
    let shuffle = defaultArg shuffle false
    let dropLast = defaultArg dropLast true
    let dtype = defaultArg dtype Dtype.Default
    let device = defaultArg device Device.Default
    let backend = defaultArg backend Backend.Default
    let targetDtype = defaultArg targetDtype dtype
    let targetDevice = defaultArg targetDevice device
    let targetBackend = defaultArg targetBackend backend
    let datalength = if dropLast then batchSize*(dataset.length/batchSize) else dataset.length
    member d.length = ((float datalength)/(float batchSize)) |> ceil |> int
    member d.epoch() =
        let indexer = if shuffle then Random.shuffledIndices datalength else id
        let indices = Seq.init datalength id |> Seq.map indexer
        let batchIndices = indices |> Seq.chunkBySize batchSize
        let batches = batchIndices |> Seq.map (Array.map dataset.item >> Array.unzip)
        batches |> Seq.mapi (fun i (data, target) -> i, data |> dsharp.stack |> dsharp.move(dtype, device, backend), target |> dsharp.stack |> dsharp.move(targetDtype, targetDevice, targetBackend))
    member d.batch() = let _, data, target = d.epoch() |> Seq.head in data, target


type TensorDataset(data:Tensor, target:Tensor) =
    inherit Dataset()
    do if data.shape.[0] <> target.shape.[0] then failwith "Expecting data and target to have the same size in the first dimension"
    override d.length = data.shape.[0]
    override d.item(i) = data.[i], target.[i]


// More datasets (MNIST, CIFAR, etc.) are implemented in DiffSharp.Data project