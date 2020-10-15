namespace rec DiffSharp.Data

open DiffSharp
open DiffSharp.Compose
open DiffSharp.Util
// open System
open System.Net
open System.IO
open System.IO.Compression

/// <namespacedoc>
///   <summary>Contains data sets and components related to data loading.</summary>
/// </namespacedoc>
///
/// <summary>Represents a data set that can load images in batches.</summary>
[<AbstractClass>]
type Dataset() =
    abstract member length: int
    abstract member item: int -> Tensor * Tensor
    member d.loader(batchSize:int, ?shuffle:bool, ?numBatches:int, ?dtype:Dtype, ?device:Device, ?backend:Backend, ?targetDtype:Dtype, ?targetDevice:Device, ?targetBackend:Backend) = DataLoader(d, batchSize=batchSize, ?shuffle=shuffle, ?numBatches=numBatches, ?dtype=dtype, ?device=device, ?backend=backend, ?targetDtype=targetDtype, ?targetDevice=targetDevice, ?targetBackend=targetBackend)
    member t.Item
        with get(i:int) =
            t.item(i)

type DataLoader(dataset:Dataset, batchSize:int, ?shuffle:bool, ?numBatches:int, ?dtype:Dtype, ?device:Device, ?backend:Backend, ?targetDtype:Dtype, ?targetDevice:Device, ?targetBackend:Backend) =
    let shuffle = defaultArg shuffle false
    let batchSize = min batchSize dataset.length
    let dtype = defaultArg dtype Dtype.Default
    let device = defaultArg device Device.Default
    let backend = defaultArg backend Backend.Default
    let targetDtype = defaultArg targetDtype dtype
    let targetDevice = defaultArg targetDevice device
    let targetBackend = defaultArg targetBackend backend
    member d.length = defaultArg numBatches (dataset.length/batchSize)
    member d.epoch() =
        let indexer = if shuffle then Random.shuffledIndices (dataset.length) else id
        let indices = Seq.init dataset.length id |> Seq.map indexer
        let batchIndices = indices |> Seq.chunkBySize batchSize
        let batches = batchIndices |> Seq.map (Array.map dataset.item >> Array.unzip)
        batches |> Seq.mapi (fun i (data, target) -> i, data |> dsharp.stack |> dsharp.move(dtype, device, backend), target |> dsharp.stack |> dsharp.move(targetDtype, targetDevice, targetBackend))

type TensorDataset(data:Tensor, target:Tensor) =
    inherit Dataset()
    do if data.shape.[0] <> target.shape.[0] then failwith "Expecting data and target to have the same size in the first dimension"
    override d.length = data.shape.[0]
    override d.item(i) = data.[i], target.[i]

type MNIST(path:string, ?urls:seq<string>, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let path = Path.Combine(path, "mnist") |> Path.GetFullPath
    let train = defaultArg train true
    let transform = defaultArg transform (fun t -> (t - 0.1307) / 0.3081)
    let targetTransform = defaultArg targetTransform id
    let urls = List.ofSeq <| defaultArg urls (Seq.ofList
                   ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
                    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
                    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
                    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"])
    let files = [for url in urls do Path.Combine(path, Path.GetFileName(url))]
    let filesProcessed = [for file in files do Path.ChangeExtension(file, ".tensor")]
    let data, target = 
        Directory.CreateDirectory(path) |> ignore
        let mutable data = dsharp.zero()
        let mutable target = dsharp.zero()
        if train then
            if not (File.Exists(files.[0])) then download urls.[0] files.[0]
            if not (File.Exists(files.[1])) then download urls.[1] files.[1]
            if File.Exists(filesProcessed.[0]) then data <-    dsharp.load(filesProcessed.[0]) else data <-    MNIST.LoadMNISTImages(files.[0]); dsharp.save(data, filesProcessed.[0])
            if File.Exists(filesProcessed.[1]) then target <- dsharp.load(filesProcessed.[1]) else target <- MNIST.LoadMNISTLabels(files.[1]); dsharp.save(target, filesProcessed.[1])
        else
            if not (File.Exists(files.[2])) then download urls.[2] files.[2]
            if not (File.Exists(files.[3])) then download urls.[3] files.[3]
            if File.Exists(filesProcessed.[2]) then data <-    dsharp.load(filesProcessed.[2]) else data <-    MNIST.LoadMNISTImages(files.[2]); dsharp.save(data, filesProcessed.[2])
            if File.Exists(filesProcessed.[3]) then target <- dsharp.load(filesProcessed.[3]) else target <- MNIST.LoadMNISTLabels(files.[3]); dsharp.save(target, filesProcessed.[3])
        data, target

    static member internal LoadMNISTImages(filename, ?n:int) =
        let r = new BinaryReader(new GZipStream(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read), CompressionMode.Decompress))
        let magicnumber = r.ReadInt32() |> IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2051 -> // Images
            let maxitems = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let rows = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let cols = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let n = defaultArg n maxitems
            r.ReadBytes(n * rows * cols)
            |> Array.map float32
            |> dsharp.tensor
            |> dsharp.view ([n; 1; 28; 28])
            |> fun t -> t / 255
        | _ -> failwith "Given file is not in the MNIST format."
    static member internal LoadMNISTLabels(filename, ?n:int) =
        let r = new BinaryReader(new GZipStream(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read), CompressionMode.Decompress))
        let magicnumber = r.ReadInt32() |> IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let n = defaultArg n maxitems
            r.ReadBytes(n)
            |> Array.map int
            |> dsharp.tensor
            |> dsharp.view ([n])
        | _ -> failwith "Given file is not in the MNIST format."
    override d.length = data.shape.[0]
    override d.item(i) = transform data.[i], targetTransform target.[i]