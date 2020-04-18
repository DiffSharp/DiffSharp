namespace DiffSharp.Data

open DiffSharp
open DiffSharp.Util
// open System
open System.Net
open System.IO
open System.IO.Compression

[<AbstractClass>]
type Dataset() =
    abstract member length: unit -> int
    abstract member item: int -> Tensor * Tensor

    member d.loader(batchSize:int, ?shuffle:bool) =
        let shuffle = defaultArg shuffle false
        let index = if shuffle then shuffledIndices (d.length()) else id
        seq {for i in 0..(d.length()/batchSize) do 
                let data, targets = [for j in 0..batchSize-1 do d.item(index(i*batchSize + j))] |> List.unzip
                i, data |> dsharp.stack, targets |> dsharp.stack}

type MNIST(path:string, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let path = Path.Combine(path, "mnist")
    let train = defaultArg train true
    let transform = defaultArg transform id
    let targetTransform = defaultArg targetTransform id
    let urls = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]
    let files = [for url in urls do Path.Combine(path, Path.GetFileName(url))]
    let filesProcessed = [for file in files do Path.ChangeExtension(file, ".tensor")]
    let data, targets = 
        Directory.CreateDirectory(path) |> ignore
        let mutable data = dsharp.zero()
        let mutable targets = dsharp.zero()
        if train then
            if not (File.Exists(files.[0])) then download urls.[0] files.[0]
            if not (File.Exists(files.[1])) then download urls.[1] files.[1]
            if File.Exists(filesProcessed.[0]) then data <-   dsharp.load(filesProcessed.[0]) else data <-   MNIST.LoadMNISTImages(files.[0]); dsharp.save(data, filesProcessed.[0])
            if File.Exists(filesProcessed.[1]) then targets <- dsharp.load(filesProcessed.[1]) else targets <- MNIST.LoadMNISTLabels(files.[1]); dsharp.save(targets, filesProcessed.[1])
        else
            if not (File.Exists(files.[2])) then download urls.[2] files.[2]
            if not (File.Exists(files.[3])) then download urls.[3] files.[3]
            if File.Exists(filesProcessed.[2]) then data <-   dsharp.load(filesProcessed.[2]) else data <-   MNIST.LoadMNISTImages(files.[2]); dsharp.save(data, filesProcessed.[2])
            if File.Exists(filesProcessed.[3]) then targets <- dsharp.load(filesProcessed.[3]) else targets <- MNIST.LoadMNISTLabels(files.[3]); dsharp.save(targets, filesProcessed.[3])
        data, targets

    static member LoadMNISTImages(filename, ?n:int) =
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
        | _ -> failwith "Given file is not in the MNIST format."
    static member LoadMNISTLabels(filename, ?n:int) =
        let r = new BinaryReader(new GZipStream(File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read), CompressionMode.Decompress))
        let magicnumber = r.ReadInt32() |> IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let n = defaultArg n maxitems
            r.ReadBytes(n)
            |> Array.map int
            |> dsharp.tensor
            |> dsharp.view ([n; -1])
        | _ -> failwith "Given file is not in the MNIST format."
    override d.length() = data.shape.[0]
    override d.item(i) = transform data.[i], targetTransform targets.[i]