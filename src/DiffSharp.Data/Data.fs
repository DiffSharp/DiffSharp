// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace DiffSharp.Data

open DiffSharp
open DiffSharp.Compose
open DiffSharp.Util
open System
open System.IO
open System.IO.Compression
open System.Text
open System.Net


/// Contains auto-opened utilities related to the DiffSharp programming model.
[<AutoOpen>]
module DataUtil =
    /// Synchronously downloads the given URL to the given local file.
    let download (url:string) (localFileName:string) =
        Directory.CreateDirectory(Path.GetDirectoryName(localFileName)) |> ignore
        if File.Exists(localFileName) then
            printfn "File exists, skipping download: %A" localFileName
        else
            let wc = new WebClient()
            printfn "Downloading %A to %A" url localFileName
            wc.DownloadFile(url, localFileName)
            wc.Dispose()

    let extractTarStream (stream:Stream) (outputDir:string) =
        // Tar standard: https://www.gnu.org/software/tar/manual/html_node/Standard.html
        let buffer:byte[] = Array.zeroCreate 100
        let mutable stop = false
        while not stop do
            stream.Read(buffer, 0, 100) |> ignore // Read 'char name[100]'
            let name = Encoding.ASCII.GetString(buffer).Trim(Convert.ToChar(0)).Trim()
            if String.IsNullOrWhiteSpace(name) then stop <- true
            else
                stream.Seek(24L, SeekOrigin.Current) |> ignore // Seek to 'char size[12]'
                stream.Read(buffer, 0, 12) |> ignore // Read 'char size[12]'
                let size = Convert.ToInt32(Encoding.ASCII.GetString(buffer, 0, 12).Trim(Convert.ToChar(0)).Trim(), 8)
                printfn "Extracting %A (%A Bytes)" name size
                stream.Seek(376L, SeekOrigin.Current) |> ignore // Seek to end of header block, beginning of file data
                let output = Path.Combine(outputDir, name)
                if not (Directory.Exists(Path.GetDirectoryName(output))) then
                    Directory.CreateDirectory(Path.GetDirectoryName(output)) |> ignore
                if size > 0 then
                    let str = File.Open(output, FileMode.OpenOrCreate, FileAccess.Write)
                    let buf:byte[] = Array.zeroCreate size
                    stream.Read(buf, 0, buf.Length) |> ignore // Read file data
                    str.Write(buf, 0, buf.Length)
                    str.Close()
                let pos = stream.Position
                let mutable offset = 512L - (pos % 512L)
                if offset = 512L then
                    offset <- 0L
                stream.Seek(offset, SeekOrigin.Current) |> ignore // Seek to next 512-byte block

    let extractTarGz (fileName:string) (outputDir:string) =
        let fs = File.OpenRead(fileName)
        let gz = new GZipStream(fs, CompressionMode.Decompress)
        let chunk = 4096
        let memstr = new MemoryStream()
        let buffer:byte[] = Array.zeroCreate chunk
        // The code below for GZipStream read was affected by a breaking change between dotnet 5.0 and 6.0
        // https://docs.microsoft.com/en-us/dotnet/core/compatibility/core-libraries/6.0/partial-byte-reads-in-streams
        // It was subsequently fixed to work correctly on both dotnet 5.0 and 6.0
        let mutable read = 1
        while read > 0 do
            read <- gz.Read(buffer, 0, chunk)
            memstr.Write(buffer, 0, read)
        gz.Close()
        fs.Close()
        memstr.Seek(0L, SeekOrigin.Begin) |> ignore
        extractTarStream memstr outputDir
        memstr.Close()


type ImageDataset(path:string, ?fileExtension:string, ?resize:int*int, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let fileExtension = defaultArg fileExtension "png"
    let transform = defaultArg transform id
    let targetTransform = defaultArg targetTransform id
    let subdirs = Directory.GetDirectories(path) |> Array.sort
    let filesInSubdirs = [|for subdir in subdirs do
                            let files = Directory.GetFiles(subdir, "*."+fileExtension)
                            if files.Length > 0 then files|]
    let _classes = filesInSubdirs.Length
    let data = [|for i in 0.._classes-1 do
                    let files = filesInSubdirs[i]
                    yield! Array.map (fun file -> file, i) files|]
    let _classNames = Array.map (fun (f:string[]) -> DirectoryInfo(f[0]).Parent.Name) filesInSubdirs
    member d.classes = _classes
    member d.classNames = _classNames
    override d.length = data.Length
    override d.item(i) =
        let fileName, category = data[i]
        transform (dsharp.loadImage(fileName, ?resize=resize, device=Device.CPU)), targetTransform (dsharp.tensor(category, device=Device.CPU))


type CIFAR10(path:string, ?url:string, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let path = Path.Combine(path, "cifar10") |> Path.GetFullPath
    let pathExtracted = Path.Combine(path, "cifar-10-batches-bin")
    let train = defaultArg train true
    let cifar10mean = dsharp.tensor([0.4914, 0.4822, 0.4465], device=Device.CPU).view([3;1;1])
    let cifar10stddev = dsharp.tensor([0.247, 0.243, 0.261], device=Device.CPU).view([3;1;1])
    let transform = defaultArg transform (fun t -> (t - cifar10mean) / cifar10stddev)
    let targetTransform = defaultArg targetTransform id
    let url = defaultArg url "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    let file = Path.Combine(path, Path.GetFileName(url))

    let loadCIFAR10 fileName =
        let br = new BinaryReader(File.OpenRead(fileName))
        [|for _ in 1..10000 do
            let label = br.ReadByte() |> dsharp.tensor(device=Device.CPU)
            let image = br.ReadBytes(3*1024) |> Array.map float32 |> dsharp.tensor(device=Device.CPU) |> dsharp.view([3; 32; 32]) // Mapping bytes to float32 before tensor construction is crucial, otherwise we have an issue with confusing byte with int8 that is destructive
            image/255, label
        |] |> Array.unzip |> fun (i, l) -> dsharp.stack(i), dsharp.stack(l)

    let data, target =
        if not (File.Exists(file)) then download url file
        if not (Directory.Exists(pathExtracted)) then extractTarGz file path
        let files = [|"data_batch_1.bin"; "data_batch_2.bin"; "data_batch_3.bin"; "data_batch_4.bin"; "data_batch_5.bin"; "test_batch.bin"|] |> Array.map (fun f -> Path.Combine(pathExtracted, f))
        if train then
            files[..4] |> Array.map loadCIFAR10 |> Array.unzip |> fun (d, t) -> dsharp.cat(d), dsharp.cat(t)
        else
            loadCIFAR10 files[5]

    let _classNames = File.ReadAllLines(Path.Combine(pathExtracted, "batches.meta.txt")) |> Array.take 10
    member d.classes = _classNames.Length
    member d.classNames = _classNames
    override d.length = data.shape[0]
    override d.item(i) = transform data[i], targetTransform target[i]


type CIFAR100(path:string, ?url:string, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor) =
    inherit Dataset()
    let path = Path.Combine(path, "cifar100") |> Path.GetFullPath
    let pathExtracted = Path.Combine(path, "cifar-100-binary")
    let train = defaultArg train true
    let cifar100mean = dsharp.tensor([0.5071, 0.4867, 0.4408], device=Device.CPU).view([3;1;1])
    let cifar100stddev = dsharp.tensor([0.2675, 0.2565, 0.2761], device=Device.CPU).view([3;1;1])
    let transform = defaultArg transform (fun t -> (t - cifar100mean) / cifar100stddev)
    let targetTransform = defaultArg targetTransform id
    let url = defaultArg url "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    let file = Path.Combine(path, Path.GetFileName(url))

    let loadCIFAR100 fileName n =
        let br = new BinaryReader(File.OpenRead(fileName))
        [|for _ in 1..n do
            let labelCoarse = br.ReadByte() |> dsharp.tensor(device=Device.CPU)
            let labelFine = br.ReadByte() |> dsharp.tensor(device=Device.CPU)
            let image = br.ReadBytes(3*1024) |> Array.map float32 |> dsharp.tensor(device=Device.CPU) |> dsharp.view([3; 32; 32]) // Mapping bytes to float32 before tensor construction is crucial, otherwise we have an issue with confusing byte with int8 that is destructive
            image/255, labelCoarse, labelFine
        |] |> Array.unzip3 |> fun (i, lc, lf) -> dsharp.stack(i), dsharp.stack(lc), dsharp.stack(lf)

    let data, _, targetFine =
        if not (File.Exists(file)) then download url file
        if not (Directory.Exists(pathExtracted)) then extractTarGz file path
        if train then loadCIFAR100 (Path.Combine(pathExtracted, "train.bin")) 50000
        else loadCIFAR100 (Path.Combine(pathExtracted, "test.bin")) 10000

    let _classNamesCoarse = File.ReadAllLines(Path.Combine(pathExtracted, "coarse_label_names.txt")) |> Array.take 20
    let _classNamesFine = File.ReadAllLines(Path.Combine(pathExtracted, "fine_label_names.txt")) |> Array.take 100
    member d.classes = _classNamesFine.Length
    member d.classNames = _classNamesFine
    override d.length = data.shape[0]
    override d.item(i) = transform data[i], targetTransform targetFine[i]    


type MNIST(path:string, ?urls:seq<string>, ?train:bool, ?transform:Tensor->Tensor, ?targetTransform:Tensor->Tensor, ?n:int) =
    inherit Dataset()
    let path = Path.Combine(path, "mnist") |> Path.GetFullPath
    let train = defaultArg train true
    let transform = defaultArg transform (fun t -> (t - 0.1307) / 0.3081)
    let targetTransform = defaultArg targetTransform id
    // let urls = List.ofSeq <| defaultArg urls (Seq.ofList
    //                ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
    //                 "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
    //                 "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
    //                 "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"])
    // Alternative URLs that work when LeCun's site is not responding
    let urls = List.ofSeq <| defaultArg urls (Seq.ofList
                   ["https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz";
                    "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz";
                    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz";
                    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"])
    let files = [for url in urls do Path.Combine(path, Path.GetFileName(url))]

    let loadMNISTImages(filename:string) =
        let r = new BinaryReader(new GZipStream(File.OpenRead(filename), CompressionMode.Decompress))
        let magicnumber = r.ReadInt32() |> IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2051 -> // Images
            let maxitems = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let rows = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let cols = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let n = min maxitems (defaultArg n maxitems)
            r.ReadBytes(n * rows * cols)
            |> Array.map float32 // Mapping bytes to float32 before tensor construction is crucial, otherwise we have an issue with confusing byte with int8 that is destructive
            |> dsharp.tensor(device=Device.CPU)
            |> dsharp.view ([n; 1; 28; 28])
            |> fun t -> t / 255
        | _ -> failwith "Given file is not in the MNIST format."
    let loadMNISTLabels(filename:string) =
        let r = new BinaryReader(new GZipStream(File.OpenRead(filename), CompressionMode.Decompress))
        let magicnumber = r.ReadInt32() |> IPAddress.NetworkToHostOrder
        match magicnumber with
        | 2049 -> // Labels
            let maxitems = r.ReadInt32() |> IPAddress.NetworkToHostOrder
            let n = min maxitems (defaultArg n maxitems)
            r.ReadBytes(n)
            |> Array.map int
            |> dsharp.tensor(device=Device.CPU)
            |> dsharp.view ([n])
        | _ -> failwith "Given file is not in the MNIST format."

    let data, target = 
        if train then
            if not (File.Exists(files[0])) then download urls[0] files[0]
            if not (File.Exists(files[1])) then download urls[1] files[1]
            loadMNISTImages files[0], loadMNISTLabels files[1]
        else
            if not (File.Exists(files[2])) then download urls[2] files[2]
            if not (File.Exists(files[3])) then download urls[3] files[3]
            loadMNISTImages files[2], loadMNISTLabels files[3]

    member d.classes = 10
    member d.classNames = Array.init 10 id |> Array.map string
    override d.length = data.shape[0]
    override d.item(i) = transform data[i], targetTransform target[i]