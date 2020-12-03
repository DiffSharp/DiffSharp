// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open System.IO
open NUnit.Framework
open DiffSharp
open DiffSharp.Data
open DiffSharp.Util

[<TestFixture>]
type TestData () =

    [<Test>]
    member _.TestMNISTDataset () =
        // Note: this test can fail if http://yann.lecun.com website goes down or file urls change
        let folder = System.IO.Path.GetTempPath()
        let urls =
           ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]
        let mnist = MNIST(folder, urls=urls, train=false) // MNIST test data
        let mnistLength = mnist.length
        let mnistLengthCorrect = 10000
        Assert.AreEqual(mnistLengthCorrect, mnistLength)

        let batchSize = 16
        let dataloader = mnist.loader(batchSize=batchSize)
        let epoch = dataloader.epoch()
        let _, x, y = epoch |> Seq.head
        let xShape = x.shape
        let xShapeCorrect = [|batchSize; 1; 28; 28|]
        let yShape = y.shape
        let yShapeCorrect = [|batchSize|]
        Assert.AreEqual(xShapeCorrect, xShape)
        Assert.AreEqual(yShapeCorrect, yShape)

        let classes = mnist.classes
        let classesCorrect = 10
        let classNames = mnist.classNames
        let classNamesCorrect = [|"0"; "1"; "2"; "3"; "4"; "5"; "6"; "7"; "8"; "9"|]
        Assert.AreEqual(classesCorrect, classes)
        Assert.AreEqual(classNamesCorrect, classNames)

    [<Test>]
    member _.TestCIFAR10Dataset () =
        // Note: this test can fail if https://www.cs.toronto.edu/~kriz website goes down or file urls change
        let folder = System.IO.Path.GetTempPath()
        let url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
        let cifar10 = CIFAR10(folder, url=url, train=false) // CIFAR10 test data
        let cifar10Length = cifar10.length
        let cifar10LengthCorrect = 10000
        Assert.AreEqual(cifar10LengthCorrect, cifar10Length)

        let batchSize = 16
        let dataloader = cifar10.loader(batchSize=batchSize)
        let epoch = dataloader.epoch()
        let _, x, y = epoch |> Seq.head
        let xShape = x.shape
        let xShapeCorrect = [|batchSize; 3; 32; 32|]
        let yShape = y.shape
        let yShapeCorrect = [|batchSize|]
        Assert.AreEqual(xShapeCorrect, xShape)
        Assert.AreEqual(yShapeCorrect, yShape)

        let classes = cifar10.classes
        let classesCorrect = 10
        let classNames = cifar10.classNames
        let classNamesCorrect = [|"airplane"; "automobile"; "bird"; "cat"; "deer"; "dog"; "frog"; "horse"; "ship"; "truck"|]
        Assert.AreEqual(classesCorrect, classes)
        Assert.AreEqual(classNamesCorrect, classNames)        

    [<Test>]
    member _.TestTensorDataset () =
        let n, din, dout = 128, 64, 16
        let x = dsharp.randn([n; din])
        let y = dsharp.randn([n; dout])
        let dataset = TensorDataset(x, y)
        let datasetLength = dataset.length
        let datasetLengthCorrect = n

        Assert.AreEqual(datasetLengthCorrect, datasetLength)

        let batchSize = 16
        let dataloader = dataset.loader(batchSize=batchSize)
        let epoch = dataloader.epoch()
        let _, x, y = epoch |> Seq.head
        let xShape = x.shape
        let xShapeCorrect = [|batchSize; din|]
        let yShape = y.shape
        let yShapeCorrect = [|batchSize; dout|]
        Assert.AreEqual(xShapeCorrect, xShape)
        Assert.AreEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestImageDataset () =
        let rootDir = Path.Join(Path.GetTempPath(), Random.UUID())
        Directory.CreateDirectory(rootDir) |> ignore

        let dataset = ImageDataset(rootDir, fileExtension="png", resize=(64, 64))
        let datasetLength = dataset.length
        let datasetLengthCorrect = 0
        let datasetClasses = dataset.classes
        let datasetClassesCorrect = 0

        Assert.AreEqual(datasetLengthCorrect, datasetLength)
        Assert.AreEqual(datasetClassesCorrect, datasetClasses)

        let catDir = Path.Join(rootDir, "cat")
        Directory.CreateDirectory(catDir) |> ignore
        dsharp.randn([3; 16; 16]).saveImage(Path.Join(catDir, "1.png"))
        dsharp.randn([3; 16; 16]).saveImage(Path.Join(catDir, "2.png"))

        let dogDir = Path.Join(rootDir, "dog")
        Directory.CreateDirectory(dogDir) |> ignore
        dsharp.randn([3; 16; 16]).saveImage(Path.Join(dogDir, "1.png"))
        dsharp.randn([3; 16; 16]).saveImage(Path.Join(dogDir, "2.png"))
        dsharp.randn([3; 16; 16]).saveImage(Path.Join(dogDir, "3.png"))
        dsharp.randn([3; 16; 16]).saveImage(Path.Join(dogDir, "4.jpg"))

        let foxDir = Path.Join(rootDir, "fox")
        Directory.CreateDirectory(foxDir) |> ignore
        dsharp.randn([3; 16; 16]).saveImage(Path.Join(foxDir, "1.jpg"))

        let dataset = ImageDataset(rootDir, fileExtension="png", resize=(64, 64))
        let datasetLength = dataset.length
        let datasetLengthCorrect = 5
        let datasetClasses = dataset.classes
        let datasetClassesCorrect = 2
        let datasetClassNames = dataset.classNames
        let datasetClassNamesCorrect = [|"cat"; "dog"|]
        let dataShape = (dataset.[0] |> fst).shape
        let dataShapeCorrect = [|3; 64; 64|]

        Assert.AreEqual(datasetLengthCorrect, datasetLength)
        Assert.AreEqual(datasetClassesCorrect, datasetClasses)
        Assert.AreEqual(datasetClassNamesCorrect, datasetClassNames)
        Assert.AreEqual(dataShapeCorrect, dataShape)

    [<Test>]
    member _.TestDataLoaderMove () =
        for combo1 in Combos.AllDevicesAndBackendsFloat32 do
            let n, din, dout = 128, 64, 16
            let x = combo1.zeros([n; din])
            let y = combo1.zeros([n; dout])
            let dataset = TensorDataset(x, y)
            for combo2 in Combos.All do
                for combo3 in Combos.All do
                    let batchSize = 16
                    let dataloader = dataset.loader(batchSize=batchSize, dtype=combo2.dtype, device=combo2.device, backend=combo2.backend, targetDtype=combo3.dtype, targetDevice=combo3.device, targetBackend=combo3.backend)
                    let epoch = dataloader.epoch()
                    let _, x, y = epoch |> Seq.head
                    let xdtype, xdevice, xbackend = x.dtype, x.device, x.backend
                    let ydtype, ydevice, ybackend = y.dtype, y.device, y.backend
                    let xdtypeCorrect, xdeviceCorrect, xbackendCorrect = combo2.dtype, combo2.device, combo2.backend
                    let ydtypeCorrect, ydeviceCorrect, ybackendCorrect = combo3.dtype, combo3.device, combo3.backend
                    Assert.AreEqual(xdtypeCorrect, xdtype)
                    Assert.AreEqual(xdeviceCorrect, xdevice)
                    Assert.AreEqual(xbackendCorrect, xbackend)
                    Assert.AreEqual(ydtypeCorrect, ydtype)
                    Assert.AreEqual(ydeviceCorrect, ydevice)
                    Assert.AreEqual(ybackendCorrect, ybackend)