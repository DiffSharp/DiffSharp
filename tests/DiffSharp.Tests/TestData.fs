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
    [<Ignore("https://github.com/DiffSharp/DiffSharp/issues/289")>]
    member _.TestMNISTDataset () =
        // Note: this test can fail if http://yann.lecun.com website goes down or file urls change
        let cd = Directory.GetCurrentDirectory()
        let dataDir = Path.Combine(cd.Substring(0, cd.LastIndexOf("tests")), "data")
        Directory.CreateDirectory(dataDir) |> ignore

        let urls =
           ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]
        let mnistTrain = MNIST(dataDir, urls=urls, train=true)
        let mnistTest = MNIST(dataDir, urls=urls, train=false)
        let mnistTrainLength = mnistTrain.length
        let mnistTrainLengthCorrect = 60000
        let mnistTestLength = mnistTest.length
        let mnistTestLengthCorrect = 10000
        Assert.AreEqual(mnistTrainLengthCorrect, mnistTrainLength)
        Assert.AreEqual(mnistTestLengthCorrect, mnistTestLength)

        let batchSize = 16
        let dataloader = mnistTrain.loader(batchSize=batchSize)
        let epoch = dataloader.epoch()
        let _, x, y = epoch |> Seq.head
        let xShape = x.shape
        let xShapeCorrect = [|batchSize; 1; 28; 28|]
        let yShape = y.shape
        let yShapeCorrect = [|batchSize|]
        Assert.AreEqual(xShapeCorrect, xShape)
        Assert.AreEqual(yShapeCorrect, yShape)

        let classes = mnistTrain.classes
        let classesCorrect = 10
        let classNames = mnistTrain.classNames
        let classNamesCorrect = [|"0"; "1"; "2"; "3"; "4"; "5"; "6"; "7"; "8"; "9"|]
        Assert.AreEqual(classesCorrect, classes)
        Assert.AreEqual(classNamesCorrect, classNames)

    [<Test>]
    member _.TestCIFAR10Dataset () =
        // Note: this test can fail if https://www.cs.toronto.edu/~kriz website goes down or file urls change
        let cd = Directory.GetCurrentDirectory()
        let dataDir = Path.Combine(cd.Substring(0, cd.LastIndexOf("tests")), "data")
        Directory.CreateDirectory(dataDir) |> ignore

        let url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
        let cifar10Train = CIFAR10(dataDir, url=url, train=true)
        let cifar10Test = CIFAR10(dataDir, url=url, train=false)
        let cifar10TrainLength = cifar10Train.length
        let cifar10TrainLengthCorrect = 50000
        let cifar10TestLength = cifar10Test.length
        let cifar10TestLengthCorrect = 10000
        Assert.AreEqual(cifar10TrainLengthCorrect, cifar10TrainLength)
        Assert.AreEqual(cifar10TestLengthCorrect, cifar10TestLength)

        let batchSize = 16
        let dataloader = cifar10Train.loader(batchSize=batchSize)
        let epoch = dataloader.epoch()
        let _, x, y = epoch |> Seq.head
        let xShape = x.shape
        let xShapeCorrect = [|batchSize; 3; 32; 32|]
        let yShape = y.shape
        let yShapeCorrect = [|batchSize|]
        Assert.AreEqual(xShapeCorrect, xShape)
        Assert.AreEqual(yShapeCorrect, yShape)

        let classes = cifar10Train.classes
        let classesCorrect = 10
        let classNames = cifar10Train.classNames
        let classNamesCorrect = [|"airplane"; "automobile"; "bird"; "cat"; "deer"; "dog"; "frog"; "horse"; "ship"; "truck"|]
        Assert.AreEqual(classesCorrect, classes)
        Assert.AreEqual(classNamesCorrect, classNames)        

    [<Test>]
    member _.TestCIFAR100Dataset () =
        // Note: this test can fail if https://www.cs.toronto.edu/~kriz website goes down or file urls change
        let cd = Directory.GetCurrentDirectory()
        let dataDir = Path.Combine(cd.Substring(0, cd.LastIndexOf("tests")), "data")
        Directory.CreateDirectory(dataDir) |> ignore

        let url = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
        let cifar100Train = CIFAR100(dataDir, url=url, train=true)
        let cifar100Test = CIFAR100(dataDir, url=url, train=false)
        let cifar100TrainLength = cifar100Train.length
        let cifar100TrainLengthCorrect = 50000
        let cifar100TestLength = cifar100Test.length
        let cifar100TestLengthCorrect = 10000
        Assert.AreEqual(cifar100TrainLengthCorrect, cifar100TrainLength)
        Assert.AreEqual(cifar100TestLengthCorrect, cifar100TestLength)

        let batchSize = 16
        let dataloader = cifar100Train.loader(batchSize=batchSize)
        let epoch = dataloader.epoch()
        let _, x, y = epoch |> Seq.head
        let xShape = x.shape
        let xShapeCorrect = [|batchSize; 3; 32; 32|]
        let yShape = y.shape
        let yShapeCorrect = [|batchSize|]
        Assert.AreEqual(xShapeCorrect, xShape)
        Assert.AreEqual(yShapeCorrect, yShape)

        let classes = cifar100Train.classes
        let classesCorrect = 100
        let classNames = cifar100Train.classNames

        let classNamesCorrect = [|"apple"; "aquarium_fish"; "baby"; "bear"; "beaver"; "bed"; "bee"; "beetle"; "bicycle"; "bottle"; "bowl"; "boy"; "bridge"; "bus"; "butterfly"; "camel"; "can"; "castle"; "caterpillar"; "cattle"; "chair"; "chimpanzee"; "clock"; "cloud"; "cockroach"; "couch"; "crab"; "crocodile"; "cup"; "dinosaur"; "dolphin"; "elephant"; "flatfish"; "forest"; "fox"; "girl"; "hamster"; "house"; "kangaroo"; "keyboard"; "lamp"; "lawn_mower"; "leopard"; "lion"; "lizard"; "lobster"; "man"; "maple_tree"; "motorcycle"; "mountain"; "mouse"; "mushroom"; "oak_tree"; "orange"; "orchid"; "otter"; "palm_tree"; "pear"; "pickup_truck"; "pine_tree"; "plain"; "plate"; "poppy"; "porcupine"; "possum"; "rabbit"; "raccoon"; "ray"; "road"; "rocket"; "rose"; "sea"; "seal"; "shark"; "shrew"; "skunk"; "skyscraper"; "snail"; "snake"; "spider"; "squirrel"; "streetcar"; "sunflower"; "sweet_pepper"; "table"; "tank"; "telephone"; "television"; "tiger"; "tractor"; "train"; "trout"; "tulip"; "turtle"; "wardrobe"; "whale"; "willow_tree"; "wolf"; "woman"; "worm"|]
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
        let x, y = dataloader.batch()
        let xShape = x.shape
        let xShapeCorrect = [|batchSize; din|]
        let yShape = y.shape
        let yShapeCorrect = [|batchSize; dout|]
        Assert.AreEqual(xShapeCorrect, xShape)
        Assert.AreEqual(yShapeCorrect, yShape)

    [<Test>]
    member _.TestDatasetSlice () =
        let x = dsharp.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18]]).float32()
        let y = dsharp.tensor([1,0,1,1,2,3]).float32()
        let dataset = TensorDataset(x, y)
        let dataset2 = dataset[1..2]
        let dataset2Length = dataset2.length
        let dataset2LengthCorrect = 2
        Assert.AreEqual(dataset2LengthCorrect, dataset2Length)

        let loader = dataset2.loader(batchSize=dataset2.length)
        let _, bx, by = loader.epoch() |> Seq.head
        Assert.True(x[1..2].allclose(bx))
        Assert.True(y[1..2].allclose(by))
  
    [<Test>]
    member _.TestDatasetFilter () =
        let x = dsharp.tensor([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15], [16,17,18]]).float32()
        let y = dsharp.tensor([1,0,1,1,2,3]).float32()
        let dataset = TensorDataset(x, y)
        let dataset2 = dataset.filter(fun _ t -> (int t) = 1)
        let dataset2Length = dataset2.length
        let dataset2LengthCorrect = 3
        Assert.AreEqual(dataset2LengthCorrect, dataset2Length)

        let loader = dataset2.loader(batchSize=dataset2.length)
        let bx, by = loader.batch()
        let bxCorrect = dsharp.tensor([[1,2,3], [7,8,9], [10,11,12]]).float32()
        let byCorrect = dsharp.tensor([1,1,1]).float32()
        Assert.True(bxCorrect.allclose(bx))
        Assert.True(byCorrect.allclose(by))

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
        let dataShape = (dataset[0] |> fst).shape
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

    [<Test>]
    member _.TestDataLoaderNumBatches () =
        let ndata = 100
        let batchSize = 10
        let x = dsharp.zeros([ndata; 10])
        let y = dsharp.zeros([ndata; 1])
        let dataset = TensorDataset(x, y)

        let loader = dataset.loader(batchSize=batchSize)
        let numBatchesCorrect = 3
        let mutable numBatches = 0
        for _ in loader.epoch(numBatchesCorrect) do
            numBatches <- numBatches + 1
        Assert.AreEqual(numBatchesCorrect, numBatches)

    [<Test>]
    member _.TestDataLoaderDroplast () =
        let ndata = 1000
        let batchSize = 16
        let x = dsharp.zeros([ndata; 10])
        let y = dsharp.zeros([ndata; 1])
        let dataset = TensorDataset(x, y)

        let loader = dataset.loader(batchSize=batchSize, dropLast=false)
        let nBatches = loader.length
        let nBatchesCorrect = 63
        let mutable nBatchesActual = 0
        for _ in loader.epoch() do nBatchesActual <- nBatchesActual + 1
        Assert.AreEqual(nBatchesCorrect, nBatches)
        Assert.AreEqual(nBatchesCorrect, nBatchesActual)

        let loaderDrop = dataset.loader(batchSize=batchSize, dropLast=true)
        let nBatchesDrop = loaderDrop.length
        let nBatchesDropCorrect = 62
        let mutable nBatchesDropActual = 0
        for _ in loaderDrop.epoch() do nBatchesDropActual <- nBatchesDropActual + 1
        Assert.AreEqual(nBatchesDropCorrect, nBatchesDrop)
        Assert.AreEqual(nBatchesDropCorrect, nBatchesDropActual)

    [<Test>]
    member _.TestDataLoaderBatch () =
        let ndata = 100
        let batchSize = 16
        let x = dsharp.zeros([ndata; 10])
        let y = dsharp.zeros([ndata; 1])
        let dataset = TensorDataset(x, y)        

        let loader = dataset.loader(batchSize=batchSize, dropLast=false)

        let batch1, batchTarget1 = loader.batch()
        let batchLen1 = batch1.shape[0]
        let batchTargetLen1 = batchTarget1.shape[0]
        let batchLenCorrect1 = 16
        Assert.AreEqual(batchLenCorrect1, batchLen1)
        Assert.AreEqual(batchLenCorrect1, batchTargetLen1)

        let len = 5
        let batch1, batchTarget1 = loader.batch(len)
        let batchLen1 = batch1.shape[0]
        let batchTargetLen1 = batchTarget1.shape[0]
        let batchLenCorrect1 = len
        Assert.AreEqual(batchLenCorrect1, batchLen1)
        Assert.AreEqual(batchLenCorrect1, batchTargetLen1)

    [<Test>]
    member _.TestTextDataset () =
        let text = "A merry little surge of electricity piped by automatic alarm from the mood organ beside his bed awakened Rick Deckard."
        let seqLen = 6
        let dataset = TextDataset(text, seqLen)

        let datasetChars = dataset.chars
        let datasetCharsCorrect = [|' '; '.'; 'A'; 'D'; 'R'; 'a'; 'b'; 'c'; 'd'; 'e'; 'f'; 'g'; 'h'; 'i'; 'k'; 'l'; 'm'; 'n'; 'o'; 'p'; 'r'; 's'; 't'; 'u'; 'w'; 'y'|]
        Assert.AreEqual(datasetCharsCorrect, datasetChars)

        let datasetLen = dataset.length
        let datasetLenCorrect = 113
        Assert.AreEqual(datasetLenCorrect, datasetLen)

        let input, target = dataset[1]
        let inputCorrect = dsharp.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        let targetCorrect = dsharp.tensor([0., 16., 9., 20., 20., 25.])
        Assert.AreEqual(inputCorrect, input)
        Assert.AreEqual(targetCorrect, target)

        let inputText = dataset.tensorToText input
        let inputTextCorrect = " merry"
        Assert.AreEqual(inputTextCorrect, inputText)

        let text = "Deckard"
        let textTensor = dataset.textToTensor text
        let textTensorCorrect = dsharp.tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        Assert.AreEqual(textTensorCorrect, textTensor)

        let text2 = dataset.tensorToText textTensor
        Assert.AreEqual(text, text2)