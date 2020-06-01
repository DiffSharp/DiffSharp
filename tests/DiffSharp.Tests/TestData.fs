namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Data


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
    member _.TestDataLoaderMove () =
        for combo1 in Combos.All do
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