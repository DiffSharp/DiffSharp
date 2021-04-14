namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestTensorShapesConstant () =
    let ShapeChecking =
        [ let devices = [ Device.CPU ]
          for device in devices do
              for dtype in [ Dtype.Float32 ] do
                yield ComboInfo(defaultBackend=Backend.ShapeChecking, defaultDevice=device, defaultDtype=dtype, defaultFetchDevices=(fun _ -> devices)) ]

    [<Test>]
    member _.``test full shape checking``() =
        for combo in ShapeChecking do 
            // An imposibly large tensor
            let shape = Shape.constant [| 2000000 ; 1000000 |]
            let t1a = combo.full(shape, 2.5)
            Assert.CheckEqual(shape, t1a.shapex)

