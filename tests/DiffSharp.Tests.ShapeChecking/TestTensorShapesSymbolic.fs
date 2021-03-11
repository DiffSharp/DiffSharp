namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.ShapeChecking

[<TestFixture>]
type TestTensorShapesSymbolic () =
    let ShapeChecking =
        [ let devices = [ Device.CPU ]
          for device in devices do
              for dtype in [ Dtype.Float32 ] do
                yield ComboInfo(defaultBackend=Backend.ShapeChecking, defaultDevice=device, defaultDtype=dtype, defaultFetchDevices=(fun _ -> devices)) ]


    [<Test>]
    member _.``test full symbolic shape``() =
        for combo in ShapeChecking do 
            let sym = SymScope()
            let shape = Shape.symbolic [| sym?M ; sym?N |]
            let t1a = combo.full(shape, 2.5)
            Assert.CheckEqual(shape, t1a.shapex)

    [<Test>]
    member _.``test view symbolic shape`` () =
        for combo in ShapeChecking do 
            let sym = SymScope()
            let N : Int = sym?N
            let M : Int = sym?M
            let t = combo.randint(0, 2, Shape.symbolic [5*2*N;5*2*M])
            let t1 = t.view(-1)
            let t1Shape = t1.shapex
            let t1ShapeCorrect = Shape.symbolic [100*N*M]
            let t2Shape = t.view([-1;50]).shapex
            let t2ShapeCorrect = Shape.symbolic [|2*N*M;Int 50|]
            let t3Shape = t.view([2;-1;50]).shapex
            let t3ShapeCorrect = Shape.symbolic [|Int 2;N*M;Int 50|]
            let t4Shape = t.view([2;-1;10]).shapex
            let t4ShapeCorrect = Shape.symbolic [|Int 2;5*N*M;Int 10|]
        
            Assert.True(t1ShapeCorrect =~= t1Shape)
            Assert.True(t2ShapeCorrect =~= t2Shape)
            Assert.True(t3ShapeCorrect =~= t3Shape)
            Assert.True(t4ShapeCorrect =~= t4Shape)
            Assert.True(t1.dtype =~= combo.dtype)
