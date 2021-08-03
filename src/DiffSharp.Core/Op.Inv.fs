namespace DiffSharp

[<AutoOpen>]
module OpBInvExtensions =

    type Tensor with
        member a.inv() =
            Shape.checkCanInvert a.shape
            Tensor.Op
                { new UnaryOp("inv") with 
                    member _.fRaw(a) = a.InverseT()
                    member _.ad_dfda(a,ad,f) = -f.matmul(ad).matmul(f)
                    member _.fd_dfda(a,f,fd) = let ft = f.transpose(-1, -2) in -ft.matmul(fd).matmul(ft)
                }
                (a)

    type dsharp with
        static member inv(a:Tensor) = a.inv()
