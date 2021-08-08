namespace DiffSharp

[<AutoOpen>]
module OpSolveExtensions =

    type Tensor with
        member a.solve(b:Tensor) =
            let _ = Shape.checkCanSolve a.shape b.shape
            Tensor.Op
                { new BinaryOp("solve") with 
                    member _.fRaw(a,b) = a.SolveTT(b)
                    member _.ad_dfda(a,ad,b,f) = a.solve(-ad.matmul(f))
                    member _.bd_dfdb(a,b,bd,f) = a.solve(bd)
                    member _.fd_dfda(a,b,f,fd) = fd.bmm(b.transpose(1, 2))
                    member _.fd_dfdb(a,b,f,fd) = a.transpose(1, 2).bmm(fd)
                }
                (a,b)

    type dsharp with
        static member solve(a:Tensor, b:Tensor) = a.solve(b)
