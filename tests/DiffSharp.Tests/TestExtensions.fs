// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp

// #nowarn "0058"

[<TestFixture>]
module TestExtensions =

    type Tensor with
        member a.sinExt() = 
            Tensor.Op
                { new UnaryOpElementwise("sinExt") with 
                    member _.fRaw(a) = a.SinT()
                    member _.dfda(a,f) = a.cosExt()
                }
                (a)

        member a.cosExt() =
            Tensor.Op
                { new UnaryOpElementwise("cosExt") with
                    member _.fRaw(a) = a.CosT()
                    member _.dfda(a,f) = -a.sinExt()
                }
                (a)

        member a.expExt() =
            Tensor.Op
                { new UnaryOpElementwise("expExt") with
                    member _.fRaw(a) = a.ExpT()
                    member _.dfda(a,f) = f
                }
                (a)

        member a.logExt() =
            Tensor.Op
                { new UnaryOpElementwise("logExt") with
                    member _.fRaw(a) = a.LogT()
                    member _.dfda(a,f) = 1/a
                }
                (a)

        member a.transposeExt() =
            Tensor.Op
                { new UnaryOp("transposeExt") with
                    member _.fRaw(a) = a.TransposeT2()
                    member _.ad_dfda(a,ad,f) = ad.transposeExt()
                    member _.fd_dfda(a,f,fd) = fd.transposeExt()
                }
                (a)

        member a.powExt(b) =
            Tensor.Op
                { new BinaryOpElementwise("powExt") with
                    member _.fRaw(a,b) = a.PowTT(b)
                    member _.dfda(a,b,f) = b * f / a
                    member _.dfdb(a,b,f) = f * a.logExt()
                }
                (a,b)

        member a.mulExt(b) =
            Tensor.Op
                { new BinaryOpElementwise("mulExt") with
                    member _.fRaw(a,b) = a.MulTT(b)
                    member _.dfda(a,b,f) = b
                    member _.dfdb(a,b,f) = a
                }
                (a,b)

        member a.matmulExt(b) =
            Tensor.Op
                { new BinaryOp("matmulExt") with
                    member _.fRaw(a,b) = a.MatMulTT(b)
                    member _.ad_dfda(a,ad,b,f) = ad.matmulExt(b)
                    member _.bd_dfdb(a,b,bd,f) = a.matmulExt(bd)
                    member _.fd_dfda(a,b,f,fd) = fd.matmulExt(b.transposeExt())
                    member _.fd_dfdb(a,b,f,fd) = a.transposeExt().matmulExt(fd)
                }
                (a,b)

    let compareUnaryOps op1 op2 count initializer =
        for i in 1..count do
            let x = initializer()
            let xd = dsharp.randnLike(x)

            let fwdx = x.forwardDiff(xd)
            let fwdz1 : Tensor = op1 fwdx
            let fwdz2 : Tensor = op2 fwdx
            let fwdzd1 = fwdz1.derivative
            let fwdzd2 = fwdz2.derivative

            let zd = dsharp.randnLike(fwdz1)
            let revx1 = x.reverseDiff()
            let revx2 = x.reverseDiff()
            let revz1 = op1 revx1
            let revz2 = op2 revx2
            revz1.reverse(zd)
            revz2.reverse(zd)
            let revxd1 = revx1.derivative
            let revxd2 = revx2.derivative

            Assert.True(fwdz1.allclose(fwdz2, 0.01))
            Assert.True(fwdzd1.allclose(fwdzd2, 0.01))
            Assert.True(revz1.allclose(revz2, 0.01))
            Assert.True(revxd1.allclose(revxd2, 0.01))

    let compareBinaryOps op1 op2 count initializer =
        for i in 1..count do
            let x, y = initializer()
            let xd = dsharp.randnLike(x)
            let yd = dsharp.randnLike(y)

            // Tensor, Tensor
            let fwdx = x.forwardDiff(xd)
            let fwdy = y.forwardDiff(yd)
            let fwdz1 : Tensor = op1 fwdx fwdy
            let fwdz2 : Tensor = op2 fwdx fwdy
            let fwdzd1 = fwdz1.derivative
            let fwdzd2 = fwdz2.derivative

            let zd = dsharp.randnLike(fwdz1)
            let revx1 = x.reverseDiff()
            let revy1 = y.reverseDiff()
            let revx2 = x.reverseDiff()
            let revy2 = y.reverseDiff()
            let revz1 = op1 revx1 revy1
            let revz2 = op2 revx2 revy2
            revz1.reverse(zd)
            revz2.reverse(zd)
            let revxd1 = revx1.derivative
            let revxd2 = revx2.derivative
            let revyd1 = revy1.derivative
            let revyd2 = revy2.derivative

            Assert.True(fwdz1.allclose(fwdz2, 0.01))
            Assert.True(fwdzd1.allclose(fwdzd2, 0.01))
            Assert.True(revz1.allclose(revz2, 0.01))
            Assert.True(revxd1.allclose(revxd2, 0.01))
            Assert.True(revyd1.allclose(revyd2, 0.01))

            // Tensor, constant Tensor
            let fwdx = x.forwardDiff(xd)
            let fwdy = y
            let fwdz1 : Tensor = op1 fwdx fwdy
            let fwdz2 : Tensor = op2 fwdx fwdy
            let fwdzd1 = fwdz1.derivative
            let fwdzd2 = fwdz2.derivative

            let zd = dsharp.randnLike(fwdz1)
            let revx1 = x.reverseDiff()
            let revy1 = y
            let revx2 = x.reverseDiff()
            let revy2 = y
            let revz1 = op1 revx1 revy1
            let revz2 = op1 revx2 revy2
            revz1.reverse(zd)
            revz2.reverse(zd)
            let revxd1 = revx1.derivative
            let revxd2 = revx2.derivative
            let revyd1 = revy1.isNoDiff
            let revyd2 = revy2.isNoDiff

            Assert.True(fwdz1.allclose(fwdz2, 0.01))
            Assert.True(fwdzd1.allclose(fwdzd2, 0.01))
            Assert.True(revz1.allclose(revz2, 0.01))
            Assert.True(revxd1.allclose(revxd2, 0.01))
            Assert.CheckEqual(revyd1, revyd2)

            // Constant Tensor, Tensor
            let fwdx = x
            let fwdy = y.forwardDiff(yd)
            let fwdz1 : Tensor = op1 fwdx fwdy
            let fwdz2 : Tensor = op2 fwdx fwdy
            let fwdzd1 = fwdz1.derivative
            let fwdzd2 = fwdz2.derivative

            let zd = dsharp.randnLike(fwdz1)
            let revx1 = x
            let revy1 = y.reverseDiff()
            let revx2 = x
            let revy2 = y.reverseDiff()
            let revz1 = op1 revx1 revy1
            let revz2 = op2 revx2 revy2
            revz1.reverse(zd)
            revz2.reverse(zd)
            let revxd1 = revx1.isNoDiff
            let revxd2 = revx2.isNoDiff
            let revyd1 = revy1.derivative
            let revyd2 = revy2.derivative            

            Assert.True(fwdz1.allclose(fwdz2, 0.01))
            Assert.True(fwdzd1.allclose(fwdzd2, 0.01))
            Assert.True(revz1.allclose(revz2, 0.01))
            Assert.CheckEqual(revxd1, revxd2)
            Assert.True(revyd1.allclose(revyd2, 0.01))

    [<Test>]
    let TestExtensions() =
        let n = 10
        compareUnaryOps (fun x -> x.sin()) (fun x -> x.sinExt()) n (fun () -> dsharp.randn(10))
        compareUnaryOps (fun x -> x.cos()) (fun x -> x.cosExt()) n (fun () -> dsharp.randn(10))
        compareUnaryOps (fun x -> x.exp()) (fun x -> x.expExt()) n (fun () -> dsharp.randn(10))
        compareUnaryOps (fun x -> x.log()) (fun x -> x.logExt()) n (fun () -> dsharp.randn(10).abs())
        compareUnaryOps (fun x -> x.transpose()) (fun x -> x.transposeExt()) n (fun () -> dsharp.randn([4; 3]))

        compareBinaryOps (fun x y -> x.pow(y)) (fun x y -> x.powExt(y)) n (fun () -> dsharp.randn(10).abs(), dsharp.randn(10))
        compareBinaryOps (fun x y -> x.mul(y)) (fun x y -> x.mulExt(y)) n (fun () -> dsharp.randn(10), dsharp.randn(10))
        compareBinaryOps (fun x y -> x.matmul(y)) (fun x y -> x.matmulExt(y)) 10 (fun () -> dsharp.randn([4; 3]), dsharp.randn([3; 5]))
