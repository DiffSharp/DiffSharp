namespace Tests

open NUnit.Framework
open DiffSharp

[<TestFixture>]
type TestDiffSharp () =

    let rosenbrock (x:Tensor) = 
        let x, y = x.[0], x.[1]
        (1. - x)**2 + 100. * (y - x**2)**2
    let rosenbrockGrad (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.tensor([-2*(1-x)-400*x*(-(x**2) + y); 200*(-(x**2) + y)])
    let rosenbrockHessian (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.tensor([[2.+1200.*x*x-400.*y, -400.*x],[-400.*x, 200.*dsharp.one()]])

    let fscalarvect3 (x:Tensor) = dsharp.stack([sin x; exp x; cos x])
    let fscalarvect3Diff (x:Tensor) = dsharp.stack([cos x; exp x; -sin x])
    let fscalarvect3Diff2 (x:Tensor) = dsharp.stack([-sin x; exp x; -cos x])
    let fscalarvect3Diff3 (x:Tensor) = dsharp.stack([-cos x; exp x; sin x])

    let fvect2vect2 (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.stack([x*x*y; 5*x+sin y])
    let fvect2vect2Jacobian (x:Tensor) = 
        let x, y = x.[0], x.[1]
        dsharp.tensor([[2*x*y; x*x];[dsharp.tensor(5.); cos y]])

    let fvect3vect2 (x:Tensor) = 
        let x, y, z = x.[0], x.[1], x.[2]
        dsharp.stack([x*y+2*y*z;2*x*y*y*z])
    let fvect3vect2Jacobian (x:Tensor) = 
        let x, y, z = x.[0], x.[1], x.[2]
        dsharp.tensor([[y;x+2*z;2*y];[2*y*y*z;4*x*y*z;2*x*y*y]])

    let fvect3vect3 (x:Tensor) = 
        let r, theta, phi = x.[0], x.[1], x.[2]
        dsharp.stack([r*(sin phi)*(cos theta); r*(sin phi)*(sin theta); r*cos phi])
    let fvect3vect3Jacobian (x:Tensor) = 
        let r, theta, phi = x.[0], x.[1], x.[2]
        dsharp.tensor([[(sin phi)*(cos theta); -r*(sin phi)*(sin theta); r*(cos phi)*(cos theta)];[(sin phi)*(sin theta); r*(sin phi)*(cos theta); r*(cos phi)*(sin theta)];[cos phi; dsharp.zero(); -r*sin phi]])

    let fvect3vect4 (x:Tensor) =
        let y1, y2, y3, y4 = x.[0], 5*x.[2], 4*x.[1]*x.[1]-2*x.[2],x.[2]*sin x.[0]
        dsharp.stack([y1;y2;y3;y4])
    let fvect3vect4Jacobian (x:Tensor) =
        let z, o = dsharp.zero(), dsharp.one()
        dsharp.tensor([[o,z,z],[z,z,5*o],[z,8*x.[1],-2*o],[x.[2]*cos x.[0],z,sin x.[0]]])

    [<SetUp>]
    member this.Setup () =
        ()

    [<Test>]
    member this.TestZeros () =
        let t = dsharp.zeros([2;3])
        let tCorrect = dsharp.tensor([[0,0,0],[0,0,0]])
        Assert.AreEqual(tCorrect, t)

    [<Test>]
    member this.TestOnes () =
        let t = dsharp.ones([2;3])
        let tCorrect = dsharp.tensor([[1,1,1],[1,1,1]])
        Assert.AreEqual(tCorrect, t)

    [<Test>]
    member this.TestRand () =
        let t = dsharp.rand([1000])
        let tMean = t.mean()
        let tMeanCorrect = dsharp.tensor(0.5)
        let tStddev = t.stddev()
        let tStddevCorrect = dsharp.tensor(1./12.) |> dsharp.sqrt
        Assert.True(tMeanCorrect.allclose(tMean, 0.1))
        Assert.True(tStddevCorrect.allclose(tStddev, 0.1))

    [<Test>]
    member this.TestRandn () =
        let t = dsharp.randn([1000])
        let tMean = t.mean()
        let tMeanCorrect = dsharp.tensor(0.)
        let tStddev = t.stddev()
        let tStddevCorrect = dsharp.tensor(1.)
        printfn "%A %A" tMean tMeanCorrect
        printfn "%A %A" tStddev tStddevCorrect
        Assert.True(tMeanCorrect.allclose(tMean, 0.1, 0.1))
        Assert.True(tStddevCorrect.allclose(tStddev, 0.1, 0.1))

    [<Test>]
    member this.TestArange () =
        let t1 = dsharp.arange(5.)
        let t1Correct = dsharp.tensor([0.,1.,2.,3.,4.])
        let t2 = dsharp.arange(startVal=1., endVal=4.)
        let t2Correct = dsharp.tensor([1.,2.,3.])
        let t3 = dsharp.arange(startVal=1., endVal=2.5, step=0.5)
        let t3Correct = dsharp.tensor([1.,1.5,2.])
        Assert.AreEqual(t1Correct, t1)
        Assert.AreEqual(t2Correct, t2)
        Assert.AreEqual(t3Correct, t3)

    [<Test>]
    member this.TestSeed () =
        dsharp.seed(123)
        let t = dsharp.rand([10])
        dsharp.seed(123)
        let t2 = dsharp.rand([10])
        Assert.AreEqual(t, t2)

    [<Test>]
    member this.TestDiff () =
        let x = dsharp.tensor(1.5)
        let fx, d = dsharp.pdiff fscalarvect3 x
        let d2 = dsharp.diff fscalarvect3 x
        let nfx, nd = dsharp.numpdiff 1e-5 fscalarvect3 x
        let nd2 = dsharp.numdiff 1e-5 fscalarvect3 x
        let fxCorrect = fscalarvect3 x
        let dCorrect = fscalarvect3Diff x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(fxCorrect, nfx)
        Assert.AreEqual(dCorrect, d)
        Assert.AreEqual(dCorrect, d2)
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))

    [<Test>]
    member this.TestDiff2 () =
        let x = dsharp.tensor(1.5)
        let fx, d = dsharp.pdiff2 fscalarvect3 x
        let d2 = dsharp.diff2 fscalarvect3 x
        let nfx, nd = dsharp.numpdiff2 1e-2 fscalarvect3 x
        let nd2 = dsharp.numdiff2 1e-2 fscalarvect3 x
        let fxCorrect = fscalarvect3 x
        let dCorrect = fscalarvect3Diff2 x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(fxCorrect, nfx)
        Assert.AreEqual(dCorrect, d)
        Assert.AreEqual(dCorrect, d2)
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))

    [<Test>]
    member this.TestDiffn () =
        let x = dsharp.tensor(1.5)
        let fx, d = dsharp.pdiffn 3 fscalarvect3 x
        let d2 = dsharp.diffn 3 fscalarvect3 x
        let fxCorrect = fscalarvect3 x
        let dCorrect = fscalarvect3Diff3 x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(dCorrect, d)
        Assert.AreEqual(dCorrect, d2)

    [<Test>]
    member this.TestGrad () =
        let x = dsharp.tensor([1.5;2.5])
        let fx, g = dsharp.pgrad rosenbrock x
        let g2 = dsharp.grad rosenbrock x
        let fxCorrect = rosenbrock x
        let gCorrect = rosenbrockGrad x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(gCorrect, g)
        Assert.AreEqual(gCorrect, g2)

    [<Test>]
    member this.TestGradv () =
        let x = dsharp.tensor([1.5;2.5])
        let v = dsharp.tensor([2.75;-3.5])
        let fx, gv = dsharp.pgradv rosenbrock x v
        let gv2 = dsharp.gradv rosenbrock x v
        let nfx, ngv = dsharp.numpgradv 1e-5 rosenbrock x v
        let ngv2 = dsharp.numgradv 1e-5 rosenbrock x v
        let fxCorrect = rosenbrock x
        let gvCorrect = dsharp.dot(rosenbrockGrad x,  v)
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(fxCorrect, nfx)
        Assert.AreEqual(gvCorrect, gv)
        Assert.AreEqual(gvCorrect, gv2)
        Assert.True(gvCorrect.allclose(ngv, 0.1))
        Assert.True(gvCorrect.allclose(ngv2, 0.1))

    [<Test>]
    member this.TestJacobianv () =
        let x = dsharp.tensor([1.5, 2.5, 3.])
        let v = dsharp.tensor([2.75, -3.5, 4.])
        let fx, jv = dsharp.pjacobianv fvect3vect2 x v
        let jv2 = dsharp.jacobianv fvect3vect2 x v
        let fxCorrect = fvect3vect2 x
        let jvCorrect = dsharp.matmul(fvect3vect2Jacobian x,  v.view([-1;1])).view(-1)
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(jvCorrect, jv)
        Assert.AreEqual(jvCorrect, jv2)

    [<Test>]
    member this.TestJacobianTv () =
        let x = dsharp.tensor([1.5, 2.5, 3.])
        let v = dsharp.tensor([2.75, -3.5])
        let fx, jTv = dsharp.pjacobianTv fvect3vect2 x v
        let jTv2 = dsharp.jacobianTv fvect3vect2 x v
        let fxCorrect = fvect3vect2 x
        let jTvCorrect = dsharp.matmul(v.view([1;-1]), fvect3vect2Jacobian x).view(-1)
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(jTvCorrect, jTv)
        Assert.AreEqual(jTvCorrect, jTv2)

    [<Test>]
    member this.TestJacobian () =
        let x = dsharp.arange(2.)
        let fx, j = dsharp.pjacobian fvect2vect2 x
        let j2 = dsharp.jacobian fvect2vect2 x
        let fxCorrect = fvect2vect2 x
        let jCorrect = fvect2vect2Jacobian x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(jCorrect, j)
        Assert.AreEqual(jCorrect, j2)

        let x = dsharp.arange(3.)
        let fx, j = dsharp.pjacobian fvect3vect2 x
        let j2 = dsharp.jacobian fvect3vect2 x
        let fxCorrect = fvect3vect2 x
        let jCorrect = fvect3vect2Jacobian x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(jCorrect, j)
        Assert.AreEqual(jCorrect, j2)

        let x = dsharp.arange(3.)
        let fx, j = dsharp.pjacobian fvect3vect3 x
        let j2 = dsharp.jacobian fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let jCorrect = fvect3vect3Jacobian x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(jCorrect, j)
        Assert.AreEqual(jCorrect, j2)

        let x = dsharp.arange(3.)
        let fx, j = dsharp.pjacobian fvect3vect4 x
        let j2 = dsharp.jacobian fvect3vect4 x
        let fxCorrect = fvect3vect4 x
        let jCorrect = fvect3vect4Jacobian x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(jCorrect, j)
        Assert.AreEqual(jCorrect, j2)

    [<Test>]
    member this.TestGradhessianv () =
        let x = dsharp.tensor([1.5, 2.5])
        let v = dsharp.tensor([0.5, -2.])
        let fx, gv, hv = dsharp.pgradhessianv rosenbrock x v
        let gv2, hv2 = dsharp.gradhessianv rosenbrock x v
        let fxCorrect = rosenbrock x
        let gvCorrect = dsharp.dot(rosenbrockGrad x,  v)        
        let hvCorrect = dsharp.matmul(rosenbrockHessian x,  v.view([-1;1])).view(-1)
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(gvCorrect, gv)
        Assert.AreEqual(gvCorrect, gv2)
        Assert.AreEqual(hvCorrect, hv)
        Assert.AreEqual(hvCorrect, hv2)

    [<Test>]
    member this.TestGradhessian () =
        let x = dsharp.tensor([1.5, 2.5])
        let fx, g, h = dsharp.pgradhessian rosenbrock x
        let g2, h2 = dsharp.gradhessian rosenbrock x
        let fxCorrect = rosenbrock x
        let gCorrect = rosenbrockGrad x
        let hCorrect = rosenbrockHessian x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(gCorrect, g)
        Assert.AreEqual(gCorrect, g2)
        Assert.AreEqual(hCorrect, h)
        Assert.AreEqual(hCorrect, h2)

    [<Test>]
    member this.TestHessianv () =
        let x = dsharp.tensor([1.5, 2.5])
        let v = dsharp.tensor([0.5, -2.])
        let fx, hv = dsharp.phessianv rosenbrock x v
        let hv2 = dsharp.hessianv rosenbrock x v
        let fxCorrect = rosenbrock x
        let hvCorrect = dsharp.matmul(rosenbrockHessian x,  v.view([-1;1])).view(-1)
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(hvCorrect, hv)
        Assert.AreEqual(hvCorrect, hv2)

    [<Test>]
    member this.TestHessian () =
        let x = dsharp.tensor([1.5, 2.5])
        let fx, h = dsharp.phessian rosenbrock x
        let h2 = dsharp.hessian rosenbrock x
        let fxCorrect = rosenbrock x
        let hCorrect = rosenbrockHessian x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(hCorrect, h)
        Assert.AreEqual(hCorrect, h2)

    [<Test>]
    member this.TestLaplacian () =
        let x = dsharp.tensor([1.5, 2.5])
        let fx, l = dsharp.plaplacian rosenbrock x
        let l2 = dsharp.laplacian rosenbrock x
        let fxCorrect = rosenbrock x
        let lCorrect = (rosenbrockHessian x).trace()
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(lCorrect, l)
        Assert.AreEqual(lCorrect, l2)

    [<Test>]
    member this.TestCurl () =
        let x = dsharp.tensor([1.5, 2.5, 0.2])
        let fx, c = dsharp.pcurl fvect3vect3 x
        let c2 = dsharp.curl fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let cCorrect = dsharp.tensor([-0.879814, -2.157828, 0.297245])
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(cCorrect.allclose(c))
        Assert.True(cCorrect.allclose(c2))

    [<Test>]
    member this.TestDivergence () =
        let x = dsharp.tensor([1.5, 2.5, 0.2])
        let fx, d = dsharp.pdivergence fvect3vect3 x
        let d2 = dsharp.divergence fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let dCorrect = dsharp.tensor(-0.695911)
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(dCorrect.allclose(d))
        Assert.True(dCorrect.allclose(d2))

    [<Test>]
    member this.TestCurlDivergence () =
        let x = dsharp.tensor([1.5, 2.5, 0.2])
        let fx, c, d = dsharp.pcurldivergence fvect3vect3 x
        let c2, d2 = dsharp.curldivergence fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let cCorrect = dsharp.tensor([-0.879814, -2.157828, 0.297245])
        let dCorrect = dsharp.tensor(-0.695911)
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(cCorrect.allclose(c))
        Assert.True(cCorrect.allclose(c2))
        Assert.True(dCorrect.allclose(d))
        Assert.True(dCorrect.allclose(d2))