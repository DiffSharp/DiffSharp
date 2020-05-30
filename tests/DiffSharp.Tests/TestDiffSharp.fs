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

    let fscalarscalar (x:Tensor) = dsharp.sin x
    let fscalarscalarDiff (x:Tensor) = dsharp.cos x

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
    member this.TestZero () =
        let t = dsharp.zero()
        let tCorrect = dsharp.tensor(0)
        Assert.AreEqual(tCorrect, t)

    [<Test>]
    member this.TestZeros () =
        let t = dsharp.zeros([2;3])
        let tCorrect = dsharp.tensor([[0,0,0],[0,0,0]])
        Assert.AreEqual(tCorrect, t)

    [<Test>]
    member this.TestOne () =
        let t = dsharp.one()
        let tCorrect = dsharp.tensor(1)
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
        for combo in Combos.All do
            combo.randint(0,10,1) |> ignore // To ensure the backend assembly is loaded before dsharp.seed is called
            dsharp.seed(123)
            let t = combo.randint(0,10,[25])
            dsharp.seed(123)
            let t2 = combo.randint(0,10,[25])
            Assert.AreEqual(t, t2)

    [<Test>]
    member this.TestDiff () =
        let x = dsharp.tensor(1.5)
        let fx, d = dsharp.fdiff fscalarvect3 x
        let d2 = dsharp.diff fscalarvect3 x
        let nfx, nd = dsharp.numfdiff 1e-5 fscalarvect3 x
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
        let fx, d = dsharp.fdiff2 fscalarvect3 x
        let d2 = dsharp.diff2 fscalarvect3 x
        let nfx, nd = dsharp.numfdiff2 1e-2 fscalarvect3 x
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
        let fx, d = dsharp.fdiffn 3 fscalarvect3 x
        let d2 = dsharp.diffn 3 fscalarvect3 x
        let fxCorrect = fscalarvect3 x
        let dCorrect = fscalarvect3Diff3 x
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(dCorrect, d)
        Assert.AreEqual(dCorrect, d2)

    [<Test>]
    member this.TestGrad () =
        let x = dsharp.tensor([1.5;2.5])
        let fx1, g1 = dsharp.fgrad rosenbrock x
        let fx2, g2 = dsharp.fg rosenbrock x
        let g3 = dsharp.grad rosenbrock x
        let g4 = dsharp.g rosenbrock x
        let nfx1, ng1 = dsharp.numfgrad 1e-6 rosenbrock x
        let nfx2, ng2 = dsharp.numfg 1e-6 rosenbrock x
        let ng3 = dsharp.numgrad 1e-6 rosenbrock x
        let ng4 = dsharp.numg 1e-6 rosenbrock x
        let fxCorrect = rosenbrock x
        let gCorrect = rosenbrockGrad x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(gCorrect, g1)
        Assert.AreEqual(gCorrect, g2)
        Assert.AreEqual(gCorrect, g3)
        Assert.AreEqual(gCorrect, g4)
        Assert.True(gCorrect.allclose(ng1, 0.1))
        Assert.True(gCorrect.allclose(ng2, 0.1))
        Assert.True(gCorrect.allclose(ng3, 0.1))
        Assert.True(gCorrect.allclose(ng4, 0.1))

    [<Test>]
    member this.TestGradScalarToScalar () =
        let x = dsharp.tensor(1.5)
        let fx1, g1 = dsharp.fgrad fscalarscalar x
        let fx2, g2 = dsharp.fg fscalarscalar x
        let g3 = dsharp.grad fscalarscalar x
        let g4 = dsharp.g fscalarscalar x
        let nfx1, ng1 = dsharp.numfgrad 1e-3 fscalarscalar x
        let nfx2, ng2 = dsharp.numfg 1e-3 fscalarscalar x
        let ng3 = dsharp.numgrad 1e-3 fscalarscalar x
        let ng4 = dsharp.numg 1e-3 fscalarscalar x
        let fxCorrect = fscalarscalar x
        let gCorrect = fscalarscalarDiff x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(gCorrect, g1)
        Assert.AreEqual(gCorrect, g2)
        Assert.AreEqual(gCorrect, g3)
        Assert.AreEqual(gCorrect, g4)
        Assert.True(gCorrect.allclose(ng1, 0.1))
        Assert.True(gCorrect.allclose(ng2, 0.1))
        Assert.True(gCorrect.allclose(ng3, 0.1))
        Assert.True(gCorrect.allclose(ng4, 0.1))

    [<Test>]
    member this.TestGradv () =
        let x = dsharp.tensor([1.5;2.5])
        let v = dsharp.tensor([2.75;-3.5])
        let fx1, gv1 = dsharp.fgradv rosenbrock x v
        let fx2, gv2 = dsharp.fgvp rosenbrock x v
        let gv3 = dsharp.gradv rosenbrock x v
        let gv4 = dsharp.gvp rosenbrock x v
        let nfx1, ngv1 = dsharp.numfgradv 1e-5 rosenbrock x v
        let nfx2, ngv2 = dsharp.numfgvp 1e-5 rosenbrock x v
        let ngv3 = dsharp.numgradv 1e-5 rosenbrock x v
        let ngv4 = dsharp.numgvp 1e-5 rosenbrock x v
        let fxCorrect = rosenbrock x
        let gvCorrect = dsharp.dot(rosenbrockGrad x,  v)
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(gvCorrect, gv1)
        Assert.AreEqual(gvCorrect, gv2)
        Assert.AreEqual(gvCorrect, gv3)
        Assert.AreEqual(gvCorrect, gv4)
        Assert.True(gvCorrect.allclose(ngv1, 0.1))
        Assert.True(gvCorrect.allclose(ngv2, 0.1))
        Assert.True(gvCorrect.allclose(ngv3, 0.1))
        Assert.True(gvCorrect.allclose(ngv4, 0.1))

    [<Test>]
    member this.TestJacobianv () =
        let x = dsharp.tensor([1.5, 2.5, 3.])
        let v = dsharp.tensor([2.75, -3.5, 4.])
        let fx1, jv1 = dsharp.fjacobianv fvect3vect2 x v
        let fx2, jv2 = dsharp.fjvp fvect3vect2 x v
        let jv3 = dsharp.jacobianv fvect3vect2 x v
        let jv4 = dsharp.jvp fvect3vect2 x v
        let nfx1, njv1 = dsharp.numfjacobianv 1e-3 fvect3vect2 x v
        let nfx2, njv2 = dsharp.numfjvp 1e-3 fvect3vect2 x v
        let njv3 = dsharp.numjacobianv 1e-3 fvect3vect2 x v
        let njv4 = dsharp.numjvp 1e-3 fvect3vect2 x v
        let fxCorrect = fvect3vect2 x
        let jvCorrect = dsharp.matmul(fvect3vect2Jacobian x,  v.view([-1;1])).view(-1)
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(jvCorrect, jv1)
        Assert.AreEqual(jvCorrect, jv2)
        Assert.AreEqual(jvCorrect, jv3)
        Assert.AreEqual(jvCorrect, jv4)
        Assert.True(jvCorrect.allclose(njv1, 0.1))
        Assert.True(jvCorrect.allclose(njv2, 0.1))
        Assert.True(jvCorrect.allclose(njv3, 0.1))
        Assert.True(jvCorrect.allclose(njv4, 0.1))

    [<Test>]
    member this.TestJacobianTv () =
        let x = dsharp.tensor([1.5, 2.5, 3.])
        let v = dsharp.tensor([2.75, -3.5])
        let fx, jTv = dsharp.fjacobianTv fvect3vect2 x v
        let jTv2 = dsharp.jacobianTv fvect3vect2 x v
        let fxCorrect = fvect3vect2 x
        let jTvCorrect = dsharp.matmul(v.view([1;-1]), fvect3vect2Jacobian x).view(-1)
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(jTvCorrect, jTv)
        Assert.AreEqual(jTvCorrect, jTv2)

    [<Test>]
    member this.TestJacobian () =
        let x = dsharp.arange(2.)
        let fx1, j1 = dsharp.fjacobian fvect2vect2 x
        let fx2, j2 = dsharp.fj fvect2vect2 x
        let j3 = dsharp.jacobian fvect2vect2 x
        let j4 = dsharp.j fvect2vect2 x
        let nfx1, nj1 = dsharp.numfjacobian 1e-4 fvect2vect2 x
        let nfx2, nj2 = dsharp.numfj 1e-4 fvect2vect2 x
        let nj3 = dsharp.numjacobian 1e-4 fvect2vect2 x
        let nj4 = dsharp.numj 1e-4 fvect2vect2 x
        let fxCorrect = fvect2vect2 x
        let jCorrect = fvect2vect2Jacobian x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(jCorrect, j1)
        Assert.AreEqual(jCorrect, j2)
        Assert.AreEqual(jCorrect, j3)
        Assert.AreEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

        let x = dsharp.arange(3.)
        let fx1, j1 = dsharp.fjacobian fvect3vect2 x
        let fx2, j2 = dsharp.fj fvect3vect2 x
        let j3 = dsharp.jacobian fvect3vect2 x
        let j4 = dsharp.j fvect3vect2 x
        let nfx1, nj1 = dsharp.numfjacobian 1e-4 fvect3vect2 x
        let nfx2, nj2 = dsharp.numfj 1e-4 fvect3vect2 x
        let nj3 = dsharp.numjacobian 1e-4 fvect3vect2 x
        let nj4 = dsharp.numj 1e-4 fvect3vect2 x
        let fxCorrect = fvect3vect2 x
        let jCorrect = fvect3vect2Jacobian x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(jCorrect, j1)
        Assert.AreEqual(jCorrect, j2)
        Assert.AreEqual(jCorrect, j3)
        Assert.AreEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

        let x = dsharp.arange(3.)
        let fx1, j1 = dsharp.fjacobian fvect3vect3 x
        let fx2, j2 = dsharp.fj fvect3vect3 x
        let j3 = dsharp.jacobian fvect3vect3 x
        let j4 = dsharp.j fvect3vect3 x
        let nfx1, nj1 = dsharp.numfjacobian 1e-4 fvect3vect3 x
        let nfx2, nj2 = dsharp.numfj 1e-4 fvect3vect3 x
        let nj3 = dsharp.numjacobian 1e-4 fvect3vect3 x
        let nj4 = dsharp.numj 1e-4 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let jCorrect = fvect3vect3Jacobian x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(jCorrect, j1)
        Assert.AreEqual(jCorrect, j2)
        Assert.AreEqual(jCorrect, j3)
        Assert.AreEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

        let x = dsharp.arange(3.)
        let fx1, j1 = dsharp.fjacobian fvect3vect4 x
        let fx2, j2 = dsharp.fj fvect3vect4 x
        let j3 = dsharp.jacobian fvect3vect4 x
        let j4 = dsharp.j fvect3vect4 x
        let nfx1, nj1 = dsharp.numfjacobian 1e-4 fvect3vect4 x
        let nfx2, nj2 = dsharp.numfj 1e-4 fvect3vect4 x
        let nj3 = dsharp.numjacobian 1e-4 fvect3vect4 x
        let nj4 = dsharp.numj 1e-4 fvect3vect4 x
        let fxCorrect = fvect3vect4 x
        let jCorrect = fvect3vect4Jacobian x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(jCorrect, j1)
        Assert.AreEqual(jCorrect, j2)
        Assert.AreEqual(jCorrect, j3)
        Assert.AreEqual(jCorrect, j4)
        Assert.True(jCorrect.allclose(nj1, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj2, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj3, 0.1, 0.1))
        Assert.True(jCorrect.allclose(nj4, 0.1, 0.1))

    [<Test>]
    member this.TestGradhessianv () =
        let x = dsharp.tensor([1.5, 2.5])
        let v = dsharp.tensor([0.5, -2.])
        let fx1, gv1, hv1 = dsharp.fgradhessianv rosenbrock x v
        let fx2, gv2, hv2 = dsharp.fghvp rosenbrock x v
        let gv3, hv3 = dsharp.gradhessianv rosenbrock x v
        let gv4, hv4 = dsharp.ghvp rosenbrock x v
        let fxCorrect = rosenbrock x
        let gvCorrect = dsharp.dot(rosenbrockGrad x,  v)        
        let hvCorrect = dsharp.matmul(rosenbrockHessian x,  v.view([-1;1])).view(-1)
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(gvCorrect, gv1)
        Assert.AreEqual(gvCorrect, gv2)
        Assert.AreEqual(gvCorrect, gv3)
        Assert.AreEqual(gvCorrect, gv4)
        Assert.AreEqual(hvCorrect, hv1)
        Assert.AreEqual(hvCorrect, hv2)
        Assert.AreEqual(hvCorrect, hv3)
        Assert.AreEqual(hvCorrect, hv4)

    [<Test>]
    member this.TestGradhessian () =
        let x = dsharp.tensor([1.5, 2.5])
        let fx1, g1, h1 = dsharp.fgradhessian rosenbrock x
        let fx2, g2, h2 = dsharp.fgh rosenbrock x
        let g3, h3 = dsharp.gradhessian rosenbrock x
        let g4, h4 = dsharp.gh rosenbrock x
        let nfx1, ng1, nh1 = dsharp.numfgradhessian 1e-3 rosenbrock x
        let nfx2, ng2, nh2 = dsharp.numfgh 1e-3 rosenbrock x
        let ng3, nh3 = dsharp.numgradhessian 1e-3 rosenbrock x
        let ng4, nh4 = dsharp.numgh 1e-3 rosenbrock x
        let fxCorrect = rosenbrock x
        let gCorrect = rosenbrockGrad x
        let hCorrect = rosenbrockHessian x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(gCorrect, g1)
        Assert.AreEqual(gCorrect, g2)
        Assert.AreEqual(gCorrect, g3)
        Assert.AreEqual(gCorrect, g4)
        Assert.AreEqual(hCorrect, h1)
        Assert.AreEqual(hCorrect, h2)
        Assert.AreEqual(hCorrect, h3)
        Assert.AreEqual(hCorrect, h4)
        Assert.True(gCorrect.allclose(ng1, 0.1))
        Assert.True(gCorrect.allclose(ng2, 0.1))
        Assert.True(gCorrect.allclose(ng3, 0.1))
        Assert.True(gCorrect.allclose(ng4, 0.1))
        Assert.True(hCorrect.allclose(nh1, 0.1))
        Assert.True(hCorrect.allclose(nh2, 0.1))
        Assert.True(hCorrect.allclose(nh3, 0.1))
        Assert.True(hCorrect.allclose(nh4, 0.1))

    [<Test>]
    member this.TestHessianv () =
        let x = dsharp.tensor([1.5, 2.5])
        let v = dsharp.tensor([0.5, -2.])
        let fx1, hv1 = dsharp.fhessianv rosenbrock x v
        let fx2, hv2 = dsharp.fhvp rosenbrock x v
        let hv3 = dsharp.hessianv rosenbrock x v
        let hv4 = dsharp.hvp rosenbrock x v
        let nfx1, nhv1 = dsharp.numfhessianv 1e-3 rosenbrock x v
        let nfx2, nhv2 = dsharp.numfhvp 1e-3 rosenbrock x v
        let nhv3 = dsharp.numhessianv 1e-3 rosenbrock x v
        let nhv4 = dsharp.numhvp 1e-3 rosenbrock x v
        let fxCorrect = rosenbrock x
        let hvCorrect = dsharp.matmul(rosenbrockHessian x,  v.view([-1;1])).view(-1)
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(hvCorrect, hv1)
        Assert.AreEqual(hvCorrect, hv2)
        Assert.AreEqual(hvCorrect, hv3)
        Assert.AreEqual(hvCorrect, hv4)
        Assert.True(hvCorrect.allclose(nhv1, 0.1))
        Assert.True(hvCorrect.allclose(nhv2, 0.1))
        Assert.True(hvCorrect.allclose(nhv3, 0.1))
        Assert.True(hvCorrect.allclose(nhv4, 0.1))

    [<Test>]
    member this.TestHessian () =
        let x = dsharp.tensor([1.5, 2.5])
        let fx1, h1 = dsharp.fhessian rosenbrock x
        let fx2, h2 = dsharp.fh rosenbrock x
        let h3 = dsharp.hessian rosenbrock x
        let h4 = dsharp.h rosenbrock x
        let nfx1, nh1 = dsharp.numfhessian 1e-3 rosenbrock x
        let nfx2, nh2 = dsharp.numfh 1e-3 rosenbrock x
        let nh3 = dsharp.numhessian 1e-3 rosenbrock x
        let nh4 = dsharp.numh 1e-3 rosenbrock x
        let fxCorrect = rosenbrock x
        let hCorrect = rosenbrockHessian x
        Assert.AreEqual(fxCorrect, fx1)
        Assert.AreEqual(fxCorrect, nfx1)
        Assert.AreEqual(fxCorrect, fx2)
        Assert.AreEqual(fxCorrect, nfx2)
        Assert.AreEqual(hCorrect, h1)
        Assert.AreEqual(hCorrect, h2)
        Assert.AreEqual(hCorrect, h3)
        Assert.AreEqual(hCorrect, h4)
        Assert.True(hCorrect.allclose(nh1, 0.1))
        Assert.True(hCorrect.allclose(nh2, 0.1))
        Assert.True(hCorrect.allclose(nh3, 0.1))
        Assert.True(hCorrect.allclose(nh4, 0.1))

    [<Test>]
    member this.TestLaplacian () =
        let x = dsharp.tensor([1.5, 2.5])
        let fx, l = dsharp.flaplacian rosenbrock x
        let l2 = dsharp.laplacian rosenbrock x
        let nfx, nl = dsharp.numflaplacian 1e-3 rosenbrock x
        let nl2 = dsharp.numlaplacian 1e-3 rosenbrock x
        let fxCorrect = rosenbrock x
        let lCorrect = (rosenbrockHessian x).trace()
        Assert.AreEqual(fxCorrect, fx)
        Assert.AreEqual(fxCorrect, nfx)
        Assert.AreEqual(lCorrect, l)
        Assert.AreEqual(lCorrect, l2)
        Assert.True(lCorrect.allclose(nl, 0.1))
        Assert.True(lCorrect.allclose(nl2, 0.1))

    [<Test>]
    member this.TestCurl () =
        let x = dsharp.tensor([1.5, 2.5, 0.2])
        let fx, c = dsharp.fcurl fvect3vect3 x
        let c2 = dsharp.curl fvect3vect3 x
        let nfx, nc = dsharp.numfcurl 1e-3 fvect3vect3 x
        let nc2 = dsharp.numcurl 1e-3 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let cCorrect = dsharp.tensor([-0.879814, -2.157828, 0.297245])
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(fxCorrect.allclose(nfx))
        Assert.True(cCorrect.allclose(c))
        Assert.True(cCorrect.allclose(c2))
        Assert.True(cCorrect.allclose(nc, 0.1))
        Assert.True(cCorrect.allclose(nc2, 0.1))

    [<Test>]
    member this.TestDivergence () =
        let x = dsharp.tensor([1.5, 2.5, 0.2])
        let fx, d = dsharp.fdivergence fvect3vect3 x
        let d2 = dsharp.divergence fvect3vect3 x
        let nfx, nd = dsharp.numfdivergence 1e-3 fvect3vect3 x
        let nd2 = dsharp.numdivergence 1e-3 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let dCorrect = dsharp.tensor(-0.695911)
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(fxCorrect.allclose(nfx))
        Assert.True(dCorrect.allclose(d))
        Assert.True(dCorrect.allclose(d2))
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))

    [<Test>]
    member this.TestCurlDivergence () =
        let x = dsharp.tensor([1.5, 2.5, 0.2])
        let fx, c, d = dsharp.fcurldivergence fvect3vect3 x
        let c2, d2 = dsharp.curldivergence fvect3vect3 x
        let nfx, nc, nd = dsharp.numfcurldivergence 1e-3 fvect3vect3 x
        let nc2, nd2 = dsharp.numcurldivergence 1e-3 fvect3vect3 x
        let fxCorrect = fvect3vect3 x
        let cCorrect = dsharp.tensor([-0.879814, -2.157828, 0.297245])
        let dCorrect = dsharp.tensor(-0.695911)
        Assert.True(fxCorrect.allclose(fx))
        Assert.True(fxCorrect.allclose(nfx))
        Assert.True(cCorrect.allclose(c))
        Assert.True(cCorrect.allclose(c2))
        Assert.True(cCorrect.allclose(nc, 0.1))
        Assert.True(cCorrect.allclose(nc2, 0.1))
        Assert.True(dCorrect.allclose(d))
        Assert.True(dCorrect.allclose(d2))
        Assert.True(dCorrect.allclose(nd, 0.1))
        Assert.True(dCorrect.allclose(nd2, 0.1))        


    [<Test>]
    member _.TestCanConfigure () =
        let device = Device.Default
        dsharp.config(device=Device.GPU)
        Assert.AreEqual(Device.GPU, Device.Default)
        dsharp.config(device=device)

        let backend = Backend.Default
        dsharp.config(backend=Backend.Torch)
        Assert.AreEqual(Backend.Torch, Backend.Default)
        dsharp.config(backend=backend)

        let dtype = Dtype.Default
        dsharp.config(dtype=Dtype.Int32)
        Assert.AreEqual(Dtype.Int32, Dtype.Default)
        dsharp.config(dtype=dtype)
