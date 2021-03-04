// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// and other contributors, see LICENSE in root of repository.
//
// BSD 2-Clause License. See LICENSE in root of repository.

namespace Tests

open NUnit.Framework
open DiffSharp
open DiffSharp.Util

#nowarn "0058"

[<TestFixture>]
type TestDerivatives () =

    [<Test>]
    member _.TestDerivativeAddTT () =
        for swap in [true; false] do
          for combo in Combos.AllDevicesAndBackendsFloat32 do
              let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
              let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([40., 50., 60.]))
              let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
              let fwdzCorrect = combo.tensor([5., 7., 9.])
              let fwdzd = fwdz.derivative
              let fwdzdCorrect = combo.tensor([50., 70., 90.])

              let revx = combo.tensor([1., 2., 3.]).reverseDiff()
              let revy = combo.tensor([4., 5., 6.]).reverseDiff()
              let revz = if swap then revy + revx else revx + revy
              let revzCorrect = combo.tensor([5., 7., 9.])
              revz.reverse(combo.tensor([100., 200., 300.]))
              let revxd = revx.derivative
              let revxdCorrect = combo.tensor([100., 200., 300.])
              let revyd = revy.derivative
              let revydCorrect = combo.tensor([100., 200., 300.])

              Assert.CheckEqual(fwdzCorrect, fwdz)
              Assert.CheckEqual(fwdzdCorrect, fwdzd)
              Assert.CheckEqual(revzCorrect, revz)
              Assert.CheckEqual(revxdCorrect, revxd)
              Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddTTConst () =
        for swap in [true; false] do
          for combo in Combos.AllDevicesAndBackendsFloat32 do
              let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
              let fwdy = combo.tensor([4., 5., 6.])
              let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
              let fwdzCorrect = combo.tensor([5., 7., 9.])
              let fwdzd = fwdz.derivative
              let fwdzdCorrect = combo.tensor([10., 20., 30.])

              let revx = combo.tensor([1., 2., 3.]).reverseDiff()
              let revy = combo.tensor([4., 5., 6.])
              let revz = if swap then revy + revx else revx + revy
              let revzCorrect = combo.tensor([5., 7., 9.])
              revz.reverse(combo.tensor([100., 200., 300.]))
              let revxd = revx.derivative
              let revxdCorrect = combo.tensor([100., 200., 300.])
              let revyd = revy.isNoDiff()
              let revydCorrect = true

              Assert.CheckEqual(fwdzCorrect, fwdz)
              Assert.CheckEqual(fwdzdCorrect, fwdzd)
              Assert.CheckEqual(revzCorrect, revz)
              Assert.CheckEqual(revxdCorrect, revxd)
              Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddTT0 () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
                let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(40.))
                let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
                let fwdzCorrect = combo.tensor([5., 6., 7.])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([50., 60., 70.])

                let revx = combo.tensor([1., 2., 3.]).reverseDiff()
                let revy = combo.tensor(4.).reverseDiff()
                let revz = if swap then revy + revx else revx + revy
                let revzCorrect = combo.tensor([5., 6., 7.])
                revz.reverse(combo.tensor([100., 200., 300.]))
                let revxd = revx.derivative
                let revxdCorrect = combo.tensor([100., 200., 300.])
                let revyd = revy.derivative
                let revydCorrect = combo.tensor(600.)

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddTT0Const () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
                let fwdy = combo.tensor(4.)
                let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
                let fwdzCorrect = combo.tensor([5., 6., 7.])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([10., 20., 30.])

                let revx = combo.tensor([1., 2., 3.]).reverseDiff()
                let revy = combo.tensor(4.)
                let revz = if swap then revy + revx else revx + revy
                let revzCorrect = combo.tensor([5., 6., 7.])
                revz.reverse(combo.tensor([100., 200., 300.]))
                let revxd = revx.derivative
                let revxdCorrect = combo.tensor([100., 200., 300.])
                let revyd = revy.isNoDiff()
                let revydCorrect = true

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddTConstT0 () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([1., 2., 3.])
                let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(40.))
                let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
                let fwdzCorrect = combo.tensor([5., 6., 7.])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([40., 40., 40.])

                let revx = combo.tensor([1., 2., 3.])
                let revy = combo.tensor(4.).reverseDiff()
                let revz = if swap then revy + revx else revx + revy
                let revzCorrect = combo.tensor([5., 6., 7.])
                revz.reverse(combo.tensor([100., 200., 300.]))
                let revxd = revx.isNoDiff()
                let revxdCorrect = true
                let revyd = revy.derivative
                let revydCorrect = combo.tensor(600.)

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddT2T1 () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([[1., 2.], [3., 4.]]).forwardDiff(combo.tensor([[10., 20.], [30., 40.]]))
                let fwdy = combo.tensor([5., 6.]).forwardDiff(combo.tensor([50., 60.]))
                let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
                let fwdzCorrect = combo.tensor([[6., 8.], [8., 10.]])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([[60., 80.], [80., 100.]])

                let revx = combo.tensor([[1., 2.], [3., 4.]]).reverseDiff()
                let revy = combo.tensor([5., 6.]).reverseDiff()
                let revz = if swap then revy + revx else revx + revy
                let revzCorrect = combo.tensor([[6., 8.], [8., 10.]])
                revz.reverse(combo.tensor([[100., 200.], [300., 400.]]))
                let revxd = revx.derivative
                let revxdCorrect = combo.tensor([[100., 200.], [300., 400.]])
                let revyd = revy.derivative
                let revydCorrect = combo.tensor([400., 600.])

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddT2T1Const () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([[1., 2.], [3., 4.]]).forwardDiff(combo.tensor([[10., 20.], [30., 40.]]))
                let fwdy = combo.tensor([5., 6.])
                let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
                let fwdzCorrect = combo.tensor([[6., 8.], [8., 10.]])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([[10., 20.], [30., 40.]])

                let revx = combo.tensor([[1., 2.], [3., 4.]]).reverseDiff()
                let revy = combo.tensor([5., 6.])
                let revz = if swap then revy + revx else revx + revy
                let revzCorrect = combo.tensor([[6., 8.], [8., 10.]])
                revz.reverse(combo.tensor([[100., 200.], [300., 400.]]))
                let revxd = revx.derivative
                let revxdCorrect = combo.tensor([[100., 200.], [300., 400.]])
                let revyd = revy.isNoDiff()
                let revydCorrect = true

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddT2ConstT1 () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([[1., 2.], [3., 4.]])
                let fwdy = combo.tensor([5., 6.]).forwardDiff(combo.tensor([50., 60.]))
                let fwdz = if swap then fwdy + fwdx else fwdx + fwdy
                let fwdzCorrect = combo.tensor([[6., 8.], [8., 10.]])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([[50., 60.], [50., 60.]])

                let revx = combo.tensor([[1., 2.], [3., 4.]])
                let revy = combo.tensor([5., 6.]).reverseDiff()
                let revz = if swap then revy + revx else revx + revy
                let revzCorrect = combo.tensor([[6., 8.], [8., 10.]])
                revz.reverse(combo.tensor([[100., 200.], [300., 400.]]))
                let revxd = revx.isNoDiff()
                let revxdCorrect = true
                let revyd = revy.derivative
                let revydCorrect = combo.tensor([400., 600.])

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeExpand () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.]; [2.]]).forwardDiff(combo.tensor([[5.]; [6.]])) // 2x1
            let fwdz = fwdx.expand([2;2;2]) // 2x2x2 = [[[1.;1]; [2.;2]]; [[1.;1]; [2.;2]]]
            let fwdzCorrect = combo.tensor([[[1.;1.]; [2.;2.]]; [[1.;1.]; [2.;2.]]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor ([[[5., 5.], [6., 6.]], [[5., 5.], [6., 6.]]])

            (* Python:
            import torch 
            t1 = torch.tensor([[1.], [2.]], requires_grad=True)
            revz = t1.expand([2,2,2])
            revz.backward(torch.tensor([[[3.,3.], [6.,6.]], [[3.,3.], [6.,6.]]]))
            t1.grad
            --> tensor([[12.],[24.]])
            *)
            let revx = combo.tensor([[1.]; [2.]]).reverseDiff()
            let revz = revx.expand([2;2;2])
            let revzCorrect = combo.tensor([[[1.;1.]; [2.;2.]]; [[1.;1.]; [2.;2.]]])
            revz.reverse(combo.tensor([[[3.;3.]; [6.;6.]]; [[3.;3.]; [6.;6.]]]))
            let revxd = revx.derivative
            // Note: The 4x'3' accumulate to the first entry, the 4x'6' accumulate to the second entry
            let revxdCorrect = combo.tensor [[12.], [24.]]

            Assert.CheckEqual(fwdz, fwdzCorrect)
            Assert.CheckEqual(fwdzd,fwdzdCorrect)
            Assert.CheckEqual(revz, revzCorrect)
            Assert.CheckEqual(revxd,revxdCorrect)

    [<Test>]
    member _.TestAddWithBroadcastSystematic () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            // This is a somewhat adhoc extra test to do a whole range of additiosn
            // with broadcast, mainly to check that not problems occur in taking the
            // derivatives.
            //
            // Systematically do all allowed broadcasts into 2x3x4
            // 2x3x4 + 1  (broadcast --> 2x3x4)
            // 2x3x4 + 4  (broadcast --> 2x3x4)
            // 2x3x4 + 1x1  (broadcast --> 2x3x4)
            // 2x3x4 + 3x1  (broadcast --> 2x3x4)
            // 2x3x4 + 1x4  (broadcast --> 2x3x4)
            // etc.
            let t1a = combo.tensor([ [ [1.; 2.; 3.; 4.]; [5.; 6.; 7.; 8.]; [9.; 10.; 11.; 12.] ];
                                      [ [13.; 14.; 15.; 16.]; [17.; 18.; 19.; 20.]; [21.; 22.; 23.; 24.] ]  ])
            
            // Get all the interesting shapes that broadcast into t1a
            let shapes = 
                [ for i1 in [0;1;2] do
                      for i2 in [0;1;3] do
                          for i3 in [0;1;4] do 
                              if i1 <> 2 || i2 <> 3 || i3 <> 4 then
                                  [| if i1 <> 0 && i2 <> 0 && i3 <> 0 then yield i1
                                     if i2 <> 0 && i3 <> 0 then yield i2
                                     if i3 <> 0 then yield i3 |] ]
                      |> List.distinct

            // For each shape, create a broadcasting addition and take forward and reverse derivatives
            for shape in shapes do 
                let t1b = combo.tensor(ArrayND.init shape (fun is -> double (Array.sum is) + 2.0))
                let t1a_deriv = t1a + 1.0
                let t1b_delta = combo.tensor(ArrayND.init shape (fun is -> double (Array.sum is) - 2.0))
                let fwda = t1a.forwardDiff(t1a_deriv)
                let fwdb = t1b.forwardDiff(t1b_delta)
                let fwdz = fwda + fwdb
                let fwdzd = fwdz.derivative

                let revx = t1a.reverseDiff()
                let revy = t1b.reverseDiff()
                let revz = revx + revy
                let revz_grad = t1a - 1.0
                revz.reverse(revz_grad)
                let revxd = revx.derivative
                let revyd = revy.derivative

                // In the simple case of broadcasting a constant, check the result against the non-broadcast case
                if t1b.sum() = combo.tensor(2.0) then 
                    let t1c = combo.tensor(ArrayND.init [| 2;3;4 |] (fun _idxs -> 2.0))
                    let t1c_deriv = combo.tensor(ArrayND.init [| 2;3;4 |] (fun _idxs -> -2.0))
                    let fwda = t1a.forwardDiff(t1a_deriv)
                    let fwdc = t1c.forwardDiff(t1c_deriv)
                    let fwdz2 = fwda + fwdc
                    let fwdzd2 = fwdz2.derivative

                    let revx2 = t1a.reverseDiff()
                    let revy2 = t1c.reverseDiff()
                    let revz2 = revx2 + revy2
                    revz2.reverse(revz_grad)
                    let revxd2 = revx2.derivative
                    let revyd2 = revy2.derivative
                    Assert.CheckEqual(fwdzd,fwdzd2)
                    Assert.CheckEqual(revxd,revxd2)
                    // note the difference in shape here, and the need to summate down
                    Assert.CheckEqual(revyd.sum(),revyd2.sum())

    [<Test>]
    member _.TestDerivativeSubTT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([40., 50., 60.]))
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -3., -3.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-30., -30., -30.])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -3., -3.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([100., 200., 300.])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([-100., -200., -300.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubTTConst () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor([4., 5., 6.])
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -3., -3.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([10., 20., 30.])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor([4., 5., 6.])
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -3., -3.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([100., 200., 300.])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubTConstT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.])
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([40., 50., 60.]))
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -3., -3.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-40., -50., -60.])

            let revx = combo.tensor([1., 2., 3.])
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -3., -3.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([-100., -200., -300.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubT0T () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.).forwardDiff(combo.tensor(10.))
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([40., 50., 60.]))
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -4., -5.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-30., -40., -50.])

            let revx = combo.tensor(1.).reverseDiff()
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -4., -5.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor(600.)
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([-100., -200., -300.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubT0TConst () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.).forwardDiff(combo.tensor(10.))
            let fwdy = combo.tensor([4., 5., 6.])
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -4., -5.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([10., 10., 10.])

            let revx = combo.tensor(1.).reverseDiff()
            let revy = combo.tensor([4., 5., 6.])
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -4., -5.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor(600.)
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubT0ConstT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.)
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([40., 50., 60.]))
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -4., -5.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-40., -50., -60.])

            let revx = combo.tensor(1.)
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -4., -5.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([-100., -200., -300.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubTT0 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(40.))
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -2., -1.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-30., -20., -10.])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor(4.).reverseDiff()
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -2., -1.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([100., 200., 300.])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor(-600.)

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubTT0Const () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor(4.)
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -2., -1.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([10., 20., 30.])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor(4.)
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -2., -1.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([100., 200., 300.])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeSubTConstT0 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.])
            let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(40.))
            let fwdz = fwdx - fwdy
            let fwdzCorrect = combo.tensor([-3., -2., -1.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-40., -40., -40.])

            let revx = combo.tensor([1., 2., 3.])
            let revy = combo.tensor(4.).reverseDiff()
            let revz = revx - revy
            let revzCorrect = combo.tensor([-3., -2., -1.])
            revz.reverse(combo.tensor([100., 200., 300.]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor(-600.)

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeMulTT () =
        for swap in [true; false] do
          for combo in Combos.AllDevicesAndBackendsFloat32 do
              let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
              let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([40., 50., 60.]))
              let fwdz = if swap then fwdy * fwdx else fwdx * fwdy
              let fwdzCorrect = combo.tensor([4., 10., 18.])
              let fwdzd = fwdz.derivative
              let fwdzdCorrect = combo.tensor([80., 200., 360.])

              let revx = combo.tensor([1., 2., 3.]).reverseDiff()
              let revy = combo.tensor([4., 5., 6.]).reverseDiff()
              let revz = if swap then revy * revx else revx * revy
              let revzCorrect = combo.tensor([4., 10., 18.])
              revz.reverse(combo.tensor([100., 200., 300.]))
              let revxd = revx.derivative
              let revxdCorrect = combo.tensor([400., 1000., 1800.])
              let revyd = revy.derivative
              let revydCorrect = combo.tensor([100., 400., 900.])

              Assert.CheckEqual(fwdzCorrect, fwdz)
              Assert.CheckEqual(fwdzdCorrect, fwdzd)
              Assert.CheckEqual(revzCorrect, revz)
              Assert.CheckEqual(revxdCorrect, revxd)
              Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeMulTTConst () =
        for swap in [true; false] do
          for combo in Combos.AllDevicesAndBackendsFloat32 do
              let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
              let fwdy = combo.tensor([4., 5., 6.])
              let fwdz = if swap then fwdy * fwdx else fwdx * fwdy
              let fwdzCorrect = combo.tensor([4., 10., 18.])
              let fwdzd = fwdz.derivative
              let fwdzdCorrect = combo.tensor([40., 100., 180.])

              let revx = combo.tensor([1., 2., 3.]).reverseDiff()
              let revy = combo.tensor([4., 5., 6.])
              let revz = if swap then revy * revx else revx * revy
              let revzCorrect = combo.tensor([4., 10., 18.])
              revz.reverse(combo.tensor([100., 200., 300.]))
              let revxd = revx.derivative
              let revxdCorrect = combo.tensor([400., 1000., 1800.])
              let revyd = revy.isNoDiff()
              let revydCorrect = true

              Assert.CheckEqual(fwdzCorrect, fwdz)
              Assert.CheckEqual(fwdzdCorrect, fwdzd)
              Assert.CheckEqual(revzCorrect, revz)
              Assert.CheckEqual(revxdCorrect, revxd)
              Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeMulTT0 () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
                let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(40.))
                let fwdz = if swap then fwdy * fwdx else fwdx * fwdy
                let fwdzCorrect = combo.tensor([4., 8., 12.])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([80., 160., 240.])

                let revx = combo.tensor([1., 2., 3.]).reverseDiff()
                let revy = combo.tensor(4.).reverseDiff()
                let revz = if swap then revy * revx else revx * revy
                let revzCorrect = combo.tensor([4., 8., 12.])
                revz.reverse(combo.tensor([100., 200., 300.]))
                let revxd = revx.derivative
                let revxdCorrect = combo.tensor([400., 800., 1200.])
                let revyd = revy.derivative
                let revydCorrect = combo.tensor(1400.)

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeMulTT0Const () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
                let fwdy = combo.tensor(4.)
                let fwdz = if swap then fwdy * fwdx else fwdx * fwdy
                let fwdzCorrect = combo.tensor([4., 8., 12.])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([40., 80., 120.])

                let revx = combo.tensor([1., 2., 3.]).reverseDiff()
                let revy = combo.tensor(4.)
                let revz = if swap then revy * revx else revx * revy
                let revzCorrect = combo.tensor([4., 8., 12.])
                revz.reverse(combo.tensor([100., 200., 300.]))
                let revxd = revx.derivative
                let revxdCorrect = combo.tensor([400., 800., 1200.])
                let revyd = revy.isNoDiff()
                let revydCorrect = true

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeMulTConstT0 () =
        for swap in [true; false] do
            for combo in Combos.AllDevicesAndBackendsFloat32 do
                let fwdx = combo.tensor([1., 2., 3.])
                let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(40.))
                let fwdz = if swap then fwdy * fwdx else fwdx * fwdy
                let fwdzCorrect = combo.tensor([4., 8., 12.])
                let fwdzd = fwdz.derivative
                let fwdzdCorrect = combo.tensor([40., 80., 120.])

                let revx = combo.tensor([1., 2., 3.])
                let revy = combo.tensor(4.).reverseDiff()
                let revz = if swap then revy * revx else revx * revy
                let revzCorrect = combo.tensor([4., 8., 12.])
                revz.reverse(combo.tensor([100., 200., 300.]))
                let revxd = revx.isNoDiff()
                let revxdCorrect = true
                let revyd = revy.derivative
                let revydCorrect = combo.tensor(1400.)

                Assert.CheckEqual(fwdzCorrect, fwdz)
                Assert.CheckEqual(fwdzdCorrect, fwdzd)
                Assert.CheckEqual(revzCorrect, revz)
                Assert.CheckEqual(revxdCorrect, revxd)
                Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeDivTT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([400., 500., 600.]))
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.4000, 0.5000])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-22.5000, -36.0000, -45.0000])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.4000, 0.5000])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([250., 400., 500.])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([ -62.5000, -160.0000, -250.0000])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeDivTTConst () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor([4., 5., 6.])
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.4000, 0.5000])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([2.5000, 4.0000, 5.0000])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor([4., 5., 6.])
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.4000, 0.5000])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([250., 400., 500.])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeDivTConstT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.])
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([400., 500., 600.]))
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.4000, 0.5000])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-25., -40., -50.])

            let revx = combo.tensor([1., 2., 3.])
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.4000, 0.5000])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([ -62.5000, -160.0000, -250.0000])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeDivT0T () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.).forwardDiff(combo.tensor(10.))
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([400., 500., 600.]))
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.2000, 0.1667])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-22.5000, -18.0000, -15.0000])

            let revx = combo.tensor(1.).reverseDiff()
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.2000, 0.1667])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor(1150.)
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([-62.5000, -80.0000, -83.3333])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeDivT0TConst () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.).forwardDiff(combo.tensor(10.))
            let fwdy = combo.tensor([4., 5., 6.])
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.2000, 0.1667])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([2.5000, 2.0000, 1.6667])

            let revx = combo.tensor(1.).reverseDiff()
            let revy = combo.tensor([4., 5., 6.])
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.2000, 0.1667])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor(1150.)
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeDivT0ConstT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.)
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([400., 500., 600.]))
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.2000, 0.1667])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-25.0000, -20.0000, -16.6667])

            let revx = combo.tensor(1.)
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.2000, 0.1667])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([-62.5000, -80.0000, -83.3333])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeDivTT0 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(400.))
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.5000, 0.7500])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-22.5000, -45.0000, -67.5000])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor(4.).reverseDiff()
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.5000, 0.7500])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([250., 500., 750.])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor(-875.)

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeDivTT0Const () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([10., 20., 30.]))
            let fwdy = combo.tensor(4.)
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.5000, 0.7500])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([2.5000, 5.0000, 7.5000])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor(4.)
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.5000, 0.7500])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([250., 500., 750.])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeDivTConstT0 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.])
            let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(400.))
            let fwdz = fwdx / fwdy
            let fwdzCorrect = combo.tensor([0.2500, 0.5000, 0.7500])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-25., -50., -75.])

            let revx = combo.tensor([1., 2., 3.])
            let revy = combo.tensor(4.).reverseDiff()
            let revz = revx / revy
            let revzCorrect = combo.tensor([0.2500, 0.5000, 0.7500])
            revz.reverse(combo.tensor([1000., 2000., 3000.]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor(-875.)

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativePowTT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([1.1,1.2,1.3]))
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([2.1,2.2,2.3]))
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([  1.,  32., 729.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([   4.4000,  144.7976, 3737.4431])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx ** revy
            let revzCorrect = combo.tensor([  1.,  32., 729.])
            revz.reverse(combo.tensor([3.1,3.2,3.3]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([  12.4000,  256.0000, 4811.3999])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([   0.0000,   70.9783, 2642.9316])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativePowTTConst () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([1.1,1.2,1.3]))
            let fwdy = combo.tensor([4., 5., 6.])
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([  1.,  32., 729.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([   4.4000,   96.0000, 1895.3999])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor([4., 5., 6.])
            let revz = revx ** revy
            let revzCorrect = combo.tensor([  1.,  32., 729.])
            revz.reverse(combo.tensor([3.1,3.2,3.3]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([  12.4000,  256.0000, 4811.3999])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativePowTConstT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.])
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([2.1,2.2,2.3]))
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([  1.,  32., 729.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([   0.0000,   48.7976, 1842.0432])

            let revx = combo.tensor([1., 2., 3.])
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx ** revy
            let revzCorrect = combo.tensor([  1.,  32., 729.])
            revz.reverse(combo.tensor([3.1,3.2,3.3]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([   0.0000,   70.9783, 2642.9316])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativePowT0T () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.5).forwardDiff(combo.tensor(1.25))
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([2.1,2.2,2.3]))
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([ 5.0625,  7.5938, 11.3906])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([21.1856, 38.4144, 67.5757])

            let revx = combo.tensor(1.5).reverseDiff()
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx ** revy
            let revzCorrect = combo.tensor([ 5.0625,  7.5938, 11.3906])
            revz.reverse(combo.tensor([3.1,3.2,3.3]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor(273.2062)
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([ 6.3633,  9.8528, 15.2411])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativePowT0TConst () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.5).forwardDiff(combo.tensor(1.25))
            let fwdy = combo.tensor([4., 5., 6.])
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([ 5.0625,  7.5938, 11.3906])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([16.8750, 31.6406, 56.9531])

            let revx = combo.tensor(1.5).reverseDiff()
            let revy = combo.tensor([4., 5., 6.])
            let revz = revx ** revy
            let revzCorrect = combo.tensor([ 5.0625,  7.5938, 11.3906])
            revz.reverse(combo.tensor([3.1,3.2,3.3]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor(273.2062)
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativePowT0ConstT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor(1.5)
            let fwdy = combo.tensor([4., 5., 6.]).forwardDiff(combo.tensor([2.1,2.2,2.3]))
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([ 5.0625,  7.5938, 11.3906])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([ 4.3106,  6.7738, 10.6226])

            let revx = combo.tensor(1.5)
            let revy = combo.tensor([4., 5., 6.]).reverseDiff()
            let revz = revx ** revy
            let revzCorrect = combo.tensor([ 5.0625,  7.5938, 11.3906])
            revz.reverse(combo.tensor([3.1,3.2,3.3]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([ 6.3633,  9.8528, 15.2411])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativePowTT0 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([1.1, 1.2, 1.3]))
            let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(2.1))
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([ 1., 16., 81.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([  4.4000,  61.6897, 327.2739])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor(4.).reverseDiff()
            let revz = revx ** revy
            let revzCorrect = combo.tensor([ 1., 16., 81.])
            revz.reverse(combo.tensor([3.1, 3.2, 3.3]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([ 12.4000, 102.4000, 356.4000])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor(329.1482)

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativePowTT0Const () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.]).forwardDiff(combo.tensor([1.1, 1.2, 1.3]))
            let fwdy = combo.tensor(4.)
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([ 1., 16., 81.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([  4.4000,  38.4000, 140.4000])

            let revx = combo.tensor([1., 2., 3.]).reverseDiff()
            let revy = combo.tensor(4.)
            let revz = revx ** revy
            let revzCorrect = combo.tensor([ 1., 16., 81.])
            revz.reverse(combo.tensor([3.1, 3.2, 3.3]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([ 12.4000, 102.4000, 356.4000])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativePowTConstT0 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1., 2., 3.])
            let fwdy = combo.tensor(4.).forwardDiff(combo.tensor(2.1))
            let fwdz = fwdx ** fwdy
            let fwdzCorrect = combo.tensor([ 1., 16., 81.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([  0.0000,  23.2897, 186.8739])

            let revx = combo.tensor([1., 2., 3.])
            let revy = combo.tensor(4.).reverseDiff()
            let revz = revx ** revy
            let revzCorrect = combo.tensor([ 1., 16., 81.])
            revz.reverse(combo.tensor([3.1, 3.2, 3.3]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor(329.1482)

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.True(revyd.allclose(revydCorrect, 0.01))


    [<Test>]
    member _.TestDerivativeMatMulT2T2 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[6.2381, 0.0393, 8.2364, 3.9906, 6.2291],
                [9.8762, 3.2263, 6.2866, 4.7111, 0.0652],
                [3.5832, 7.9801, 1.9854, 4.4965, 4.1712]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[4.6453, 8.4388, 4.6549, 9.5680, 1.5756],
                [3.2066, 4.2429, 2.2028, 9.1037, 3.4022],
                [4.2324, 4.5508, 3.4755, 2.7196, 5.5344]]))
            let fwdy = combo.tensor([[4.4220, 3.7293],
                [6.1928, 2.1446],
                [0.0525, 1.2494],
                [7.5281, 1.4816],
                [5.0328, 2.2756]])
            let fwdy = fwdy.forwardDiff(combo.tensor([[1.4749, 9.7608],
                [3.6599, 7.9553],
                [3.5503, 1.3757],
                [8.3172, 6.6748],
                [2.2959, 0.6784]]))
            let fwdz = dsharp.matmul(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([[ 89.6516, 53.7260],
                [ 99.7751, 58.7331],
                [120.2113, 49.1116]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[239.0819, 162.3930],
                [214.2522, 207.2430],
                [183.9220, 180.5424]])

            let revx = combo.tensor([[6.2381, 0.0393, 8.2364, 3.9906, 6.2291],
                [9.8762, 3.2263, 6.2866, 4.7111, 0.0652],
                [3.5832, 7.9801, 1.9854, 4.4965, 4.1712]]).reverseDiff()
            let revy = combo.tensor([[4.4220, 3.7293],
                [6.1928, 2.1446],
                [0.0525, 1.2494],
                [7.5281, 1.4816],
                [5.0328, 2.2756]]).reverseDiff()
            let revz = dsharp.matmul(revx, revy)
            let revzCorrect = combo.tensor([[ 89.6516, 53.7260],
                [ 99.7751, 58.7331],
                [120.2113, 49.1116]])
            revz.reverse(combo.tensor([[7.3984, 0.1849],
                [1.2520, 9.5731],
                [6.8201, 9.5221]]))            
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[33.4050, 46.2136,  0.6191, 55.9696, 37.6556],
                [41.2370, 28.2842, 12.0266, 23.6085, 28.0854],
                [65.6689, 62.6571, 12.2551, 65.4497, 55.9926]])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([[ 82.9549,129.8180],
                [ 58.7551,106.8801],
                [ 82.3474, 80.6097],
                [ 66.0888, 88.6534],
                [ 74.6154, 41.4950]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeMatMulT2T2Const () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[6.2381, 0.0393, 8.2364, 3.9906, 6.2291],
                [9.8762, 3.2263, 6.2866, 4.7111, 0.0652],
                [3.5832, 7.9801, 1.9854, 4.4965, 4.1712]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[4.6453, 8.4388, 4.6549, 9.5680, 1.5756],
                [3.2066, 4.2429, 2.2028, 9.1037, 3.4022],
                [4.2324, 4.5508, 3.4755, 2.7196, 5.5344]]))
            let fwdy = combo.tensor([[4.4220, 3.7293],
                [6.1928, 2.1446],
                [0.0525, 1.2494],
                [7.5281, 1.4816],
                [5.0328, 2.2756]])
            let fwdz = dsharp.matmul(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([[ 89.6516, 53.7260],
                [ 99.7751, 58.7331],
                [120.2113, 49.1116]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[153.0042,  58.9988],
                [126.2268,  45.0400],
                [ 95.4073,  46.5093]])

            let revx = combo.tensor([[6.2381, 0.0393, 8.2364, 3.9906, 6.2291],
                [9.8762, 3.2263, 6.2866, 4.7111, 0.0652],
                [3.5832, 7.9801, 1.9854, 4.4965, 4.1712]]).reverseDiff()
            let revy = combo.tensor([[4.4220, 3.7293],
                [6.1928, 2.1446],
                [0.0525, 1.2494],
                [7.5281, 1.4816],
                [5.0328, 2.2756]])
            let revz = dsharp.matmul(revx, revy)
            let revzCorrect = combo.tensor([[ 89.6516, 53.7260],
                [ 99.7751, 58.7331],
                [120.2113, 49.1116]])
            revz.reverse(combo.tensor([[7.3984, 0.1849],
                [1.2520, 9.5731],
                [6.8201, 9.5221]]))            
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[33.4050, 46.2136,  0.6191, 55.9696, 37.6556],
                [41.2370, 28.2842, 12.0266, 23.6085, 28.0854],
                [65.6689, 62.6571, 12.2551, 65.4497, 55.9926]])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeMatMulT2ConstT2 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[6.2381, 0.0393, 8.2364, 3.9906, 6.2291],
                [9.8762, 3.2263, 6.2866, 4.7111, 0.0652],
                [3.5832, 7.9801, 1.9854, 4.4965, 4.1712]])
            let fwdy = combo.tensor([[4.4220, 3.7293],
                [6.1928, 2.1446],
                [0.0525, 1.2494],
                [7.5281, 1.4816],
                [5.0328, 2.2756]])
            let fwdy = fwdy.forwardDiff(combo.tensor([[1.4749, 9.7608],
                [3.6599, 7.9553],
                [3.5503, 1.3757],
                [8.3172, 6.6748],
                [2.2959, 0.6784]]))
            let fwdz = dsharp.matmul(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([[ 89.6516, 53.7260],
                [ 99.7751, 58.7331],
                [120.2113, 49.1116]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[ 86.0781, 103.3946],
                [ 88.0265, 162.2042],
                [ 88.5149, 134.0333]])

            let revx = combo.tensor([[6.2381, 0.0393, 8.2364, 3.9906, 6.2291],
                [9.8762, 3.2263, 6.2866, 4.7111, 0.0652],
                [3.5832, 7.9801, 1.9854, 4.4965, 4.1712]])
            let revy = combo.tensor([[4.4220, 3.7293],
                [6.1928, 2.1446],
                [0.0525, 1.2494],
                [7.5281, 1.4816],
                [5.0328, 2.2756]]).reverseDiff()
            let revz = dsharp.matmul(revx, revy)
            let revzCorrect = combo.tensor([[ 89.6516, 53.7260],
                [ 99.7751, 58.7331],
                [120.2113, 49.1116]])
            revz.reverse(combo.tensor([[7.3984, 0.1849],
                [1.2520, 9.5731],
                [6.8201, 9.5221]]))            
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([[ 82.9549,129.8180],
                [ 58.7551,106.8801],
                [ 82.3474, 80.6097],
                [ 66.0888, 88.6534],
                [ 74.6154, 41.4950]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestTensorStackTs () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdxa = combo.tensor([1.; 2.]).forwardDiff(combo.tensor([10.; 20.]))
            let fwdxb = combo.tensor([3.; 4.]).forwardDiff(combo.tensor([30.; 40.]))
            let fwdxc = combo.tensor([5.; 6.]).forwardDiff(combo.tensor([50.; 60.]))
            let fwdz = dsharp.stack([fwdxa;fwdxb;fwdxc])
            let fwdzCorrect = combo.tensor([[1.;2.];[3.;4.];[5.;6.]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[10.;20.];[30.;40.];[50.;60.]])

            let revxa = combo.tensor([1.; 2.]).reverseDiff()
            let revxb = combo.tensor([3.; 4.]).reverseDiff()
            let revxc = combo.tensor([5.; 6.]).reverseDiff()
            let revz = dsharp.stack([revxa;revxb;revxc])
            let revzCorrect = combo.tensor([[1.;2.];[3.;4.];[5.;6.]])
            revz.reverse(combo.tensor([[10.;20.];[30.;40.];[50.;60.]]))
            let revxda = revxa.derivative
            let revxdaCorrect = combo.tensor([10.; 20.])
            let revxdb = revxb.derivative
            let revxdbCorrect = combo.tensor([30.; 40.])
            let revxdc = revxc.derivative
            let revxdcCorrect = combo.tensor([50.; 60.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdaCorrect, revxda)
            Assert.CheckEqual(revxdbCorrect, revxdb)
            Assert.CheckEqual(revxdcCorrect, revxdc)

    [<Test>]
    member _.TestDerivativeNeg () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1.; 2.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = -fwdx
            let fwdzCorrect = combo.tensor([-1.; -2.; -3.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-2.; -3.; -4.])

            let revx = combo.tensor([1.; 2.; 3.]).reverseDiff()
            let revz = -revx
            let revzCorrect = combo.tensor([-1.; -2.; -3.])
            revz.reverse(combo.tensor([5.; 5.; 5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([-5.; -5.; -5.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeGather () =
        let fwdx = dsharp.tensor([1,2,3,4,5]).forwardDiff(dsharp.tensor([10,20,30,40,50]))
        let fwdz = dsharp.gather(fwdx, 0, dsharp.tensor([0,2,3], dtype=Dtype.Int32))
        let fwdzCorrect = dsharp.tensor([1,3,4])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([10,30,40])

        let revx = dsharp.tensor([1,2,3,4,5]).reverseDiff()
        let revz = dsharp.gather(revx, 0, dsharp.tensor([0,2,3], dtype=Dtype.Int32))
        let revzCorrect = dsharp.tensor([1,3,4])
        revz.reverse(dsharp.tensor([10,30,40]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([10.,  0., 30., 40.,  0.])

        Assert.CheckEqual(fwdzCorrect, fwdz)
        Assert.CheckEqual(fwdzdCorrect, fwdzd)
        Assert.CheckEqual(revzCorrect, revz)
        Assert.CheckEqual(revxdCorrect, revxd)

        let fwdx = dsharp.tensor([[1,2,3],[4,5,6]]).forwardDiff(dsharp.tensor([[10,20,30],[40,50,60]]))
        let fwdz = dsharp.gather(fwdx, 0, dsharp.tensor([[1,0,1],[0,1,1]], dtype=Dtype.Int32))
        let fwdzCorrect = dsharp.tensor([[4,2,6],[1,5,6]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[40,20,60],[10,50,60]])

        let revx = dsharp.tensor([[1,2,3],[4,5,6]]).reverseDiff()
        let revz = dsharp.gather(revx, 0, dsharp.tensor([[1,0,1],[0,1,1]], dtype=Dtype.Int32))
        let revzCorrect = dsharp.tensor([[4,2,6],[1,5,6]])
        revz.reverse(dsharp.tensor([[40,20,60],[10,50,60]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10,20,0],[40,50,120]])

        Assert.CheckEqual(fwdzCorrect, fwdz)
        Assert.CheckEqual(fwdzdCorrect, fwdzd)
        Assert.CheckEqual(revzCorrect, revz)
        Assert.CheckEqual(revxdCorrect, revxd)

        let fwdx = dsharp.tensor([[1,2,3],[4,5,6]]).forwardDiff(dsharp.tensor([[10,20,30],[40,50,60]]))
        let fwdz = dsharp.gather(fwdx, 1, dsharp.tensor([[1,0,1],[0,1,1]], dtype=Dtype.Int32))
        let fwdzCorrect = dsharp.tensor([[2,1,2],[4,5,5]])
        let fwdzd = fwdz.derivative
        let fwdzdCorrect = dsharp.tensor([[20,10,20],[40,50,50]])

        let revx = dsharp.tensor([[1,2,3],[4,5,6]]).reverseDiff()
        let revz = dsharp.gather(revx, 1, dsharp.tensor([[1,0,1],[0,1,1]], dtype=Dtype.Int32))
        let revzCorrect = dsharp.tensor([[2,1,2],[4,5,5]])
        revz.reverse(dsharp.tensor([[20,10,20],[40,50,50]]))
        let revxd = revx.derivative
        let revxdCorrect = dsharp.tensor([[10,40,0],[40,100,0]])

        Assert.CheckEqual(fwdzCorrect, fwdz)
        Assert.CheckEqual(fwdzdCorrect, fwdzd)
        Assert.CheckEqual(revzCorrect, revz)
        Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeSum () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1.; 2.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = fwdx.sum()
            let fwdzCorrect = combo.tensor(6.)
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor(9.)

            let revx = combo.tensor([1.; 2.; 3.]).reverseDiff()
            let revz = revx.sum()
            let revzCorrect = combo.tensor(6.)
            revz.reverse(combo.tensor(5.))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([5.; 5.; 5.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeSumT2Dim0 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.; 2.]; [3.; 4.]]).forwardDiff(combo.tensor([[2.; 3.]; [4.; 5.]]))
            let fwdz = fwdx.sumT2Dim0()
            let fwdzCorrect = combo.tensor([4.; 6.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([6.; 8.])

            let revx = combo.tensor([[1.; 2.]; [3.; 4.]]).reverseDiff()
            let revz = revx.sumT2Dim0()
            let revzCorrect = combo.tensor([4.; 6.])
            revz.reverse(combo.tensor([5.; 6.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[5.; 6.]; [5.; 6.]])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeMean () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1.; 2.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = fwdx.mean()
            let fwdzCorrect = combo.tensor(2.)
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor(3.)

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)

            (* Python:
            input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
            loss = input.mean()
            loss.backward(torch.tensor(3.0))
            input.grad
            --> tensor([1., 1., 1.])
            *)
            let revx = combo.tensor([1.; 2.; 3.]).reverseDiff()
            let revz = revx.mean()
            let revzCorrect = combo.tensor(2.)
            revz.reverse(combo.tensor(30.))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([10.; 10.; 10.])

            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeVariance () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1.; 2.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = fwdx.variance()
            let fwdzCorrect = combo.tensor(1.0)
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor(2.0)

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)

            (* Python:
            input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
            loss = input.var()
            loss.backward(torch.tensor(3.0))
            input.grad
            --> tensor([-3.,  0.,  3.])
            *)
            let revx = combo.tensor([1.; 2.; 3.]).reverseDiff()
            let revz = revx.variance()
            let revzCorrect = combo.tensor(1.)
            revz.reverse(combo.tensor(3.))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([-3.; 0.; 3.])

            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

            // keepDim = true, forward
            let fwdx = combo.tensor([1.; 2.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = fwdx.variance(0,keepDim=true)
            let fwdzCorrect = combo.tensor([1.0])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([2.0])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)

            // keepDim = true, backward
            (* Python:
            input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
            loss = input.var(0, keepdim=True)
            loss.backward(torch.tensor([3.0]))
            input.grad
            --> tensor([-3.,  0.,  3.])
            *)
            let revx = combo.tensor([1.; 2.; 3.]).reverseDiff()
            let revz = revx.variance(0,keepDim=true)
            let revzCorrect = combo.tensor([1.])
            revz.reverse(combo.tensor([3.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([-3.; 0.; 3.])

            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeStddev () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1.; 2.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = fwdx.stddev()
            let fwdzCorrect = combo.tensor(1.0)
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor(1.0)

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)

            (* Python:
            import torch
            input = torch.tensor([1.0,2.0,3.0], requires_grad=True)
            loss = input.std()
            loss.backward(torch.tensor(3.0))
            input.grad
            --> tensor([-1.5000,  0.0000,  1.5000])
            *)
            let revx = combo.tensor([1.; 2.; 3.]).reverseDiff()
            let revz = revx.stddev()
            let revzCorrect = combo.tensor(1.)
            revz.reverse(combo.tensor(3.))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([-1.5; 0.; 1.5])

            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativePermuteT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[[ 0.,  1.],
                                       [ 2.,  3.],
                                       [ 4.,  5.]],

                                      [[ 6.,  7.],
                                       [ 8.,  9.],
                                       [10., 11.]]]).forwardDiff(combo.tensor([[[  0.,  10.],
                                                                                 [ 20.,  30.],
                                                                                 [ 40.,  50.]],

                                                                                [[ 60.,  70.],
                                                                                 [ 80.,  90.],
                                                                                 [100., 110.]]]))
            // Note, this is a swap
            let fwdz = fwdx.permute([2;1;0])
            let fwdzCorrect = combo.tensor([[[ 0.,  6.],
                                               [ 2.,  8.],
                                               [ 4., 10.]],

                                              [[ 1.,  7.],
                                               [ 3.,  9.],
                                               [ 5., 11.]]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[[  0.,  60.],
                                               [ 20.,  80.],
                                               [ 40., 100.]],

                                              [[ 10.,  70.],
                                               [ 30.,  90.],
                                               [ 50., 110.]]])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)

            // Python:
            (*
            import torch
            revx = torch.tensor([[[ 0.,  1.],[ 2.,  3.],[ 4.,  5.]],[[ 6.,  7.],[ 8.,  9.],[10., 11.]]], requires_grad=True)
            revz = revx.permute([1,2,0])
            revz.backward(torch.tensor([[[ 0.,  1.],[ 2.,  3.]],[[ 4.,  5.],[ 6.,  7.]],[[ 8.,  9.],[10., 11.]]]))
            revz
            revx.grad
            *)

            let revx = combo.tensor([[[ 0.,  1.],
                                      [ 2.,  3.],
                                      [ 4.,  5.]],

                                     [[ 6.,  7.],
                                      [ 8.,  9.],
                                      [10., 11.]]]).reverseDiff()

            // Note, this is a rotation
            let revz = revx.permute([1;2;0])
            let revzCorrect = combo.tensor([[[ 0.,  6.],
                                             [ 1.,  7.]],
                                            [[ 2.,  8.],
                                             [ 3.,  9.]],
                                            [[ 4., 10.],
                                             [ 5., 11.]]])

            revz.reverse(combo.tensor([[[ 0.,  1.],
                                        [ 2.,  3.]],
                                       [[ 4.,  5.],
                                        [ 6.,  7.]],
                                       [[ 8.,  9.],
                                        [10., 11.]]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[[ 0.,  2.],
                                              [ 4.,  6.],
                                              [ 8., 10.]],

                                             [[ 1.,  3.],
                                              [ 5.,  7.],
                                              [ 9., 11.]]])

            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeTransposeT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[[ 0.,  1.],
                                       [ 2.,  3.],
                                       [ 4.,  5.]],

                                      [[ 6.,  7.],
                                       [ 8.,  9.],
                                       [10., 11.]]]).forwardDiff(combo.tensor([[[  0.,  10.],
                                                                                 [ 20.,  30.],
                                                                                 [ 40.,  50.]],

                                                                                [[ 60.,  70.],
                                                                                 [ 80.,  90.],
                                                                                 [100., 110.]]]))
            let fwdz = fwdx.transpose(0,2)
            let fwdzCorrect = combo.tensor([[[ 0.,  6.],
                                               [ 2.,  8.],
                                               [ 4., 10.]],

                                              [[ 1.,  7.],
                                               [ 3.,  9.],
                                               [ 5., 11.]]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[[  0.,  60.],
                                               [ 20.,  80.],
                                               [ 40., 100.]],

                                              [[ 10.,  70.],
                                               [ 30.,  90.],
                                               [ 50., 110.]]])

            let revx = combo.tensor([[[ 0.,  1.],
                                         [ 2.,  3.],
                                         [ 4.,  5.]],

                                        [[ 6.,  7.],
                                         [ 8.,  9.],
                                         [10., 11.]]]).reverseDiff()
            let revz = revx.transpose(0,2)
            let revzCorrect = combo.tensor([[[ 0.,  6.],
                                             [ 2.,  8.],
                                             [ 4., 10.]],

                                            [[ 1.,  7.],
                                             [ 3.,  9.],
                                             [ 5., 11.]]])
            revz.reverse(combo.tensor([[[  0., 120.],
                                         [ 40., 160.],
                                         [ 80., 200.]],

                                        [[ 20., 140.],
                                         [ 60., 180.],
                                         [100., 220.]]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[[  0.,  20.],
                                               [ 40.,  60.],
                                               [ 80., 100.]],

                                              [[120., 140.],
                                               [160., 180.],
                                               [200., 220.]]])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeTransposeT2 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]]).forwardDiff(combo.tensor([[2.; 3.; 4.]; [10.; 20.; 30.]]))
            let fwdz = fwdx.transpose()
            let fwdzCorrect = combo.tensor([[1.; 4.]; [2.; 5.]; [3.; 6.]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[2.; 10.]; [3.; 20.]; [4.; 30.]])

            let revx = combo.tensor([[1.; 2.; 3.]; [4.; 5.; 6.]]).reverseDiff()
            let revz = revx.transpose()
            let revzCorrect = combo.tensor([[1.; 4.]; [2.; 5.]; [3.; 6.]])
            revz.reverse(combo.tensor([[5.; 5.]; [2.; 5.]; [3.; 7.]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[5.; 2.; 3.]; [5.; 5.; 7.]])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeSignT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([-1.; 0.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = fwdx.sign()
            let fwdzCorrect = combo.tensor([-1.; 0.; 1.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.; 0.; 0.])

            let revx = combo.tensor([-1.; 0.; 3.]).reverseDiff()
            let revz = revx.sign()
            let revzCorrect = combo.tensor([-1.; 0.; 1.])
            revz.reverse(combo.tensor([5.; 5.; 5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.; 0.; 0.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeFloorT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.floor()
            let fwdzCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            let revx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.floor()
            let revzCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeCeilT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.ceil()
            let fwdzCorrect = combo.tensor([1.; 1.; 1.; 1.; 1.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            let revx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.ceil()
            let revzCorrect = combo.tensor([1.; 1.; 1.; 1.; 1.])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeRoundT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.round()
            let fwdzCorrect = combo.tensor([1.; 0.; 0.; 1.; 1.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            let revx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.round()
            let revzCorrect = combo.tensor([1.; 0.; 0.; 1.; 1.])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.; 0.; 0.; 0.; 0.])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeAbsT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([-1.; 0.; 3.]).forwardDiff(combo.tensor([2.; 3.; 4.]))
            let fwdz = fwdx.abs()
            let fwdzCorrect = combo.tensor([1.; 0.; 3.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-2.; 0.; 4.])

            let revx = combo.tensor([-1.; 0.; 3.]).reverseDiff()
            let revz = revx.abs()
            let revzCorrect = combo.tensor([1.; 0.; 3.])
            revz.reverse(combo.tensor([5.; 5.; 5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([-5.; 0.; 5.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeReluT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([-1.; -2.; 0.; 3.; 10.]).forwardDiff(combo.tensor([2.; 3.; 4.; 5.; 6.]))
            let fwdz = fwdx.relu()
            let fwdzCorrect = combo.tensor([0.; 0.; 0.; 3.; 10.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.; 0.; 0.; 5.; 6.])

            let revx = combo.tensor([-1.; -2.; 0.; 3.; 10.]).reverseDiff()
            let revz = revx.relu()
            let revzCorrect = combo.tensor([0.; 0.; 0.; 3.; 10.])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.; 0.; 0.; 5.; -5.])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeLeakyRelu () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([-1.; -2.; 0.; 3.; 10.]).forwardDiff(combo.tensor([2.; 3.; 4.; 5.; 6.]))
            let fwdz = fwdx.leakyRelu()
            let fwdzCorrect = combo.tensor([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])
            let fwdzd = fwdz.derivative
            // TODO: behavior of derivative at 0 (where it is undefined) can be reconsidered
            // let fwdzdCorrect = combo.tensor([0.0200; 0.0300; 0.0400; 5.; 6.])
            let fwdzdCorrect = combo.tensor([0.0200; 0.0300; 2.02; 5.; 6.])

            let revx = combo.tensor([-1.; -2.; 0.; 3.; 10.]).reverseDiff()
            let revz = revx.leakyRelu()
            let revzCorrect = combo.tensor([-1.0000e-02; -2.0000e-02;  0.0000e+00;  3.0000e+00;  1.0000e+01])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            // TODO: behavior of derivative at 0 (where it is undefined) can be reconsidered
            // let revxdCorrect = combo.tensor([0.0500; 0.0500; 0.0500; 5.; -5.])
            let revxdCorrect = combo.tensor([0.0500; 0.0500; 2.52; 5.; -5.])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSoftplusT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05]).forwardDiff(combo.tensor([  765080.1250,  1507281.3750,  -646660.5000,   -90687.9375, 821899.7500,  -180674.6875, -1726284.8750,   212356.4219]))
            let fwdz = fwdx.softplus()
            let fwdzCorrect = combo.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([ 344587.4062, 1072155.8750, -232257.6719,  -69829.2578,       0.0000, -180674.6875,      -0.0000,       0.0000])

            let revx = combo.tensor([-1.9908e-01,  9.0179e-01, -5.7899e-01,  1.2083e+00, -4.0689e+04, 2.8907e+05, -6.5848e+05, -1.2992e+05]).reverseDiff()
            let revz = revx.softplus()
            let revzCorrect = combo.tensor([5.9855e-01, 1.2424e+00, 4.4498e-01, 1.4697e+00, 0.0000e+00, 2.8907e+05, 0.0000e+00, 0.0000e+00])
            revz.reverse(combo.tensor([  765080.1250,  1507281.3750,  -646660.5000,   -90687.9375, 821899.7500,  -180674.6875, -1726284.8750,   212356.4219]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([ 344587.4062, 1072155.8750, -232257.6719,  -69829.2578,       0.0000, -180674.6875,      -0.0000,       0.0000])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSigmoidT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.sigmoid()
            let fwdzCorrect = combo.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.3456; 0.0684; 0.3681; 0.2893; 0.1215])

            let revx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.sigmoid()
            let revzCorrect = combo.tensor([0.7206; 0.6199; 0.5502; 0.6415; 0.6993])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([1.0067;  1.1781;  1.2374;  1.1499; -1.0514])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeExpT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).forwardDiff(combo.tensor([-1.9015; 0.4606; -0.1030; 0.0466; -0.2321]))
            let fwdz = fwdx.exp()
            let fwdzCorrect = combo.tensor([1.3305; 0.3487; 2.7628; 1.5230; 3.5895])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-2.5300; 0.1606; -0.2845; 0.0710; -0.8331])

            let revx = combo.tensor([0.2856; -1.0535; 1.0162; 0.4207; 1.2780]).reverseDiff()
            let revz = revx.exp()
            let revzCorrect = combo.tensor([1.3305; 0.3487; 2.7628; 1.5230; 3.5895])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([6.6526; 1.7435; 13.8140; 7.6152; -17.9474])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeLogT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.log()
            let fwdzCorrect = combo.tensor([-0.0541; 0.3982; -1.6021; -0.5417; -0.1697])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([1.8118; 0.1951; 7.3820; 2.1624; 0.6847])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.log()
            let revzCorrect = combo.tensor([-0.0541; 0.3982; -1.6021; -0.5417; -0.1697])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([5.2780; 3.3576; 24.8177; 8.5945; -5.9248])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeLog10T () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.log10()
            let fwdzCorrect = combo.tensor([-0.0235;  0.1729; -0.6957; -0.2352; -0.0737])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.7869; 0.0847; 3.2054; 0.9391; 0.2974])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.log10()
            let revzCorrect = combo.tensor([-0.0235;  0.1729; -0.6957; -0.2352; -0.0737])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([2.2923;  1.4582; 10.7765;  3.7323; -2.5731])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
        
    [<Test>]
    member _.TestDerivativeSqrtT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).forwardDiff(combo.tensor([8.8405; 2.7188; 1.5814; 8.7951; 0.1119]))
            let fwdz = fwdx.sqrt()
            let fwdzCorrect = combo.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.5972; 0.1617; 0.1971; 0.5093; 0.0061])

            let revx = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).reverseDiff()
            let revz = revx.sqrt()
            let revzCorrect = combo.tensor([7.4022; 8.4050; 4.0108; 8.6342; 9.1067])
            revz.reverse(combo.tensor([7.0478; 2.0493; 1.8341; 0.0166; 9.4089]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.4761; 0.1219; 0.2286; 0.0010; 0.5166])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.05))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.05))
            Assert.True(revz.allclose(revzCorrect, 0.05))
            Assert.True(revxd.allclose(revxdCorrect, 0.05))

    [<Test>]
    member _.TestDerivativeSinT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.sin()
            let fwdzCorrect = combo.tensor([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([1.0022; 0.0237; 1.4571; 1.0510; 0.3840])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.sin()
            let revzCorrect = combo.tensor([0.8118; 0.9967; 0.2001; 0.5495; 0.7472])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([2.9194;  0.4080;  4.8988;  4.1774; -3.3228])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeCosT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.cos()
            let fwdzCorrect = combo.tensor([0.5839; 0.0816; 0.9798; 0.8355; 0.6646])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-1.3934; -0.2895; -0.2976; -0.6913; -0.4318])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.cos()
            let revzCorrect = combo.tensor([0.5839; 0.0816; 0.9798; 0.8355; 0.6646])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([-4.0592; -4.9833; -1.0007; -2.7476;  3.7362])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeTanT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.tan()
            let fwdzCorrect = combo.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([5.0347; 43.6222;  1.5493;  1.8022;  1.3083])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.tan()
            let revzCorrect = combo.tensor([1.3904; 12.2132;  0.2043;  0.6577;  1.1244])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([14.6665; 750.8119;   5.2086;   7.1631; -11.3217])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
    
    [<Test>]
    member _.TestDerivativeSinhT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.sinh()
            let fwdzCorrect = combo.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([2.5459; 0.6767; 1.5175; 1.4770; 0.7960])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.sinh()
            let revzCorrect = combo.tensor([1.0955; 2.1038; 0.2029; 0.6152; 0.9477])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([7.4163; 11.6467;  5.1018;  5.8704; -6.8886])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeCoshT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.cosh()
            let fwdzCorrect = combo.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([1.8803; 0.6111; 0.3017; 0.7739; 0.5476])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.cosh()
            let revzCorrect = combo.tensor([1.4833; 2.3293; 1.0204; 1.1741; 1.3777])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([5.4774; 10.5188;  1.0143;  3.0759; -4.7385])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeTanhT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.tanh()
            let fwdzCorrect = combo.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.7802; 0.0535; 1.4284; 0.9126; 0.3044])

            let revx = combo.tensor([0.9473; 1.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.tanh()
            let revzCorrect = combo.tensor([0.7386; 0.9032; 0.1988; 0.5240; 0.6879])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([2.2727;  0.9215;  4.8024;  3.6273; -2.6342])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
    
    [<Test>]
    member _.TestDerivativeAsinT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.asin()
            let fwdzCorrect = combo.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([5.3579; 0.3331; 1.5183; 1.5467; 1.0770])

            let revx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.asin()
            let revzCorrect = combo.tensor([1.2447; 0.5111; 0.2029; 0.6209; 1.0045])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([15.6080;  5.7324;  5.1047;  6.1476; -9.3197])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeAcosT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.acos()
            let fwdzCorrect = combo.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([-5.3579; -0.3331; -1.5183; -1.5467; -1.0770])

            let revx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.acos()
            let revzCorrect = combo.tensor([0.3261; 1.0597; 1.3679; 0.9499; 0.5663])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([-15.6080;  -5.7324;  -5.1047;  -6.1476;   9.3197])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeAtanT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).forwardDiff(combo.tensor([1.7164; 0.2905; 1.4872; 1.2580; 0.5778]))
            let fwdz = fwdx.atan()
            let fwdzCorrect = combo.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([0.9046; 0.2344; 1.4292; 0.9399; 0.3375])

            let revx = combo.tensor([0.9473; 0.4891; 0.2015; 0.5818; 0.8439]).reverseDiff()
            let revz = revx.atan()
            let revzCorrect = combo.tensor([0.7583; 0.4549; 0.1988; 0.5269; 0.7009])
            revz.reverse(combo.tensor([5.; 5.; 5.; 5.; -5.]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([2.6352;  4.0348;  4.8049;  3.7355; -2.9203])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeStackTs () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdxa = combo.tensor([1.; 2.]).forwardDiff(combo.tensor([10.;20.]))
            let fwdxb = combo.tensor([3.; 4.]).forwardDiff(combo.tensor([30.;40.]))
            let fwdxc = combo.tensor([5.; 6.]).forwardDiff(combo.tensor([50.;60.]))
            let fwdz = dsharp.stack([fwdxa;fwdxb;fwdxc])
            let fwdzCorrect = combo.tensor([[1.;2.];[3.;4.];[5.;6.]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[10.;20.];[30.;40.];[50.;60.]])

            let revxa = combo.tensor([1.; 2.]).reverseDiff()
            let revxb = combo.tensor([3.; 4.]).reverseDiff()
            let revxc = combo.tensor([5.; 6.]).reverseDiff()
            let revz = dsharp.stack([revxa;revxb;revxc])
            let revzCorrect = combo.tensor([[1.;2.];[3.;4.];[5.;6.]])
            revz.reverse(combo.tensor([[10.;20.];[30.;40.];[50.;60.]]))
            let revxda = revxa.derivative
            let revxdb = revxb.derivative
            let revxdc = revxc.derivative
            let revxdaCorrect = combo.tensor([10.; 20.])
            let revxdbCorrect = combo.tensor([30.; 40.])
            let revxdcCorrect = combo.tensor([50.; 60.])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxda.allclose(revxdaCorrect, 0.01))
            Assert.True(revxdb.allclose(revxdbCorrect, 0.01))
            Assert.True(revxdc.allclose(revxdcCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeUnstackT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.;2.];[3.;4.];[5.;6.]]).forwardDiff(combo.tensor([[10.;20.];[30.;40.];[50.;60.]]))
            let fwdz = dsharp.unstack(fwdx) |> Seq.toArray
            let fwdza = fwdz.[0]
            let fwdzb = fwdz.[1]
            let fwdzc = fwdz.[2]
            let fwdzda = fwdza.derivative
            let fwdzdb = fwdzb.derivative
            let fwdzdc = fwdzc.derivative
            let fwdzaCorrect = combo.tensor([1.; 2.])
            let fwdzbCorrect = combo.tensor([3.; 4.])
            let fwdzcCorrect = combo.tensor([5.; 6.])
            let fwdzdaCorrect = combo.tensor([10.; 20.])
            let fwdzdbCorrect = combo.tensor([30.; 40.])
            let fwdzdcCorrect = combo.tensor([50.; 60.])

            let revx = combo.tensor([[1.;2.];[3.;4.];[5.;6.]]).reverseDiff()
            let revz = dsharp.unstack(revx) |> Seq.toArray
            let revza = revz.[0]
            let revzb = revz.[1]
            let revzc = revz.[2]
            let revzaCorrect = combo.tensor([1.; 2.])
            let revzbCorrect = combo.tensor([3.; 4.])
            let revzcCorrect = combo.tensor([5.; 6.])
            revza.reverse(combo.tensor([10.; 20.]))
            revzb.reverse(combo.tensor([30.; 40.]), zeroDerivatives=false)
            revzc.reverse(combo.tensor([50.; 60.]), zeroDerivatives=false)
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[10.;20.];[30.;40.];[50.;60.]])

            Assert.True(fwdza.allclose(fwdzaCorrect, 0.01))
            Assert.True(fwdzb.allclose(fwdzbCorrect, 0.01))
            Assert.True(fwdzc.allclose(fwdzcCorrect, 0.01))
            Assert.True(fwdzda.allclose(fwdzdaCorrect, 0.01))
            Assert.True(fwdzdb.allclose(fwdzdbCorrect, 0.01))
            Assert.True(fwdzdc.allclose(fwdzdcCorrect, 0.01))
            Assert.True(revza.allclose(revzaCorrect, 0.01))
            Assert.True(revzb.allclose(revzbCorrect, 0.01))
            Assert.True(revzc.allclose(revzcCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeUnstackT_Dim1 () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.;2.];[3.;4.];[5.;6.]]).forwardDiff(combo.tensor([[10.;20.];[30.;40.];[50.;60.]]))
            let fwdz = fwdx.unstack(dim=1) |> Seq.toArray
            let fwdza = fwdz.[0]
            let fwdzb = fwdz.[1]
            let fwdzda = fwdza.derivative
            let fwdzdb = fwdzb.derivative
            let fwdzaCorrect = combo.tensor([1.; 3.; 5.])
            let fwdzbCorrect = combo.tensor([2.; 4.; 6.])
            let fwdzdaCorrect = combo.tensor([10.; 30.; 50.])
            let fwdzdbCorrect = combo.tensor([20.; 40.; 60.])

            let revx = combo.tensor([[1.;2.];[3.;4.];[5.;6.]]).reverseDiff()
            let revz = revx.unstack(dim=1) |> Seq.toArray
            let revza = revz.[0]
            let revzb = revz.[1]
            let revzaCorrect = combo.tensor([1.; 3.; 5.])
            let revzbCorrect = combo.tensor([2.; 4.; 6.])
            revza.reverse(combo.tensor([10.; 30.; 50.]))
            revzb.reverse(combo.tensor([20.; 40.; 60.]), zeroDerivatives=false)
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[10.;20.];[30.;40.];[50.;60.]])

            Assert.True(fwdza.allclose(fwdzaCorrect, 0.01))
            Assert.True(fwdzb.allclose(fwdzbCorrect, 0.01))
            Assert.True(fwdzda.allclose(fwdzdaCorrect, 0.01))
            Assert.True(fwdzdb.allclose(fwdzdbCorrect, 0.01))
            Assert.True(revza.allclose(revzaCorrect, 0.01))
            Assert.True(revzb.allclose(revzbCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeCatTs () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdxa = combo.tensor([1.; 2.]).forwardDiff(combo.tensor([10.;20.]))
            let fwdxb = combo.tensor([3.; 4.]).forwardDiff(combo.tensor([30.;40.]))
            let fwdxc = combo.tensor([5.; 6.]).forwardDiff(combo.tensor([50.;60.]))
            let fwdz = Tensor.cat([fwdxa;fwdxb;fwdxc])
            let fwdzCorrect = combo.tensor([1.;2.;3.;4.;5.;6.])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([10.;20.;30.;40.;50.;60.])

            let revxa = combo.tensor([1.; 2.]).reverseDiff()
            let revxb = combo.tensor([3.; 4.]).reverseDiff()
            let revxc = combo.tensor([5.; 6.]).reverseDiff()
            let revz = Tensor.cat([revxa;revxb;revxc])
            let revzCorrect = combo.tensor([1.;2.;3.;4.;5.;6.])
            revz.reverse(combo.tensor([10.;20.;30.;40.;50.;60.]))
            let revxda = revxa.derivative
            let revxdb = revxb.derivative
            let revxdc = revxc.derivative
            let revxdaCorrect = combo.tensor([10.; 20.])
            let revxdbCorrect = combo.tensor([30.; 40.])
            let revxdcCorrect = combo.tensor([50.; 60.])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxda.allclose(revxdaCorrect, 0.01))
            Assert.True(revxdb.allclose(revxdbCorrect, 0.01))
            Assert.True(revxdc.allclose(revxdcCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSplitT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1.;2.;3.;4.;5.;6.]).forwardDiff(combo.tensor([10.;20.;30.;40.;50.;60.]))
            let fwdz = fwdx.split([| 1;3;2 |])
            let fwdza = fwdz.[0]
            let fwdzb = fwdz.[1]
            let fwdzc = fwdz.[2]
            let fwdzda = fwdza.derivative
            let fwdzdb = fwdzb.derivative
            let fwdzdc = fwdzc.derivative
            let fwdzaCorrect = combo.tensor([1.])
            let fwdzbCorrect = combo.tensor([2.; 3.; 4.])
            let fwdzcCorrect = combo.tensor([5.; 6.])
            let fwdzdaCorrect = combo.tensor([10.])
            let fwdzdbCorrect = combo.tensor([20.;30.; 40.])
            let fwdzdcCorrect = combo.tensor([50.; 60.])

            let revx = combo.tensor([1.;2.;3.;4.;5.;6.]).reverseDiff()
            let revz = revx.split([| 1;3;2 |])
            let revza = revz.[0]
            let revzb = revz.[1]
            let revzc = revz.[2]
            let revzaCorrect = combo.tensor([1.])
            let revzbCorrect = combo.tensor([2.;3.; 4.])
            let revzcCorrect = combo.tensor([5.; 6.])
            revza.reverse(combo.tensor([10.]))
            revzb.reverse(combo.tensor([20.;30.; 40.]), zeroDerivatives=false)
            revzc.reverse(combo.tensor([50.; 60.]), zeroDerivatives=false)
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([10.;20.;30.;40.;50.;60.])

            Assert.True(fwdza.allclose(fwdzaCorrect, 0.01))
            Assert.True(fwdzb.allclose(fwdzbCorrect, 0.01))
            Assert.True(fwdzc.allclose(fwdzcCorrect, 0.01))
            Assert.True(fwdzda.allclose(fwdzdaCorrect, 0.01))
            Assert.True(fwdzdb.allclose(fwdzdbCorrect, 0.01))
            Assert.True(fwdzdc.allclose(fwdzdcCorrect, 0.01))
            Assert.True(revza.allclose(revzaCorrect, 0.01))
            Assert.True(revzb.allclose(revzbCorrect, 0.01))
            Assert.True(revzc.allclose(revzcCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSliceT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).forwardDiff(combo.tensor([8.8405; 2.7188; 1.5814; 8.7951; 0.1119]))
            let fwdz = fwdx.[2..]
            let fwdzCorrect = combo.tensor([16.0868; 74.5486; 82.9318])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([1.5814; 8.7951; 0.1119])

            let revx = combo.tensor([54.7919; 70.6440; 16.0868; 74.5486; 82.9318]).reverseDiff()
            let revz = revx.[2..]
            let revzCorrect = combo.tensor([16.0868; 74.5486; 82.9318])
            revz.reverse(combo.tensor([0.9360; 0.8748; 0.4353]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.; 0.; 0.9360; 0.8748; 0.4353])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1; 2; 3]).forwardDiff(combo.tensor([10; 20; 30]))
            let fwdz = fwdx.[..0] // In Python this is [:1] because in Python upper limits are exclusive whereas in F# they are inclusive
            let fwdzCorrect = combo.tensor([1])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([10])

            let revx = combo.tensor([1; 2; 3]).reverseDiff()
            let revz = revx.[..0] // In Python this is [:1] because in Python upper limits are exclusive whereas in F# they are inclusive
            let revzCorrect = combo.tensor([1])
            revz.reverse(combo.tensor([10]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([10; 0; 0])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([1; 2; 3]).forwardDiff(combo.tensor([10; 20; 30]))
            let fwdz = fwdx.[2..]
            let fwdzCorrect = combo.tensor([3])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([30])

            let revx = combo.tensor([1; 2; 3]).reverseDiff()
            let revz = revx.[2..]
            let revzCorrect = combo.tensor([3])
            revz.reverse(combo.tensor([30]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0; 0; 30])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1; 2; 3]; [4; 5; 6]; [7; 8; 9]]).forwardDiff(combo.tensor([[10; 20; 30]; [40; 50; 60]; [70; 80; 90]]))
            let fwdz = fwdx.[..0] // In Python this is [:1] because in Python upper limits are exclusive whereas in F# they are inclusive
            let fwdzCorrect = combo.tensor([[1; 2; 3]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[10; 20; 30]])

            let revx = combo.tensor([[1; 2; 3]; [4; 5; 6]; [7; 8; 9]]).reverseDiff()
            let revz = revx.[..0] // In Python this is [:1] because in Python upper limits are exclusive whereas in F# they are inclusive
            let revzCorrect = combo.tensor([[1; 2; 3]])
            revz.reverse(combo.tensor([[10; 20; 30]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[10; 20; 30]; [0; 0; 0]; [0; 0; 0]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1; 2; 3]; [4; 5; 6]; [7; 8; 9]]).forwardDiff(combo.tensor([[10; 20; 30]; [40; 50; 60]; [70; 80; 90]]))
            let fwdz = fwdx.[2..]
            let fwdzCorrect = combo.tensor([[7; 8; 9]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[70; 80; 90]])

            let revx = combo.tensor([[1; 2; 3]; [4; 5; 6]; [7; 8; 9]]).reverseDiff()
            let revz = revx.[2..]
            let revzCorrect = combo.tensor([[7; 8; 9]])
            revz.reverse(combo.tensor([[70; 80; 90]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[0; 0; 0]; [0; 0; 0]; [70; 80; 90]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeAddTTSlice () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[-0.0552;  0.6113; -0.2341];
                [ 1.4232; -1.2062;  0.3189];
                [ 0.6859; -0.3385; -0.1263];
                [-0.5159; -1.1882; -1.3437]]))
            let fwdy = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]])            
            let fwdy = fwdy.forwardDiff(combo.tensor([[ 1.3431; 23.0647];
                [71.1838; 39.8339]]))        
            let fwdz = fwdx.addSlice([0;1], fwdy)
            let fwdzCorrect = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[-5.5237e-02;  1.9544e+00;  2.2831e+01];
                [ 1.4232e+00;  6.9978e+01;  4.0153e+01];
                [ 6.8592e-01; -3.3845e-01; -1.2635e-01];
                [-5.1592e-01; -1.1882e+00; -1.3437e+00]])

            let revx = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]]).reverseDiff()
            let revy = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]]).reverseDiff()
            let revz = revx.addSlice([0;1], revy)
            let revzCorrect = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])
            revz.reverse(combo.tensor([[ 1.2453;  1.2199; -0.5281];
                [ 1.2203; -0.8378; -0.3876];
                [ 0.3626; -0.1200; -0.1496];
                [-0.6304;  1.0198; -0.4969]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[ 1.2453;  1.2199; -0.5281];
                [ 1.2203; -0.8378; -0.3876];
                [ 0.3626; -0.1200; -0.1496];
                [-0.6304;  1.0198; -0.4969]])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([[1.2199; -0.5281]; [-0.8378; -0.3876]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSliceT_2D () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do

            let revx = combo.tensor([[54.7919; 70.6440; 16.0868; 74.5486; 82.9318]; 
                                     [54.7919; 70.6440; 16.0868; 74.5486; 82.9318]]).reverseDiff()
            let revz = revx.[*,2..2]
            let revzCorrect = combo.tensor([16.0868; 16.0868])
            revz.reverse(combo.tensor([0.9360;0.9360]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[0.; 0.; 0.9360; 0.; 0.];[0.; 0.; 0.9360; 0.; 0.]])

            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeAddTTConstSlice () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]])
            let fwdx = fwdx.forwardDiff(combo.tensor([[-0.0552;  0.6113; -0.2341];
                [ 1.4232; -1.2062;  0.3189];
                [ 0.6859; -0.3385; -0.1263];
                [-0.5159; -1.1882; -1.3437]]))
            let fwdy = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]])  
            let fwdz = fwdx.addSlice([0;1], fwdy)
            let fwdzCorrect = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[-5.5237e-02;  0.6113;  -0.2337];
                [ 1.4232e+00;  -1.2058;  0.3191];
                [ 6.8592e-01; -3.3845e-01; -1.2635e-01];
                [-5.1592e-01; -1.1882e+00; -1.3437e+00]])

            let revx = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]]).reverseDiff()
            let revy = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]])
            let revz = revx.addSlice([0;1], revy)
            let revzCorrect = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])
            revz.reverse(combo.tensor([[ 1.2453;  1.2199; -0.5281];
                [ 1.2203; -0.8378; -0.3876];
                [ 0.3626; -0.1200; -0.1496];
                [-0.6304;  1.0198; -0.4969]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[ 1.2453;  1.2199; -0.5281];
                [ 1.2203; -0.8378; -0.3876];
                [ 0.3626; -0.1200; -0.1496];
                [-0.6304;  1.0198; -0.4969]])
            let revyd = revy.isNoDiff()
            let revydCorrect = true

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.CheckEqual(revydCorrect, revyd)

    [<Test>]
    member _.TestDerivativeAddTConstTSlice () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]])
            let fwdy = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]])            
            let fwdy = fwdy.forwardDiff(combo.tensor([[ 1.3431; 23.0647];
                [71.1838; 39.8339]]))        
            let fwdz = fwdx.addSlice([0;1], fwdy)
            let fwdzCorrect = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[0.;  1.3431;  23.0647];
                [ 0.;  71.1838;  39.8339];
                [ 0.; 0.; 0.];
                [0.; 0.; 0.]])

            let revx = combo.tensor([[-0.2754;  0.0172;  0.7105];
                [-0.1890;  1.7664;  0.5377];
                [-0.5313; -2.2530; -0.6235];
                [ 0.6776;  1.5844; -0.5686]])
            let revy = combo.tensor([[-111.8892;   -7.0328];
                [  18.7557;  -86.2308]]).reverseDiff()
            let revz = revx.addSlice([0;1], revy)
            let revzCorrect = combo.tensor([[  -0.2754; -111.8720;   -6.3222];
                [  -0.1890;   20.5221;  -85.6932];
                [  -0.5313;   -2.2530;   -0.6235];
                [   0.6776;    1.5844;   -0.5686]])
            revz.reverse(combo.tensor([[ 1.2453;  1.2199; -0.5281];
                [ 1.2203; -0.8378; -0.3876];
                [ 0.3626; -0.1200; -0.1496];
                [-0.6304;  1.0198; -0.4969]]))
            let revxd = revx.isNoDiff()
            let revxdCorrect = true
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([[1.2199; -0.5281]; [-0.8378; -0.3876]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.CheckEqual(revxdCorrect, revxd)
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeSqueezeT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[[1.; 2.]]; [[3.;4.]]]).forwardDiff(combo.tensor([[[10.; 20.]]; [[30.;40.]]]))
            let fwdz = fwdx.squeeze()
            let fwdzCorrect =  combo.tensor([[1.;2.];[3.;4.]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect =  combo.tensor([[10.;20.];[30.;40.]])

            let revx = combo.tensor([[[1.; 2.]]; [[3.;4.]]]).reverseDiff()
            let revz = revx.squeeze()
            let revzCorrect =  combo.tensor([[1.;2.];[3.;4.]])
            revz.reverse(combo.tensor([[10.;20.];[30.;40.]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[[10.; 20.]]; [[30.;40.]]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeUnsqueezeT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.;2.];[3.;4.]]).forwardDiff(combo.tensor([[10.;20.];[30.;40.]]))
            let fwdz = fwdx.unsqueeze(1)
            let fwdzCorrect =  combo.tensor([[[1.; 2.]]; [[3.;4.]]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect =  combo.tensor([[[10.; 20.]]; [[30.;40.]]])

            let revx = combo.tensor([[1.;2.];[3.;4.]]).reverseDiff()
            let revz = revx.unsqueeze(1)
            let revzCorrect =  combo.tensor([[[1.; 2.]]; [[3.;4.]]])
            revz.reverse(combo.tensor([[[10.; 20.]]; [[30.;40.]]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[10.;20.];[30.;40.]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeFlipT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.;2.];[3.;4.]]).forwardDiff(combo.tensor([[10.;20.];[30.;40.]]))
            let fwdz = fwdx.flip([|0; 1|])
            let fwdzCorrect =  combo.tensor([[4.; 3.]; [2.;1.]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect =  combo.tensor([[40.; 30.]; [20.;10.]])

            let revx = combo.tensor([[1.;2.];[3.;4.]]).reverseDiff()
            let revz = revx.flip([|0; 1|])
            let revzCorrect =  combo.tensor([[4.; 3.]; [2.;1.]])
            revz.reverse(combo.tensor([[40.; 30.]; [20.;10.]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[10.;20.];[30.;40.]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeDilateT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.;2.];[3.;4.]]).forwardDiff(combo.tensor([[10.;20.];[30.;40.]]))
            let fwdz = fwdx.dilate([|2; 2|])
            let fwdzCorrect =  combo.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect =  combo.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]])

            let revx = combo.tensor([[1.;2.];[3.;4.]]).reverseDiff()
            let revz = revx.dilate([|2; 2|])
            let revzCorrect =  combo.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]])
            revz.reverse(combo.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[10.;20.];[30.;40.]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeUndilateT () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]]).forwardDiff(combo.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]]))
            let fwdz = fwdx.undilate([|2; 2|])
            let fwdzCorrect =  combo.tensor([[1.;2.];[3.;4.]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect =  combo.tensor([[10.;20.];[30.;40.]])

            let revx = combo.tensor([[1.; 0.; 2.]; [0.; 0.; 0.]; [3.; 0.; 4.]]).reverseDiff()
            let revz = revx.undilate([|2; 2|])
            let revzCorrect =  combo.tensor([[1.;2.];[3.;4.]])
            revz.reverse(combo.tensor([[10.;20.];[30.;40.]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[10.; 0.; 20.]; [0.; 0.; 0.]; [30.; 0.; 40.]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeClampT () =
        for combo in Combos.SignedIntegralAndFloatingPointExcept16s do 
            let fwdx = combo.tensor([-4,-3,-2,-1,0,1,2,3,4]).forwardDiff(combo.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90]))
            let fwdz = fwdx.clamp(-2, 3)
            let fwdzCorrect = combo.tensor([-2, -2, -2, -1,  0,  1,  2,  3,  3])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([ 0,  0, 30, 40, 50, 60, 70, 80,  0])

            let revx = combo.tensor([-4,-3,-2,-1,0,1,2,3,4]).reverseDiff()
            let revz = revx.clamp(-2, 3)
            let revzCorrect = combo.tensor([-2, -2, -2, -1,  0,  1,  2,  3,  3])
            revz.reverse(combo.tensor([100, 200, 300, 400, 500, 600, 700, 800, 900]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([  0,   0, 300, 400, 500, 600, 700, 800,   0])

            Assert.CheckEqual(fwdzCorrect, fwdz)
            Assert.CheckEqual(fwdzdCorrect, fwdzd)
            Assert.CheckEqual(revzCorrect, revz)
            Assert.CheckEqual(revxdCorrect, revxd)

    [<Test>]
    member _.TestDerivativeSoftmax () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([[4.6815; 5.6441; 7.4689];
                [9.1976; 8.1241; 7.4521]]).forwardDiff(combo.tensor([[8.0030; 7.0798; 6.8637];
                    [9.5760; 7.4524; 2.6404]]))
            let fwdz = fwdx.softmax(dim=1)
            let fwdzCorrect = combo.tensor([[0.0504; 0.1319; 0.8178];
                [0.6595; 0.2254; 0.1151]])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([[0.0530; 0.0172; -0.0702]; [0.8422; -0.1908; -0.6514]])

            let revx = combo.tensor([[4.6815; 5.6441; 7.4689];
                [9.1976; 8.1241; 7.4521]]).reverseDiff()
            let revz = revx.softmax(dim=1)
            let revzCorrect = combo.tensor([[0.0504; 0.1319; 0.8178];
                [0.6595; 0.2254; 0.1151]])
            revz.reverse(combo.tensor([[6.0933; 9.6456; 7.0996];
                [0.2617; 1.7002; 4.9711]]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([[-0.0649; 0.2988; -0.2329]; [-0.5713; 0.1291; 0.4426]])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeMaxBinary () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).forwardDiff(combo.tensor([1.9788; 0.2861; 4.2025; 0.5602; 7.9510]))
            let fwdy = combo.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).forwardDiff(combo.tensor([9.6600; 6.9111; 9.7303; 0.1491; 7.7003]))
            let fwdz = dsharp.max(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([19.3520;  8.9730;  4.2405; -3.5977; 12.2048])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([1.9788; 0.2861; 9.7303; 0.5602; 7.7003])

            let revx = combo.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).reverseDiff()
            let revy = combo.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).reverseDiff()
            let revz = dsharp.max(revx, revy)
            let revzCorrect = combo.tensor([19.3520;  8.9730;  4.2405; -3.5977; 12.2048])
            revz.reverse(combo.tensor([  9.7293; -10.2704; -13.7527;  -3.9050;  -1.6439]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([9.7293; -10.2704; 0.; -3.9050; 0.])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([0.; 0.; -13.7527; 0.; -1.6439])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

    [<Test>]
    member _.TestDerivativeMinBinary () =
        for combo in Combos.AllDevicesAndBackendsFloat32 do
            let fwdx = combo.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).forwardDiff(combo.tensor([1.9788; 0.2861; 4.2025; 0.5602; 7.9510]))
            let fwdy = combo.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).forwardDiff(combo.tensor([9.6600; 6.9111; 9.7303; 0.1491; 7.7003]))
            let fwdz = dsharp.min(fwdx, fwdy)
            let fwdzCorrect = combo.tensor([-17.1885;  -4.0684; -23.6274; -21.7158; -20.3245])
            let fwdzd = fwdz.derivative
            let fwdzdCorrect = combo.tensor([9.6600; 6.9111; 4.2025; 0.1491; 7.9510])

            let revx = combo.tensor([ 19.3520;   8.9730; -23.6274;  -3.5977; -20.3245]).reverseDiff()
            let revy = combo.tensor([-17.1885;  -4.0684;   4.2405; -21.7158;  12.2048]).reverseDiff()
            let revz = dsharp.min(revx, revy)
            let revzCorrect = combo.tensor([-17.1885;  -4.0684; -23.6274; -21.7158; -20.3245])
            revz.reverse(combo.tensor([  9.7293; -10.2704; -13.7527;  -3.9050;  -1.6439]))
            let revxd = revx.derivative
            let revxdCorrect = combo.tensor([0.; 0.; -13.7527; 0.; -1.6439])
            let revyd = revy.derivative
            let revydCorrect = combo.tensor([9.7293; -10.2704; 0.; -3.9050; 0.])

            Assert.True(fwdz.allclose(fwdzCorrect, 0.01))
            Assert.True(fwdzd.allclose(fwdzdCorrect, 0.01))
            Assert.True(revz.allclose(revzCorrect, 0.01))
            Assert.True(revxd.allclose(revxdCorrect, 0.01))
            Assert.True(revyd.allclose(revydCorrect, 0.01))

