//
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright 2014 National University of Ireland Maynooth.
// All rights reserved.
//
// Written by:
//
//   Atilim Gunes Baydin
//   atilimgunes.baydin@nuim.ie
//
//   Barak A. Pearlmutter
//   barak@cs.nuim.ie
//
//   Hamilton Institute & Department of Computer Science
//   National University of Ireland Maynooth
//   Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

//
// Reference for "triplets": Dixon L., 2001, Automatic Differentiation: Calculation of the Hessian (http://dx.doi.org/10.1007/0-306-48332-7_17)
//

#light

/// Forward AD module, using triplets for gradient and Hessian computation
module DiffSharp.AD.ForwardTriplet

open DiffSharp.Util
open DiffSharp.Util.General

/// DualT numeric type, keeping a triplet of primal value, a vector of gradient components, and a matrix of Hessian components
// NOT FULLY OPTIMIZED
type DualT =
    | DualT of float * Vector * Matrix
    override d.ToString() = let (DualT(p, g, h)) = d in sprintf "DualT (%f, %A, %A)" p g h
    static member op_Explicit(p) = DualT(p, Vector.Zero, Matrix.Zero)
    static member op_Explicit(DualT(p, _, _)) = p
    static member DivideByInt(DualT(p, g, m), i:int) = DualT(p / float i, g / float i, m / float i)
    static member Zero = DualT(0., Vector.Zero, Matrix.Zero)
    static member One = DualT(1., Vector.Zero, Matrix.Zero)
    static member (+) (DualT(a, ag, ah), DualT(b, bg, bh)) = DualT(a + b, ag + bg, ah + bh)
    static member (-) (DualT(a, ag, ah), DualT(b, bg, bh)) = DualT(a - b, ag - bg, ah - bh)
    static member (*) (DualT(a, ag, ah), DualT(b, bg, bh)) = DualT(a * b, ag * b + a * bg, Matrix.Create(ah.Rows, ah.Cols, fun i j -> ag.[j] * bg.[i] + a * bh.[i, j] + bg.[j] * ag.[i] + b * ah.[i, j]))
    static member (/) (DualT(a, ag, ah), DualT(b, bg, bh)) = DualT(a / b, (ag * b - a * bg) / (b * b), Matrix.Create(ah.Rows, ah.Cols, fun i j -> (2. * a * bg.[j] * bg.[i] + b * b * ah.[i, j] - b * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + a * bh.[i, j])) / (b * b * b)))
    static member Pow (DualT(a, ag, ah), DualT(b, bg, bh)) = let loga = log a in DualT(a ** b, (a ** b) * ((b * ag / a) + (loga * bg)), Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a ** (b - 2.)) * (b * b * ag.[j] * ag.[i] + b * (ag.[j] * (-ag.[i] + loga * a * bg.[i]) + a * (loga * bg.[j] * ag.[i] + ah.[i, j])) + a * (ag.[j] * bg.[i] + bg.[j] * (ag.[i] + loga * loga * a * bg.[i]) + loga * a * bh.[i, j]))))
    static member (+) (DualT(a, ag, ah), b) = DualT(a + b, ag, ah)
    static member (-) (DualT(a, ag, ah), b) = DualT(a - b, ag, ah)
    static member (*) (DualT(a, ag, ah), b) = DualT(a * b, ag * b, Matrix.Create(ah.Rows, ah.Cols, fun i j -> ah.[i, j] * b))
    static member (/) (DualT(a, ag, ah), b) = DualT(a / b, ag / b, Matrix.Create(ah.Rows, ah.Cols, fun i j -> ah.[i, j] / b))
    static member Pow (DualT(a, ag, ah), b) = DualT(a ** b, (a ** b) * (b * ag / a), Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a ** (b - 2.)) * (b * b * ag.[j] * ag.[i] + b * (a * ah.[i, j] - ag.[j] * ag.[i]))))
    static member (+) (a, DualT(b, bg, bh)) = DualT(a + b, bg, bh)
    static member (-) (a, DualT(b, bg, bh)) = DualT(a - b, -bg, -bh)
    static member (*) (a, DualT(b, bg, bh)) = DualT(a * b, a * bg, Matrix.Create(bh.Rows, bh.Cols, fun i j -> a * bh.[i, j]))
    static member (/) (a, DualT(b, bg, bh)) = DualT(a / b, (-a * bg) / (b * b), Matrix.Create(bh.Rows, bh.Cols, fun i j -> a * (2. * bg.[j] * bg.[i] - b * bh.[i, j]) / (b * b * b)))
    static member Pow (a, DualT(b, bg, bh)) = let loga = log a in DualT(a ** b, (a ** b) * loga * bg, Matrix.Create(bh.Rows, bh.Cols, fun i j -> (a ** (b - 2.)) * a * loga * (bg.[j] * loga * a * bg.[i] + a * bh.[i, j])))
    static member Log (DualT(a, ag, ah)) = DualT(log a, ag / a, Matrix.Create(ah.Rows, ah.Cols, fun i j -> - ag.[i] * ag.[j] / (a * a) + ah.[i, j] / a))
    static member Exp (DualT(a, ag, ah)) = let expa = exp a in DualT(expa, expa * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> expa * ag.[i] * ag.[j] + expa * ah.[i, j]))
    static member Sin (DualT(a, ag, ah)) = let (sina, cosa) = (sin a, cos a) in DualT(sina, cosa * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> -sina * ag.[i] * ag.[j] + cosa * ah.[i, j]))
    static member Cos (DualT(a, ag, ah)) = let (sina, cosa) = (sin a, cos a) in DualT(cosa, -sina * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> -cosa * ag.[i] * ag.[j] - sina * ah.[i, j]))
    static member Tan (DualT(a, ag, ah)) = let (tana, seca) = (tan a, 1. / cos a) in DualT(tana, (seca * seca) * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> 2. * seca * seca * tana * ag.[i] * ag.[j] + seca * seca * ah.[i, j]))
    static member (~-) (DualT(a, ag, ah)) = DualT(-a, -ag, -ah)
    static member Sqrt (DualT(a, ag, ah)) = let s = 1. / (2. * sqrt a) in DualT(sqrt a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (s / (-2. * a)) * ag.[i] * ag.[j] + s * ah.[i,j]))
    static member Sinh (DualT(a, ag, ah)) = let (sinha, cosha) = (sinh a, cosh a) in DualT(sinha, cosha * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> sinha * ag.[i] * ag.[j] + cosha * ah.[i, j]))
    static member Cosh (DualT(a, ag, ah)) = let (sinha, cosha) = (sinh a, cosh a) in DualT(cosha, sinha * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> cosha * ag.[i] * ag.[j] + sinha * ah.[i, j]))
    static member Tanh (DualT(a, ag, ah)) = let (tanha, secha) = (tanh a, 1. / cosh a) in DualT(tanha, secha * secha * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> -2. * secha * secha * tanha * ag.[i] * ag.[j] + secha * secha * ah.[i, j]))
    static member Asin (DualT(a, ag, ah)) = let s = 1. / sqrt (1. - a * a) in DualT(asin a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a / (1. - a * a)) * s * ag.[i] * ag.[j] + s * ah.[i, j]))
    static member Acos (DualT(a, ag, ah)) = let s = -1. / sqrt (1. - a * a) in DualT(acos a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (a / (1. - a * a)) * s * ag.[i] * ag.[j] + s * ah.[i, j]))
    static member Atan (DualT(a, ag, ah)) = let s = 1. / (1. + a * a) in DualT(atan a, s * ag, Matrix.Create(ah.Rows, ah.Cols, fun i j -> (-2. * a / (1. + a * a)) * s * ag.[i] * ag.[j] + s * ah.[i, j]))


/// DualT operations module (automatically opened)
[<AutoOpen>]
module DualTOps =
    /// Make DualT, with primal value `p`, gradient dimension `m`, and all gradient and Hessian components 0
    let inline dualT p m = DualT(p, Vector.Create(m, 0.), Matrix.Create(m, m, 0.))
    /// Make DualT, with primal value `p`, gradient array `g`, and Hessian 2d array `h`
    let inline dualTSet (p, g, h) = DualT(p, Vector.Create(g), Matrix.Create(h))
    /// Make active DualT (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the gradient component with index `i` having value 1, the rest of the gradient components 0, and Hessian components 0
    let inline dualTAct p m i = DualT(p, Vector.Create(m, i, 1.), Matrix.Create(m, m, 0.))
    /// Make an array of active DualT, with primal values given in array `x`. For a DualT with index _i_, the gradient is the unit vector with 1 in the _i_th place, and the Hessian components are 0.
    let inline dualTActArray (x:float[]) = Array.init x.Length (fun i -> dualTAct x.[i] x.Length i)
    /// Get the primal value of a DualT
    let inline primal (DualT(p, _, _)) = p
    /// Get the gradient array of a DualT
    let inline gradient (DualT(_, g, _)) = g.V
    /// Get the Hessian 2d array of a DualT
    let inline hessian (DualT(_, _, h)) = h.M
    /// Get the primal and the first gradient component of a DualT, as a tuple
    let inline tuple (DualT(p, g, _)) = (p, g.V.[0])
    /// Get the primal and the gradient array of a DualT, as a tuple
    let inline tupleG (DualT(p, g, _)) = (p, g.V)
    /// Get the primal and Hessian 2d array of a DualT, as a tuple
    let inline tupleH (DualT(p, _, h)) = (p, h.M)
    /// Get the primal, the gradient array, and the Hessian 2d array of a DualT, as a tuple
    let inline tupleGH (DualT(p, g, h)) = (p, g.V, h.M)


[<AutoOpen>]
module ForwardTripletOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f =
        fun x -> dualTAct x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f =
        diff' f >> snd

    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f =
        dualTActArray >> f >> tupleG

    /// Gradient of a vector-to-scalar function `f`
    let inline grad f =
        grad' f >> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f =
        fun x ->
            let a = dualTActArray x |> f
            (Array.map primal a, Matrix.Create(a.Length, fun i -> gradient a.[i]).M)

    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f =
        jacobian' f >> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f =
        fun x -> let (v, j) = jacobian' f x in (v, transpose j)
    
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f =
        jacobianT' f >> snd

    /// Original value and Hessian of a vector-to-scalar function `f`
    let inline hessian' f =
        dualTActArray >> f >> tupleH

    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f =
        hessian' f >> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f =
        dualTActArray >> f >> tupleGH
    
    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f =
        gradhessian' f >> sndtrd

    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f =
        fun x -> let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f =
        laplacian' f >> snd

