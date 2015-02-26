//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (C) 2014, 2015, National University of Ireland Maynooth.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with DiffSharp. If not, see <http://www.gnu.org/licenses/>.
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
//   Maynooth, Co. Kildare
//   Ireland
//
//   www.bcl.hamilton.ie
//

//
// Reference for "triplets": Dixon L., 2001, Automatic Differentiation: Calculation of the Hessian (http://dx.doi.org/10.1007/0-306-48332-7_17)
//

#light

/// Forward mode AD module, keeping vectors of gradient components and matrices of Hessian components
module DiffSharp.AD.ForwardGH

open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General

/// DualGH numeric type, keeping a triplet of primal value, a vector of gradient components, and a matrix of Hessian components
// NOT FULLY OPTIMIZED
[<CustomEquality; CustomComparison>]
type DualGH =
    // Primal, vector of gradient components, matrix of Hessian components
    | DualGH of float * Vector<float> * Matrix<float>
    override d.ToString() = let (DualGH(p, g, h)) = d in sprintf "DualGH (%A, %A, %A)" p g h
    static member op_Explicit(p) = DualGH(p, Vector.Zero, Matrix.Zero)
    static member op_Explicit(DualGH(p, _, _)) = p
    static member DivideByInt(DualGH(p, g, m), i:int) = DualGH(p / float i, g / float i, m / float i)
    static member Zero = DualGH(0., Vector.Zero, Matrix.Zero)
    static member One = DualGH(1., Vector.Zero, Matrix.Zero)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? DualGH as d2 -> let DualGH(a, _, _), DualGH(b, _, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this DualGH with another type of object."
    override d.Equals(other) = 
        match other with
        | :? DualGH as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (DualGH(a, b, c)) = d in hash [|a; b; c|]
    // DualGH - DualGH binary operations
    static member (+) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a + b, ag + bg, Matrix.SymmetricOp(ah, bh, fun i j -> ah.[i, j] + bh.[i, j]))
    static member (-) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a - b, ag - bg, Matrix.SymmetricOp(ah, bh, fun i j -> ah.[i, j] - bh.[i, j]))
    static member (*) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = DualGH(a * b, ag * b + a * bg, Matrix.SymmetricOp(ah, bh, fun i j -> ag.[j] * bg.[i] + a * bh.[i, j] + bg.[j] * ag.[i] + b * ah.[i, j]))
    static member (/) (DualGH(a, ag, ah), DualGH(b, bg, bh)) = let bsq, atimes2, oneoverbcube = b * b, a * 2., 1. / (b * b * b) in DualGH(a / b, (ag * b - a * bg) / bsq, Matrix.SymmetricOp(ah, bh, fun i j -> (atimes2 * bg.[j] * bg.[i] + bsq * ah.[i, j] - b * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + a * bh.[i, j])) * oneoverbcube))
    static member Pow (DualGH(a, ag, ah), DualGH(b, bg, bh)) = let apowb, loga, apowbminus2, bsq = a ** b, log a, a ** (b - 2.), b * b in DualGH(apowb, apowb * ((b * ag / a) + (loga * bg)), Matrix.SymmetricOp(ah, bh, fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (ag.[j] * (-ag.[i] + loga * a * bg.[i]) + a * (loga * bg.[j] * ag.[i] + ah.[i, j])) + a * (ag.[j] * bg.[i] + bg.[j] * (ag.[i] + loga * loga * a * bg.[i]) + loga * a * bh.[i, j]))))
    static member Atan2 (DualGH(a, ag, ah), DualGH(b, bg, bh)) = let asq, bsq = a * a, b * b in DualGH(atan2 a b, (ag * b - a * bg) / (asq + bsq), Matrix.SymmetricOp(ah, bh, fun i j -> (bsq * (-bg.[j] * ag.[i] - ag.[j] * bg.[i] + b * ah.[i, j]) + asq * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + b * ah.[i,j]) - (asq * a) * bh.[i, j] - a * b * (2. * ag.[j] * ag.[i] - 2. * bg.[j] * bg.[i] + b * bh.[i, j])) / (asq + bsq) ** 2.))
    // DualGH - float binary operations
    static member (+) (DualGH(a, ag, ah), b) = DualGH(a + b, ag, ah)
    static member (-) (DualGH(a, ag, ah), b) = DualGH(a - b, ag, ah)
    static member (*) (DualGH(a, ag, ah), b) = DualGH(a * b, ag * b, Matrix.SymmetricOp(ah, fun i j -> ah.[i, j] * b))
    static member (/) (DualGH(a, ag, ah), b) = DualGH(a / b, ag / b, Matrix.SymmetricOp(ah, fun i j -> ah.[i, j] / b))
    static member Pow (DualGH(a, ag, ah), b) = let apowb, bsq, apowbminus2 = a ** b, b * b, a ** (b - 2.) in DualGH(apowb, b * (a ** (b - 1.)) * ag, Matrix.SymmetricOp(ah, fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (a * ah.[i, j] - ag.[j] * ag.[i]))))
    static member Atan2 (DualGH(a, ag, ah), b) = let asq, bsq = a * a, b * b in DualGH(atan2 a b, (b * ag) / (bsq + asq), Matrix.SymmetricOp(ah, fun i j -> (b * (-2. * a * ag.[j] * ag.[i] + bsq * ah.[i,j] + asq * ah.[i, j])) / (bsq + asq) ** 2.))
    // float - DualGH binary operations
    static member (+) (a, DualGH(b, bg, bh)) = DualGH(a + b, bg, bh)
    static member (-) (a, DualGH(b, bg, bh)) = DualGH(a - b, -bg, -bh)
    static member (*) (a, DualGH(b, bg, bh)) = DualGH(a * b, a * bg, Matrix.SymmetricOp(bh, fun i j -> a * bh.[i, j]))
    static member (/) (a, DualGH(b, bg, bh)) = let aoverbcube = a / (b * b * b) in DualGH(a / b, -aoverbcube * b * bg, Matrix.SymmetricOp(bh, fun i j -> (2. * bg.[j] * bg.[i] - b * bh.[i, j]) * aoverbcube))
    static member Pow (a, DualGH(b, bg, bh)) = let apowb, loga, term = a ** b, log a, (a ** (b - 2.)) * a * log a in DualGH(apowb, apowb * loga * bg, Matrix.SymmetricOp(bh, fun i j -> term * (bg.[j] * loga * a * bg.[i] + a * bh.[i, j])))
    static member Atan2 (a, DualGH(b, bg, bh)) = let asq, bsq = a * a, b * b in DualGH(atan2 a b, -(a * bg) / (asq + bsq), Matrix.SymmetricOp(bh, fun i j -> -((a *(-2. * b * bg.[j] * bg.[i] + asq * bh.[i, j] + bsq * bh.[i, j])) / (asq + bsq) ** 2.)))
    // DualGH - int binary operations
    static member (+) (a:DualGH, b:int) = a + float b
    static member (-) (a:DualGH, b:int) = a - float b
    static member (*) (a:DualGH, b:int) = a * float b
    static member (/) (a:DualGH, b:int) = a / float b
    static member Pow (a:DualGH, b:int) = DualGH.Pow(a, float b)
    static member Atan2 (a:DualGH, b:int) = DualGH.Atan2(a, float b)
    // int - DualGH binary operations
    static member (+) (a:int, b:DualGH) = (float a) + b
    static member (-) (a:int, b:DualGH) = (float a) - b
    static member (*) (a:int, b:DualGH) = (float a) * b
    static member (/) (a:int, b:DualGH) = (float a) / b
    static member Pow (a:int, b:DualGH) = DualGH.Pow(float a, b)
    static member Atan2 (a:int, b:DualGH) = DualGH.Atan2(float a, b)
    // DualGH unary operations
    static member Log (DualGH(a, ag, ah)) = 
        if a <= 0. then invalidArgLog()
        let asq = a * a in DualGH(log a, ag / a, Matrix.SymmetricOp(ah, fun i j -> -ag.[i] * ag.[j] / asq + ah.[i, j] / a))
    static member Log10 (DualGH(a, ag, ah)) = 
        if a <= 0. then invalidArgLog10()
        let alog10 = a * log10val in DualGH(log10 a, ag / alog10, Matrix.SymmetricOp(ah, fun i j -> -ag.[i] * ag.[j] / (a * alog10) + ah.[i, j] / alog10))
    static member Exp (DualGH(a, ag, ah)) = let expa = exp a in DualGH(expa, expa * ag, Matrix.SymmetricOp(ah, fun i j -> expa * ag.[i] * ag.[j] + expa * ah.[i, j]))
    static member Sin (DualGH(a, ag, ah)) = let sina, cosa = sin a, cos a in DualGH(sina, cosa * ag, Matrix.SymmetricOp(ah, fun i j -> -sina * ag.[i] * ag.[j] + cosa * ah.[i, j]))
    static member Cos (DualGH(a, ag, ah)) = let sina, cosa = sin a, cos a in DualGH(cosa, -sina * ag, Matrix.SymmetricOp(ah, fun i j -> -cosa * ag.[i] * ag.[j] - sina * ah.[i, j]))
    static member Tan (DualGH(a, ag, ah)) = 
        let cosa = cos a
        if cosa = 0. then invalidArgTan()
        let tana, secsqa = tan a, 1. / ((cosa) * (cosa)) in DualGH(tana, secsqa * ag, Matrix.SymmetricOp(ah, fun i j -> 2. * secsqa * tana * ag.[i] * ag.[j] + secsqa * ah.[i, j]))
    static member (~-) (DualGH(a, ag, ah)) = DualGH(-a, -ag, -ah)
    static member Sqrt (DualGH(a, ag, ah)) = 
        if a <= 0. then invalidArgSqrt()
        let term = 1. / (2. * sqrt a) in DualGH(sqrt a, term * ag, Matrix.SymmetricOp(ah, fun i j -> (term / (-2. * a)) * ag.[i] * ag.[j] + term * ah.[i,j]))
    static member Sinh (DualGH(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in DualGH(sinha, cosha * ag, Matrix.SymmetricOp(ah, fun i j -> sinha * ag.[i] * ag.[j] + cosha * ah.[i, j]))
    static member Cosh (DualGH(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in DualGH(cosha, sinha * ag, Matrix.SymmetricOp(ah, fun i j -> cosha * ag.[i] * ag.[j] + sinha * ah.[i, j]))
    static member Tanh (DualGH(a, ag, ah)) = let tanha, sechsqa = tanh a, 1. / ((cosh a) * (cosh a)) in DualGH(tanha, sechsqa * ag, Matrix.SymmetricOp(ah, fun i j -> -2. * sechsqa * tanha * ag.[i] * ag.[j] + sechsqa * ah.[i, j]))
    static member Asin (DualGH(a, ag, ah)) = 
        if (abs a) >= 1. then invalidArgAsin()
        let term, term2 = 1. / sqrt (1. - a * a), (a / (1. - a * a)) in DualGH(asin a, term * ag, Matrix.SymmetricOp(ah, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Acos (DualGH(a, ag, ah)) = 
        if (abs a) >= 1. then invalidArgAcos()
        let term, term2 = -1. / sqrt (1. - a * a), (a / (1. - a * a)) in DualGH(acos a, term * ag, Matrix.SymmetricOp(ah, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Atan (DualGH(a, ag, ah)) = let term, term2 = 1. / (1. + a * a), (-2. * a / (1. + a * a)) in DualGH(atan a, term * ag, Matrix.SymmetricOp(ah, fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Abs (DualGH(a, ag, ah)) = 
        if a = 0. then invalidArgAbs()
        DualGH(abs a, ag * float (sign a), Matrix.SymmetricOp(ah, fun i j -> ah.[i, j] * float (sign a)))
    static member Floor (DualGH(a, ag, ah)) =
        if isInteger a then invalidArgFloor()
        DualGH(floor a, Vector.Create(ag.Length, 0.), Matrix.Create(ah.Rows, ah.Cols, 0.))
    static member Ceiling (DualGH(a, ag, ah)) =
        if isInteger a then invalidArgCeil()
        DualGH(ceil a, Vector.Create(ag.Length, 0.), Matrix.Create(ah.Rows, ah.Cols, 0.))
    static member Round (DualGH(a, ag, ah)) =
        if isHalfway a then invalidArgRound()
        DualGH(round a, Vector.Create(ag.Length, 0.), Matrix.Create(ah.Rows, ah.Cols, 0.))

/// DualGH operations module (automatically opened)
[<AutoOpen>]
module DualGHOps =
    /// Make DualGH, with primal value `p`, gradient dimension `m`, and all gradient and Hessian components 0
    let inline dualGH p m = DualGH(float p, Vector.Create(m, 0.), Matrix.Create(m, m, 0.))
    /// Make DualGH, with primal value `p`, gradient array `g`, and Hessian 2d array `h`
    let inline dualGHSet (p, g, h:float[,]) = DualGH(float p, Vector.Create(g), Matrix.Create(h))
    /// Make active DualGH (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the gradient component with index `i` having value 1, the rest of the gradient components 0, and Hessian components 0
    let inline dualGHAct p m i = DualGH(float p, Vector.Create(m, i, 1.), Matrix.Create(m, m, 0.))
    /// Make an array of active DualGH, with primal values given in array `x`. For a DualGH with index _i_, the gradient is the unit vector with 1 in the _i_th place, and the Hessian components are 0.
    let inline dualGHActArray (x:_[]) = Array.init x.Length (fun i -> dualGHAct x.[i] x.Length i)
    /// Get the primal value of a DualGH
    let inline primal (DualGH(p, _, _)) = p
    /// Get the gradient array of a DualGH
    let inline gradient (DualGH(_, g, _)) = Vector.toArray g
    /// Get the Hessian 2d array of a DualGH
    let inline hessian (DualGH(_, _, h)) = Matrix.toArray2D h
    /// Get the primal and the first gradient component of a DualGH, as a tuple
    let inline tuple (DualGH(p, g, _)) = (p, g.FirstItem)
    /// Get the primal and the gradient array of a DualGH, as a tuple
    let inline tupleG (DualGH(p, g, _)) = (p, Vector.toArray g)
    /// Get the primal and Hessian 2d array of a DualGH, as a tuple
    let inline tupleH (DualGH(p, _, h)) = (p, Matrix.toArray2D h)
    /// Get the primal, the gradient array, and the Hessian 2d array of a DualGH, as a tuple
    let inline tupleGH (DualGH(p, g, h)) = (p, Vector.toArray g, Matrix.toArray2D h)


/// ForwardGH differentiation operations module (automatically opened)
[<AutoOpen>]
module ForwardGHOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f (x:float) =
        dualGHAct x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        diff' f x |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f (x:float[]) =
        dualGHActArray x |> f |> tupleG

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f (x:float[]) =
        let a = dualGHActArray x |> f
        (Array.map primal a, Matrix.toArray2D (Matrix.Create(a.Length, fun i -> gradient a.[i])))

    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian f x =
        jacobian' f x |> snd

    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' f x =
        let (v, j) = jacobian' f x in (v, transpose j)
    
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT f x =
        jacobianT' f x |> snd

    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' f (x:float[]) =
        dualGHActArray x |> f |> tupleH

    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x =
        hessian' f x |> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f (x:float[]) =
        dualGHActArray x |> f |> tupleGH
    
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian f x =
        gradhessian' f x |> sndtrd

    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' f x =
        let (v, h) = hessian' f x in (v, trace h)

    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian f x =
        laplacian' f x |> snd

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' f x =
        let v, j = jacobian' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurl()
        v, [|j.[2, 1] - j.[1, 2]; j.[0, 2] - j.[2, 0]; j.[1, 0] - j.[0, 1]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl f x =
        curl' f x |> snd

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' f x =
        let v, j = jacobian' f x
        if Array2D.length1 j <> Array2D.length2 j then invalidArgDiv()
        v, trace j

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div f x =
        div' f x |> snd

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' f x =
        let v, j = jacobian' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then invalidArgCurlDiv()
        v, [|j.[2, 1] - j.[1, 2]; j.[0, 2] - j.[2, 0]; j.[1, 0] - j.[0, 1]|], trace j

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv f x =
        curldiv' f x |> sndtrd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' (f:DualGH->DualGH) x = ForwardGHOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:DualGH->DualGH) x = ForwardGHOps.diff f x
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<DualGH>->DualGH) x = ForwardGHOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<DualGH>->DualGH) x = ForwardGHOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<DualGH>->DualGH) x = ForwardGHOps.laplacian' (vector >> f) (Vector.toArray x)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian (f:Vector<DualGH>->DualGH) x = ForwardGHOps.laplacian (vector >> f) (Vector.toArray x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' (f:Vector<DualGH>->DualGH) x = ForwardGHOps.hessian' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, Matrix.ofArray2d b)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian (f:Vector<DualGH>->DualGH) x = ForwardGHOps.hessian (vector >> f) (Vector.toArray x) |> Matrix.ofArray2d
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' (f:Vector<DualGH>->DualGH) x = ForwardGHOps.gradhessian' (vector >> f) (Vector.toArray x) |> fun (a, b, c) -> (a, vector b, Matrix.ofArray2d c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian (f:Vector<DualGH>->DualGH) x = ForwardGHOps.gradhessian (vector >> f) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<DualGH>->Vector<DualGH>) x = ForwardGHOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
