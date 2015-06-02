//
// This file is part of
// DiffSharp: Automatic Differentiation Library
//
// Copyright (c) 2014--2015, National University of Ireland Maynooth (Atilim Gunes Baydin, Barak A. Pearlmutter)
// 
// Released under LGPL license.
//
//   DiffSharp is free software: you can redistribute it and/or modify
//   it under the terms of the GNU Lesser General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   DiffSharp is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public License
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
//   Brain and Computation Lab
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

/// Non-nested forward mode AD, keeping vectors of gradient components and matrices of Hessian components
namespace DiffSharp.AD.Specialized.ForwardGH

open DiffSharp.Util
open FsAlg.Generic

/// Numeric type keeping a triplet of primal value, a vector of gradient components, and a matrix of Hessian components
[<CustomEquality; CustomComparison>]
type D =
    | D of float * Vector<float> * Matrix<float> // Primal, vector of gradient components, matrix of Hessian components
    override d.ToString() = let (D(p, g, h)) = d in sprintf "D (%A, %A, %A)" p g h
    static member op_Explicit(D(p, _, _)):float = p
    static member op_Explicit(D(p, _, _)):int = int p
    static member DivideByInt(D(p, g, m), i:int) = D(p / float i, g / float i, m / float i)
    static member Zero = D(0., Vector.Zero, Matrix.Zero)
    static member One = D(1., Vector.Zero, Matrix.Zero)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> let D(a, _, _), D(b, _, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this D with another type of object."
    override d.Equals(other) = 
        match other with
        | :? D as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (D(a, b, c)) = d in hash [|a; b; c|]
    // D - D binary operations
    static member (+) (D(a, ag, ah), D(b, bg, bh)) = D(a + b, ag + bg, Matrix.initSymmetric (max ag.Length bg.Length) (fun i j -> ah.[i, j] + bh.[i, j]))
    static member (-) (D(a, ag, ah), D(b, bg, bh)) = D(a - b, ag - bg, Matrix.initSymmetric (max ag.Length bg.Length) (fun i j -> ah.[i, j] - bh.[i, j]))
    static member (*) (D(a, ag, ah), D(b, bg, bh)) = D(a * b, ag * b + a * bg, Matrix.initSymmetric (max ag.Length bg.Length) (fun i j -> ag.[j] * bg.[i] + a * bh.[i, j] + bg.[j] * ag.[i] + b * ah.[i, j]))
    static member (/) (D(a, ag, ah), D(b, bg, bh)) = let bsq, atimes2, oneoverbcube = b * b, a * 2., 1. / (b * b * b) in D(a / b, (ag * b - a * bg) / bsq, Matrix.initSymmetric (max ag.Length bg.Length) (fun i j -> (atimes2 * bg.[j] * bg.[i] + bsq * ah.[i, j] - b * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + a * bh.[i, j])) * oneoverbcube))
    static member Pow (D(a, ag, ah), D(b, bg, bh)) = let apowb, loga, apowbminus2, bsq = a ** b, log a, a ** (b - 2.), b * b in D(apowb, apowb * ((b * ag / a) + (loga * bg)), Matrix.initSymmetric (max ag.Length bg.Length) (fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (ag.[j] * (-ag.[i] + loga * a * bg.[i]) + a * (loga * bg.[j] * ag.[i] + ah.[i, j])) + a * (ag.[j] * bg.[i] + bg.[j] * (ag.[i] + loga * loga * a * bg.[i]) + loga * a * bh.[i, j]))))
    static member Atan2 (D(a, ag, ah), D(b, bg, bh)) = let asq, bsq = a * a, b * b in D(atan2 a b, (ag * b - a * bg) / (asq + bsq), Matrix.initSymmetric (max ag.Length bg.Length) (fun i j -> (bsq * (-bg.[j] * ag.[i] - ag.[j] * bg.[i] + b * ah.[i, j]) + asq * (bg.[j] * ag.[i] + ag.[j] * bg.[i] + b * ah.[i,j]) - (asq * a) * bh.[i, j] - a * b * (2. * ag.[j] * ag.[i] - 2. * bg.[j] * bg.[i] + b * bh.[i, j])) / (asq + bsq) ** 2.))
    // D - float binary operations
    static member (+) (D(a, ag, ah), b) = D(a + b, ag, ah)
    static member (-) (D(a, ag, ah), b) = D(a - b, ag, ah)
    static member (*) (D(a, ag, ah), b) = D(a * b, ag * b, Matrix.initSymmetric ag.Length (fun i j -> ah.[i, j] * b))
    static member (/) (D(a, ag, ah), b) = D(a / b, ag / b, Matrix.initSymmetric ag.Length (fun i j -> ah.[i, j] / b))
    static member Pow (D(a, ag, ah), b) = let apowb, bsq, apowbminus2 = a ** b, b * b, a ** (b - 2.) in D(apowb, b * (a ** (b - 1.)) * ag, Matrix.initSymmetric ag.Length (fun i j -> apowbminus2 * (bsq * ag.[j] * ag.[i] + b * (a * ah.[i, j] - ag.[j] * ag.[i]))))
    static member Atan2 (D(a, ag, ah), b) = let asq, bsq = a * a, b * b in D(atan2 a b, (b * ag) / (bsq + asq), Matrix.initSymmetric ag.Length (fun i j -> (b * (-2. * a * ag.[j] * ag.[i] + bsq * ah.[i,j] + asq * ah.[i, j])) / (bsq + asq) ** 2.))
    // float - D binary operations
    static member (+) (a, D(b, bg, bh)) = D(a + b, bg, bh)
    static member (-) (a, D(b, bg, bh)) = D(a - b, -bg, -bh)
    static member (*) (a, D(b, bg, bh)) = D(a * b, a * bg, Matrix.initSymmetric bg.Length (fun i j -> a * bh.[i, j]))
    static member (/) (a, D(b, bg, bh)) = let aoverbcube = a / (b * b * b) in D(a / b, -aoverbcube * b * bg, Matrix.initSymmetric bg.Length (fun i j -> (2. * bg.[j] * bg.[i] - b * bh.[i, j]) * aoverbcube))
    static member Pow (a, D(b, bg, bh)) = let apowb, loga, term = a ** b, log a, (a ** (b - 2.)) * a * log a in D(apowb, apowb * loga * bg, Matrix.initSymmetric bg.Length (fun i j -> term * (bg.[j] * loga * a * bg.[i] + a * bh.[i, j])))
    static member Atan2 (a, D(b, bg, bh)) = let asq, bsq = a * a, b * b in D(atan2 a b, -(a * bg) / (asq + bsq), Matrix.initSymmetric bg.Length (fun i j -> -((a *(-2. * b * bg.[j] * bg.[i] + asq * bh.[i, j] + bsq * bh.[i, j])) / (asq + bsq) ** 2.)))
    // D - int binary operations
    static member (+) (a:D, b:int) = a + float b
    static member (-) (a:D, b:int) = a - float b
    static member (*) (a:D, b:int) = a * float b
    static member (/) (a:D, b:int) = a / float b
    static member Pow (a:D, b:int) = D.Pow(a, float b)
    static member Atan2 (a:D, b:int) = D.Atan2(a, float b)
    // int - D binary operations
    static member (+) (a:int, b:D) = (float a) + b
    static member (-) (a:int, b:D) = (float a) - b
    static member (*) (a:int, b:D) = (float a) * b
    static member (/) (a:int, b:D) = (float a) / b
    static member Pow (a:int, b:D) = D.Pow(float a, b)
    static member Atan2 (a:int, b:D) = D.Atan2(float a, b)
    // D unary operations
    static member Log (D(a, ag, ah)) = 
        if a <= 0. then invalidArgLog()
        let asq = a * a in D(log a, ag / a, Matrix.initSymmetric ag.Length (fun i j -> -ag.[i] * ag.[j] / asq + ah.[i, j] / a))
    static member Log10 (D(a, ag, ah)) = 
        if a <= 0. then invalidArgLog10()
        let alog10 = a * log10val in D(log10 a, ag / alog10, Matrix.initSymmetric ag.Length (fun i j -> -ag.[i] * ag.[j] / (a * alog10) + ah.[i, j] / alog10))
    static member Exp (D(a, ag, ah)) = let expa = exp a in D(expa, expa * ag, Matrix.initSymmetric ag.Length (fun i j -> expa * ag.[i] * ag.[j] + expa * ah.[i, j]))
    static member Sin (D(a, ag, ah)) = let sina, cosa = sin a, cos a in D(sina, cosa * ag, Matrix.initSymmetric ag.Length (fun i j -> -sina * ag.[i] * ag.[j] + cosa * ah.[i, j]))
    static member Cos (D(a, ag, ah)) = let sina, cosa = sin a, cos a in D(cosa, -sina * ag, Matrix.initSymmetric ag.Length (fun i j -> -cosa * ag.[i] * ag.[j] - sina * ah.[i, j]))
    static member Tan (D(a, ag, ah)) = 
        let cosa = cos a
        if cosa = 0. then invalidArgTan()
        let tana, secsqa = tan a, 1. / ((cosa) * (cosa)) in D(tana, secsqa * ag, Matrix.initSymmetric ag.Length (fun i j -> 2. * secsqa * tana * ag.[i] * ag.[j] + secsqa * ah.[i, j]))
    static member (~-) (D(a, ag, ah)) = D(-a, -ag, -ah)
    static member Sqrt (D(a, ag, ah)) = 
        if a <= 0. then invalidArgSqrt()
        let term = 1. / (2. * sqrt a) in D(sqrt a, term * ag, Matrix.initSymmetric ag.Length (fun i j -> (term / (-2. * a)) * ag.[i] * ag.[j] + term * ah.[i,j]))
    static member Sinh (D(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in D(sinha, cosha * ag, Matrix.initSymmetric ag.Length (fun i j -> sinha * ag.[i] * ag.[j] + cosha * ah.[i, j]))
    static member Cosh (D(a, ag, ah)) = let sinha, cosha = sinh a, cosh a in D(cosha, sinha * ag, Matrix.initSymmetric ag.Length (fun i j -> cosha * ag.[i] * ag.[j] + sinha * ah.[i, j]))
    static member Tanh (D(a, ag, ah)) = let tanha, sechsqa = tanh a, 1. / ((cosh a) * (cosh a)) in D(tanha, sechsqa * ag, Matrix.initSymmetric ag.Length (fun i j -> -2. * sechsqa * tanha * ag.[i] * ag.[j] + sechsqa * ah.[i, j]))
    static member Asin (D(a, ag, ah)) = 
        if (abs a) >= 1. then invalidArgAsin()
        let term, term2 = 1. / sqrt (1. - a * a), (a / (1. - a * a)) in D(asin a, term * ag, Matrix.initSymmetric ag.Length (fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Acos (D(a, ag, ah)) = 
        if (abs a) >= 1. then invalidArgAcos()
        let term, term2 = -1. / sqrt (1. - a * a), (a / (1. - a * a)) in D(acos a, term * ag, Matrix.initSymmetric ag.Length (fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Atan (D(a, ag, ah)) = let term, term2 = 1. / (1. + a * a), (-2. * a / (1. + a * a)) in D(atan a, term * ag, Matrix.initSymmetric ag.Length (fun i j -> term2 * term * ag.[i] * ag.[j] + term * ah.[i, j]))
    static member Abs (D(a, ag, ah)) = 
        if a = 0. then invalidArgAbs()
        D(abs a, ag * float (sign a), Matrix.initSymmetric ag.Length (fun i j -> ah.[i, j] * float (sign a)))
    static member Floor (D(a, ag, ah)) =
        if isInteger a then invalidArgFloor()
        D(floor a, Vector.create ag.Length 0., Matrix.create ag.Length ag.Length 0.)
    static member Ceiling (D(a, ag, ah)) =
        if isInteger a then invalidArgCeil()
        D(ceil a, Vector.create ag.Length 0., Matrix.create ag.Length ag.Length 0.)
    static member Round (D(a, ag, ah)) =
        if isHalfway a then invalidArgRound()
        D(round a, Vector.create ag.Length 0., Matrix.create ag.Length ag.Length 0.)

/// D operations module (automatically opened)
[<AutoOpen>]
module DOps =
    /// Make D, with primal value `p`, gradient dimension `m`, and all gradient and Hessian components 0
    let inline makeD p m = D(float p, Vector.create m 0., Matrix.create m m 0.)
    /// Make D, with primal value `p`, gradient array `g`, and Hessian 2d array `h`
    let inline makeDPT (p, g, h:float[,]) = D(float p, vector g, Matrix.ofArray2D h)
    /// Make active D (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the gradient component with index `i` having value 1, the rest of the gradient components 0, and Hessian components 0
    let inline makeDP1 p m i = D(float p, Vector.standardBasis m i, Matrix.create m m 0.)
    /// Make an array of active D, with primal values given in array `x`. For a D with index _i_, the gradient is the unit vector with 1 in the _i_th place, and the Hessian components are 0.
    let inline makeDP1Array (x:_[]) = Array.init x.Length (fun i -> makeDP1 x.[i] x.Length i)
    /// Get the primal value of a D
    let inline primal (D(p, _, _)) = p
    /// Get the gradient array of a D
    let inline gradient (D(_, g, _)) = Vector.toArray g
    /// Get the Hessian 2d array of a D
    let inline hessian (D(_, _, h)) = Matrix.toArray2D h
    /// Get the primal and the first gradient component of a D, as a tuple
    let inline tuple (D(p, g, _)) = (p, g.FirstItem)
    /// Get the primal and the gradient array of a D, as a tuple
    let inline tupleG (D(p, g, _)) = (p, Vector.toArray g)
    /// Get the primal and Hessian 2d array of a D, as a tuple
    let inline tupleH (D(p, _, h)) = (p, Matrix.toArray2D h)
    /// Get the primal, the gradient array, and the Hessian 2d array of a D, as a tuple
    let inline tupleGH (D(p, g, h)) = (p, Vector.toArray g, Matrix.toArray2D h)


/// ForwardGH differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f (x:float) =
        makeDP1 x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f x =
        diff' f x |> snd

    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f (x:float[]) =
        makeDP1Array x |> f |> tupleG

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f (x:float[]) =
        let a = makeDP1Array x |> f
        (Array.map primal a, array2D (Array.init a.Length (fun i -> gradient a.[i])))

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
        makeDP1Array x |> f |> tupleH

    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian f x =
        hessian' f x |> snd

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' f (x:float[]) =
        makeDP1Array x |> f |> tupleGH
    
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
    let inline diff' (f:D->D) x = DiffOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:D->D) x = DiffOps.diff f x
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<D>->D) x = DiffOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<D>->D) x = DiffOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian' (f:Vector<D>->D) x = DiffOps.laplacian' (vector >> f) (Vector.toArray x)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    let inline laplacian (f:Vector<D>->D) x = DiffOps.laplacian (vector >> f) (Vector.toArray x)
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<D>->Vector<D>) x = DiffOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<D>->Vector<D>) x = DiffOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian' (f:Vector<D>->D) x = DiffOps.hessian' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, Matrix.ofArray2D b)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    let inline hessian (f:Vector<D>->D) x = DiffOps.hessian (vector >> f) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian' (f:Vector<D>->D) x = DiffOps.gradhessian' (vector >> f) (Vector.toArray x) |> fun (a, b, c) -> (a, vector b, Matrix.ofArray2D c)
    /// Gradient and Hessian of a vector-to-scalar function `f`, at point `x`
    let inline gradhessian (f:Vector<D>->D) x = DiffOps.gradhessian (vector >> f) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<D>->Vector<D>) x = DiffOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<D>->Vector<D>) x = DiffOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<D>->Vector<D>) x = DiffOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<D>->Vector<D>) x = DiffOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<D>->Vector<D>) x = DiffOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<D>->Vector<D>) x = DiffOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
