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
// Reference for "doublets": Dixon L., 2001, Automatic Differentiation: Calculation of the Hessian (http://dx.doi.org/10.1007/0-306-48332-7_17)
//

#light

/// Non-nested forward mode AD, keeping vectors of gradient components
namespace DiffSharp.AD.Specialized.ForwardG

open DiffSharp.Util
open FsAlg.Generic

/// DualG numeric type, keeping a doublet of primal value and a vector of gradient components
[<CustomEquality; CustomComparison>]
type DualG =
    // Primal, vector of gradient components
    | DualG of float * Vector<float>
    override d.ToString() = let (DualG(p, g)) = d in sprintf "DualG (%A, %A)" p g
    static member op_Explicit(p) = DualG(p, Vector.Zero)
    static member op_Explicit(DualG(p, _)) = p
    static member DivideByInt(DualG(p, g), i:int) = DualG(p / float i, g / float i)
    static member Zero = DualG(0., Vector.Zero)
    static member One = DualG(1., Vector.Zero)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? DualG as d2 -> let DualG(a, _), DualG(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this DualG with another type of object."
    override d.Equals(other) = 
        match other with
        | :? DualG as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (DualG(a, b)) = d in hash [|a; b|]
    // DualG - DualG binary operations
    static member (+) (DualG(a, ag), DualG(b, bg)) = DualG(a + b, ag + bg)
    static member (-) (DualG(a, ag), DualG(b, bg)) = DualG(a - b, ag - bg)
    static member (*) (DualG(a, ag), DualG(b, bg)) = DualG(a * b, ag * b + a * bg)
    static member (/) (DualG(a, ag), DualG(b, bg)) = DualG(a / b, (ag * b - a * bg) / (b * b))
    static member Pow (DualG(a, ag), DualG(b, bg)) = let apowb = a ** b in DualG(apowb, apowb * ((b * ag / a) + ((log a) * bg)))
    static member Atan2 (DualG(a, ag), DualG(b, bg)) = DualG(atan2 a b, (ag * b - a * bg) / (a * a + b * b))
    // DualG - float binary operations
    static member (+) (DualG(a, ag), b) = DualG(a + b, ag)
    static member (-) (DualG(a, ag), b) = DualG(a - b, ag)
    static member (*) (DualG(a, ag), b) = DualG(a * b, ag * b)
    static member (/) (DualG(a, ag), b) = DualG(a / b, ag / b)
    static member Pow (DualG(a, ag), b) = DualG(a ** b, b * (a ** (b - 1.)) * ag)
    static member Atan2 (DualG(a, ag), b) = DualG(atan2 a b, (b * ag) / (b * b + a * a))
    // float - DualG binary operations
    static member (+) (a, DualG(b, bg)) = DualG(b + a, bg)
    static member (-) (a, DualG(b, bg)) = DualG(a - b, -bg)
    static member (*) (a, DualG(b, bg)) = DualG(b * a, bg * a)
    static member (/) (a, DualG(b, bg)) = DualG(a / b, -a * bg / (b * b))
    static member Pow (a, DualG(b, bg)) = let apowb = a ** b in DualG(apowb, apowb * (log a) * bg)
    static member Atan2 (a, DualG(b, bg)) = DualG(atan2 a b, -(a * bg) / (a * a + b * b))
    // DualG - int binary operations
    static member (+) (a:DualG, b:int) = a + float b
    static member (-) (a:DualG, b:int) = a - float b
    static member (*) (a:DualG, b:int) = a * float b
    static member (/) (a:DualG, b:int) = a / float b
    static member Pow (a:DualG, b:int) = DualG.Pow(a, float b)
    static member Atan2 (a:DualG, b:int) = DualG.Atan2(a, float b)
    // int - DualG binary operations
    static member (+) (a:int, b:DualG) = (float a) + b
    static member (-) (a:int, b:DualG) = (float a) - b
    static member (*) (a:int, b:DualG) = (float a) * b
    static member (/) (a:int, b:DualG) = (float a) / b
    static member Pow (a:int, b:DualG) = DualG.Pow(float a, b)
    static member Atan2 (a:int, b:DualG) = DualG.Atan2(float a, b)
    // DualG unary operations
    static member Log (DualG(a, ag)) = 
        if a <= 0. then invalidArgLog()
        DualG(log a, ag / a)
    static member Log10 (DualG(a, ag)) = 
        if a <= 0. then invalidArgLog10()
        DualG(log10 a, ag / (a * log10val))
    static member Exp (DualG(a, ag)) = let expa = exp a in DualG(expa, ag * expa)
    static member Sin (DualG(a, ag)) = DualG(sin a, ag * cos a)
    static member Cos (DualG(a, ag)) = DualG(cos a, -ag * sin a)
    static member Tan (DualG(a, ag)) = 
        let cosa = cos a
        if cosa = 0. then invalidArgTan()
        DualG(tan a, ag / (cosa * cosa))
    static member (~-) (DualG(a, ag)) = DualG(-a, -ag)
    static member Sqrt (DualG(a, ag)) =
        if a <= 0. then invalidArgSqrt()
        let sqrta = sqrt a in DualG(sqrta, ag / (2. * sqrta))
    static member Sinh (DualG(a, ag)) = DualG(sinh a, ag * cosh a)
    static member Cosh (DualG(a, ag)) = DualG(cosh a, ag * sinh a)
    static member Tanh (DualG(a, ag)) = let cosha = cosh a in DualG(tanh a, ag / (cosha * cosha))
    static member Asin (DualG(a, ag)) =
        if (abs a) >= 1. then invalidArgAsin()
        DualG(asin a, ag / sqrt (1. - a * a))
    static member Acos (DualG(a, ag)) = 
        if (abs a) >= 1. then invalidArgAcos()
        DualG(acos a, -ag / sqrt (1. - a * a))
    static member Atan (DualG(a, ag)) = DualG(atan a, ag / (1. + a * a))
    static member Abs (DualG(a, ag)) = 
        if a = 0. then invalidArgAbs()
        DualG(abs a, ag * float (sign a))
    static member Floor (DualG(a, ag)) =
        if isInteger a then invalidArgFloor()
        DualG(floor a, Vector.create ag.Length 0.)
    static member Ceiling (DualG(a, ag)) =
        if isInteger a then invalidArgCeil()
        DualG(ceil a, Vector.create ag.Length 0.)
    static member Round (DualG(a, ag)) =
        if isHalfway a then invalidArgRound()
        DualG(round a, Vector.create ag.Length 0.)

/// DualG operations module (automatically opened)
[<AutoOpen>]
module DualGOps =
    /// Make DualG, with primal value `p`, gradient dimension `m`, and all gradient components 0
    let inline dualG p m = DualG(float p, Vector.create m 0.)
    /// Make DualG, with primal value `p` and gradient array `g`
    let inline dualGPT p g = DualG(float p, vector g)
    /// Make active DualG (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the component with index `i` having value 1, and the rest of the components 0
    let inline dualGP1 p m i = DualG(float p, Vector.standardBasis m i)
    /// Make an array of active DualG, with primal values given in array `x`. For a DualG with index _i_, the gradient is the unit vector with 1 in the _i_th place.
    let inline dualGP1Array (x:_[]) = Array.init x.Length (fun i -> dualGP1 x.[i] x.Length i)
    /// Get the primal value of a DualG
    let inline primal (DualG(p, _)) = p
    /// Get the gradient array of a DualG
    let inline gradient (DualG(_, g)) = Vector.toArray g
    /// Get the first gradient component of a DualG
    let inline tangent (DualG(_, g)) = g.FirstItem
    /// Get the primal value and the first gradient component of a DualG, as a tuple
    let inline tuple (DualG(p, g)) = (p, g.FirstItem)
    /// Get the primal value and the gradient array of a DualG, as a tuple
    let inline tupleG (DualG(p, g)) = (p, Vector.toArray g)


/// ForwardG differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f (x:float) =
        dualGP1 x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f (x:float) =
        dualGP1 x 1 0 |> f |> tangent
       
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' f (x:float[]) =
        dualGP1Array x |> f |> tupleG

    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad f x =
        grad' f x |> snd
    
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' f (x:float[]) =
        let a = dualGP1Array x |> f
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
    let inline diff' (f:DualG->DualG) x = DiffOps.diff' f x
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff (f:DualG->DualG) x = DiffOps.diff f x
    /// Original value and gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad' (f:Vector<DualG>->DualG) x = DiffOps.grad' (vector >> f) (Vector.toArray x) |> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    let inline grad (f:Vector<DualG>->DualG) x = DiffOps.grad (vector >> f) (Vector.toArray x) |> vector
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<DualG>->Vector<DualG>) x = DiffOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<DualG>->Vector<DualG>) x = DiffOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<DualG>->Vector<DualG>) x = DiffOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<DualG>->Vector<DualG>) x = DiffOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' (f:Vector<DualG>->Vector<DualG>) x = DiffOps.curl' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, vector b)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl (f:Vector<DualG>->Vector<DualG>) x = DiffOps.curl (vector >> f >> Vector.toArray) (Vector.toArray x) |> vector
    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' (f:Vector<DualG>->Vector<DualG>) x = DiffOps.div' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div (f:Vector<DualG>->Vector<DualG>) x = DiffOps.div (vector >> f >> Vector.toArray) (Vector.toArray x)
    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' (f:Vector<DualG>->Vector<DualG>) x = DiffOps.curldiv' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b, c) -> (vector a, vector b, c)
    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv (f:Vector<DualG>->Vector<DualG>) x = DiffOps.curldiv (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, b)
