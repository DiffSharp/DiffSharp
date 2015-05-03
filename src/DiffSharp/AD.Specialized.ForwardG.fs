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

/// Numeric type keeping a doublet of primal value and a vector of gradient components
[<CustomEquality; CustomComparison>]
type D =
    | D of float * Vector<float> // Primal, vector of gradient components
    override d.ToString() = let (D(p, g)) = d in sprintf "D (%A, %A)" p g
    static member op_Explicit(D(p, _)):float = p
    static member op_Explicit(D(p, _)):int = int p
    static member DivideByInt(D(p, g), i:int) = D(p / float i, g / float i)
    static member Zero = D(0., Vector.Zero)
    static member One = D(1., Vector.Zero)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> let D(a, _), D(b, _) = d, d2 in compare a b
            | _ -> failwith "Cannot compare this D with another type of object."
    override d.Equals(other) = 
        match other with
        | :? D as d2 -> compare d d2 = 0
        | _ -> false
    override d.GetHashCode() = let (D(a, b)) = d in hash [|a; b|]
    // D - D binary operations
    static member (+) (D(a, ag), D(b, bg)) = D(a + b, ag + bg)
    static member (-) (D(a, ag), D(b, bg)) = D(a - b, ag - bg)
    static member (*) (D(a, ag), D(b, bg)) = D(a * b, ag * b + a * bg)
    static member (/) (D(a, ag), D(b, bg)) = D(a / b, (ag * b - a * bg) / (b * b))
    static member Pow (D(a, ag), D(b, bg)) = let apowb = a ** b in D(apowb, apowb * ((b * ag / a) + ((log a) * bg)))
    static member Atan2 (D(a, ag), D(b, bg)) = D(atan2 a b, (ag * b - a * bg) / (a * a + b * b))
    // D - float binary operations
    static member (+) (D(a, ag), b) = D(a + b, ag)
    static member (-) (D(a, ag), b) = D(a - b, ag)
    static member (*) (D(a, ag), b) = D(a * b, ag * b)
    static member (/) (D(a, ag), b) = D(a / b, ag / b)
    static member Pow (D(a, ag), b) = D(a ** b, b * (a ** (b - 1.)) * ag)
    static member Atan2 (D(a, ag), b) = D(atan2 a b, (b * ag) / (b * b + a * a))
    // float - D binary operations
    static member (+) (a, D(b, bg)) = D(b + a, bg)
    static member (-) (a, D(b, bg)) = D(a - b, -bg)
    static member (*) (a, D(b, bg)) = D(b * a, bg * a)
    static member (/) (a, D(b, bg)) = D(a / b, -a * bg / (b * b))
    static member Pow (a, D(b, bg)) = let apowb = a ** b in D(apowb, apowb * (log a) * bg)
    static member Atan2 (a, D(b, bg)) = D(atan2 a b, -(a * bg) / (a * a + b * b))
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
    static member Log (D(a, ag)) = 
        if a <= 0. then invalidArgLog()
        D(log a, ag / a)
    static member Log10 (D(a, ag)) = 
        if a <= 0. then invalidArgLog10()
        D(log10 a, ag / (a * log10val))
    static member Exp (D(a, ag)) = let expa = exp a in D(expa, ag * expa)
    static member Sin (D(a, ag)) = D(sin a, ag * cos a)
    static member Cos (D(a, ag)) = D(cos a, -ag * sin a)
    static member Tan (D(a, ag)) = 
        let cosa = cos a
        if cosa = 0. then invalidArgTan()
        D(tan a, ag / (cosa * cosa))
    static member (~-) (D(a, ag)) = D(-a, -ag)
    static member Sqrt (D(a, ag)) =
        if a <= 0. then invalidArgSqrt()
        let sqrta = sqrt a in D(sqrta, ag / (2. * sqrta))
    static member Sinh (D(a, ag)) = D(sinh a, ag * cosh a)
    static member Cosh (D(a, ag)) = D(cosh a, ag * sinh a)
    static member Tanh (D(a, ag)) = let cosha = cosh a in D(tanh a, ag / (cosha * cosha))
    static member Asin (D(a, ag)) =
        if (abs a) >= 1. then invalidArgAsin()
        D(asin a, ag / sqrt (1. - a * a))
    static member Acos (D(a, ag)) = 
        if (abs a) >= 1. then invalidArgAcos()
        D(acos a, -ag / sqrt (1. - a * a))
    static member Atan (D(a, ag)) = D(atan a, ag / (1. + a * a))
    static member Abs (D(a, ag)) = 
        if a = 0. then invalidArgAbs()
        D(abs a, ag * float (sign a))
    static member Floor (D(a, ag)) =
        if isInteger a then invalidArgFloor()
        D(floor a, Vector.create ag.Length 0.)
    static member Ceiling (D(a, ag)) =
        if isInteger a then invalidArgCeil()
        D(ceil a, Vector.create ag.Length 0.)
    static member Round (D(a, ag)) =
        if isHalfway a then invalidArgRound()
        D(round a, Vector.create ag.Length 0.)

/// D operations module (automatically opened)
[<AutoOpen>]
module DOps =
    /// Make D, with primal value `p`, gradient dimension `m`, and all gradient components 0
    let inline makeD p m = D(float p, Vector.create m 0.)
    /// Make D, with primal value `p` and gradient array `g`
    let inline makeDPT p g = D(float p, vector g)
    /// Make active D (i.e. variable of differentiation), with primal value `p`, gradient dimension `m`, the component with index `i` having value 1, and the rest of the components 0
    let inline makeDP1 p m i = D(float p, Vector.standardBasis m i)
    /// Make an array of active D, with primal values given in array `x`. For a D with index _i_, the gradient is the unit vector with 1 in the _i_th place.
    let inline makeDP1Array (x:_[]) = Array.init x.Length (fun i -> makeDP1 x.[i] x.Length i)
    /// Get the primal value of a D
    let inline primal (D(p, _)) = p
    /// Get the gradient array of a D
    let inline gradient (D(_, g)) = Vector.toArray g
    /// Get the first gradient component of a D
    let inline tangent (D(_, g)) = g.FirstItem
    /// Get the primal value and the first gradient component of a D, as a tuple
    let inline tuple (D(p, g)) = (p, g.FirstItem)
    /// Get the primal value and the gradient array of a D, as a tuple
    let inline tupleG (D(p, g)) = (p, Vector.toArray g)


/// ForwardG differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// Original value and first derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff' f (x:float) =
        makeDP1 x 1 0 |> f |> tuple

    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    let inline diff f (x:float) =
        makeDP1 x 1 0 |> f |> tangent
       
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
    /// Original value and transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT' (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobianT (f:Vector<D>->Vector<D>) x = DiffOps.jacobianT (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
    /// Original value and Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian' (f:Vector<D>->Vector<D>) x = DiffOps.jacobian' (vector >> f >> Vector.toArray) (Vector.toArray x) |> fun (a, b) -> (vector a, Matrix.ofArray2D b)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    let inline jacobian (f:Vector<D>->Vector<D>) x = DiffOps.jacobian (vector >> f >> Vector.toArray) (Vector.toArray x) |> Matrix.ofArray2D
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
