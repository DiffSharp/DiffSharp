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

/// Interoperability with .NET languages
namespace DiffSharp.Interop

/// Numeric type keeping dual numbers for forward mode and adjoints and tapes for reverse mode AD, with nesting capability, using tags to avoid perturbation confusion
type D(x:DiffSharp.AD.D) =
    new(x:float) = D(DiffSharp.AD.D(x))
    member internal this.toADD() = x
    static member internal ADDToD (x:DiffSharp.AD.D) = new D(x)
    static member internal DToADD (x:D) = x.toADD()

    /// Primal value of this D
    member d.P = d.toADD().P |> D.ADDToD
    /// Tangent value of this D
    member d.T = d.toADD().T |> D.ADDToD
    /// Adjoint value of this D
    member d.A = d.toADD().A |> D.ADDToD

    override d.ToString() =
        let rec s (d:DiffSharp.AD.D) =
            match d with
            | DiffSharp.AD.D(p) -> sprintf "D %A" p
            | DiffSharp.AD.DF(p,t,_) -> sprintf "DF (%A, %A)" (s p) (s t)
            | DiffSharp.AD.DR(p,a,_,_,_) -> sprintf "DR (%A, %A)" (s p) (s !a)
        s (d.toADD())
    static member op_Implicit(d:D):float = float (d.toADD())
    static member op_Implicit(a:float):D = D(a)
    static member Zero = D(0.)
    static member One = D(1.)
    interface System.IComparable with
        override d.CompareTo(other) =
            match other with
            | :? D as d2 -> compare (d.toADD()) (d2.toADD())
            | _ -> invalidArg "" "Cannot compare this D with another type of object."
    override d.Equals(other) =
        match other with
        | :? D as d2 -> compare (d.toADD()) (d2.toADD()) = 0
        | _ -> false
    override d.GetHashCode() = d.toADD().GetHashCode()
    // D - D binary operations
    static member (+) (a:D, b:D) = D(a.toADD() + b.toADD())
    static member (-) (a:D, b:D) = D(a.toADD() - b.toADD())
    static member (*) (a:D, b:D) = D(a.toADD() * b.toADD())
    static member (/) (a:D, b:D) = D(a.toADD() / b.toADD())
    static member Pow (a:D, b:D) = D(a.toADD() ** b.toADD())
    static member Atan2 (a:D, b:D) = D(atan2 (a.toADD()) (b.toADD()))
    // D - float binary operations
    static member (+) (a:D, b:float) = a + (D b)
    static member (-) (a:D, b:float) = a - (D b)
    static member (*) (a:D, b:float) = a * (D b)
    static member (/) (a:D, b:float) = a / (D b)
    static member Pow (a:D, b:float) = a ** (D b)
    static member Atan2 (a:D, b:float) = atan2 a (D b)
    // float - D binary operations
    static member (+) (a:float, b:D) = (D a) + b
    static member (-) (a:float, b:D) = (D a) - b
    static member (*) (a:float, b:D) = (D a) * b
    static member (/) (a:float, b:D) = (D a) / b
    static member Pow (a:float, b:D) = (D a) ** b
    static member Atan2 (a:float, b:D) = atan2 (D a) b
    // D - int binary operations
    static member (+) (a:D, b:int) = a + (D (float b))
    static member (-) (a:D, b:int) = a - (D (float b))
    static member (*) (a:D, b:int) = a * (D (float b))
    static member (/) (a:D, b:int) = a / (D (float b))
    static member Pow (a:D, b:int) = D.Pow(a, (D (float b)))
    static member Atan2 (a:D, b:int) = D.Atan2(a, (D (float b)))
    // int - D binary operations
    static member (+) (a:int, b:D) = (D (float a)) + b
    static member (-) (a:int, b:D) = (D (float a)) - b
    static member (*) (a:int, b:D) = (D (float a)) * b
    static member (/) (a:int, b:D) = (D (float a)) / b
    static member Pow (a:int, b:D) = D.Pow((D (float a)), b)
    static member Atan2 (a:int, b:D) = D.Atan2((D (float a)), b)
    // D unary operations
    static member Log (a:D) = D(log (a.toADD()))
    static member Log10 (a:D) = D(log10 (a.toADD()))
    static member Exp (a:D) = D(exp (a.toADD()))
    static member Sin (a:D) = D(sin (a.toADD()))
    static member Cos (a:D) = D(cos (a.toADD()))
    static member Tan (a:D) = D(tan (a.toADD()))
    static member Neg (a:D) = D(-(a.toADD()))
    static member Sqrt (a:D) = D(sqrt (a.toADD()))
    static member Sinh (a:D) = D(sinh (a.toADD()))
    static member Cosh (a:D) = D(cosh (a.toADD()))
    static member Tanh (a:D) = D(tanh (a.toADD()))
    static member Asin (a:D) = D(asin (a.toADD()))
    static member Acos (a:D) = D(acos (a.toADD()))
    static member Atan (a:D) = D(atan (a.toADD()))
    static member Abs (a:D) = D(abs (a.toADD()))
    static member Floor (a:D) = D(floor (a.toADD()))
    static member Ceiling (a:D) = D(ceil (a.toADD()))
    static member Round (a:D) = D(round (a.toADD()))

/// Nested forward and reverse mode automatic differentiation module
type AD =
    /// First derivative of a scalar-to-scalar function `f`
    static member Diff(f:System.Func<D,D>) = System.Func<D,D>(D.DToADD >> (DiffSharp.AD.DiffOps.diff (D.ADDToD >> f.Invoke >> D.DToADD)) >> D.ADDToD)
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff(f:System.Func<D,D>, x:D) = D.ADDToD <| DiffSharp.AD.DiffOps.diff (D.ADDToD >> f.Invoke >> D.DToADD) (x |> D.DToADD)
    /// Second derivative of a scalar-to-scalar function `f`
    static member Diff2(f:System.Func<D,D>) = System.Func<D,D>(D.DToADD >> (DiffSharp.AD.DiffOps.diff2 (D.ADDToD >> f.Invoke >> D.DToADD)) >> D.ADDToD)
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff2(f:System.Func<D,D>, x:D) = D.ADDToD <| DiffSharp.AD.DiffOps.diff2 (D.ADDToD >> f.Invoke >> D.DToADD) (x |> D.DToADD)
    /// `n`-th derivative of a scalar-to-scalar function `f`
    static member Diffn(n:int, f:System.Func<D,D>) = System.Func<D,D>(D.DToADD >> (DiffSharp.AD.DiffOps.diffn n (D.ADDToD >> f.Invoke >> D.DToADD)) >> D.ADDToD)
    /// `n`-th derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diffn(n:int, f:System.Func<D,D>, x:D) = D.ADDToD <| DiffSharp.AD.DiffOps.diffn n (D.ADDToD >> f.Invoke >> D.DToADD) (x |> D.DToADD)
    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    static member Gradv(f:System.Func<D[],D>, x:D[], v:D[]) = D.ADDToD <| DiffSharp.AD.DiffOps.gradv ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD) (x |> (Array.map D.DToADD)) (v |> (Array.map D.DToADD))
    /// Gradient of a vector-to-scalar function `f`
    static member Grad(f:System.Func<D[],D>) = System.Func<D[],D[]>((Array.map D.DToADD) >> (DiffSharp.AD.DiffOps.grad ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD)) >> (Array.map D.ADDToD))
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    static member Grad(f:System.Func<D[],D>, x:D[]) = (Array.map D.ADDToD) <| DiffSharp.AD.DiffOps.grad ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD) (x |> (Array.map D.DToADD))
    /// Laplacian of a vector-to-scalar function `f`
    static member Laplacian(f:System.Func<D[],D>) = System.Func<D[],D>((Array.map D.DToADD) >> (DiffSharp.AD.DiffOps.laplacian ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD)) >> D.ADDToD)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    static member Laplacian(f:System.Func<D[],D>, x:D[]) = D.ADDToD <| DiffSharp.AD.DiffOps.laplacian ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD) (x |> (Array.map D.DToADD))
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    static member Jacobianv(f:System.Func<D[],D[]>, x:D[], v:D[]) = (Array.map D.ADDToD) <| DiffSharp.AD.DiffOps.jacobianv ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD)) (x |> (Array.map D.DToADD)) (v |> (Array.map D.DToADD))
    /// Transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    static member JacobianTv(f:System.Func<D[],D[]>, x:D[], v:D[]) = (Array.map D.ADDToD) <| DiffSharp.AD.DiffOps.jacobianTv ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD)) (x |> (Array.map D.DToADD)) (v |> (Array.map D.DToADD))
    /// Jacobian of a vector-to-vector function `f`
    static member Jacobian(f:System.Func<D[],D[]>) = System.Func<D[],D[,]>((Array.map D.DToADD) >> (DiffSharp.AD.DiffOps.jacobian ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD))) >> (Array2D.map D.ADDToD))
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    static member Jacobian(f:System.Func<D[],D[]>, x:D[]) = (Array2D.map D.ADDToD) <| DiffSharp.AD.DiffOps.jacobian ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD)) (x |> (Array.map D.DToADD))
    /// Transposed Jacobian of a vector-to-vector function `f`
    static member JacobianT(f:System.Func<D[],D[]>) = System.Func<D[],D[,]>((Array.map D.DToADD) >> (DiffSharp.AD.DiffOps.jacobianT ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD))) >> (Array2D.map D.ADDToD))
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    static member JacobianT(f:System.Func<D[],D[]>, x:D[]) = (Array2D.map D.ADDToD) <| DiffSharp.AD.DiffOps.jacobianT ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD)) (x |> (Array.map D.DToADD))
    /// Hessian of a vector-to-scalar function `f`
    static member Hessian(f:System.Func<D[],D>) = System.Func<D[],D[,]>((Array.map D.DToADD) >> (DiffSharp.AD.DiffOps.hessian ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD)) >> (Array2D.map D.ADDToD))
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    static member Hessian(f:System.Func<D[],D>, x:D[]) = (Array2D.map D.ADDToD) <| DiffSharp.AD.DiffOps.hessian ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD) (x |> (Array.map D.DToADD))
    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`
    static member Hessianv(f:System.Func<D[],D>, x:D[], v:D[]) = (Array.map D.ADDToD) <| DiffSharp.AD.DiffOps.hessianv ((Array.map D.ADDToD) >> f.Invoke >> D.DToADD) (x |> (Array.map D.DToADD)) (v |> (Array.map D.DToADD))
    /// Curl of a vector-to-vector function `f`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<D[],D[]>) = System.Func<D[],D[]>((Array.map D.DToADD) >> (DiffSharp.AD.DiffOps.curl ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD))) >> (Array.map D.ADDToD))
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<D[],D[]>, x:D[]) = (Array.map D.ADDToD) <| DiffSharp.AD.DiffOps.curl ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD)) (x |> (Array.map D.DToADD))
    /// Divergence of a vector-to-vector function `f`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<D[],D[]>) = System.Func<D[],D>((Array.map D.DToADD) >> (DiffSharp.AD.DiffOps.div ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD))) >> D.ADDToD)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<D[],D[]>, x:D[]) = D.ADDToD <| DiffSharp.AD.DiffOps.div ((Array.map D.ADDToD) >> f.Invoke >> (Array.map D.DToADD)) (x |> (Array.map D.DToADD))
    /// Returns a specified number raised to the specified power.
    static member Pow(a:D, b:D) = a ** b
    /// Returns the angle whose tangent is the quotient of two specified numbers.
    static member Atan2(a:D, b:D) = atan2 a b
    /// Returns the logarithm of a specified number.
    static member Log(a:D) = log a
    /// Returns the base 10 logarithm of a specified number.
    static member Log10(a:D) = log10 a
    /// Returns e raised to the specified power.
    static member Exp(a:D) = exp a
    /// Returns the sine of the specified angle.
    static member Sin(a:D) = sin a
    /// Returns the cosine of the specified angle.
    static member Cos(a:D) = cos a
    /// Returns the tangent of the specified angle.
    static member Tan(a:D) = tan a
    /// Returns the square root of a specified number.
    static member Sqrt(a:D) = sqrt a
    /// Returns the hyperbolic sine of the specified angle.
    static member Sinh(a:D) = sinh a
    /// Returns the hyperbolic cosine of the specified angle.
    static member Cosh(a:D) = cosh a
    /// Returns the hyperbolic tangent of the specified angle.
    static member Tanh(a:D) = tanh a
    /// Returns the angle whose sine is the specified number.
    static member Asin(a:D) = asin a
    /// Returns the angle whose cosine is the specified number.
    static member Acos(a:D) = acos a
    /// Returns the angle whose tangent is the specified number.
    static member Atan(a:D) = atan a
    /// Returns the absolute value of a specified number.
    static member Abs(a:D) = abs a
    /// Returns the largest integer less than or equal to the specified number.
    static member Floor(a:D) = floor a
    /// Returns the smallest integer greater than or equal to the specified number.
    static member Ceiling(a:D) = ceil a
    /// Rounds a value to the nearest integer or to the specified number of fractional digits.
    static member Round(a:D) = round a
    /// Returns the larger of two specified numbers.
    static member Max(a:D, b:D) = max a b
    /// Returns the smaller of two numbers.
    static member Min(a:D, b:D) = min a b

/// Numerical differentiation module
type Numerical =
    /// First derivative of a scalar-to-scalar function `f`
    static member Diff(f:System.Func<float,float>) = System.Func<float, float>(DiffSharp.Numerical.DiffOps.diff f.Invoke)
    /// First derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff(f:System.Func<float,float>, x:float) = DiffSharp.Numerical.DiffOps.diff f.Invoke x
    /// Second derivative of a scalar-to-scalar function `f`
    static member Diff2(f:System.Func<float,float>) = System.Func<float, float>(DiffSharp.Numerical.DiffOps.diff2 f.Invoke)
    /// Second derivative of a scalar-to-scalar function `f`, at point `x`
    static member Diff2(f:System.Func<float,float>, x:float) = DiffSharp.Numerical.DiffOps.diff2 f.Invoke x
    /// Gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at point `x`, along vector `v`
    static member Gradv(f:System.Func<float[],float>, x:float[], v:float[]) = DiffSharp.Numerical.DiffOps.gradv f.Invoke x v
    /// Gradient of a vector-to-scalar function `f`
    static member Grad(f:System.Func<float[],float>) = System.Func<float[],float[]>(DiffSharp.Numerical.DiffOps.grad f.Invoke)
    /// Gradient of a vector-to-scalar function `f`, at point `x`
    static member Grad(f:System.Func<float[],float>, x:float[]) = DiffSharp.Numerical.DiffOps.grad f.Invoke x
    /// Hessian-vector product of a vector-to-scalar function `f`, at point `x`, along vector `v`
    static member Hessianv(f:System.Func<float[],float>, x:float[], v:float[]) = DiffSharp.Numerical.DiffOps.hessianv f.Invoke x v
    /// Hessian of a vector-to-scalar function `f`
    static member Hessian(f:System.Func<float[],float>) = System.Func<float[],float[,]>(DiffSharp.Numerical.DiffOps.hessian f.Invoke)
    /// Hessian of a vector-to-scalar function `f`, at point `x`
    static member Hessian(f:System.Func<float[],float>, x:float[]) = DiffSharp.Numerical.DiffOps.hessian f.Invoke x
    /// Laplacian of a vector-to-scalar function `f`
    static member Laplacian(f:System.Func<float[],float>) = System.Func<float[],float>(DiffSharp.Numerical.DiffOps.laplacian f.Invoke)
    /// Laplacian of a vector-to-scalar function `f`, at point `x`
    static member Laplacian(f:System.Func<float[],float>, x:float[]) = DiffSharp.Numerical.DiffOps.laplacian f.Invoke x
    /// Jacobian-vector product of a vector-to-vector function `f`, at point `x`, along vector `v`
    static member Jacobianv(f:System.Func<float[],float[]>, x:float[], v:float[]) = DiffSharp.Numerical.DiffOps.jacobianv f.Invoke x v
    /// Jacobian of a vector-to-vector function `f`
    static member Jacobian(f:System.Func<float[],float[]>) = System.Func<float[],float[,]>(DiffSharp.Numerical.DiffOps.jacobian f.Invoke)
    /// Jacobian of a vector-to-vector function `f`, at point `x`
    static member Jacobian(f:System.Func<float[],float[]>, x:float[]) = DiffSharp.Numerical.DiffOps.jacobian f.Invoke x
    /// Transposed Jacobian of a vector-to-vector function `f`
    static member JacobianT(f:System.Func<float[],float[]>) = System.Func<float[],float[,]>(DiffSharp.Numerical.DiffOps.jacobianT f.Invoke)
    /// Transposed Jacobian of a vector-to-vector function `f`, at point `x`
    static member JacobianT(f:System.Func<float[],float[]>, x:float[]) = DiffSharp.Numerical.DiffOps.jacobianT f.Invoke x
    /// Curl of a vector-to-vector function `f`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<float[],float[]>) = System.Func<float[],float[]>(DiffSharp.Numerical.DiffOps.curl f.Invoke)
    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    static member Curl(f:System.Func<float[],float[]>, x:float[]) = DiffSharp.Numerical.DiffOps.curl f.Invoke x
    /// Divergence of a vector-to-vector function `f`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<float[],float[]>) = System.Func<float[],float>(DiffSharp.Numerical.DiffOps.div f.Invoke)
    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    static member Div(f:System.Func<float[],float[]>, x:float[]) = DiffSharp.Numerical.DiffOps.div f.Invoke x

