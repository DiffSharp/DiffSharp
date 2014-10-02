//
// This file is part of
// DiffSharp -- F# Automatic Differentiation Library
//
// Copyright (C) 2014, National University of Ireland Maynooth.
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
// Symbolic differentiation
//
// - Currently limited to closed-form algebraic functions, i.e. no control flow
// - Can drill into method bodies of other functions called from the current one, provided these have the [<ReflectedDefinition>] attribute set
// - Can compute higher order derivatives and all combinations of partial derivatives
//

#light

/// Symbolic differentiation module
module DiffSharp.Symbolic

open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General
open DiffSharp.Util.Quotations
open Swensen.Unquote

/// Symbolic differentiation expression operations module (automatically opened)
[<AutoOpen>]
module ExprOps =
    // We need to get MethodInfo information for operators on float type
    // Because we should be able to recursively compute higher order derivatives, this is better than
    // using static methods on a custom type implementing math operators, as used in other examples elsewhere
    let floatAdd = methInf <@@ 1. + 1. @@>  // MethodInfo for Double op_Addition
    let floatSub = methInf <@@ 1. - 1. @@>  // MethodInfo for Double op_Subtraction
    let floatMul = methInf <@@ 1. * 1. @@>  // MethodInfo for Double op_Multiply
    let floatDiv = methInf <@@ 1. / 1. @@>  // MethodInfo for Double op_Division
    let floatNeg = methInf <@@ - (1.) @@>   // MethodInfo for Double op_UnaryNegation
    let floatPow = methInf <@@ 1. ** 1. @@> // MethodInfo for Double op_Exponentiation
    let floatLog = methInf <@@ log 1. @@>   // MethodInfo for Double log
    let floatExp = methInf <@@ exp 1. @@>   // MethodInfo for Double exp
    let floatSin = methInf <@@ sin 1. @@>   // MethodInfo for Double sin
    let floatCos = methInf <@@ cos 1. @@>   // MethodInfo for Double cos
    let floatTan = methInf <@@ tan 1. @@>   // MethodInfo for Double tan
    let floatSqrt = methInf <@@ sqrt 1. @@> // MethodInfo for Double sqrt
    let floatSinh = methInf <@@ sinh 1. @@> // MethodInfo for Double sinh
    let floatCosh = methInf <@@ cosh 1. @@> // MethodInfo for Double cosh
    let floatTanh = methInf <@@ tanh 1. @@> // MethodInfo for Double tanh
    let floatAsin = methInf <@@ asin 1. @@> // MethodInfo for Double asin
    let floatAcos = methInf <@@ acos 1. @@> // MethodInfo for Double acos
    let floatAtan = methInf <@@ atan 1. @@> // MethodInfo for Double atan

    /// Recursively traverse and differentiate Expr `expr` with respect to Var `v`
    // UNOPTIMIZED
    let rec diffExpr v expr =
        match expr with
        | Value(_) -> Expr.Value(0.)
        | SpecificCall <@ (+) @> (_, _, [f; g]) -> Expr.Call(floatAdd, [diffExpr v f; diffExpr v g])
        | SpecificCall <@ (-) @> (_, _, [f; g]) -> Expr.Call(floatSub, [diffExpr v f; diffExpr v g])
        | SpecificCall <@ (*) @> (_, _, [f; g]) -> Expr.Call(floatAdd, [Expr.Call(floatMul, [diffExpr v f; g]); Expr.Call(floatMul, [f; diffExpr v g])])
        | SpecificCall <@ (/) @> (_, _, [f; g]) -> Expr.Call(floatDiv, [Expr.Call(floatSub, [Expr.Call(floatMul, [diffExpr v f; g]); Expr.Call(floatMul, [f; diffExpr v g])]); Expr.Call(floatMul, [g; g])])
        | SpecificCall <@ (~-) @> (_, _, [f]) -> Expr.Call(floatNeg, [diffExpr v f])
        | SpecificCall <@ op_Exponentiation @> (_, _, [f; g]) -> Expr.Call(floatMul, [Expr.Call(floatPow, [f; Expr.Call(floatSub, [g; Expr.Value(1.)])]); Expr.Call(floatAdd, [Expr.Call(floatMul, [g; diffExpr v f]); Expr.Call(floatMul, [Expr.Call(floatMul, [f; Expr.Call(floatLog, [f])]); diffExpr v g])])]) // This should cover all cases: (f(x) ^ (g(x) - 1))(g(x) * f'(x) + f(x) * log(f(x)) * g'(x))
        | SpecificCall <@ log @> (_, _, [f]) -> Expr.Call(floatDiv, [diffExpr v f; f])
        | SpecificCall <@ exp @> (_, _, [f]) -> Expr.Call(floatMul, [diffExpr v f; Expr.Call(floatExp, [f])])
        | SpecificCall <@ sin @> (_, _, [f]) -> Expr.Call(floatMul, [diffExpr v f; Expr.Call(floatCos, [f])])
        | SpecificCall <@ cos @> (_, _, [f]) -> Expr.Call(floatMul, [diffExpr v f; Expr.Call(floatNeg, [Expr.Call(floatSin, [f])])])
        | SpecificCall <@ tan @> (_, _, [f]) -> Expr.Call(floatMul, [diffExpr v f; Expr.Call(floatPow, [Expr.Call(floatDiv, [Expr.Value(1.); Expr.Call(floatCos, [f])]); Expr.Value(2.)])])
        | SpecificCall <@ sqrt @> (_, _, [f]) -> Expr.Call(floatDiv, [diffExpr v f; Expr.Call(floatMul, [Expr.Value(2.); Expr.Call(floatSqrt, [f])])])
        | SpecificCall <@ sinh @> (_, _, [f]) -> Expr.Call(floatMul, [diffExpr v f; Expr.Call(floatCosh, [f])])
        | SpecificCall <@ cosh @> (_, _, [f]) -> Expr.Call(floatMul, [diffExpr v f; Expr.Call(floatSinh, [f])])
        | SpecificCall <@ tanh @> (_, _, [f]) -> Expr.Call(floatMul, [diffExpr v f; Expr.Call(floatPow, [Expr.Call(floatDiv, [Expr.Value(1.); Expr.Call(floatCosh, [f])]); Expr.Value(2.)])])
        | SpecificCall <@ asin @> (_, _, [f]) -> Expr.Call(floatDiv, [diffExpr v f; Expr.Call(floatSqrt, [Expr.Call(floatSub, [Expr.Value(1.); Expr.Call(floatMul, [f; f])])])])
        | SpecificCall <@ acos @> (_, _, [f]) -> Expr.Call(floatDiv, [diffExpr v f; Expr.Call(floatNeg, [Expr.Call(floatSqrt, [Expr.Call(floatSub, [Expr.Value(1.); Expr.Call(floatMul, [f; f])])])])])
        | SpecificCall <@ atan @> (_, _, [f]) -> Expr.Call(floatDiv, [diffExpr v f; Expr.Call(floatAdd, [Expr.Value(1.); Expr.Call(floatMul, [f; f])])])
        | ShapeVar(var) -> if var = v then Expr.Value(1.) else Expr.Value(0.)
        | ShapeLambda(arg, body) -> Expr.Lambda(arg, diffExpr v body)
        | ShapeCombination(shape, args) -> RebuildShapeCombination(shape, List.map (diffExpr v) args)

    /// Symbolically differentiate Expr `expr` with respect to variable name `vname`
    let diffSym vname expr =
        let eexpr = expand expr
        let args = getExprArgs eexpr
        let xvar = Array.tryFind (fun (a:Var) -> a.Name = vname) args
        match xvar with
        | Some(v) -> eexpr |> diffExpr v
        | None -> failwith "Given expression is not a function of a variable with the given name."

    /// Evaluate scalar-to-scalar Expr `expr`, at point `x`
    let evalSS (x:float) expr =
        Expr.Application(expr, Expr.Value(x))
        |> evalRaw<float>

    /// Evaluate vector-to-scalar Expr `expr`, at point `x`
    let evalVS (x:float[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float>)])
        Expr.Applications(expr, args)
        |> evalRaw<float>
    
    /// Evaluate vector-to-vector Expr `expr`, at point `x`
    let evalVV (x:float[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float>)])
        Expr.Applications(expr, args)
        |> evalRaw<float[]>

    /// Compute the `n`-th derivative of an Expr, with respect to Var `v`
    let rec diffExprN v n =
        match n with
        | a when a < 0 -> failwith "Order of derivative cannot be negative."
        | 0 -> fun (x:Expr) -> x
        | 1 -> fun x -> diffExpr v x
        | _ -> fun x -> diffExprN v (n - 1) (diffExpr v x)

/// Symbolic differentiation operations module (automatically opened)
[<AutoOpen>]
module SymbolicOps =
    /// First derivative of a scalar-to-scalar function `f`
    let diff (f:Expr) =
        fun x ->
            let fe = expand f
            let args = getExprArgs fe
            diffExpr args.[0] fe
            |> evalSS x

    /// Original value and first derivative of a scalar-to-scalar function `f`
    let diff' (f:Expr<float->float>) =
        fun x -> (evalSS x f, diff f x)

    /// `n`-th derivative of a scalar-to-scalar function `f`
    let diffn n (f:Expr) =
        fun x -> 
            let fe = expand f
            let args = getExprArgs fe
            diffExprN args.[0] n fe
            |> evalSS x

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`
    let diffn' n f =
        fun x -> (evalSS x f, diffn n f x)
    
    /// Second derivative of a scalar-to-scalar function `f`
    let diff2 (f:Expr) =
        diffn 2 f

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let diff2' f =
        fun x -> (evalSS x f, diff2 f x)

    /// Gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad (f:Expr) =
        fun x ->
            let fe = expand f
            fe
            |> getExprArgs
            |> Array.map (fun a -> diffExpr a fe)
            |> Array.map (evalVS x)
    
    /// Original value and gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad' f =
        fun x -> (evalVS x f, grad f x)

    /// Transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT (f:Expr) =
        fun x ->
            let fe = expand f
            fe
            |> getExprArgs
            |> Array.map (fun a -> diffExpr a fe)
            |> Array.map (evalVV x)
            |> array2D

    /// Original value and transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT' f =
        fun x -> (evalVV x f, jacobianT f x)

    /// Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian f =
        jacobianT f >> transpose

    /// Original value and Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian' f =
        jacobianT' f >> fun (r, j) -> (r, transpose j)

    /// Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian (f:Expr) =
        fun x ->
            let fe = expand f
            fe
            |> getExprArgs
            |> Array.map (fun a -> diffExpr a (diffExpr a fe))
            |> Array.sumBy (evalVS x)

    /// Original value and Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian' f =
        fun x -> (evalVS x f, laplacian f x)

    /// Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let hessian (f:Expr) =
        fun (x:float[]) ->
            let fe = expand f
            let args = getExprArgs fe
            let ret:float[,] = Array2D.create x.Length x.Length 0.
            for i = 0 to x.Length - 1 do
                let di = diffExpr args.[i] fe
                for j = i to x.Length - 1 do
                    ret.[i, j] <- evalVS x (diffExpr args.[j] di)
            symmetricFromUpperTriangular ret

    /// Original value and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let hessian' f =
        fun x -> (evalVS x f, hessian f x)

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let gradhessian' f =
        fun x ->
            let (v, g) = grad' f x in (v, g, hessian f x)

    /// Gradient and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let gradhessian f =
        gradhessian' f >> sndtrd


/// Module with differentiation operators using Vector and Matrix input and output, instead of float[] and float[,]
module Vector =
    /// Original value and first derivative of a scalar-to-scalar function `f`
    let inline diff' f = SymbolicOps.diff' f
    /// First derivative of a scalar-to-scalar function `f`
    let inline diff f = SymbolicOps.diff f
    /// Original value and second derivative of a scalar-to-scalar function `f`
    let inline diff2' f = SymbolicOps.diff2' f
    /// Second derivative of a scalar-to-scalar function `f`
    let inline diff2 f = SymbolicOps.diff2 f
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn' n f = SymbolicOps.diffn' n f
    /// `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn n f = SymbolicOps.diffn n f
    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f = array >> SymbolicOps.grad' f >> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`
    let inline grad f = array >> SymbolicOps.grad f >> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f = array >> SymbolicOps.laplacian' f
    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f = array >> SymbolicOps.laplacian f
    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f = array >> SymbolicOps.jacobianT' f >> fun (a, b) -> (vector a, matrix b)
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f = array >> SymbolicOps.jacobianT f >> matrix
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f = array >> SymbolicOps.jacobian' f >> fun (a, b) -> (vector a, matrix b)
    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f = array >> SymbolicOps.jacobian f >> matrix
    /// Original value and Hessian of a vector-to-scalar function `f`
    let inline hessian' f = array >> SymbolicOps.hessian' f >> fun (a, b) -> (a, matrix b)
    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f = array >> SymbolicOps.hessian f >> matrix
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f = array >> SymbolicOps.gradhessian' f >> fun (a, b, c) -> (a, vector b, matrix c)
    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f = array >> SymbolicOps.gradhessian f >> fun (a, b) -> (vector a, matrix b)