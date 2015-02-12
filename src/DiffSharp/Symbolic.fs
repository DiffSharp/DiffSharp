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

//
// TO DO: Add support for abs, log10, floor, ceiling, round
//

#light

/// Symbolic differentiation module
module DiffSharp.Symbolic

open System.Reflection
open Microsoft.FSharp.Reflection
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open Microsoft.FSharp.Quotations.ExprShape
open FSharp.Quotations.Evaluator
open DiffSharp.Util.LinearAlgebra
open DiffSharp.Util.General
open DiffSharp.Util.Quotations

/// Symbolic differentiation expression operations module (automatically opened)
[<AutoOpen>]
module ExprOps =

    let coreass = typeof<unit>.Assembly
    let coremod = coreass.GetModule("FSharp.Core.dll")
    let coreops = coremod.GetType("Microsoft.FSharp.Core.Operators")
    let coreprim = coremod.GetType("Microsoft.FSharp.Core.LanguagePrimitives")

    let opAdd = coreops.GetMethod("op_Addition")
    let opSub = coreops.GetMethod("op_Subtraction")
    let opMul = coreops.GetMethod("op_Multiply")
    let opDiv = coreops.GetMethod("op_Division")
    let opPow = coreops.GetMethod("op_Exponentiation")
    let opNeg = coreops.GetMethod("op_UnaryNegation")
    let opAbs = coreops.GetMethod("Abs")
    let opLog = coreops.GetMethod("Log")
    let opExp = coreops.GetMethod("Exp")
    let opSin = coreops.GetMethod("Sin")
    let opCos = coreops.GetMethod("Cos")
    let opTan = coreops.GetMethod("Tan")
    let opSqrt = coreops.GetMethod("Sqrt")
    let opSinh = coreops.GetMethod("Sinh")
    let opCosh = coreops.GetMethod("Cosh")
    let opTanh = coreops.GetMethod("Tanh")
    let opAsin = coreops.GetMethod("Asin")
    let opAcos = coreops.GetMethod("Acos")
    let opAtan = coreops.GetMethod("Atan")
    let primGen0 = coreprim.GetMethod("GenericZero")
    let primGen1 = coreprim.GetMethod("GenericOne")

    let call(genmi:MethodInfo, types, args) =
        Expr.Call(genmi.MakeGenericMethod(Array.ofList types), args)

    let callGen0(t) =
        call(primGen0, [t], [])

    let callGen1(t) =
        call(primGen1, [t], [])

    let callGen2(t) =
        call(opAdd, [t; t; t], [callGen1(t); callGen1(t)])

    /// Recursively traverse and differentiate Expr `expr` with respect to Var `v`
    // UNOPTIMIZED
    let rec diffExpr (v:Var) expr =
        match expr with
        | Value(v, vt) -> callGen0(vt)
        | Call(_, primGen1, []) -> callGen0(v.Type)
        | SpecificCall <@ (+) @> (_, ts, [f; g]) -> call(opAdd, ts, [diffExpr v f; diffExpr v g])
        | SpecificCall <@ (-) @> (_, ts, [f; g]) -> call(opSub, ts, [diffExpr v f; diffExpr v g])
        | SpecificCall <@ (*) @> (_, ts, [f; g]) -> call(opAdd, ts, [call(opMul, ts, [diffExpr v f; g]); call(opMul, ts, [f; diffExpr v g])])
        | SpecificCall <@ (/) @> (_, ts, [f; g]) -> call(opDiv, ts, [call(opSub, ts, [call(opMul, ts, [diffExpr v f; g]); call(opMul, ts, [f; diffExpr v g])]); call(opMul, ts, [g; g])])
        //This should cover all the cases: (f(x) ^ (g(x) - 1))(g(x) * f'(x) + f(x) * log(f(x)) * g'(x))
        | SpecificCall <@ op_Exponentiation @> (_, [t1; t2], [f; g]) -> call(opMul, [t1; t1; t1], [call(opPow, [t1; t1], [f; call(opSub, [t1; t1; t1], [g; callGen1(t1)])]); call(opAdd, [t1; t1; t1], [call(opMul, [t1; t1; t1], [g; diffExpr v f]); call(opMul, [t1; t1; t1], [call(opMul, [t1; t1; t1], [f; call(opLog, [t1], [f])]); diffExpr v g])])])
        | SpecificCall <@ atan2 @> (_, [t1; t2], [f; g]) -> call(opDiv, [t1; t1; t1], [call(opSub, [t1; t1; t1], [call(opMul, [t1; t1; t1], [g; diffExpr v f]); call(opMul, [t1; t1; t1], [f; diffExpr v g])]) ; call(opAdd, [t1; t1; t1], [call(opMul, [t1; t1; t1], [f; f]); call(opMul, [t1; t1; t1], [g; g])])])
        | SpecificCall <@ (~-) @> (_, ts, [f]) -> call(opNeg, ts, [diffExpr v f])
        | SpecificCall <@ log @> (_, [t], [f]) -> call(opDiv, [t; t; t], [diffExpr v f; f])
        | SpecificCall <@ exp @> (_, [t], [f]) -> call(opMul, [t; t; t], [diffExpr v f; call(opExp, [t], [f])])
        | SpecificCall <@ sin @> (_, [t], [f]) -> call(opMul, [t; t; t], [diffExpr v f; call(opCos, [t], [f])])
        | SpecificCall <@ cos @> (_, [t], [f]) -> call(opMul, [t; t; t], [diffExpr v f; call(opNeg, [t], [call(opSin, [t], [f])])])
        | SpecificCall <@ tan @> (_, [t], [f]) -> call(opMul, [t; t; t], [diffExpr v f; call(opMul, [t; t; t], [call(opDiv, [t; t; t], [callGen1(t); call(opCos, [t], [f])]); call(opDiv, [t; t; t], [callGen1(t); call(opCos, [t], [f])])])])
        | SpecificCall <@ sqrt @> (_, [t1; t2], [f]) -> call(opDiv, [t1; t1; t1], [diffExpr v f; call(opMul, [t1; t1; t1], [callGen2(t1); call(opSqrt, [t1; t1], [f])])])
        | SpecificCall <@ sinh @> (_, [t], [f]) -> call(opMul, [t; t; t], [diffExpr v f; call(opCosh, [t], [f])])
        | SpecificCall <@ cosh @> (_, [t], [f]) -> call(opMul, [t; t; t], [diffExpr v f; call(opSinh, [t], [f])])
        | SpecificCall <@ tanh @> (_, [t], [f]) -> call(opMul, [t; t; t], [diffExpr v f; call(opMul, [t; t; t], [call(opDiv, [t; t; t], [callGen1(t); call(opCosh, [t], [f])]); call(opDiv, [t; t; t], [callGen1(t); call(opCosh, [t], [f])])])])
        | SpecificCall <@ asin @> (_, [t], [f]) -> call(opDiv, [t; t; t], [diffExpr v f; call(opSqrt, [t; t], [call(opSub, [t; t; t], [callGen1(t); call(opMul, [t; t; t], [f; f])])])])
        | SpecificCall <@ acos @> (_, [t], [f]) -> call(opDiv, [t; t; t], [diffExpr v f; call(opNeg, [t], [call(opSqrt, [t; t], [call(opSub, [t; t; t], [callGen1(t); call(opMul, [t; t; t], [f; f])])])])])
        | SpecificCall <@ atan @> (_, [t], [f]) -> call(opDiv, [t; t; t], [diffExpr v f; call(opAdd, [t; t; t], [callGen1(t); call(opMul, [t; t; t], [f; f])])])
        | ShapeVar(var) -> if var = v then callGen1(var.Type) else callGen0(var.Type)
        | ShapeLambda(arg, body) -> Expr.Lambda(arg, diffExpr v body)
        | ShapeCombination(shape, args) -> RebuildShapeCombination(shape, List.map (diffExpr v) args)
    
    /// Symbolically differentiate Expr `expr` with respect to variable name `vname`
    let diffSym vname expr =
        let eexpr = expr
        let args = getExprArgs eexpr
        let xvar = Array.tryFind (fun (a:Var) -> a.Name = vname) args
        match xvar with
        | Some(v) -> eexpr |> diffExpr v
        | None -> eexpr |> diffExpr (Var(vname, args.[0].Type))
    
    /// Compute the `n`-th derivative of an Expr, with respect to Var `v`
    let rec diffExprN v n =
        match n with
        | a when a < 0 -> failwith "Order of derivative cannot be negative."
        | 0 -> fun (x:Expr) -> x
        | 1 -> fun x -> diffExpr v x
        | _ -> fun x -> diffExprN v (n - 1) (diffExpr v x)

    /// Evaluate scalar-to-scalar Expr `expr`, at point `x`
    let evalSS (x:float) expr =
        Expr.Application(expr, Expr.Value(x))
        |> QuotationEvaluator.CompileUntyped
        :?> float

    /// Evaluate vector-to-scalar Expr `expr`, at point `x`
    let evalVS (x:float[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float>)])
        Expr.Applications(expr, args)
        |> QuotationEvaluator.CompileUntyped
        :?> float
    
    /// Evaluate vector-to-vector Expr `expr`, at point `x`
    let evalVV (x:float[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float>)])
        Expr.Applications(expr, args)
        |> QuotationEvaluator.CompileUntyped
        :?> float[]


/// Symbolic differentiation operations module (automatically opened)
[<AutoOpen>]
module SymbolicOps =
    /// First derivative of a scalar-to-scalar function `f`
    let diff (f:Expr<float->float>) =
        let fe = expand f
        let args = getExprArgs fe
        diffExpr args.[0] fe
        |> QuotationEvaluator.CompileUntyped
        :?> (float->float)

    /// Original value and first derivative of a scalar-to-scalar function `f`
    let diff' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diff f
        fun x -> (fe x, fd x)

    /// `n`-th derivative of a scalar-to-scalar function `f`
    let diffn n (f:Expr<float->float>) =
        let fe = expand f
        let args = getExprArgs fe
        diffExprN args.[0] n fe
        |> QuotationEvaluator.CompileUntyped
        :?> (float->float)

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`
    let diffn' n f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diffn n f
        fun x -> (fe x, fd x)
    
    /// Second derivative of a scalar-to-scalar function `f`
    let diff2 f =
        diffn 2 f

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let diff2' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diff2 f
        fun x -> (fe x, fd x)

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`
    let inline diff2'' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float->float)
        let fd = diff f
        let fd2 = diff2 f
        fun x -> (fe, fd x, fd2 x)

    /// Gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad (f:Expr) =
        let fe = expand f
        let fg = Array.map (fun a -> diffExpr a fe) (getExprArgs fe)
        fun x -> Array.map (evalVS x) fg
    
    /// Original value and gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad' f =
        let fg = grad f
        fun x -> (evalVS x f, fg x)

    /// Transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT (f:Expr) =
        let fe = expand f
        let fj = Array.map (fun a -> diffExpr a fe) (getExprArgs fe)
        fun x -> Array.map (evalVV x) fj |> array2D

    /// Original value and transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT' f =
        let fj = jacobianT f
        fun x -> (evalVV x f, fj x)

    /// Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian f =
        let fj = jacobianT f
        fun x -> fj x |> transpose

    /// Original value and Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian' f =
        let fj = jacobianT' f
        fun x -> fj x |> fun (r, j) -> (r, transpose j)

    /// Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian (f:Expr) =
        let fe = expand f
        let fd = Array.map (fun a -> diffExpr a (diffExpr a fe)) (getExprArgs fe)
        fun x -> Array.sumBy (evalVS x) fd

    /// Original value and Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian' f =
        let fd = laplacian f
        fun x -> (evalVS x f, fd x)

    /// Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let hessian (f:Expr) =
        let fe = expand f
        let args = getExprArgs fe
        let fd:Expr[,] = Array2D.create args.Length args.Length (Expr.Value(0.))
        for i = 0 to args.Length - 1 do
            let di = diffExpr args.[i] fe
            for j = i to args.Length - 1 do
                fd.[i, j] <- diffExpr args.[j] di
        fun (x:float[]) ->
            let ret:float[,] = Array2D.create x.Length x.Length 0.
            for i = 0 to x.Length - 1 do
                for j = i to x.Length - 1 do
                    ret.[i, j] <- evalVS x fd.[i, j]
            copyUpperToLower ret

    /// Original value and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let hessian' f =
        let fd = hessian f
        fun x -> (evalVS x f, fd x)

    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let gradhessian' f =
        let fh = hessian f
        let fg = grad f
        fun x -> (evalVS x f, fg x, fh x)

    /// Gradient and Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let gradhessian f =
        let fgh = gradhessian' f
        fun x -> fgh x |> sndtrd


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
    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`
    let inline diff2'' f = SymbolicOps.diff2'' f
    /// Original value and the `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn' n f = SymbolicOps.diffn' n f
    /// `n`-th derivative of a scalar-to-scalar function `f`
    let inline diffn n f = SymbolicOps.diffn n f
    /// Original value and gradient of a vector-to-scalar function `f`
    let inline grad' f = Vector.toArray >> SymbolicOps.grad' f >> fun (a, b) -> (a, vector b)
    /// Gradient of a vector-to-scalar function `f`
    let inline grad f x = Vector.toArray >> SymbolicOps.grad f >> vector
    /// Original value and Laplacian of a vector-to-scalar function `f`
    let inline laplacian' f x = Vector.toArray >> SymbolicOps.laplacian' f
    /// Laplacian of a vector-to-scalar function `f`
    let inline laplacian f x = Vector.toArray >> SymbolicOps.laplacian f
    /// Original value and transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT' f x = Vector.toArray >> SymbolicOps.jacobianT' f >> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Transposed Jacobian of a vector-to-vector function `f`
    let inline jacobianT f x = Vector.toArray >> SymbolicOps.jacobianT f >> Matrix.ofArray2d
    /// Original value and Jacobian of a vector-to-vector function `f`
    let inline jacobian' f x = Vector.toArray >> SymbolicOps.jacobian' f >> fun (a, b) -> (vector a, Matrix.ofArray2d b)
    /// Jacobian of a vector-to-vector function `f`
    let inline jacobian f x = Vector.toArray >> SymbolicOps.jacobian f >> Matrix.ofArray2d
    /// Original value and Hessian of a vector-to-scalar function `f`
    let inline hessian' f x = Vector.toArray >> SymbolicOps.hessian' f >> fun (a, b) -> (a, Matrix.ofArray2d b)
    /// Hessian of a vector-to-scalar function `f`
    let inline hessian f x = Vector.toArray >> SymbolicOps.hessian f >> Matrix.ofArray2d
    /// Original value, gradient, and Hessian of a vector-to-scalar function `f`
    let inline gradhessian' f x = Vector.toArray >> SymbolicOps.gradhessian' f >> fun (a, b, c) -> (a, vector b, Matrix.ofArray2d c)
    /// Gradient and Hessian of a vector-to-scalar function `f`
    let inline gradhessian f x = Vector.toArray >> SymbolicOps.gradhessian f >> fun (a, b) -> (vector a, Matrix.ofArray2d b)