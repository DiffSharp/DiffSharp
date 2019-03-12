// This file is part of DiffSharp: Differentiable Functional Programming - https://diffsharp.github.io
// Copyright (c) 2016-     University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
// Copyright (c) 2017-     Microsoft Research, Cambridge, UK (Don Syme <dsyme@microsoft.com>)
// Copyright (c) 2014-     National University of Ireland Maynooth (Barak A. Pearlmutter <barak@pearlmutter.net>)
// Copyright (c) 2014-2016 National University of Ireland Maynooth (Atilim Gunes Baydin)
// This code is licensed under the BSD license (see LICENSE file for details)

/// Symbolic differentiation module
module DiffSharp.Symbolic.Float32

open FSharp.Quotations
open FSharp.Quotations.Patterns
open FSharp.Quotations.DerivedPatterns
open FSharp.Quotations.ExprShape
open FSharp.Quotations.Evaluator
open DiffSharp.Util
open DiffSharp.Config

/// Symbolic differentiation expression operations module (automatically opened)
[<AutoOpen>]
module ExprOps =

    /// Recursively traverse and differentiate Expr `expr` with respect to Var `v`
    let rec diffExpr (v:Var) expr =
        match expr with
        | Single(_) -> <@@ 0.f @@>
        | SpecificCall <@ (+) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ (%%at:float32) + %%bt @@>
        | SpecificCall <@ (-) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ (%%at:float32) - %%bt @@>
        | SpecificCall <@ (*) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ (%%at:float32) * %%b + %%a * %%bt @@>
        | SpecificCall <@ (/) @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ ((%%at:float32) * %%b - %%a * %%bt) / (%%b * %%b)@@>
        | SpecificCall <@ op_Exponentiation @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ ((%%a:float32) ** %%b) * ((%%b * %%at / %%a) + ((log %%a) * %%bt)) @@>
        | SpecificCall <@ atan2 @> (_, _, [a; b]) -> let at, bt = diffExpr v a, diffExpr v b in <@@ ((%%at:float32) * %%b - %%a * %%bt) / (%%a * %%a + %%b * %%b) @@>
        | SpecificCall <@ (~-) @> (_, _, [a]) -> let at = diffExpr v a in <@@ -(%%at:float32) @@>
        | SpecificCall <@ log @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) / %%a @@>
        | SpecificCall <@ log10 @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) / (%%a * log10ValFloat32) @@>
        | SpecificCall <@ exp @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) * %%expr @@>
        | SpecificCall <@ sin @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) * cos %%a @@>
        | SpecificCall <@ cos @> (_, _, [a]) -> let at = diffExpr v a in <@@ -(%%at:float32) * sin %%a @@>
        | SpecificCall <@ tan @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) / ((cos %%a) ** 2.f) @@>
        | SpecificCall <@ sqrt @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) / (2.f * %%expr) @@>
        | SpecificCall <@ sinh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) * cosh %%a @@>
        | SpecificCall <@ cosh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) * sinh %%a @@>
        | SpecificCall <@ tanh @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) / ((cosh %%a) ** 2.f) @@>
        | SpecificCall <@ asin @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) / sqrt (1.f - %%a * %%a) @@>
        | SpecificCall <@ acos @> (_, _, [a]) -> let at = diffExpr v a in <@@ -(%%at:float32) / sqrt (1.f - %%a * %%a) @@>
        | SpecificCall <@ atan @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) / (1.f + %%a * %%a) @@>
        | SpecificCall <@ abs @> (_, _, [a]) -> let at = diffExpr v a in <@@ (%%at:float32) * float32 (sign %%a) @@> //The derivative of abs is not defined at 0.f
        | SpecificCall <@ floor @> (_, _, [a]) -> <@@ 0.f @@> // The derivative of floor is not defined for integer values.
        | SpecificCall <@ ceil @> (_, _, [a]) -> <@@ 0.f @@> // The derivative of ceil is not defined for integer values.
        | SpecificCall <@ round @> (_, _, [a]) -> <@@ 0.f @@> // The derivative of round is not defined for values halfway between integers.
        | ShapeVar(var) -> if var = v then <@@ 1.f @@> else <@@ 0.f @@>
        | ShapeLambda(arg, body) -> Expr.Lambda(arg, diffExpr v body)
        | ShapeCombination(shape, args) -> RebuildShapeCombination(shape, List.map (diffExpr v) args)

    /// Simplify Expr `expr`
    let simplify expr =
        let rec simplifyExpr expr =
            match expr with
            | SpecificCall <@ (*) @> (_, _, [a; Single(0.f)]) -> <@@ 0.f @@>
            | SpecificCall <@ (*) @> (_, _, [Single(0.f); b]) -> <@@ 0.f @@>
            | SpecificCall <@ (*) @> (_, _, [a; Single(1.f)]) -> a
            | SpecificCall <@ (*) @> (_, _, [Single(1.f); b]) -> b
            | SpecificCall <@ (+) @> (_, _, [a; Single(0.f)]) -> a
            | SpecificCall <@ (+) @> (_, _, [Single(0.f); b]) -> b
            | SpecificCall <@ (-) @> (_, _, [a; Single(0.f)]) -> a
            | SpecificCall <@ (-) @> (_, _, [Single(0.f); b]) -> <@@ -(%%b:float32) @@>
            | SpecificCall <@ (/) @> (_, _, [a; Single(1.f)]) -> a
            | SpecificCall <@ op_Exponentiation @> (_, _, [Single(1.f); _]) -> <@@ 1.f @@>
            | ShapeVar(var) -> Expr.Var(var)
            | ShapeLambda(arg, body) -> Expr.Lambda(arg, simplifyExpr body)
            | ShapeCombination(shape, args) -> RebuildShapeCombination(shape, List.map simplifyExpr args)
        let s = Seq.unfold (fun s ->
                        let e = simplifyExpr (fst s)
                        let el = e.ToString().Length
                        if el <> (snd s) then
                            Some(e, (e, el))
                        else
                            None
                        ) (expr, expr.ToString().Length)
        if Seq.isEmpty s then expr else Seq.last s

    /// Completely expand Expr `expr`
    let expand expr =
        let rec expandExpr vars expr =
            let expanded =
                match expr with
                | ShapeVar v when Map.containsKey v vars -> vars.[v]
                | ShapeVar v -> Expr.Var v
                | Call(body, MethodWithReflectedDefinition meth, args) ->
                    let this = match body with Some b -> Expr.Application(meth, b) | _ -> meth
                    let res = Expr.Applications(this, [for a in args -> [a]])
                    expandExpr vars res
                | ShapeLambda(v, expr) -> Expr.Lambda(v, expandExpr vars expr)
                | ShapeCombination(o, exprs) -> RebuildShapeCombination(o, List.map (expandExpr vars) exprs)
            match expanded with
            | Application(ShapeLambda(v, body), assign)
            | Let(v, assign, body) -> expandExpr (Map.add v (expandExpr vars assign) vars) body
            | _ -> expanded
        expandExpr Map.empty expr

    /// Get the arguments of a function given in Expr `expr`, as a Var array
    let getExprArgs expr =
        let rec getLambdaArgs (args:Var[]) = function
            | Lambda(arg, body) -> getLambdaArgs (Array.append args [|arg|]) body
            | _ -> args
        getLambdaArgs Array.empty expr

    /// Compute the `n`-th derivative of an Expr, with respect to Var `v`
    let rec diffExprN v n =
        match n with
        | a when a < 0 -> failwith "Order of derivative cannot be negative."
        | 0 -> fun (x:Expr) -> x
        | 1 -> fun x -> diffExpr v x
        | _ -> fun x -> diffExprN v (n - 1) (diffExpr v x)

    /// Evaluate scalar-to-scalar Expr `expr`, at point `x`
    let evalSS (x:float32) expr =
        Expr.Application(expr, Expr.Value(x))
        |> QuotationEvaluator.CompileUntyped
        :?> float32

    /// Evaluate vector-to-scalar Expr `expr`, at point `x`
    let evalVS (x:float32[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float32>)])
        Expr.Applications(expr, args)
        |> QuotationEvaluator.CompileUntyped
        :?> float32

    /// Evaluate vector-to-vector Expr `expr`, at point `x`
    let evalVV (x:float32[]) expr =
        let args = List.ofArray x |> List.map (fun a -> [Expr.Value(a, typeof<float32>)])
        Expr.Applications(expr, args)
        |> QuotationEvaluator.CompileUntyped
        :?> float32[]


/// Symbolic differentiation operations module (automatically opened)
[<AutoOpen>]
module DiffOps =
    /// First derivative of a scalar-to-scalar function `f`
    let diff (f:Expr<float32->float32>) =
        let fe = expand f
        let args = getExprArgs fe
        diffExpr args.[0] fe
        |> simplify
        |> QuotationEvaluator.CompileUntyped
        :?> (float32->float32)

    /// Original value and first derivative of a scalar-to-scalar function `f`
    let diff' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float32->float32)
        let fd = diff f
        fun x -> (fe x, fd x)

    /// `n`-th derivative of a scalar-to-scalar function `f`
    let diffn n (f:Expr<float32->float32>) =
        let fe = expand f
        let args = getExprArgs fe
        diffExprN args.[0] n fe
        |> simplify
        |> QuotationEvaluator.CompileUntyped
        :?> (float32->float32)

    /// Original value and `n`-th derivative of a scalar-to-scalar function `f`
    let diffn' n f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float32->float32)
        let fd = diffn n f
        fun x -> (fe x, fd x)

    /// Second derivative of a scalar-to-scalar function `f`
    let diff2 f =
        diffn 2 f

    /// Original value and second derivative of a scalar-to-scalar function `f`
    let diff2' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float32->float32)
        let fd = diff2 f
        fun x -> (fe x, fd x)

    /// Original value, first derivative, and second derivative of a scalar-to-scalar function `f`
    let inline diff2'' f =
        let fe = (QuotationEvaluator.CompileUntyped f) :?> (float32->float32)
        let fd = diff f
        let fd2 = diff2 f
        fun x -> (fe, fd x, fd2 x)

    /// Gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad (f:Expr) =
        let fe = expand f
        let fg = Array.map (fun a -> simplify (diffExpr a fe)) (getExprArgs fe)
        fun x -> Array.map (evalVS x) fg

    /// Original value and gradient of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let grad' f =
        let fg = grad f
        fun x -> (evalVS x f, fg x)

    /// Transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT (f:Expr) =
        let fe = expand f
        let fj = Array.map (fun a -> simplify (diffExpr a fe)) (getExprArgs fe)
        fun x -> Array.map (evalVV x) fj |> array2D

    /// Original value and transposed Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobianT' f =
        let fj = jacobianT f
        fun x -> (evalVV x f, fj x)

    /// Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian f =
        let fj = jacobianT f
        fun x -> fj x |> GlobalConfig.Float32Backend.Transpose_M

    /// Original value and Jacobian of a vector-to-vector function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let jacobian' f =
        let fj = jacobianT' f
        fun x -> fj x |> fun (r, j) -> (r, GlobalConfig.Float32Backend.Transpose_M j)

    /// Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian (f:Expr) =
        let fe = expand f
        let fd = Array.map (fun a -> simplify (diffExpr a (diffExpr a fe))) (getExprArgs fe)
        fun x -> Array.sumBy (evalVS x) fd

    /// Original value and Laplacian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let laplacian' f =
        let fd = laplacian f
        fun x -> (evalVS x f, fd x)

    /// Hessian of a vector-to-scalar function `f`. Function should have multiple variables in curried form, instead of an array variable as in other parts of the library.
    let hessian (f:Expr) =
        let fe = expand f
        let args = getExprArgs fe
        let fd:Expr[,] = Array2D.create args.Length args.Length (Expr.Value(0.f))
        for i = 0 to args.Length - 1 do
            let di = diffExpr args.[i] fe
            for j = i to args.Length - 1 do
                fd.[i, j] <- simplify (diffExpr args.[j] di)
        fun (x:float32[]) ->
            let ret:float32[,] = Array2D.create x.Length x.Length 0.f
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
        fun x -> fgh x |> drop1Of3

    /// Original value and curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Curl of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curl f x =
        let j = jacobianT f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurl()
        [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|]

    /// Original value and divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div' f x =
        let v, j = jacobianT' f x
        if Array2D.length1 j <> Array2D.length2 j then ErrorMessages.InvalidArgDiv()
        v, GlobalConfig.Float32Backend.Sum_V(GlobalConfig.Float32Backend.Diagonal_M(j))

    /// Divergence of a vector-to-vector function `f`, at point `x`. Defined only for functions with a square Jacobian matrix.
    let inline div f x =
        let j = jacobianT f x
        if Array2D.length1 j <> Array2D.length2 j then ErrorMessages.InvalidArgDiv()
        GlobalConfig.Float32Backend.Sum_V(GlobalConfig.Float32Backend.Diagonal_M(j))

    /// Original value, curl, and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv' f x =
        let v, j = jacobianT' f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        v, [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], j.[0, 0] + j.[1, 1] + j.[2, 2]

    /// Curl and divergence of a vector-to-vector function `f`, at point `x`. Supported only for functions with a three-by-three Jacobian matrix.
    let inline curldiv f x =
        let j = jacobianT f x
        if (Array2D.length1 j, Array2D.length2 j) <> (3, 3) then ErrorMessages.InvalidArgCurlDiv()
        [|j.[1, 2] - j.[2, 1]; j.[2, 0] - j.[0, 2]; j.[0, 1] - j.[1, 0]|], j.[0, 0] + j.[1, 1] + j.[2, 2]
