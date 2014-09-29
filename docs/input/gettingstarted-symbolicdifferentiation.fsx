(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Symbolic Differentiation
========================

In addition to AD, DiffSharp also implements symbolic differentiation, which works by symbolic manipulation of mathematical expressions using rules of differential calculus.

For a complete list of available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.Symbolic
------------------

This is a symbolic differentiation module, used with the [**Expr**](http://msdn.microsoft.com/en-us/library/ee370577.aspx) type representing F# code expressions. A common way of generating F# code expressions is to use [code quotations](http://msdn.microsoft.com/en-us/library/dd233212.aspx), with the <@ and @> symbols delimiting an expression.
*)

open DiffSharp.Symbolic

// Derivative of Sin(3 * Sqrt(x)) at x = 2
let d = diff <@ fun x -> sin (3. * sqrt x) @> 2.

(**
Function definitions should be marked with the [**ReflectedDefinition**](http://msdn.microsoft.com/en-us/library/ee353643.aspx) attribute for allowing access to quotation expressions at runtime.
*)

// f: float -> float
[<ReflectedDefinition>]
let f x = sin (3. * sqrt x)

// Derivative of f at 2
let df = diff <@ f @> 2.

(**
Different from the **DiffSharp.AD** and **DiffSharp.Numerical** parts of the library, functions with vector domains are expected to be in curried form, instead of taking an array as a parameter.
*)

// g: float -> float -> float
[<ReflectedDefinition>]
let g x y = sin (x * y)

// Gradient of g at (2, 3)
let gg = grad <@ g @> [|2.; 3.|]

(**

Functions can be marked with the **ReflectedDefinition** attribute one by one, or they can be put into a module marked with this attribute to make it apply recursively to all.

Operations will delve into the definitions of other functions referenced from a given function (the referenced function will be _inlined_ into the body of the calling function), as long as they have the **ReflectedDefinition** attribute.

*)

[<ReflectedDefinition>]
module m =
    // f: float -> float
    let f x = sqrt x
    // g: float -> float -> float
    let g x y = sin (x * (f y))

// Hessian of g at (2, 3)
let hg = hessian <@ m.g @> [|2.; 3.|]