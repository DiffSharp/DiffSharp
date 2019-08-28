(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Symbolic Differentiation
========================

In addition to AD, the DiffSharp library also implements [symbolic differentiation](https://en.wikipedia.org/wiki/Symbolic_computation), which works by the symbolic manipulation of closed-form expressions using rules of differential calculus.

For a complete list of the available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.Symbolic
------------------

This is a symbolic differentiation module, used with the [**Expr**](https://msdn.microsoft.com/en-us/library/ee370577.aspx) type representing F# code expressions. A common way of generating F# code expressions is to use [code quotations](https://msdn.microsoft.com/en-us/library/dd233212.aspx), with the <@ and @> symbols delimiting an expression.

Symbolic differentiation operators construct the wanted derivative as a new expression and return this as a compiled function that can be used subsequently for evaluating the derivative. Once the derivative expression is compiled and returned, it is significantly faster to run it with specific numerical arguments, compared to the initial time it takes to compile the function. You can see example compilation and running times on the [Benchmarks](benchmarks.html) page.
*)

open DiffSharp.Symbolic.Float64

// Derivative of Sin(3 * Sqrt(x))
// This returns a compiled function that gives the derivative
let d = diff <@ fun x -> sin (3. * sqrt x) @>

// Compute the derivative at x = 2
let d2 = d 2.

(**
Function definitions should be marked with the [**ReflectedDefinition**](https://msdn.microsoft.com/en-us/library/ee353643.aspx) attribute for allowing access to quotation expressions at runtime.
*)

// f: float -> float
[<ReflectedDefinition>]
let f x = sin (3. * sqrt x)

// Derivative of f at 2
let df = diff <@ f @> 2.

(**
Different from the **DiffSharp.AD** and **DiffSharp.Numerical** parts of the library, multivariate functions are expected to be in [curried](https://msdn.microsoft.com/en-us/library/dd233213.aspx?f=255&MSPPError=-2147217396) form.
*)

// g: float -> float -> float
[<ReflectedDefinition>]
let g x y = sin (x * y)

// Gradient of g at (2, 3)
let gg = grad <@ g @> [|2.; 3.|]

(**

Functions can be marked with the **ReflectedDefinition** attribute one by one, or they can be put into a module marked with this attribute to make it apply to all.

Differentiation operations will delve into the definition of any other function referenced from a given function (the referenced function will be _inlined_ into the body of the calling function), as long as they have the **ReflectedDefinition** attribute.

*)

[<ReflectedDefinition>]
module m =
    // f: float -> float
    let f x = sqrt x
    // g: float -> float -> float
    let g x y = sin (x * (f y))

// Hessian of g at (2, 3)
let hg = hessian <@ m.g @> [|2.; 3.|]