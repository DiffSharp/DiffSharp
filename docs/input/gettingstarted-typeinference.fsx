(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Type Inference
===============

F# is a statically typed language with [type inference](http://msdn.microsoft.com/en-us/library/dd233180.aspx). Parts of the DiffSharp library work via AD-enabled numeric types, such as the **Diffsharp.AD.Forward.Dual** type implementing [dual numbers](http://en.wikipedia.org/wiki/Dual_number).

There are several ways the F# type inference system can work together with the DiffSharp library.

Without Change in Existing Code
-------------------------------

In some cases, just opening the DiffSharp library and using a differentiation operation is sufficient for an existing function to get inferred to use AD-enabled types.

(You can hover the pointer over the examples to check the types.)

*)

// f1: float -> float
let f1 x =
    sin (sqrt x)

(** *)

open DiffSharp.AD.Forward

// f2 has the same definition as f1, here Dual type is inferred automatically
// f2: Dual -> Dual
let f2 x =
    sin (sqrt x)

let f2' = diff f2

(**
"Injecting" AD-enabled Types
----------------------------

Functions with numeric literals in their definition cannot be used as in the previous case, because literals cause the compiler to infer **float** or **int** arithmetic.
*)

// f3: float -> float
let f3 x =
    sin (3. * sqrt x)

(** 
In such cases, AD-enabled types should be explicitly used in one or more places in the function definition. 

A function's signature can be usually changed without having to change the type of all the involved values. For example, "injecting" some **Dual**s into a large **float** expression can cause the whole function to have **Dual** arithmetic.

Explicitly marking a parameter as **Dual**:
*)

// f4: Dual -> Dual
let f4 (x:Dual) =
    sin (3. * sqrt x)

(**
Converting a **float** into a **Dual**:
*)

// f5: Dual -> Dual
let f5 x =
    sin ((dual 3.) * sqrt x)

(**
Numeric Literals for DiffSharp.AD.Forward.Dual
----------------------------------------------

The library provides the _Q_ and _R_ numeric literals for the **Dual** type. A _Q-literal_ produces a **Dual** with tangent (or, derivative) value 0 and an _R-literal_ produces a **Dual** with tangent value 1 (representing the variable of differentiation, because the derivative of the variable of differentiation is, by definition, 1).

Using numeric literals to cause **Dual** inference:
*)

// f6: Dual -> Dual
let f6 x =
    sin (3Q * sqrt x)

(**

Using numeric literals to calculate partial derivatives with **DiffSharp.AD.Forward**:
*)

// A multivariate function
// f7: Dual -> Dual -> Dual
let f7 x y =
    sin (x * y)

// df7 / dx at (3, 7)
let f7'x = tangent (f7 3R 7Q)

// df7 / dx at (3, 7)
let f7'y = tangent (f7 3Q 7R)

(**
Generic Functions
-----------------

F# supports [generic numeric functions](http://tomasp.net/blog/fsharp-generic-numeric.aspx/) that allow computations with a variety of numeric types without changing the code.
*)

// A simple generic function
let inline sum a b =
    a + b

// Use sum with float
let g1 = sum 2. 2.

// Use sum with int
let g2 = sum 2 2

// Use sum with bigint
let g3 = sum 2I 2I

(** 
DiffSharp can be used with generic numeric functions.
*)

// A generic implementation of cosine, 8-th degree approximation
// cosine: 'a -> 'a
let inline cosine (x: ^a) =
    let one: ^a = LanguagePrimitives.GenericOne
    Seq.initInfinite(fun i -> LanguagePrimitives.DivideByInt (-x*x) ((2*i+1) * (2*i+2)))
    |> Seq.scan (*) one
    |> Seq.take 8
    |> Seq.sum

// Derivative of cosine at 3
let g6 = diff cosine 3.