(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Type Inference
===============

F# is a statically typed language with [type inference](http://msdn.microsoft.com/en-us/library/dd233180.aspx). Parts of the DiffSharp library work via AD-enabled numeric types, such as the **Diffsharp.AD.Forward.Dual** type implementing [dual numbers](http://en.wikipedia.org/wiki/Dual_number).

There are several ways the F# type inference system can work together with the DiffSharp library.

Lambda Expressions
------------------

The simplest and easiest way is to define functions using [lambda expressions](http://msdn.microsoft.com/en-us/library/dd233201.aspx) after differentiation operators. The expression will automatically assume the AD-enabled numeric type corresponding to the DiffSharp module and operator you are using.

(You can hover the pointer over the examples to check the types.)
*)

open DiffSharp.AD.Forward

// The lambda expression after "diff" has type Dual -> Dual
let f1 = diff (fun x -> sin (sqrt x))

(**
Existing Functions
------------------

When you have an existing function, in some cases, just opening the DiffSharp library and using a differentiation operation is sufficient for getting it interpreted as using AD-enabled types.

*)

// f2: float -> float
let f2 x =
    sin (sqrt x)

(** *)

open DiffSharp.AD.Forward

// f3 has the same definition as f2, here Dual type is inferred automatically
// f3: Dual -> Dual
let f3 x =
    sin (sqrt x)

let df3 = diff f3

(**
"Injecting" AD-enabled Types
----------------------------

Functions with numeric literals in their definition cannot be used as in the previous case, because literals cause the compiler to infer **float** or **int** arithmetic.
*)

// f4: float -> float
let f4 x =
    sin (3. * sqrt x)

(** 
In such cases, AD-enabled types should be explicitly used in one or more places in the function definition. 

A function's signature can be usually changed without having to change the type of all the involved values. For example, "injecting" some **Dual**s into a large **float** expression can cause the whole function to have **Dual** arithmetic.

Explicitly marking a parameter as **Dual**:
*)

// f5: Dual -> Dual
let f5 (x:Dual) =
    sin (3. * sqrt x)

(**
Converting a **float** into a **Dual**:
*)

// f6: Dual -> Dual
let f6 x =
    sin ((dual 3.) * sqrt x)

(**
Numeric Literals for DiffSharp.AD.Forward.Dual
----------------------------------------------

The library provides the _Q_ and _R_ numeric literals for the **Dual** type. A _Q-literal_ produces a **Dual** with tangent (or, derivative) value 0 and an _R-literal_ produces a **Dual** with tangent value 1 (representing the variable of differentiation, because the derivative of the variable of differentiation is, by definition, 1).

Using numeric literals to cause **Dual** inference:
*)

// f7: Dual -> Dual
let f7 x =
    sin (3Q * sqrt x)

(**

Using numeric literals to calculate partial derivatives with **DiffSharp.AD.Forward**:
*)

// A multivariate function
// f8: Dual -> Dual -> Dual
let f8 x y =
    sin (x * y)

// df8 / dx at (3, 7)
let df8x = tangent (f8 3R 7Q)

// df8 / dx at (3, 7)
let df8y = tangent (f8 3Q 7R)

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