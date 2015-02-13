(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Type Inference
==============

F# is a statically typed language with [type inference](http://msdn.microsoft.com/en-us/library/dd233180.aspx). Parts of the DiffSharp library work via AD-enabled numeric types, such as the **Diffsharp.AD.Forward.Dual** type implementing [dual numbers](http://en.wikipedia.org/wiki/Dual_number) for forward AD and the **DiffSharp.AD.Reverse.Adj** type implementing adjoints for reverse AD.

There are several ways the F# type inference system can work together with the DiffSharp library.

Lambda Expressions
------------------

The simplest and easiest way is to define functions using [lambda expressions](http://msdn.microsoft.com/en-us/library/dd233201.aspx) after differentiation operators. The expression will automatically assume the AD-enabled numeric type corresponding to the DiffSharp module and operator you are using.

(You can hover the pointer over the examples to check their types.)
*)

open DiffSharp.AD.Forward

// The lambda expression after "diff" has type Dual -> Dual
// a: float -> float, the derivative of Sin(Sqrt(x))
let a = diff (fun x -> sin (sqrt x))

// Use a to compute the derivative at 2
let da = a 2.

(**
Existing Functions
------------------

When you have an existing function, just opening the DiffSharp library and using a differentiation operation is sufficient in some cases for getting it interpreted as using AD-enabled types.

*)

// b: float -> float
let b x =
    sin (sqrt x)

(** *)

open DiffSharp.AD.Forward

// c has the same definition with b, here Dual type is inferred automatically
// c: Dual -> Dual
let c x =
    sin (sqrt x)

let dc = diff c

(**
"Injecting" AD-enabled Types
----------------------------

Functions with numeric literals in their definition cannot be used as in the previous case, because literals cause the compiler to infer other numeric types (such as 3. for **float** or 3 for **int** arithmetic).
*)

// d: float -> float
let d x =
    sin (3. * sqrt x)

(** 
In such cases, AD-enabled types should be explicitly used in one or more places in the function definition.

A function's signature can be usually changed without having to change the type of all the involved values. For example, "injecting" some **Dual**s into a large **float** expression can cause the whole function to have **Dual** arithmetic.

Explicitly marking a parameter as **Dual**:
*)

// e: Dual -> Dual
let e (x:Dual) =
    sin (3. * sqrt x)

(**
Converting a **float** into a **Dual**:
*)

// f: Dual -> Dual
let f x =
    sin ((dual 3.) * sqrt x)

(**
Numeric Literals for DiffSharp.AD.Forward.Dual
----------------------------------------------

The library provides the _Q_ and _R_ numeric literals for the **Dual** type. A _Q-literal_ produces a **Dual** with tangent (or, derivative) value 0 and an _R-literal_ produces a **Dual** with tangent value 1 (representing the variable of differentiation, because the derivative of the variable of differentiation is, by definition, 1).

Using numeric literals to cause **Dual** inference:
*)

// g: Dual -> Dual
let g x =
    sin (3Q * sqrt x)

(**

Using numeric literals to calculate partial derivatives with **DiffSharp.AD.Forward**:
*)

// A multivariate function
// h: Dual -> Dual -> Dual
let h x y =
    sin (x * y)

// dh / dx at (3, 7)
let dhx = tangent (h 3R 7Q)

// dh / dy at (3, 7)
let dhy = tangent (h 3Q 7R)

(**
Generic Functions
-----------------

F# supports [generic numeric functions](http://tomasp.net/blog/fsharp-generic-numeric.aspx/) that can work with multiple different numeric types.
*)

// A simple generic function
let inline sum a b =
    a + b

// Use sum with float
let i1 = sum 2. 2.

// Use sum with int
let i2 = sum 2 2

// Use sum with bigint
let i3 = sum 2I 2I

// Use sum with Dual
let i4 = sum 2Q 2Q

(** 
Using a generic function with the DiffSharp library.
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
let j = diff cosine 3.