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
// f: float -> float, the derivative of Sin(Sqrt(x))
let f = diff (fun x -> sin (sqrt x))

// Use a to compute the derivative at 2
let df = f 2.

(**

The library also provides differentiation operators that return the original value and the derivative values at the same time. This is advantageous because, due to the way AD works, a function's value and its derivative can be computed via executing it just once using the AD-enabled type.

*)

// f2: float -> (float * float), the original value and the derivative of Sin(Sqrt(x))
let f2 = diff' (fun x -> sin (sqrt x))

// Compute f2 and its derivative at 2
let vf2, df2 = f2 2.

(**
Existing Functions
------------------

When you have an existing function, just opening the DiffSharp library and using a differentiation operation is sufficient in some cases for giving it the AD-enabled signature.

*)

// f3: float -> float
let f3 x = sin (sqrt x)

(** *)

open DiffSharp.AD.Forward

// f4 has the same definition with f3, here Dual type is inferred automatically
// f4: Dual -> Dual
let f4 x = sin (sqrt x)

let df4 = diff f4

(**

In the above example, **f4** assumes the **Dual -> Dual** type and therefore cannot be used with other types, for example **float**. We can get around this by defining [generic numeric functions](http://tomasp.net/blog/fsharp-generic-numeric.aspx/) that can work with multiple types, via using **inline**.

*)
// f5 is the generic version of f4
let inline f5 x = sin (sqrt x)

// Here f5 behaves as Dual -> Dual
let df5 = diff f5 2.

// Here f5 behaves as float -> float
let vf5 = f5 2.

(**
"Injecting" AD-enabled Types
----------------------------

Functions with numeric literals in their definition cannot be used as in the previous case, because literals cause the compiler to infer other numeric types (such as 3. for **float** or 3 for **int** arithmetic).
*)

// f6: float -> float
let f6 x = sin (3. * sqrt x)

(** 
In such cases, AD-enabled types should be explicitly used in one or more places in the function definition.

Usually, a function's signature can be changed without having to change the type of all involved values. For example, "injecting" some **Dual**s into a large **float** expression can cause the whole function to assume **Dual** type.

Explicitly marking an argument as **Dual**:
*)

// f7: Dual -> Dual
let f7 (x:Dual) = sin (3. * sqrt x)

(**
Converting a **float** into a **Dual**:
*)

// f8: Dual -> Dual
let f8 x = sin ((dual 3.) * sqrt x)
