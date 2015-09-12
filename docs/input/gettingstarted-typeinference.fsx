(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Type Inference
==============

The recommended way of using the library is through the **DiffSharp.AD.D** type. This type instantiates [dual numbers](http://en.wikipedia.org/wiki/Dual_number) (for forward AD) and/or [adjoints](http://en.wikipedia.org/wiki/Adjoint) (for reverse AD) as needed, automatically selecting the best one for a given case.

The library supports nesting, which means that you can evaluate derivatives of functions that may themselves be internally taking derivatives, up to arbitrary levels. All emerging higher-order derivatives are automatically handled by the library and computed exactly and efficiently.

In summary, you need to write the part of your numeric code where you need derivatives (e.g., for optimization) using the **D** numeric type, the results of which you may convert later to an integral type such as **float**.

Converting to and from D
------------------------
*)

open DiffSharp.AD

let v1 = D 2.6    // Create D with value 2.6
let v2 = float v1 // Convert D to float

let v3 = 2.6      // Create float
let v4 = D v3     // Convert float to D

(**
Defining Your Functions
-----------------------

There are several ways the type inference system can work together with DiffSharp, when defining your functions.

### Lambda Expressions

The simplest and easiest way is to define functions using [lambda expressions](http://msdn.microsoft.com/en-us/library/dd233201.aspx) after differentiation operators. The expression will automatically assume the required signature.

(You can hover the pointer over the examples to check their types.)
*)

open DiffSharp.AD

// The lambda expression after "diff" has type D -> D
// df: D -> D, the derivative of Sin(Sqrt(x))
let df = diff (fun x -> sin (sqrt x))

// Use df to compute the derivative at 2
let vdf = df (D 2.)

(**

The library also provides differentiation operators that return the original value and the derivative values at the same time. This is advantageous because, in many cases, original values and derivatives can be computed during the same execution of the code.

*)

// df2: D -> (D * D), the original value and the derivative of Sin(Sqrt(x))
let df2 = diff' (fun x -> sin (sqrt x))

// Compute Sin(Sqrt(x)) and its derivative at 2
let vf2, vdf2 = df2 (D 2.)

(**
### Existing Functions

When you have an existing function, just opening the DiffSharp library and using a differentiation operation is sufficient in some cases for enforcing the required signature.

*)

// f3: float -> float
let f3 x = sin (sqrt x)

(** *)

open DiffSharp.AD

// f4 has the same definition with f3, yet here D type is inferred automatically
// f4: D -> D
let f4 x = sin (sqrt x)

let df4 = diff f4

(**

In the above example, **f4** assumes the **D -> D** type and therefore cannot be used with other types, for example **float**. We can get around this by defining [generic numeric functions](http://tomasp.net/blog/fsharp-generic-numeric.aspx/) that can work with multiple types, via using **inline**.

*)
// f5 is the generic version of f4
// f5: 'a -> 'b
let inline f5 x = sin (sqrt x)

// Here f5 behaves as D -> D
let df5 = diff f5 (D 2.)

// Here f5 behaves as float -> float
let vf5 = f5 2.

(**
### "Injecting" AD-enabled Types

Functions with numeric literals in their definition cannot be used as in the previous case, because literals cause the compiler to infer other types of arithmetic (such as 3. for **float** or 3 for **int** ).
*)

// f6: float -> float
let f6 x = sin (3. * sqrt x)

(** 
In such cases, AD-enabled types should be explicitly used in one or more places in the function definition.

Usually, a function's signature can be altered without having to change the type of all involved values. For example, "injecting" some **D**s into a large **float** expression can cause the whole expression to assume **D** type.

Explicitly marking an argument as **D**:
*)

// f7: D -> D
let f7 (x:D) = sin (3. * sqrt x)

(**
Converting a **float** into **D**:
*)

// f8: D -> D
let f8 x = sin ((D 3.) * sqrt x)
