(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Type Inference
==============

Differentiation can be applied to functions using the **D**, **DV**, and **DM** types respectively for scalar, vector, and matrix values. 32- and 64-bit floating point varieties of these types are provided by the **DiffSharp.AD.Float32** and **DiffSharp.AD.Float64** modules. On many current systems, 32-bit (single) precision floating point operations run significantly faster than 64-bit (double) precision. It is therefore recommended to use the 32-bit module if this precision is sufficient for your usage case.

The library automatically instantiates [dual numbers](https://en.wikipedia.org/wiki/Dual_number) (for forward AD) and/or [adjoints](https://en.wikipedia.org/wiki/Adjoint) (for reverse AD) as needed, using the best one for a given differentiation operation.

DiffSharp supports nested AD, which means that you can evaluate derivatives of functions that may themselves be internally using derivatives, up to arbitrary level. All emerging higher-order derivatives are automatically handled by the library and computed exactly and efficiently.

In summary, you need to write the part of your numeric code where you need derivatives (e.g., for optimization) using the **D**, **DV**, and **DM** numeric types, which you may convert to or from integral types such as **float**, **float[]**, and **float[,]**.

Converting to and from D, DV, DM
--------------------------------
*)

open DiffSharp.AD.Float64

let s1 = D 2.6            // Create D from float
let s2 = float s1         // Convert D to float
let s3:float = convert s1 // Convert D to float

let v1 = DV [|1.; 2.; 3.|]       // Create DV from float[]
let v2 = toDV [1.; 2.; 3.]       // Create DV from sequence of floats
let v3 = toDV [|1.; 2.; 3.|]     // Create DV from sequence of floats
let v4 = toDV [D 1.; D 2.; D 3.] // Create DV from sequence of Ds
let v5:float[] = convert v1      // Convert to array of floats

let m1 = DM (array2D [[1.; 2.] ; [3.; 4.]]) // Create DM from float[,]
let m2 = toDM [[1.; 2.]; [3.; 4.]]          // Create DM from sequence of sequences of floats
let m3:float[,] = convert m1                // Convert to float[,]

(**
Defining Your Functions
-----------------------

There are several ways the type inference system can work together with DiffSharp, when defining your functions.

### Lambda Expressions

The simplest and easiest way is to define functions using [lambda expressions](https://msdn.microsoft.com/en-us/library/dd233201.aspx) after differentiation operators. The expression will automatically assume the required signature.

(You can hover the pointer over the examples to check their types.)
*)

open DiffSharp.AD.Float64

// The lambda expression after "diff": D -> D
// df: D -> D, the derivative of Sin(Sqrt(x))
let df = diff (fun x -> sin (sqrt x))

// Use df to compute the derivative at 2
let vdf = df (D 2.)

(**

The library also provides differentiation operators that return the original function value and the derivative values at the same time. This is advantageous because, in many cases, original values and derivatives are computed during the same execution of the code.

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

open DiffSharp.AD.Float64

// f4 has the same definition with f3, yet here D type is inferred automatically
// f4: D -> D
let f4 x = sin (sqrt x)

let df4 = diff f4

(**
### Generic Functions

In the previous example, **f4** assumes the **D -> D** type and therefore cannot be used with other types, for example **float**. We can get around this by defining [generic numeric functions](https://tomasp.net/blog/fsharp-generic-numeric.aspx/) that can work with multiple types, by using **inline**.

*)
// f5 is the generic version of f4
// f5: 'a -> 'b
let inline f5 x = sin (sqrt x)

// Here f5 behaves as float -> float
let vf5 = f5 2.

// Here f5 behaves as D -> D
let df5 = diff f5 (D 2.)

(**
### "Injecting" AD-Types

Existing functions with numeric literals in their definition cannot assume the required signature, because literals cause the compiler to infer specific types of arithmetic (such as 3. for **float** or 3 for **int** ).
*)

// f6: float -> float
let f6 x = sin (3. * sqrt x)

(** 
In such cases, AD-enabled types should be explicitly used in one or more places in the function definition.

Usually, a function's signature can be altered without having to change the type of all involved values. For example, "injecting" some **D**s into a large **float** expression will cause the whole expression to assume the **D** type.

Explicitly marking an argument as **D**:
*)

// f7: D -> D
let f7 (x:D) = sin (3. * sqrt x)

(**
Converting a **float** into **D**:
*)

// f8: D -> D
let f8 x = sin ((D 3.) * sqrt x)
