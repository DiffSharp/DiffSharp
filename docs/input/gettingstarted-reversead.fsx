(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Reverse AD
==========

DiffSharp currently provides a single implementation of reverse AD, using what is known in AD literature as a _tape_. In future releases of the library, we will provide several other implementations.

For a complete list of available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.AD.Reverse
--------------------

This is a reverse AD module that works by recording a trace of operations in forward evaluation and using this in the reverse sweep for backpropagating adjoints. The technique is essentially equivalent to the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) method for training artificial neural networks in machine learning, which is just a special case of reverse AD.

Reverse AD provides a performance advantage when computing gradients of vector-to-scalar functions.

This module is used with the **Adj** numeric type.
*)

open DiffSharp.AD.Reverse

// g: Adj[] -> Adj
let g (x:Adj[]) = sin (x.[0] * x.[1] * x.[2])

// Gradient of g
let ga = grad g

(**
Alternatively, we can use a lambda expression to alleviate the need to explicitly define the type **x:Adj[]**, as it will be automatically inferred.
*)

// The same gradient
let gb = grad (fun x -> sin (x.[0] * x.[1] * x.[2]))
