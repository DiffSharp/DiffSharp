(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
Reverse AD
==========

DiffSharp currently provides a single implementation of reverse AD, using what is known in AD literature as a _tape_. In future releases of the library, we will provide several other implementations.

For a complete list of available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

DiffSharp.AD.Reverse
--------------------

This is a reverse AD module that works by recording a trace of operations in forward evaluation and using this in the reverse sweep for backpropagating adjoints.

Reverse AD provides a performance advantage when computing gradients of vector-to-scalar functions, because it can calculate all partial derivatives in just one reverse sweep.

This module is used with the **Adj** numeric type.
*)

open DiffSharp.AD.Reverse

// g: Adj[] -> Adj
let g (x:Adj[]) = sin (x.[0] * x.[1] * x.[2])

// Gradient of g
let ga = grad g

(**
Alternatively, we can use a lambda expression to alleviate the need to explicitly state the **Adj[]** type, as it will be automatically inferred.
*)

// The same gradient
let gb = grad (fun x -> sin (x.[0] * x.[1] * x.[2]))

(**
Using the Reverse AD Trace
==========================

In addition to using the differentiation API provided by the reverse AD module (such as **diff**, **grad**, **jacobian** ), you can make use of the exposed [trace](http://en.wikipedia.org/wiki/Tracing_%28software%29) functionality. 

For code using the **Adj** numeric type, **DiffSharp.AD.Reverse.Trace** builds a trace of all executed mathematical operations, which subsequently allows a reverse sweep of these operations for propagating adjoint values in reverse. 
*)

// Reset the trace
Trace.Clear()

// Perform some operation (or a series of operations) using Adj type
let x = adj 0.5
let y = adj 1.2
let z = (sin x) * (cos y)

// Set the adjoint value of z to 1 (dz/dz = 1)
// i.e. calculate partial derivatives of z with respect to other variables
z.A <- 1.

// Propagate the adjoint values in reverse
Trace.ReverseSweep()

// You can calculate all partial derivatives in just one reverse sweep!
let dzdx = x.A
let dzdy = y.A

(**
The technique is equivalent to the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) method commonly used for training artificial neural networks in machine learning, which is essentially just a special case of reverse AD. You can see an implementation of the backpropagation algorithm, making use of the reverse AD trace, in the [neural networks](http://en.wikipedia.org/wiki/Backpropagation) example.
*)