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

In addition to using the differentiation API provided by the reverse AD module (such as **diff**, **grad**, **jacobian** ), you can make use of the exposed [trace](http://en.wikipedia.org/wiki/Tracing_%28software%29) functionality. For code using the **Adj** numeric type, **DiffSharp.AD.Reverse.Trace** builds a trace of all executed mathematical operations, which subsequently allows a reverse sweep of these operations for propagating adjoint values in reverse. 

The technique is equivalent to the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) method commonly used for training artificial neural networks in machine learning, which is essentially just a special case of reverse AD. You can see an implementation of the backpropagation algorithm, making use of the reverse AD trace, in the [neural networks](http://en.wikipedia.org/wiki/Backpropagation) example.

For example, consider the computation

$$$
  e = (\sin a) (a + b) \; ,

using the values $a = 0.5$ and $b = 1.2$.

During the execution of a program, this computation is carried out by the sequence of operations

$$$
 \begin{eqnarray*}
 a &=& 0.5 \; ,\\
 b &=& 1.2 \; , \\
 c &=& \sin a \; , \\
 d &=& a + b \; , \\
 e &=& c \times d \; ,
 \end{eqnarray*}

the dependencies between which can be represented by the graph below.

<div class="row">
    <div class="span6 offset2">
        <img src="img/gettingstarted-reversead-graph.png" alt="Chart" style="width:350px;"/>
    </div>
</div>
*)

(**

Reverse AD works by propagating adjoint values from the output (e.g. $\bar{e} = \frac{\partial e}{\partial e}$) towards the inputs (e.g. $\bar{a} = \frac{\partial e}{\partial a}$ and $\bar{b} = \frac{\partial e}{\partial b}$), using adjoint propagation rules derived from the computational graph:

$$$
 \begin{eqnarray*}
 \bar{d} &=& \frac{\partial e}{\partial d} &=& \frac{\partial e}{\partial e} \frac{\partial e}{\partial d} &=& \bar{e} c\; , \\
 \bar{c} &=& \frac{\partial e}{\partial c} &=& \frac{\partial e}{\partial e} \frac{\partial e}{\partial c} &=& \bar{e} d\; , \\
 \bar{b} &=& \frac{\partial e}{\partial b} &=& \frac{\partial e}{\partial d} \frac{\partial d}{\partial b} &=& \bar{d} \; , \\
 \bar{a} &=& \frac{\partial e}{\partial a} &=& \frac{\partial e}{\partial c} \frac{\partial c}{\partial a} + \frac{\partial e}{\partial d} \frac{\partial d}{\partial a} &=& \bar{c} (\cos a) + \bar{d} \; .\\
 \end{eqnarray*}
  
Using the DiffSharp library, we get access to these values as follows.
*)

// Reset the trace
Trace.Clear()

// Perform a series of operations involving Adj type
let a = adj 0.5
let b = adj 1.2
let e = (sin a) * (a + b)

// Set the adjoint value of z to 1 (dz/dz = 1)
// i.e. calculate partial derivatives of z with respect to other variables
e.A <- 1.

// Propagate the adjoint values in reverse
Trace.ReverseSweep()

// Read the adjoint values of the inputs
// You can calculate all partial derivatives in just one reverse sweep!
let deda = a.A
let dedb = b.A

(*** hide, define-output: o ***)
printf "val a : Adj = Adj(0.5, 1.971315894)\nval b : Adj = Adj(1.2, 0.4794255386)\nval e : Adj = Adj(0.8150234156, 1.0)\nval deda : float = 1.971315894\nval dedb : float = 0.4794255386"
(*** include-output: o ***)

(** 
In addition to the partial derivatives of $e$ with respect to the independent variables $a$ and $b$, you can also extract the partial derivatives of $e$ with respect to any intermediate variable involved in this computation.
*)

//Reset the trace
Trace.Clear()

let a' = adj 0.5
let b' = adj 1.2
let c' = sin a'
let d' = a' + b'
let e' = c' * d' // e' = (sin a') * (a' + b')

// Set the adjoint of e'
e'.A <- 1.

// Propagate the adjoint values in reverse
Trace.ReverseSweep()

// Read the adjoint values
// You can calculate all partial derivatives in just one reverse sweep!
let de'da' = a'.A
let de'db' = b'.A
let de'dc' = c'.A
let de'dd' = d'.A

(*** hide, define-output: o2 ***)
printf "val a' : Adj = Adj(0.5, 1.971315894)\nval b' : Adj = Adj(1.2, 0.4794255386)\nval c' : Adj = Adj(0.4794255386, 1.7)\nval d' : Adj = Adj(1.7, 0.4794255386)\nval e' : Adj = Adj(0.8150234156, 1.0)\nval de'da' : float = 1.971315894\nval de'db' : float = 0.4794255386\nval de'dc' : float = 1.7\nval de'dd' : float = 0.4794255386"
(*** include-output: o2 ***)
