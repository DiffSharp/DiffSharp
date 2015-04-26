(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"


(**
Nested AD
=========

The main functionality of DiffSharp is found under the **DiffSharp.AD** namespace. Opening this namespace allows you to automatically evaluate derivatives of code via forward and/or reverse AD.

For a complete list of the available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

Background
----------

The library supports nested invocations of differentiation operations. So, for example, you can compute higher-order derivatives or take derivatives of functions that are themselves internally computing derivatives. 
*)

open DiffSharp.AD

let y x = sin (sqrt x)

// Derivative of y
let d1 = diff y

// 2nd derivative of y
let d2 = diff (diff y)

// 3rd derivative of y
let d3 = diff d2

(**

Moreover, DiffSharp can handle cases such as computing the derivative of a function $f$ that takes an argument $x$, which, in turn, computes the derivative of another function $g$ nested inside $f$ that has a free reference to $x$, the argument to the surrounding function.

$$$
  \frac{d}{dx} \left. \left( x \left( \left. \frac{d}{dy} x y \; \right|_{y=3} \right) \right) \right|_{x=2}
*)

let d4 = diff (fun x -> x * (diff (fun y -> x * y) (D 3.))) (D 2.)

(*** hide, define-output: o ***)
printf "val d4 : D = D 4.0"
(*** include-output: o ***)

(**

This allows you to write, for example, nested optimization algorithms of the form

$$$
  \mathbf{min} \left( \lambda x \; . \; (f \; x) + \mathbf{min} \left( \lambda y \; . \; g \; x \; y \right) \right)\; ,

for functions $f$ and $g$ and a gradient-based minimization procedure $\mathbf{min}$.

Correctly nesting AD in a functional framework is achieved through the method of "tagging", which serves to prevent a class of bugs called "perturbation confusion" where a system fails to distinguish between distinct perturbations introduced by distinct invocations of differentiation operations. You can refer to the following articles, among others, to understand the bug and its solution:


_Jeffrey Mark Siskind and Barak A. Pearlmutter. Perturbation Confusion and Referential Transparency: Correct Functional Implementation of Forward-Mode AD. In Proceedings of the 17th International Workshop on Implementation and Application of Functional Languages (IFL2005), Dublin, Ireland, Sep. 19-21, 2005._

_Jeffrey Mark Siskind and Barak A. Pearlmutter. Nesting forward-mode AD in a functional framework. Higher Order and Symbolic Computation 21(4):361-76, 2008. [doi:10.1007/s10990-008-9037-1](http://dx.doi.org/10.1007/s10990-008-9037-1) _

_Barak A. Pearlmutter and Jeffrey Mark Siskind. Reverse-Mode AD in a functional framework: Lambda the ultimate backpropagator. TOPLAS 30(2):1-36, Mar. 2008. [doi:10.1145/1330017.1330018](http://dx.doi.org/10.1145/1330017.1330018) _

Forward and Reverse AD Operations
---------------------------------

DiffSharp automatically selects forward or reverse AD, or any combination of these, for a given operation. (If you need, you can force only forward or only reverse AD by using **DiffSharp.AD.Forward** or **DiffSharp.AD.Reverse**.)

The following are just a small selection of operations.

*)

open DiffSharp.AD

// f: D -> D
let f (x:D) = sin (3. * sqrt x)

// Derivative of f at 2
// Uses forward AD
let df = diff f (D 2.)

// g: D[] -> D
let g (x:D[]) = sin (x.[0] * x.[1])

// Directional derivative of g at (2, 3) with direction (4, 1)
// Uses forward AD
let ddg = gradv g [|D 2.; D 3.|] [|D 4.; D 1.|]

// Gradient of g at (2, 3)
// Uses reverse AD
let gg = grad g [|D 2.; D 3.|]

// Hessian-vector product of g at (2, 3) with vector (4, 1)
// Uses reverse-on-forward AD
let hvg = hessianv g [|D 2.; D 3.|] [|D 4.; D 1.|]

// Hessian of g at (2, 3)
// Uses reverse-on-forward AD
let hg = hessian g [|D 2.; D 3.|]

// h: D[] -> D[]
let h (x:D[]) = [| sin x.[0]; cos x.[1] |]

// Jacobian-vector product of h at (2, 3) with vector (4, 1)
// Uses forward AD
let jvh = jacobianv h [|D 2.; D 3.|] [|D 4.; D 1.|]

// Transposed Jacobian-vector product of h at (2, 3) with vector (4, 1)
// Uses reverse AD
let tjvh = jacobianTv h [|D 2.; D 3.|] [|D 4.; D 1.|]

// Jacobian of h at (2, 3)
// Uses forward or reverse AD depending on the number of inputs and outputs
let jh = jacobian h [|D 2.; D 3.|]

(**
Using the Reverse AD Trace
--------------------------

In addition to the differentiation API that uses reverse AD (such as **grad**, **jacobianTv** ), you can make use of the exposed [trace](http://en.wikipedia.org/wiki/Tracing_%28software%29) functionality. Reverse AD automatically builds a global trace (or "tape", in AD literature) of all executed numeric operations, which subsequently allows a reverse sweep of these operations for propagating adjoint values in reverse. 

The technique is equivalent to the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) method commonly used for training artificial neural networks, which is essentially just a special case of reverse AD. You can see an implementation of the backpropagation algorithm using reverse AD in the [neural networks](examples-neuralnetworks.html) example.

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

the dependencies between which can be represented by the computational graph below.

<div class="row">
    <div class="span6 offset2">
        <img src="img/gettingstarted-reversead-graph.png" alt="Chart" style="width:350px;"/>
    </div>
</div>
*)

(**

Reverse mode AD works by propagating adjoint values from the output (e.g. $\bar{e} = \frac{\partial e}{\partial e}$) towards the inputs (e.g. $\bar{a} = \frac{\partial e}{\partial a}$ and $\bar{b} = \frac{\partial e}{\partial b}$), using adjoint propagation rules dictated by the dependencies in the computational graph:

$$$
 \begin{eqnarray*}
 \bar{d} &=& \frac{\partial e}{\partial d} &=& \frac{\partial e}{\partial e} \frac{\partial e}{\partial d} &=& \bar{e} c\; , \\
 \bar{c} &=& \frac{\partial e}{\partial c} &=& \frac{\partial e}{\partial e} \frac{\partial e}{\partial c} &=& \bar{e} d\; , \\
 \bar{b} &=& \frac{\partial e}{\partial b} &=& \frac{\partial e}{\partial d} \frac{\partial d}{\partial b} &=& \bar{d} \; , \\
 \bar{a} &=& \frac{\partial e}{\partial a} &=& \frac{\partial e}{\partial c} \frac{\partial c}{\partial a} + \frac{\partial e}{\partial d} \frac{\partial d}{\partial a} &=& \bar{c} (\cos a) + \bar{d} \; .\\
 \end{eqnarray*}
  
In order to write code using low-level AD functionality, you should understand the tagging method used for avoiding perturbation confusion. Users would need to write such code sporadically. The normal way of interacting with the library is through the high-level differentiation API, which handles these issues internally.

You can get access to adjoints as follows.
*)

open DiffSharp.AD

// Get a fresh global tag for this run of reverse AD
let i = DiffSharp.Util.GlobalTagger.Next

// Initialize input values for reverse AD
let a = D 0.5 |> makeDR i
let b = D 1.2 |> makeDR i

// Perform a series of operations involving the D type
let e = (sin a) * (a + b)

// Propagate the adjoint value of 1 backward from e (or de/de = 1)
// i.e., calculate partial derivatives of e with respect to other variables
e |> reverseProp (D 1.)

// Read the adjoint values of the inputs
// You can calculate all partial derivatives in just one reverse sweep!
let deda = a.A
let dedb = b.A

(*** hide, define-output: o2 ***)
printf "val a : D =
  DR (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},550202u)
val b : D =
  DR (D 1.2,{contents = D 0.4794255386;},Noop,{contents = 0u;},550202u)
val e : D =
  DR
    (D 0.8150234156,{contents = D 1.0;},
     Mul
       (DR
          (D 0.4794255386,{contents = D 1.7;},
           Sin
             (DR
                (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},
                 550202u)),{contents = 0u;},550202u),
        DR
          (D 1.7,{contents = D 0.4794255386;},
           Add
             (DR
                (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},
                 550202u),
              DR
                (D 1.2,{contents = D 0.4794255386;},Noop,{contents = 0u;},
                 550202u)),{contents = 0u;},550202u)),{contents = 0u;},550202u)
val deda : D = D 1.971315894
val dedb : D = D 0.4794255386"
(*** include-output: o2 ***)

(** 
In addition to the partial derivatives of the dependent variable $e$ with respect to the independent variables $a$ and $b$, you can also extract the partial derivatives of $e$ with respect to any intermediate variable involved in this computation.
*)

// Get a fresh global tag for this run of reverse AD
let i' = DiffSharp.Util.GlobalTagger.Next

// Initialize input values for reverse AD
let a' = D 0.5 |> makeDR i'
let b' = D 1.2 |> makeDR i'

// Perform a series of operations involving the D type
let c' = sin a'
let d' = a' + b'
let e' = c' * d' // e' = (sin a') * (a' + b')

// Propagate the adjoint value of 1 backward from e
e' |> reverseProp (D 1.)

// Read the adjoint values
// You can calculate all partial derivatives in just one reverse sweep!
let de'da' = a'.A
let de'db' = b'.A
let de'dc' = c'.A
let de'dd' = d'.A

(*** hide, define-output: o3 ***)
printf "val a' : D =
  DR (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},550203u)
val b' : D =
  DR (D 1.2,{contents = D 0.4794255386;},Noop,{contents = 0u;},550203u)
val c' : D =
  DR
    (D 0.4794255386,{contents = D 1.7;},
     Sin
       (DR (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},550203u)),
     {contents = 0u;},550203u)
val d' : D =
  DR
    (D 1.7,{contents = D 0.4794255386;},
     Add
       (DR (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},550203u),
        DR (D 1.2,{contents = D 0.4794255386;},Noop,{contents = 0u;},550203u)),
     {contents = 0u;},550203u)
val e' : D =
  DR
    (D 0.8150234156,{contents = D 1.0;},
     Mul
       (DR
          (D 0.4794255386,{contents = D 1.7;},
           Sin
             (DR
                (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},
                 550203u)),{contents = 0u;},550203u),
        DR
          (D 1.7,{contents = D 0.4794255386;},
           Add
             (DR
                (D 0.5,{contents = D 1.971315894;},Noop,{contents = 0u;},
                 550203u),
              DR
                (D 1.2,{contents = D 0.4794255386;},Noop,{contents = 0u;},
                 550203u)),{contents = 0u;},550203u)),{contents = 0u;},550203u)
val de'da' : D = D 1.971315894
val de'db' : D = D 0.4794255386
val de'dc' : D = D 1.7
val de'dd' : D = D 0.4794255386"
(*** include-output: o3 ***)
