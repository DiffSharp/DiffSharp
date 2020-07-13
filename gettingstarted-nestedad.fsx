(**
Nested AD
=========

The main functionality of DiffSharp is found under the **DiffSharp.AD** namespace. Opening this namespace allows you to automatically evaluate derivatives of functions via forward and/or reverse AD. Internally, for any case involving a function $f: \mathbb{R}^n \to \mathbb{R}^m$, DiffSharp uses forward AD when $n \ll m$ and reverse AD when $n \gg m$. Combinations such as reverse-on-forward or forward-on-reverse AD can be also handled.

For a complete list of the available differentiation operations, please refer to [API Overview](api-overview.html) and [API Reference](reference/index.html).

Background
----------

The library supports nested invocations of differentiation operations. So, for example, you can compute exact higher-order derivatives or take derivatives of functions that are themselves internally computing derivatives. 
*)
open DiffSharp.AD.Float64

let y x = sin (sqrt x)

// Derivative of y
let d1 = diff y

// 2nd derivative of y
let d2 = diff (diff y)

// 3rd derivative of y
let d3 = diff d2
(**
Nesting capability means more than just being able to compute higher-order derivatives of one function.

DiffSharp can handle complex nested cases such as computing the derivative of a function $f$ that takes an argument $x$, which, in turn, computes the derivative of another function $g$ nested inside $f$ that has a free reference to $x$, the argument to the surrounding function.

egin{equation}
  \frac{d}{dx} \left. \left( x \left( \left. \frac{d}{dy} x y \; \right|_{y=3} \right) \right) \right|_{x=2}
\end{equation}
*)
let d4 = diff (fun x -> x * (diff (fun y -> x * y) (D 3.))) (D 2.)(* output: 
val d4 : D = D 4.0*)
(**
This allows you to write, for example, nested optimization algorithms of the form

egin{equation}
  \mathbf{min} \left( \lambda x \; . \; (f \; x) + \mathbf{min} \left( \lambda y \; . \; g \; x \; y \right) \right)\; ,
\end{equation}

for functions $f$ and $g$ and a gradient-based minimization procedure $\mathbf{min}$.

Correctly nesting AD in a functional framework is achieved through the method of "tagging", which serves to prevent a class of bugs called "perturbation confusion" where a system fails to distinguish between distinct perturbations introduced by distinct invocations of differentiation operations. You can refer to the following articles, among others, to understand the issue and how it should be handled correctly:


_Jeffrey Mark Siskind and Barak A. Pearlmutter. Perturbation Confusion and Referential Transparency: Correct Functional Implementation of Forward-Mode AD. In Proceedings of the 17th International Workshop on Implementation and Application of Functional Languages (IFL2005), Dublin, Ireland, Sep. 19-21, 2005._

_Jeffrey Mark Siskind and Barak A. Pearlmutter. Nesting forward-mode AD in a functional framework. Higher Order and Symbolic Computation 21(4):361-76, 2008. [doi:10.1007/s10990-008-9037-1](https://dx.doi.org/10.1007/s10990-008-9037-1) _

_Barak A. Pearlmutter and Jeffrey Mark Siskind. Reverse-Mode AD in a functional framework: Lambda the ultimate backpropagator. TOPLAS 30(2):1-36, Mar. 2008. [doi:10.1145/1330017.1330018](https://dx.doi.org/10.1145/1330017.1330018) _

Forward and Reverse AD Operations
---------------------------------

DiffSharp automatically selects forward or reverse AD, or a combination of these, for a given operation.

The following are just a small selection of operations.

*)
open DiffSharp.AD.Float64

// f: D -> D
let f (x:D) = sin (3. * sqrt x)

// Derivative of f at 2
// Uses forward AD
let df = diff f (D 2.)

// g: DV -> D
let g (x:DV) = sin (x.[0] * x.[1])

// Directional derivative of g at (2, 3) with direction (4, 1)
// Uses forward AD
let ddg = gradv g (toDV [2.; 3.]) (toDV [4.; 1.])

// Gradient of g at (2, 3)
// Uses reverse AD
let gg = grad g (toDV [2.; 3.])

// Hessian-vector product of g at (2, 3) with vector (4, 1)
// Uses reverse-on-forward AD
let hvg = hessianv g (toDV [2.; 3.]) (toDV [4.; 1.])

// Hessian of g at (2, 3)
// Uses reverse-on-forward AD
let hg = hessian g (toDV [2.; 3.])

// h: DV -> DV
let h (x:DV) = toDV [sin x.[0]; cos x.[1]]

// Jacobian-vector product of h at (2, 3) with vector (4, 1)
// Uses forward AD
let jvh = jacobianv h (toDV [2.; 3.]) (toDV [4.; 1.])

// Transposed Jacobian-vector product of h at (2, 3) with vector (4, 1)
// Uses reverse AD
let tjvh = jacobianTv h (toDV [2.; 3.]) (toDV [4.; 1.])

// Jacobian of h at (2, 3)
// Uses forward or reverse AD depending on the number of inputs and outputs
let jh = jacobian h (toDV [2.; 3.])
(**
Using the Reverse AD Trace
--------------------------

In addition to the high-level differentiation API that uses reverse AD (such as **grad**, **jacobianTv** ), you can make use of the exposed low-level [trace](https://en.wikipedia.org/wiki/Tracing_%28software%29) functionality. Reverse AD automatically builds a global trace (or "tape", in AD literature) of all executed numeric operations, which allows a subsequent reverse sweep of these operations for propagating adjoint values in reverse. 

The technique is equivalent to the [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) method commonly used for training artificial neural networks, which is essentially just a special case of reverse AD. (You can see an implementation of the backpropagation algorithm using reverse AD in the [neural networks example](examples-neuralnetworks.html).)

For example, consider the computation

egin{equation}
  e = (\sin a) (a + b) \; ,
\end{equation}

using the values $a = 0.5$ and $b = 1.2$.

During the execution of a program, this computation is carried out by the sequence of operations

egin{equation}
 \begin{eqnarray*}
 a &=& 0.5 \; ,\\
 b &=& 1.2 \; , \\
 c &=& \sin a \; , \\
 d &=& a + b \; , \\
 e &=& c \times d \; ,
 \end{eqnarray*}
\end{equation}

the dependencies between which can be represented by the computational graph below.

<div class="row">
<div class="row">
    <div class="span6 offset2">
        <img src="img/gettingstarted-reversead-graph.png" alt="Chart" style="width:350px;"/>
    </div>
</div>
Reverse mode AD works by propagating adjoint values from the output (e.g. $\bar{e} = \frac{\partial e}{\partial e}$) towards the inputs (e.g. $\bar{a} = \frac{\partial e}{\partial a}$ and $\bar{b} = \frac{\partial e}{\partial b}$), using adjoint propagation rules dictated by the dependencies in the computational graph:

egin{equation}
 \begin{eqnarray*}
 \bar{d} &=& \frac{\partial e}{\partial d} &=& \frac{\partial e}{\partial e} \frac{\partial e}{\partial d} &=& \bar{e} c\; , \\
 \bar{c} &=& \frac{\partial e}{\partial c} &=& \frac{\partial e}{\partial e} \frac{\partial e}{\partial c} &=& \bar{e} d\; , \\
 \bar{b} &=& \frac{\partial e}{\partial b} &=& \frac{\partial e}{\partial d} \frac{\partial d}{\partial b} &=& \bar{d} \; , \\
 \bar{a} &=& \frac{\partial e}{\partial a} &=& \frac{\partial e}{\partial c} \frac{\partial c}{\partial a} + \frac{\partial e}{\partial d} \frac{\partial d}{\partial a} &=& \bar{c} (\cos a) + \bar{d} \; .\\
 \end{eqnarray*}
\end{equation}
  
In order to write code using low-level AD functionality, you should understand the tagging method used for avoiding perturbation confusion. Users would need to write such code sporadically. The normal way of interacting with the library is through the high-level differentiation API, which handles these issues internally.

You can get access to adjoints as follows.
*)
open DiffSharp.AD.Float64

// Get a fresh global tag for this run of reverse AD
let i = DiffSharp.Util.GlobalTagger.Next

// Initialize input values for reverse AD
let a = D 0.5 |> makeReverse i
let b = D 1.2 |> makeReverse i

// Perform a series of operations involving the D type
let e = (sin a) * (a + b)

// Propagate the adjoint value of 1 backward from e (or de/de = 1)
// i.e., calculate partial derivatives of e with respect to other variables
let adjoints = e |> computeAdjoints

// Read the adjoint values of the inputs
// You can calculate all partial derivatives in just one reverse sweep!
let deda = adjoints.[a]
let dedb = adjoints.[b](* output: 
val a : D = DR (D 0.5,Noop,219u)
val b : D = DR (D 1.2,Noop,219u)
val e : D =
  DR
    (D 0.8150234156,
     Mul_D_D
       (DR (D 0.4794255386,Sin_D (DR (D 0.5,Noop,219u)),219u),
        DR (D 1.7,Add_D_D (DR (D 0.5,Noop,219u),DR (D 1.2,Noop,219u)),219u)),
     219u)
val adjoints : Adjoints
val deda : D = D 1.971315894
val dedb : D = D 0.4794255386*)
(**
In addition to the partial derivatives of the dependent variable $e$ with respect to the independent variables $a$ and $b$, you can also extract the partial derivatives of $e$ with respect to any intermediate variable involved in this computation.
*)
// Get a fresh global tag for this run of reverse AD
let i' = DiffSharp.Util.GlobalTagger.Next

// Initialize input values for reverse AD
let a' = D 0.5 |> makeReverse i'
let b' = D 1.2 |> makeReverse i'

// Perform a series of operations involving the D type
let c' = sin a'
let d' = a' + b'
let e' = c' * d' // e' = (sin a') * (a' + b')

// Propagate the adjoint value of 1 backward from e
let adjoints' = e' |> computeAdjoints

// Read the adjoint values
// You can calculate all partial derivatives in just one reverse sweep!
let de'da' = adjoints'.[a']
let de'db' = adjoints'.[b']
let de'dc' = adjoints'.[c']
let de'dd' = adjoints'.[d'](* output: 
val a' : D = DR (D 0.5,Noop,221u)
val b' : D = DR (D 1.2,Noop,221u)
val c' : D = DR (D 0.4794255386,Sin_D (DR (D 0.5,Noop,221u)),221u)
val d' : D =
  DR (D 1.7,Add_D_D (DR (D 0.5,Noop,221u),DR (D 1.2,Noop,221u)),221u)
val e' : D =
  DR
    (D 0.8150234156,
     Mul_D_D
       (DR (D 0.4794255386,Sin_D (DR (D 0.5,Noop,221u)),221u),
        DR (D 1.7,Add_D_D (DR (D 0.5,Noop,221u),DR (D 1.2,Noop,221u)),221u)),
     221u)
val adjoints' : Adjoints
val de'da' : D = D 1.971315894
val de'db' : D = D 0.4794255386
val de'dc' : D = D 1.7
val de'dd' : D = D 0.4794255386*)
