(*** hide ***)
#r "../../src/DiffSharp.Core/bin/Debug/netstandard2.1/DiffSharp.Core.dll"
#r "../../src/DiffSharp.Backends.Reference/bin/Debug/netstandard2.1/DiffSharp.Backends.Reference.dll"

//#load "../../packages/FSharp.Charting/FSharp.Charting.fsx"
#load "helpers.fsx"

(**
Stochastic Gradient Descent
===========================

[Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) is a [stochastic](https://en.wikipedia.org/wiki/Stochastic) variant of the gradient descent algorithm that is used for minimizing loss functions with the form of a sum

$$$
  Q(\mathbf{w}) = \sum_{i=1}^{d} Q_i(\mathbf{w}) \; ,

where $\mathbf{w}$ is a "weight" vector that is being optimized. The component $Q_i$ is the contribution of the $i$-th sample to the overall loss $Q$, which is to be minimized using a training set of $d$ samples.

Using the standard gradient descent algorithm, $Q$ can be minimized by the iteration

$$$
  \begin{eqnarray*}
  \mathbf{w}_{t+1} &=& \mathbf{w}_t - \eta \nabla Q \\
   &=& \mathbf{w}_t - \eta \sum_{i=1}^{d} \nabla Q_i(\mathbf{w}_t) \; ,\\
  \end{eqnarray*}

where $\eta > 0$ is a step size. This "batch" update rule has to compute the full loss in each step, the evaluation time of which is proportional to the size of the training set $d$.

Alternatively, in stochastic gradient descent, $Q$ is minimized using

$$$
  \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla Q_i (\mathbf{w}_t) \; ,

updating the weights $\mathbf{w}$ in each step using just one sample $i$ randomly chosen from the training set. This is advantageous for big sample sizes, because it makes the evaluation time of each step independent from $d$. Another advantage is that it can process samples on the fly, in an [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) task.

In practice, instead of $\eta$, the algorithm is used with a decreasing sequence of step sizes $\eta_t$, for convergence.

Let's implement stochastic gradient descent with constant step size.

*)

open DiffSharp

let rnd = new System.Random()

// Stochastic gradient descent
// f: function, w0: starting weights, eta: step size, epsilon: threshold, t: training set
let sgd f w0 (eta:Scalar) epsilon (t:(Vec*Vec)[]) =
    let rec desc w =
        let x, y = t.[rnd.Next(t.Length)]
        let g = dsharp.grad (fun wi -> Vec.l2norm (y - (f wi x))) w
        if Vec.l2norm g < epsilon then w else desc (w - eta * g)
    desc w0

(**

In this implementation $Q_i$ has the form

$$$
  Q_i(\mathbf{w}) = \left\Vert \mathbf{y}_i - f_{\mathbf{w}} (\mathbf{x}_i) \right\Vert \; ,

where $f_{\mathbf{w}} : \mathbb{R}^n \to \mathbb{R}^m$ is a model function for our data (parameterized by $\mathbf{w}$) and $\mathbf{x}_i \in \mathbb{R}^n$ and $\mathbf{y}_i \in \mathbb{R}^m$ are the input–output pair of the $i$-th sample in the training set. Finding the parameters $\mathbf{w}$ minimizing $Q(\mathbf{w}) = \sum_{i=1}^{d} Q_i (\mathbf{w})$ thus fits the model function $f_{\mathbf{w}}$ to our data.

We can test this via [fitting a curve](https://en.wikipedia.org/wiki/Curve_fitting)

$$$
  f_{\mathbf{w}} (x) = w_1 x^2 + w_2 x + w_3

to the points $(0.5, 2), (3.2, 1), (5.2, 4)$.

*)

// Model function
let inline f (w:Vec) (x:Vec) =
    w.[0] * x.[0] * x.[0] + w.[1] * x.[0] + w.[2] 

// Points
let points = [|0.5, 2.
               3.2, 1.
               5.2, 4.|]

// Construct training set using the points
let train = Array.map (fun x -> (vec [fst x]), (vec [snd x])) points

// Find w minimizing the error of fit
let wopt = sgd f (vec [0.; 0.; 0.]) (v 0.0001) (v 0.01) train

(*** hide, define-output: o ***)
printf "val wopt : Vec = Vec [|0.3854891266; -1.761000211; 2.731836811|]"
(*** include-output: o ***)

(**
 
We can plot the points in the training set and the fitted curve.
   
*)

//open FSharp.Charting

//Chart.Combine([Chart.Line([for x in 0. .. 0.1 .. 6. -> (x, float <|f wopt (vec [x]))])
//               Chart.Point(points, MarkerSize = 10)])

(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-stochasticgradientdescent.png" alt="Chart" style="width:550px"/>
    </div>
</div>

*)