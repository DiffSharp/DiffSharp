(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.10/FSharp.Charting.fsx"

(**
Stochastic Gradient Descent
===========================

[Stochastic gradient descent](http://en.wikipedia.org/wiki/Stochastic_gradient_descent) is a [stochastic](http://en.wikipedia.org/wiki/Stochastic) variant of the gradient descent algorithm that is used for minimizing objective functions with the form of a sum

$$$
  Q(\mathbf{w}) = \sum_{i=1}^{d} Q_i(\mathbf{w}) \; ,

where $\mathbf{w}$ is a weight vector parametrizing $Q$. The component $Q_i$ is the contribution of the $i$-th sample to the objective function $Q$, which is to be minimized using a training set of $n$ samples.

Using the standard gradient descent algorithm, $Q$ can be minimized by the iteration

$$$
  \begin{eqnarray*}
  \mathbf{w}_{t+1} &=& \mathbf{w}_t - \eta \nabla Q \\
   &=& \mathbf{w}_t - \eta \sum_{i=1}^{d} \nabla Q_i(\mathbf{w}) \; ,\\
  \end{eqnarray*}

where $\eta > 0$ is a step size. This "batch" update rule has to compute the full objective function in each step, the evaluation time of which is proportional to the size of the training set $d$.

Alternatively, in stochastic gradient descent, $Q$ is minimized using

$$$
  \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla Q_i (\mathbf{w}) \; ,

updating the weights $\mathbf{w}$ in each step using just one sample $i$ randomly chosen from the training set. This is advantageous for big sample sizes, because it makes the evaluation time of each step independent from $d$. Another advantage is that it can process samples on the fly, in an [online learning](http://en.wikipedia.org/wiki/Online_machine_learning) task.

In practice, instead of $\eta$, the algorithm is used with a decreasing sequence of step sizes $\eta_t$, updated in each step.

Let us implement stochastic gradient descent with the DiffSharp library, using constant step size.

*)

open DiffSharp.AD
open DiffSharp.AD.Vector
open FsAlg.Generic

let rnd = new System.Random()

// Stochastic gradient descent
// f: function, w0: starting weights, eta: step size, epsilon: threshold, t: training set
let sgd f w0 (eta:D) epsilon (t:(Vector<float>*Vector<float>)[]) =
    let ta = Array.map (fun (x, y) -> Vector.map D x, Vector.map D y) t
    let rec desc w =
        let x, y = ta.[rnd.Next(ta.Length)]
        let g = grad (fun wi -> Vector.normSq (y - (f wi x))) w
        if Vector.normSq g < epsilon then w else desc (w - eta * g)
    desc w0

(**

In this implementation $Q_i$ has the form

$$$
  Q_i(\mathbf{w}) = \left( \mathbf{y}_i - f_{\mathbf{w}} (\mathbf{x}_i) \right)^2 \; ,

where $f_{\mathbf{w}} : \mathbb{R}^n \to \mathbb{R}^m$ is a model function for our data (parametrized by $\mathbf{w}$) and $\mathbf{x}_i \in \mathbb{R}^n$ and $\mathbf{y}_i \in \mathbb{R}^m$ are the input and the expected output of the $i$-th sample in the training set. Finding the parameters $\mathbf{w}$ minimizing $Q(\mathbf{w}) = \sum_{i=1}^{d} Q_i (\mathbf{w})$ thus fits the model function $f_{\mathbf{w}}$ to our data.

We can test this via [fitting a curve](http://en.wikipedia.org/wiki/Curve_fitting)

$$$
  f_{\mathbf{w}} (x) = w_1 x^2 + w_2 x + w_3

to the points $(0.5, 2), (3.2, 1), (5.2, 4)$.

*)

// Model function
let inline f (w:Vector<_>) (x:Vector<_>) =
    w.[0] * x.[0] * x.[0] + w.[1] * x.[0] + w.[2] 

// Points
let points = [|0.5, 2.
               3.2, 1.
               5.2, 4.|]

// Construct training set using the points
let train = Array.map (fun x -> (vector [fst x]), (vector [snd x])) points

// Find w minimizing the error of fit
let wopt = sgd f (vector [D 0.; D 0.; D 0.]) (D 0.0001) (D 0.01) train |> Vector.map float

(*** hide, define-output: o ***)
printf "val wopt : Vector<float> = Vector [|0.3874125148; -1.77368708; 2.745850698|]"
(*** include-output: o ***)

(**
 
We can plot the points in the training set and the fitted curve.
   
*)

open FSharp.Charting

Chart.Combine([Chart.Line([for x in 0. .. 0.1 .. 6. -> (x, f wopt (vector [x]))])
               Chart.Point(points, MarkerSize = 10)])

(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-stochasticgradientdescent.png" alt="Chart" style="width:550px"/>
    </div>
</div>

*)