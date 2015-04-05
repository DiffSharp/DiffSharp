(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#load "../../packages/FSharp.Charting.0.90.9/FSharp.Charting.fsx"

(**
Helmholtz Energy Function
=========================

The following formula, giving the [Helmholtz free energy](http://en.wikipedia.org/wiki/Helmholtz_free_energy) of a mixed fluid based on the [Peng-Robinson equation of state](http://en.wikipedia.org/wiki/Equation_of_state#Peng-Robinson_equation_of_state), has been used in automatic differentiation literature for benchmarking gradient calculations:

$$$
 f(\mathbf{x}) = R \, T \sum_{i = 0}^{n} \log \frac{x_i}{1 - \mathbf{b^T} \mathbf{x}} - \frac{\mathbf{x^T} \mathbf{A} \mathbf{x}}{\sqrt{8} \mathbf{b^T} \mathbf{x}} \log \frac{1 + (1 + \sqrt{2}) \mathbf{b^T} \mathbf{x}}{1 + (1 - \sqrt{2}) \mathbf{b^T} \mathbf{x}} \; ,

where $R$ is the universal gas constant, $T$ is the absolute temperature, $\mathbf{b} \in \mathbb{R}^n$ is a vector of constants, $\mathbf{A} \in \mathbb{R}^{n \times n}$ is a symmetric matrix of constants, and $\mathbf{x} \in \mathbb{R}^n$ is the vector of independent variables describing the system.

In practice, gradients of formulae such as this need to be evaluated at thousands of points for the purposes of phase equilibrium calculations, stability analysis, and energy density calculations of mixed fluids.

Let us compute the gradient of this function with the **DiffSharp.AD.Reverse** module. $f: \mathbb{R}^n \to \mathbb{R}$ being a scalar valued function of many variables, this is an ideal case for using reverse mode AD, which needs only one forward and one reverse evaluation of $f$ to compute all the partial derivatives $\frac{\partial f}{\partial x_i}$.
*)

open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open FsAlg.Generic

let rnd = System.Random()

let helmholtz R T (b:Vector<Adj>) (A:Matrix<Adj>) (x:Vector<Adj>) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * (Vector.sumBy (fun a -> a * log (a / oneminbx)) x) 
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

// Compute the gradient of the Helmholtz function with n variables
let testHelmholtz n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> adj (0.1 * rnd.NextDouble()))
    let A = Matrix.init n n (fun _ _ -> adj (0.1 * rnd.NextDouble()))
    let x = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    grad (helmholtz R T b A) x

// Compute the gradient with 6 variables
let test = testHelmholtz 6


(**

<table class="pre"><tr><td><pre><code>val test : Vector&lt;float&gt; =
  Vector
    [|-1.349719714; -1.458964954; -1.637555967; -2.891547509; -2.136274086;
      -2.676240035|]
</code></pre></td></tr></table>

We can investigate how the time needed to compute this gradient scales with the number of independent variables $n$.

First we define a simple way of measuring the time spent during the evaluation of a function, averaged over a number of runs to smooth out inconsistencies.

*)

// Measure the time (in miliseconds) spent evaluating function f,
// averaged over n executions
let duration n f =
    let s = new System.Diagnostics.Stopwatch()
    s.Start() |> ignore
    for i in 1..n do
        f() |> ignore
    s.Stop() |> ignore
    let dur = s.ElapsedMilliseconds
    (float dur) / (float n)

(**

Now we can plot the time needed as a function of $n$.

*)

open FSharp.Charting

Chart.Line([for n in 1 .. 50 -> duration 1000 (fun _ -> testHelmholtz n)])
          .WithXAxis(Title = "n (num. variables)", Min = 0.)
          .WithYAxis(Title = "Time (ms)")

(**

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-helmholtzenergyfunction-chart1.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Let us compare this with the performance of numerical differentiation (**DiffSharp.Numerical**) and forward mode AD (**DiffSharp.AD.Forward** and **DiffSharp.AD.ForwardG**).
*)

open DiffSharp.Numerical.Vector

let helmholtzFloat R T (b:Vector<float>) (A:Matrix<float>) (x:Vector<float>) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * (Vector.sumBy (fun a -> a * log (a / oneminbx)) x) 
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

let testHelmholtzFloat n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    let A = Matrix.init n n (fun _ _ -> (0.1 * rnd.NextDouble()))
    let x = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    grad (helmholtzFloat R T b A) x

open DiffSharp.AD.Forward
open DiffSharp.AD.Forward.Vector

let helmholtzDual R T (b:Vector<Dual>) (A:Matrix<Dual>) (x:Vector<Dual>) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * (Vector.sumBy (fun a -> a * log (a / oneminbx)) x) 
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

let testHelmholtzDual n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> dual (0.1 * rnd.NextDouble()))
    let A = Matrix.init n n (fun _ _ -> dual (0.1 * rnd.NextDouble()))
    let x = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    grad (helmholtzDual R T b A) x

open DiffSharp.AD.ForwardG
open DiffSharp.AD.ForwardG.Vector

let helmholtzDualG R T (b:Vector<DualG>) (A:Matrix<DualG>) (x:Vector<DualG>) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * (Vector.sumBy (fun a -> a * log (a / oneminbx)) x) 
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

let testHelmholtzDualG n =
    let R = 1.
    let T = 1.
    let b = Vector.init n (fun _ -> dualG (0.1 * rnd.NextDouble()) n)
    let A = Matrix.init n n (fun _ _ -> dualG (0.1 * rnd.NextDouble()) n)
    let x = Vector.init n (fun _ -> (0.1 * rnd.NextDouble()))
    grad (helmholtzDualG R T b A) x

(**
As shown by the regular and logarithmic plots below, reverse mode AD performs substantially better than forward mode AD and numerical differentiation, as expected. For instance, for $n = 50$, reverse mode AD performs approximately ten times faster than the other methods. Also, simple forward mode AD (**DiffSharp.AD.Forward**) performs worse than vectorized forward mode (**DiffSharp.AD.ForwardG**), which is optimized for functions with many inputs.

In general, for a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$, reverse mode AD will have a performance advantage over forward mode AD when $n \gg m$.

It should also be noted that numerical differentiation is shown here only for illustrating its computational cost. Both forward mode AD and reverse mode AD give the exact value of the gradient up to machine precision (equal to what one would get from symbolic differentiation) whereas numerical differentiation only provides a finite difference approximation with the associated truncation and roundoff errors. Automatic differentiation is superior to numerical differentiation in terms of both accuracy and speed.
*)

Chart.Combine([Chart.Line([for n in 1 .. 50 -> duration 1000 (fun _ -> testHelmholtzFloat n)],
                          Name ="Numerical diff.")
               Chart.Line([for n in 1 .. 50 -> duration 1000 (fun _ -> testHelmholtzDual n)], 
                          Name ="Forward AD")
               Chart.Line([for n in 1 .. 50 -> duration 1000 (fun _ -> testHelmholtzDualG n)], 
                          Name ="ForwardG AD")
               Chart.Line([for n in 1 .. 50 -> duration 1000 (fun _ -> testHelmholtz n)], 
                          Name ="Reverse AD")])
               .WithLegend()
               .WithXAxis(Title = "n (num. variables)", Min = 0.)
               .WithYAxis(Title = "Time (ms)")

(**

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-helmholtzenergyfunction-chart2.png" alt="Chart" style="width:550px"/>
    </div>
</div>
*)

Chart.Combine([Chart.Line([for n in 1..7..50->n,0.000001+duration 1000 (fun _->testHelmholtzFloat n)],
                          Name ="Numerical diff.")
               Chart.Line([for n in 1..7..50->n,0.000001+duration 1000 (fun _->testHelmholtzDual n)], 
                          Name ="Forward AD")
               Chart.Line([for n in 1..7..50->n,0.000001+duration 1000 (fun _->testHelmholtzDualG n)], 
                          Name ="ForwardG AD")
               Chart.Line([for n in 1..7..50->n,0.000001+duration 1000 (fun _->testHelmholtz n)], 
                          Name ="Reverse AD")])
               .WithLegend(Alignment = System.Drawing.StringAlignment.Far)
               .WithXAxis(Title = "n (num. variables)", Min = 0.)
               .WithYAxis(Title = "Log time (ms)", Log = true)

(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-helmholtzenergyfunction-chart2b.png" alt="Chart" style="width:550px"/>
    </div>
</div>
   
*)