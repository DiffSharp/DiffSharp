(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting/FSharp.Charting.fsx"

(**
Helmholtz Energy Function
=========================

The following formula, giving the [Helmholtz free energy](https://en.wikipedia.org/wiki/Helmholtz_free_energy) of a mixed fluid based on the [Peng-Robinson equation of state](https://en.wikipedia.org/wiki/Equation_of_state#Peng-Robinson_equation_of_state), has been used in automatic differentiation literature for benchmarking gradient calculations:

$$$
 f(\mathbf{x}) = R \, T \sum_{i = 0}^{n} x_i \log \frac{x_i}{1 - \mathbf{b^T} \mathbf{x}} - \frac{\mathbf{x^T} \mathbf{A} \mathbf{x}}{\sqrt{8} \mathbf{b^T} \mathbf{x}} \log \frac{1 + (1 + \sqrt{2}) \mathbf{b^T} \mathbf{x}}{1 + (1 - \sqrt{2}) \mathbf{b^T} \mathbf{x}} \; ,

where $R$ is the universal gas constant, $T$ is the absolute temperature, $\mathbf{b} \in \mathbb{R}^n$ is a vector of constants, $\mathbf{A} \in \mathbb{R}^{n \times n}$ is a symmetric matrix of constants, and $\mathbf{x} \in \mathbb{R}^n$ is the vector of independent variables describing the system.

In practice, gradients of formulae such as this need to be evaluated at thousands of points for the purposes of phase equilibrium calculations, stability analysis, and energy density calculations of mixed fluids.

Let's compute the gradient of this function with DiffSharp. $f: \mathbb{R}^n \to \mathbb{R}$ being a scalar valued function of many variables, this is an ideal case for reverse mode AD, which needs only one forward and one reverse evaluation of $f$ to compute all partial derivatives ${\partial f}/{\partial x_i}$.
*)

open DiffSharp.AD.Float64

let rnd = System.Random()

let helmholtz R T (b:DV) (A:DM) (x:DV) =
    let bx = b * x
    let oneminbx = 1. - bx
    R * T * DV.sum(x .* log (x / oneminbx))
    - ((x * A * x) / (bx * sqrt 8.)) 
    * log ((1. + (1. + sqrt 2.) * bx) / (1. + (1. - sqrt 2.) * bx))

// Compute the Helmholtz function, n dimensions
let testHelmholtz n =
    let R = 1.
    let T = 1.
    let b = DV.init n (fun _ -> 0.1 * rnd.NextDouble())
    let A = DM.init n n (fun _ _ -> 0.1 * rnd.NextDouble())
    let x = DV.init n (fun _ -> 0.1 * rnd.NextDouble())
    helmholtz R T b A x

// Compute the gradient of the Helmholtz function, n dimensions
let testHelmholtzGrad n =
    let R = 1.
    let T = 1.
    let b = DV.init n (fun _ -> 0.1 * rnd.NextDouble())
    let A = DM.init n n (fun _ _ -> 0.1 * rnd.NextDouble())
    let x = DV.init n (fun _ -> 0.1 * rnd.NextDouble())
    grad (helmholtz R T b A) x

// Compute the Helmholtz function, 100 dimensions
let h = testHelmholtz 100

// Compute the Helmholtz gradient, 100 dimensions
let hg = testHelmholtzGrad 100

(**

<table class="pre"><tr><td><pre><code>val h : D = D -13.75993246
val hg : DV =
  DV
    [|-1.4842828; -1.282739032; -1.667366526; -1.287808084; -1.571103144;
      -1.594745918; -2.084607212; -1.995920314; -2.207931642; -1.297893157;
      -1.018172854; -1.50501561; -2.484189159; -2.359668278; -2.098546078;
      -1.862580008; -2.473887617; -0.8838544933; -2.460401099; -2.747744062;
      -2.289848228; -1.220797233; -2.379743703; -1.376533519; -2.200374723;
      -1.504920002; -1.684602781; -3.657835076; -1.302331671; -1.420049031;
      -7.290263813; -1.716436424; -1.463548812; -1.453457075; -2.641135289;
      -1.143515454; -1.992469935; -1.686939101; -3.866042781; -1.393953931;
      -1.654542729; -1.34234604; -1.920989005; -1.943331378; -1.243882459;
      -1.967065692; -3.24722634; -1.253789246; -2.045403465; -2.521874965;
      -1.55959586; -2.230446245; -2.482885653; -0.9426452727; -3.109204376;
      -0.9193115501; -1.937010989; -0.9204217881; -2.294830477; -4.213303259;
      -1.295249675; -2.640273197; -1.24467688; -1.293075034; -1.399349403;
      -2.575902894; -1.486756148; -2.171570365; -5.182173734; -1.481108708;
      -1.249585009; -0.9841810701; -1.62198927; -2.124936697; -1.68330899;
      -1.95914448; -2.681662583; -1.659985765; -3.699546333; -0.9646666506;
      -1.755901163; -1.009102119; -1.300431176; -2.238218078; -1.210907907;
      -1.298877971; -1.988219529; -6.103086998; -1.277308334; -2.202099136;
      -2.575852458; -1.517786017; -3.431578846; -4.863001764; -4.068044673;
      -1.817614786; -1.509747264; -1.213446903; -1.157966538; -1.912585298|]
</code></pre></td></tr></table>

We can investigate how the time needed to compute the Helmholtz gradient scales with the number of independent variables $n$, normalized by the time it takes to compute the original Helmholtz function.

First we define a function for measuring the time spent during the evaluation of a function, averaged over a number of runs to smooth out inconsistencies.
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

Now we can run the experiment.

*)

open FSharp.Charting

Chart.Line([for n in 1..25..500 -> n, (duration 1000 (fun _ -> testHelmholtzGrad n)) 
                                        / (duration 1000 (fun _ -> testHelmholtz n))])
          .WithXAxis(Title = "n (num. variables)", Min = 0.)
          .WithYAxis(Title = "Time factor")

(**

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-helmholtzenergyfunction-chart1.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Computing derivatives with AD has complexity guarantees, where derivative evaluations introduce only a small constant factor of overhead.

In general, for a function $f: \mathbb{R}^n \to \mathbb{R}^m$, if we denote the operation count to evaluate the original function by $\textrm{ops}(f)$, we need $n \, c_f \, \textrm{ops}(f)$ operations to evaluate the full Jacobian $\mathbf{J} \in \mathbb{R}^{m \times n}$ with forward mode AD. The same computation can be done with reverse mode AD in $m \, c_r \, \textrm{ops}(f)$, where $c_r$ is a constant guaranteed to be $c_r < 6$ and typically $c_r \sim [2, 3]$.
*)
