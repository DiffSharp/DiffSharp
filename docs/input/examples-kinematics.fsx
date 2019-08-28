(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting/FSharp.Charting.fsx"

(**
Kinematics
==========

Let us use the DiffSharp library for describing the simple [kinematics](https://en.wikipedia.org/wiki/Kinematics) of a point particle moving in one dimension.

Take the function $ x(t) = t^3 - 6 t^2 + 10t $, giving the position $x$ of a particle at time $t$.
*)

// Position x(t)
let x t = t * t * t - 6. * t * t + 10. * t

(**
Plot $x(t)$ between $t=0$ and $t=4$.
*)

open FSharp.Charting

// Plot x(t) between t = 0 and t = 4
Chart.Line([for t in 0.0..0.01..4.0 -> (t, x t)]).WithXAxis(Title="t").WithYAxis(Title="x")


(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-kinematics-plot1.png" alt="Chart" style="width:550px"/>
    </div>
</div>

We can calculate the position $x(t)$, the velocity $v(t)=\frac{\partial x(t)}{\partial t}$, and the acceleration $a(t)=\frac{\partial ^ 2 x(t)}{\partial t ^ 2}$ of the particle at the same time, using the **diff2''** operation that returns the original value, the first derivative, and the second derivative of a given function.
*)

open DiffSharp.AD.Float64

// diff2'' returns the tuple (original value, first derivative, second derivative)
let xva = D >> diff2'' (fun t -> t * t * t - 6 * t * t + 10 * t)

(**
The following gives us a combined plot of $x(t)$, $v(t)$, and $a(t)$.
*)

// Functions for extracting the position, velocity, acceleration values from a 3-tuple
let pos (x, _, _) = x |> float
let vel (_, v, _) = v |> float
let acc (_, _, a) = a |> float

// Draw x(t), v(t), and a(t) between t = 0 and t = 4
Chart.Combine([Chart.Line([for t in 0.0..0.01..4.0 -> (t, pos (xva t))], Name="x(t)")
               Chart.Line([for t in 0.0..0.01..4.0 -> (t, vel (xva t))], Name="v(t)")
               Chart.Line([for t in 0.0..0.01..4.0 -> (t, acc (xva t))], Name="a(t)")])
               .WithLegend().WithXAxis(Title = "t")

(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-kinematics-plot2.png" alt="Chart" style="width:550px"/>
    </div>
</div>
*)
