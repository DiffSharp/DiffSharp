(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting/FSharp.Charting.fsx"
#load "EventEx-0.1.fsx"

(**
Inverse Kinematics
==================

[Inverse kinematics](https://en.wikipedia.org/wiki/Inverse_kinematics) is a technique in robotics, computer graphics, and animation to find physical configurations of a structure that would put an end-effector in a desired position in space. In other words, it answers the question: "given the desired position of a robot's hand, what should be the angles of all the joints in the robot's body, to take the hand to that position?"

For example, take the system drawn below, attached to a stationary structure on the left. This "arm" has two links of length $L_1$ and $L_2$, two joints at points $(0,0)$ and $(x_1,y_1)$, and the end point at $(x_2,y_2)$. 

<div class="row">
    <div class="span6 offset3">
        <img src="img/examples-inversekinematics.png" alt="Chart" style="width:300px;"/>
    </div>
</div>

It is straightforward to write the equations describing the forward kinematics of this mechanism, providing the coordinates of its parts for given angles of joints $a_1$ and $a_2$.

$$$
 \begin{eqnarray*}
 x_1 &=& L_1 \cos a_1\\
 y_1 &=& L_1 \sin a_1\\
 x_2 &=& x_1 + L_2 \cos (a_1 + a_2)\\
 y_2 &=& y_1 + L_2 \sin (a_1 + a_2)\\
 \end{eqnarray*}

A common approach to the inverse kinematics problem involves the use of Jacobian matrices for linearizing the system describing the position of the end point, in this example, $(x_2,y_2)$. This defines how the position of the end point changes locally, relative to the instantaneous changes in the joint angles.

$$$
 \mathbf{J} = \begin{bmatrix}
              \frac{\partial x_2}{\partial a_1} & \frac{\partial x_2}{\partial a_2} \\
              \frac{\partial y_2}{\partial a_1} & \frac{\partial y_2}{\partial a_2} \\
              \end{bmatrix}

The Jacobian $\mathbf{J}$ approximates the movement of the end point $\mathbf{x} = (x_2, y_2)$ with respect to changes in joint angles $\mathbf{a} = (a_1, a_2)$:

$$$
 \Delta \mathbf{x} \approx \mathbf{J} \Delta \mathbf{a} \; .

Starting from a given position of the end point $\mathbf{x} = \mathbf{x}_0$ and given a target position $\mathbf{x}_T$, we can use the inverse of the Jacobian to compute small updates in the angles $\mathbf{a}$ using

$$$
 \Delta \mathbf{a} = \eta \, (\mathbf{J^{-1}} \mathbf{e}) \; ,

where $\mathbf{e} = \mathbf{x}_T - \mathbf{x}$ is the current position error and $\eta > 0$ is a small step size. Computing the new position $\mathbf{x}$ with the updated angles $\mathbf{a}$ and repeating this process until the error $\mathbf{e}$ approaches zero moves the end point of the mechanism towards the target point $\mathbf{x}_T$.

Let's use DiffSharp for implementing this system.

*)

open DiffSharp.AD.Float64

// Set the lengths of the arm segments
let l1 = 4.5
let l2 = 2.5

// Set the initial angles
let mutable a = toDV [1.1; -0.9]

// Transform angles into (x1, y1) and (x2, y2) positions
let transform (a:DV) =
    let x1, y1 = l1 * cos a.[0], l1 * sin a.[0]
    let x2, y2 = x1 + l2 * cos (a.[0] + a.[1]), y1 + l2 * sin (a.[0] + a.[1])
    toDV [x1; y1; x2; y2]
    
// Forward kinematics of the tip of the arm (x2, y2)
let inline forwardK (a:DV) =
    let t = transform a
    toDV [t.[2]; t.[3]]

// Inverse kinematics using inverse Jacobian-vector product
// target is the target position of the tip (x2, y2)
// eta is the update coefficient
// timeout is the maximum number of iterations
let inverseK (target:DV) (eta:float) (timeout:int) =
    seq {for i in 0 .. timeout do
            let pos, j = jacobian' forwardK a
            let error = target - pos
            let da = (DM.inverse j) * error
            a <- a + eta * da
            yield (DV.norm error, a)
            }
    |> Seq.takeWhile (fun x -> fst x > D 0.4)
    |> Seq.map snd

(**

The given link sizes $L_1 = 4.5$, $L_2 = 2.5$ and initial angles $(a_1, a_2) = (1.1, -0.9)$ put the starting position approximately at $(x_2, y_2) = (4.5, 4.5)$.

Let us take the end point of the robot arm to positions $(4.5, 4.5) \to (5.0, 0.0) \to (3.5, -4.0)$ and then back to $(4.5, 4.5)$.

*)

let movement1 = inverseK (toDV [5.0; 0.0]) 0.025 100000
let movement2 = inverseK (toDV [3.5; -4.0]) 0.025 100000
let movement3 = inverseK (toDV [4.5; 4.5]) 0.025 100000

(*** hide, define-output: o ***)
printf "val movement1 : seq<Vector<D>>
val movement2 : seq<Vector<D>>
val movement3 : seq<Vector<D>>"
(*** include-output: o ***)

(**

The following code draws the movement of the arm.

*)

open FSharp.Charting

let armPoints (a:DV) =
    let t = a |> transform |> primal |> convert
    seq [0., 0.; t.[0], t.[1]; t.[2], t.[3]]

let drawArmLive (aa:seq<DV>) =
    let pp = aa |> Array.ofSeq |> Array.map armPoints
    let pp2 = Array.copy pp
    Chart.Combine(
            [ LiveChart.Line(Event.cycle 10 pp, Color = System.Drawing.Color.Purple)
              LiveChart.Point(Event.cycle 10 pp2, MarkerSize = 20) ])
              .WithXAxis(Min = 0.0, Max = 10.0, 
                    LabelStyle = ChartTypes.LabelStyle(Interval = 2.), 
                    MajorGrid = ChartTypes.Grid(Interval = 2.),
                    MajorTickMark = ChartTypes.TickMark(Enabled = false))
              .WithYAxis(Min = -5.0, Max = 5.0, 
                    LabelStyle = ChartTypes.LabelStyle(Interval = 2.), 
                    MajorGrid = ChartTypes.Grid(Interval = 2.), 
                    MajorTickMark = ChartTypes.TickMark(Enabled = false))

drawArmLive (movement3 |> Seq.append movement2 |> Seq.append movement1)


(**

<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-inversekinematics-inverse.gif" alt="Chart" style="width:450px;"/>
    </div>
</div>

It is known that one can use the transposed Jacobian $\mathbf{J}^\textrm{T}$ instead of the inverse Jacobian $\mathbf{J^{-1}}$ and obtain similar results, albeit with slightly different behavior.

The **DiffSharp.AD** module provides the operation **jacobianTv** making use of reverse mode AD to calculate the transposed Jacobian-vector product in a matrix-free and highly efficient way. This is in contrast to the above code, which comes with the cost of computing the full Jacobian matrix and its inverse in every step of the iteration. (See the [API Overview](api-overview.html) and [Benchmarks](benchmarks.html) pages for a comparison of these operations.)

Using this method with the same arm positions gives us the following result:
*)


// Inverse kinematics using transposed Jacobian-vector product
let inverseK' (target:DV) (eta:float) (timeout:int) =
    seq {for i in 0 .. timeout do
            let pos, jTv = jacobianTv'' forwardK a
            let error = target - pos
            let da = jTv error
            a <- a + eta * da
            yield (DV.norm error, a)
            }
    |> Seq.takeWhile (fun x -> fst x > D 0.4)
    |> Seq.map snd

(**
 
    
<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-inversekinematics-transpose.gif" alt="Chart" style="width:450px;"/>
    </div>
</div>

   
*)