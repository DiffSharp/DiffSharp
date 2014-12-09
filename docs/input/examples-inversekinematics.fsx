(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"
#load "../../packages/FSharp.Charting.0.90.7/EventEx-0.1.fsx"

(**
Inverse Kinematics
==================

[Inverse kinematics](http://en.wikipedia.org/wiki/Inverse_kinematics)

<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-inversekinematics.png" alt="Chart" style="width:300px;"/>
    </div>
</div>

*)

open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open DiffSharp.Util.LinearAlgebra

// Set the lengths of the arm segments
let l1 = 4.5
let l2 = 2.5

// Set the initial angles
let mutable a = vector [0.3; 1.01]

// Transform angles into (x1, y1) and (x2, y2) positions
let transform (a:Vector<Adj>) =
    let x1, y1 = l1 * cos a.[0], l1 * sin a.[0]
    let x2, y2 = x1 + l2 * cos (a.[0] + a.[1]), y1 + l2 * sin (a.[0] + a.[1])
    vector [x1; y1; x2; y2]
    
// Forward kinematics of the tip of the arm (x2, y2)
let inline forwardK (a:Vector<Adj>) =
    let t = transform a
    vector [t.[2]; t.[3]]

(**
    
*)

let inverseK (target:Vector<float>) (eta:float) (timeout:int) =
    seq {for i in 0 .. timeout do
            let pos, jTv = jacobianTv'' forwardK a
            let error = target - pos
            let da = jTv error
            a <- a + eta * da
            yield (Vector.norm error, a)
            }
    |> Seq.takeWhile (fun x -> fst x > 0.4)
    |> Seq.map snd

let inverseK' (target:Vector<float>) (eta:float) (timeout:int) =
    seq {for i in 0 .. timeout do
            let pos, j = jacobian' forwardK a
            let error = target - pos
            let da = (Matrix.inverse j) * error
            a <- a + eta * da
            yield (Vector.norm error, a)
            }
    |> Seq.takeWhile (fun x -> fst x > 0.4)
    |> Seq.map snd

let movement1 = inverseK (vector [6.0; 1.0]) 0.008 10000
let pos1 = a |> Vector.map adj |> forwardK |> Vector.copy
let movement2 = inverseK (vector [4.1; -3.0]) 0.01 10000
let pos2 = a |> Vector.map adj |> forwardK |> Vector.copy
let movement3 = inverseK (vector [4.7; 3.8]) 0.01 10000
let pos3 = a |> Vector.map adj |> forwardK |> Vector.copy

(**
    
*)

open FSharp.Charting

let armPoints (a:Vector<float>) =
    let t = a |> Vector.map adj |> transform |> Vector.map primal
    seq [0., 0.; t.[0], t.[1]; t.[2], t.[3]]

let drawArmLive (aa:seq<Vector<float>>) =
    let pp = aa |> Array.ofSeq |> Array.map armPoints
    let pp2 = Array.copy pp
    Chart.Combine(
            [ LiveChart.Line(Event.cycle 10 pp)
              LiveChart.Point(Event.cycle 10 pp2, MarkerSize = 10) ])
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
        <img src="img/examples-inversekinematics.gif" alt="Chart" style="width:450px;"/>
    </div>
</div>

*)