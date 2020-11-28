#I "../tests/DiffSharp.Tests/bin/Debug/netcoreapp3.0"
#r "Microsoft.Z3.dll"
#r "DiffSharp.Core.dll"
#r "DiffSharp.Backends.Torch.dll"
#r "DiffSharp.Backends.ShapeChecking.dll"
open System
open DiffSharp
open DiffSharp.Model
open DiffSharp.Optim
open DiffSharp.Data
open DiffSharp.ShapeChecking





[<ShapeCheck>]
module SomeTensorCode =
    
    [<ShapeCheck>]
    let someFunction (C: Int, H: Int) = 
          let res = C + H + 2
          res
    

(*    
[<ShapeCheck>]
type SomeModel(sym: SymScope, x1: Int) =

    //do Assert (x1 >~ 0)
    
    [<ShapeCheck("𝐵,3,(𝐻/4)+3,(𝑊/4)+3", "5,64,16,16")>]
    member _.Add(input: Tensor) = 
        //let res = dsharp.convTranspose2d(input, filters, stride=Int 2, padding=Int 9, outputPadding=Int 1)   // , paddings=[sym?FH/Int 2; sym?FW/Int 2])
        input
        *)

(*
open DiffSharp.ShapeChecking

Model.AnalyseShapes<Linear> ()
Model.AnalyseShapes<Linear> (Shape.symbolic [| sym?N; sym?M; |])
Model.AnalyseShapes<VAE> ()
Model.AnalyseShapes<Conv1d> (Shape.symbolic [| sym?N; sym?C; sym?L; |])
Model.AnalyseShapes<Conv2d> (Shape.symbolic [| sym?N; sym?C; sym?H; sym?W; |])
Model.AnalyseShapes<Conv3d> (Shape.symbolic [| sym?N; sym?C; sym?D; sym?H; sym?W; |])
Model.AnalyseShapes<ConvTranspose1d> (Shape.symbolic [| sym?N; sym?C; sym?L; |])
Model.AnalyseShapes<ConvTranspose2d> (Shape.symbolic [| sym?N; sym?C; sym?H; sym?W; |])
Model.AnalyseShapes<ConvTranspose3d> (Shape.symbolic [| sym?N; sym?C; sym?D; sym?H; sym?W; |])
Model.AnalyseShapes<Conv1d> (Shape.symbolic [| sym?N; sym?C; sym?L; |], optionals=false)
Model.AnalyseShapes<Conv2d> (Shape.symbolic [| sym?N; sym?C; sym?H; sym?W; |], optionals=false)
Model.AnalyseShapes<Conv3d> (Shape.symbolic [| sym?N; sym?C; sym?D; sym?H; sym?W; |], optionals=false)
Model.AnalyseShapes<ConvTranspose1d> (Shape.symbolic [| sym?N; sym?C; sym?L; |], optionals=false)
Model.AnalyseShapes<ConvTranspose2d> (Shape.symbolic [| sym?N; sym?C; sym?H; sym?W; |], optionals=false)
Model.AnalyseShapes<ConvTranspose3d> (Shape.symbolic [| sym?N; sym?C; sym?D; sym?H; sym?W; |], optionals=false)
Model.AnalyseShapes<Dropout> (Shape [| 30; 40; |] )
Model.AnalyseShapes<Dropout> ()
Model.AnalyseShapes<Dropout2d> ()
Model.AnalyseShapes<Dropout3d> ()
Model.AnalyseShapes<BatchNorm1d> ()
Model.AnalyseShapes<BatchNorm2d> ()
Model.AnalyseShapes<BatchNorm3d> ()

*)

(*
open Microsoft.Z3

let ctx = Context()

let solver = ctx.MkSolver()

let c = ctx.MkFreshConst("c", ctx.IntSort)
let d = ctx.MkFreshConst("d", ctx.IntSort)
let res, out = solver.Consequences([ ctx.MkEq(c, ctx.MkInt(1)); ctx.MkEq(d, ctx.MkInt(1))], [c; d])
solver.Assert( ctx.MkEq(c, ctx.MkInt(1)))
solver.Assert( ctx.MkEq(c, d))
let res, out = solver.Consequences([  ], [c; d])
*)
