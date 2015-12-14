#r "bin/Debug/DiffSharp.dll"
#r "../../packages/FSharp.Quotations.Evaluator.1.0.6/lib/net40/FSharp.Quotations.Evaluator.dll"
#r "../../packages/FSharp.Charting.0.90.12/lib/net40/FSharp.Charting.dll"
#r @"C:\Program Files (x86)\Reference Assemblies\Microsoft\Framework\.NETFramework\v4.6\System.Windows.Forms.DataVisualization.dll"

open DiffSharp.AD.Float32
open DiffSharp.Util

open FSharp.Charting

open System.IO

let rng = System.Random()

let makeUniformRandomDM(hidden_size, input_size) =
    let scale = (2.0f / sqrt(hidden_size+input_size |> float32))
    let t = DV [|for x=1 to hidden_size*input_size do yield (rng.NextDouble()-0.5 |> float32)*scale|]
    DV.ReshapeToDM(hidden_size, t)

let makeUniformRandomDV hidden_size =
    let scale = (2.0f / sqrt(hidden_size+1 |> float32))
    DV [|for x=1 to hidden_size do yield (rng.NextDouble()-0.5 |> float32)*scale|]

type LSTMLayer =
    {mutable W_z:DM  // Input weight matrix for the block input
     mutable U_z:DM  // Recurrent weight matrix for the block input
     mutable b_z:DV  // Bias vector for the block input

     mutable W_i:DM  // Input weight matrix for the input gate
     mutable U_i:DM  // Recurrent weight matrix for the input gate
     mutable b_i:DV  // Bias vector for the input gate
     mutable P_i:DM  // Peephole weight matrix for the input gate

     mutable W_f:DM  // Input weight matrix for the forget gate
     mutable U_f:DM  // Recurrent weight matrix for the forget gate
     mutable b_f:DV  // Bias vector for the forget gate
     mutable P_f:DM  // Peephole weight matrix for the forget gate

     mutable W_o:DM  // Input weight matrix for the output gate
     mutable U_o:DM  // Recurrent weight matrix for the output gate
     mutable b_o:DV  // Bias vector for the output gate
     mutable P_o:DM  // Peephole weight matrix for the output gate

     block_input_a : DM -> DM
     block_output_a : DM -> DM
     } with

    static member createRandomLSTMLayer hidden_size input_size block_input_a block_output_a =
        {
        W_z = makeUniformRandomDM(hidden_size, input_size)
        U_z = makeUniformRandomDM(hidden_size, hidden_size)
        b_z = makeUniformRandomDV(hidden_size)

        W_i = makeUniformRandomDM(hidden_size, input_size)
        U_i = makeUniformRandomDM(hidden_size, hidden_size)
        b_i = makeUniformRandomDV(hidden_size)
        P_i = makeUniformRandomDM(hidden_size, hidden_size)

        W_f = makeUniformRandomDM(hidden_size, input_size)
        U_f = makeUniformRandomDM(hidden_size, hidden_size)
        b_f = makeUniformRandomDV(hidden_size)
        P_f = makeUniformRandomDM(hidden_size, hidden_size)

        W_o = makeUniformRandomDM(hidden_size, input_size)
        U_o = makeUniformRandomDM(hidden_size, hidden_size)
        b_o = makeUniformRandomDV(hidden_size)
        P_o = makeUniformRandomDM(hidden_size, hidden_size)

        block_input_a = block_input_a
        block_output_a = block_output_a
        }

    member l.tagReverse tag =
         l.W_z <- l.W_z |> makeReverse tag
         l.U_z <- l.U_z |> makeReverse tag
         l.b_z <- l.b_z |> makeReverse tag

         l.W_i <- l.W_i |> makeReverse tag
         l.U_i <- l.U_i |> makeReverse tag
         l.b_i <- l.b_i |> makeReverse tag
         l.P_i <- l.P_i |> makeReverse tag

         l.W_f <- l.W_f |> makeReverse tag
         l.U_f <- l.U_f |> makeReverse tag
         l.b_f <- l.b_f |> makeReverse tag
         l.P_f <- l.P_f |> makeReverse tag

         l.W_o <- l.W_o |> makeReverse tag
         l.U_o <- l.U_o |> makeReverse tag
         l.b_o <- l.b_o |> makeReverse tag
         l.P_o <- l.P_o |> makeReverse tag
    
    /// Returns all the weights in an array.
    member l.ToArray = [|l.W_z;l.U_z;l.W_i;l.U_i;l.P_i;l.W_f;l.U_f;l.P_f;l.W_o;l.U_o;l.P_o|],[|l.b_z;l.b_i;l.b_f;l.b_o|]
    static member fromArray (a: DM[]) block_input_a block_output_a =
        {
         W_z = a.[0]
         U_z = a.[1]
         b_z = a.[2] |> DM.toDV

         W_i = a.[3]
         U_i = a.[4]
         b_i = a.[5] |> DM.toDV
         P_i = a.[6]

         W_f = a.[7]
         U_f = a.[8]
         b_f = a.[9] |> DM.toDV
         P_f = a.[10]

         W_o = a.[11]
         U_o = a.[12]
         b_o = a.[13] |> DM.toDV
         P_o = a.[14]

         block_input_a = block_input_a
         block_output_a = block_output_a
        }

    member l.addAdjoints (learning_rate: float32) =
         l.W_z <- l.W_z.P - learning_rate * l.W_z.A
         l.U_z <- l.U_z.P - learning_rate * l.U_z.A
         l.b_z <- l.b_z.P - learning_rate * l.b_z.A

         l.W_i <- l.W_i.P - learning_rate * l.W_i.A
         l.U_i <- l.U_i.P - learning_rate * l.U_i.A
         l.b_i <- l.b_i.P - learning_rate * l.b_i.A
         l.P_i <- l.P_i.P - learning_rate * l.P_i.A

         l.W_f <- l.W_f.P - learning_rate * l.W_f.A
         l.U_f <- l.U_f.P - learning_rate * l.U_f.A
         l.b_f <- l.b_f.P - learning_rate * l.b_f.A
         l.P_f <- l.P_f.P - learning_rate * l.P_f.A

         l.W_o <- l.W_o.P - learning_rate * l.W_o.A
         l.U_o <- l.U_o.P - learning_rate * l.U_o.A
         l.b_o <- l.b_o.P - learning_rate * l.b_o.A
         l.P_o <- l.P_o.P - learning_rate * l.P_o.A

    member l.runLayer (x:DM) (y:DM) (c:DM) =
        let block_input = l.W_z*x+l.U_z*y + l.b_z |> l.block_input_a
        let input_gate = l.W_i*x+l.U_i*y+l.P_i*c + l.b_i |> sigmoid
        let forget_gate = l.W_f*x+l.U_f*y+l.P_f*c + l.b_f |> sigmoid
        let c' = block_input.*input_gate + c.*forget_gate
        let output_gate = l.W_o*x+l.U_o*y+l.P_o*c' + l.b_o |> sigmoid
        (l.block_output_a c') .* output_gate, c'

    member l.runLayerNoH (x:DM) =
        let block_input = l.W_z*x + l.b_z |> l.block_input_a
        let input_gate = l.W_i*x + l.b_i |> sigmoid
        let forget_gate = l.W_f*x + l.b_f |> sigmoid
        let c' = block_input .* input_gate
        let output_gate = l.W_o*x+l.P_o*c' + l.b_o |> sigmoid
        (l.block_output_a c') .* output_gate, c'

    member l.runLayerNoI (y:DM) (c:DM) =
        let block_input = l.U_z*y + l.b_z |> l.block_input_a
        let input_gate = l.U_i*y+l.P_i*c + l.b_i |> sigmoid
        let forget_gate = l.U_f*y+l.P_f*c + l.b_f |> sigmoid
        let c' = block_input.*input_gate+c.*forget_gate
        let output_gate = l.U_o*y+l.P_o*c' + l.b_o |> sigmoid
        (l.block_output_a c') .* output_gate, c'

// A recurrent layer of neurons
type Layer =
    {
    mutable W:DM  // Input weight matrix
    mutable U:DM  // Recurrent weight matrix
    mutable b:DV  // Bias vector
    a:DM->DM
    } with     // Activation function

    static member makeUniformRandomDM(hidden_size, input_size) =
        let scale = (2.0f / sqrt(hidden_size+input_size |> float32))
        let t = DV [|for x=1 to hidden_size*input_size do yield (rng.NextDouble()-0.5 |> float32)*scale|]
        DV.ReshapeToDM(hidden_size, t)

    static member makeUniformRandomDV hidden_size =
        let scale = (2.0f / sqrt(hidden_size+1 |> float32))
        DV [|for x=1 to hidden_size do yield (rng.NextDouble()-0.5 |> float32)*scale|]

    static member createRandomLayer hidden_size input_size act =
        {
        W = makeUniformRandomDM(hidden_size, input_size)
        U = makeUniformRandomDM(hidden_size, hidden_size)
        b = makeUniformRandomDV(hidden_size)

        a = act
        }
     
    member l.ToArray = 
        [|l.W;l.U|], [|l.b|]

    member l.tagReverse tag =
         l.W <- l.W |> makeReverse tag
         l.U <- l.U |> makeReverse tag
         l.b <- l.b |> makeReverse tag

    member l.addAdjoints (learning_rate: float32) =
         l.W <- l.W.P - learning_rate * l.W.A
         l.U <- l.U.P - learning_rate * l.U.A
         l.b <- l.b.P - learning_rate * l.b.A

    static member fromArray (a : DM[]) act =
        {
         W = a.[0]
         U = a.[1]
         b = a.[2] |> DM.toDV
         a = act
        }

    // For the section with no previous hidden state.
    member l.runLayerNoH (x:DM) =
        l.W*x + l.b |> l.a
    
    // For the section with no input
    member l.runLayerNoI (y:DM) =
        l.U*y + l.b |> l.a

    // For the section with previous hidden state
    member l.runLayer (x:DM) (y:DM) =
        l.W*x + l.U*y + l.b |> l.a

let cross_entropy_cost (targets:DM) (inputs:DM) =
    ((targets .* (DM.Log inputs) + (1.0f-targets) .* DM.Log (1.0f-inputs)) |> DM.Sum) / (-inputs.Cols)

let squareSum (targets:DM) (inputs:DM) =
    let r = targets - inputs
    (DM.Pow(r,2) |> DM.Sum) / (2*targets.Cols)

#load "embedded_reber.fsx"
open Embedded_reber

let reber_set = make_reber_set 3000

let make_data_from_set target_length =
    let twenties = reber_set |> Seq.filter (fun (a,b,c) -> a.Length = target_length) |> Seq.toArray
    let batch_size = (twenties |> Seq.length)

    let d_training_data =
        [|
        for i=0 to target_length-1 do
            let input = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield input.[i] |] |> Array.concat
            yield DV.ReshapeToDM(batch_size, DV input) |> DM.Transpose|]

    let d_target_data =
        [|
        for i=1 to target_length-1 do // The targets are one less than the inputs. This has the effect of shifting them to the left.
            let output = [|
                for k=0 to batch_size-1 do
                    let example = twenties.[k]
                    let s, input, output = example
                    yield output.[i] |] |> Array.concat
            yield DV.ReshapeToDM(batch_size, DV output) |> DM.Transpose|]

    d_training_data, d_target_data

let train_lstm_reber num_iters learning_rate (data: DM[]) (targets: DM[]) clip_coef (l1: LSTMLayer) (l2: Layer) =
    [|
    let tag = DiffSharp.Util.GlobalTagger.Next

    let mutable i=1
    let mutable rr=0.0f
    while i <= num_iters && System.Single.IsNaN rr = false do
        l1.tagReverse tag
        l2.tagReverse tag

        let costs = [|
            let mutable a, c = l1.runLayerNoH data.[0]
            let b = l2.runLayerNoH a
            let r = squareSum targets.[0] b
            yield r

            for i=1 to data.Length-2 do
                let a',c' = l1.runLayer data.[i] a c
                a <- a'; c <- c'
                let b = l2.runLayerNoH a
                let r = squareSum targets.[i] b
                yield r
            
            |]

        let r = (Array.sum costs) / float32 costs.Length

        printfn "The cost is %f at iteration %i" (float32 r.P) i

        rr <- float32 r.P

        r |> reverseProp (D 1.0f)

        // Add gradients.
        l1.addAdjoints learning_rate
        l2.addAdjoints learning_rate

        i <- i+1
        yield rr |]

let d_training_data_20, d_target_data_20 = make_data_from_set 20

let hidden_size = 64
let l1 = LSTMLayer.createRandomLSTMLayer hidden_size 7 tanh tanh
let l2 = Layer.createRandomLayer 7 hidden_size sigmoid

DiffSharp.Config.GlobalConfig.SetBackend("Cuda")

#time
let t = [|
    for i=1 to 1 do
        yield train_lstm_reber 10 5.0f d_training_data_20 d_target_data_20 1.0f l1 l2 
        System.GC.Collect() |] |> Array.concat
#time

(Chart.Line t).ShowChart()

//OpenBLAS: 4.7s
//Cuda: 67s.
