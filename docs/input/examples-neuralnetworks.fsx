(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.7/FSharp.Charting.fsx"

(**
Neural Networks
===============

[Artificial neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) are computational architectures based on the properties of biological neural systems, capable of learning and pattern recognition.

Let us create a [feedforward neural network](http://en.wikipedia.org/wiki/Feedforward_neural_network) model and use the DiffSharp library for implementing the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) algorithm for training it. 

We start by defining our neural network structure.

*)

open DiffSharp.AD.Reverse
open DiffSharp.AD.Reverse.Vector
open DiffSharp.Util.LinearAlgebra

// A neuron
type Neuron =
    {mutable w:Vector<Adj> // Weight vector of this neuron
     mutable b:Adj} // Bias of this neuron
 
// A layer of neurons
type Layer =
    {n:Neuron[]} // The neurons forming this layer

// A feedforward network of neuron layers
type Network =
    {l:Layer[]} // The layers forming this network

(** 

Each neuron works by taking inputs $x_1, \dots, x_n$ and calculating the activation (output)

$$$
  a = \sigma \left(\sum_{i} w_i x_i + b\right) \; ,

where $w_i$ are synapse weights associated with each input, $b$ is a bias, and $\sigma$ is an [activation function](http://en.wikipedia.org/wiki/Activation_function) representing the rate of [action potential](http://en.wikipedia.org/wiki/Action_potential) firing in the neuron.

<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-neuralnetworks-neuron.png" alt="Chart" style="width:400px;"/>
    </div>
</div>

The activation function is commonly taken as the [sigmoid function](http://en.wikipedia.org/wiki/Sigmoid_function)

$$$
 \sigma (z) = \frac{1}{1 + e^{-z}} \; ,

due to its "nice" and simple derivative and gain control properties.

Now let us write the network evaluation code and a function for creating a given network configuration and initializing the weights and biases with small random values.

*)

let sigmoid (x:Adj) = 1. / (1. + exp -x)

let runNeuron (x:Vector<Adj>) (n:Neuron) =
    x * n.w + n.b
    |> sigmoid

let runLayer (x:Vector<Adj>) (l:Layer) =
    Array.map (runNeuron x) l.n
    |> vector

let runNetwork (x:Vector<Adj>) (n:Network) =
    Seq.fold (fun o l -> runLayer o l) x n.l

let rnd = new System.Random()

// Initialize a fully connected feedforward neural network
// Weights and biases between -0.5 and 0.5
let createNetwork (inputs:int) (layers:int[]) =
    {l = Array.init layers.Length (fun i -> 
        {n = Array.init layers.[i] (fun j -> 
            {w = Vector.init
                     (if i = 0 then inputs else layers.[i - 1])
                     (fun k -> adj (-0.5 + rnd.NextDouble()))
             b = adj (-0.5 + rnd.NextDouble())})})}
(**

This gives us a highly scalable feedforward network architecture capable of expressing any number of inputs, outputs, and hidden layers we want. The network is fully connected, meaning that each neuron in a layer receives as input all the outputs of the previous layer.

For example, using the code

*)

let net1 = createNetwork 3 [|4; 2|]

(**

would give us the following network with 3 inputs, a hidden layer with 4 neurons, and an output layer with 2 neurons:

<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-neuralnetworks-network.png" alt="Chart" style="width:400px;"/>
    </div>
</div>

We can also have more than one hidden layer.

For training networks, we will make use of reverse automatic differentiation (the **DiffSharp.AD.Reverse** module) for propagating the error at the output backwards through the network synapse weights. This will give us the partial derivative of the error at the output with respect to each weight $w_i$ and bias $b_i$ in the network, which we will use in an update rule

$$$
 \begin{eqnarray*}
 \Delta w_i &=& -\eta \frac{\partial E}{\partial w_i} \; ,\\
 \Delta b_i &=& -\eta \frac{\partial E}{\partial b_i} \; ,\\
 \end{eqnarray*}

where $E$ is the error at the output and $\eta$ is the learning rate.

It is important to note that the backpropagation algorithm is just a special case of reverse AD, with which it shares a common history. Please see the [Reverse AD](gettingstarted-reversead.html) page for an explanation of the usage of adjoints and their backwards propagation.

*)

// The backpropagation algorithm
// t is the training set consisting of input and output vectors
// eta is the learning rate
// n is the network to be trained
let backprop (t:(Vector<float>*Vector<float>)[]) (eta:float) (timeout:int) (n:Network) =
    let ta = Array.map (fun x -> Vector.map adj (fst x), Vector.map adj (snd x)) t
    seq {for i in 0 .. timeout do // A timeout value
            Trace.Clear()
            let error = 
                (1. / float t.Length) * Array.sumBy 
                    (fun t -> Vector.norm ((snd t) - runNetwork (fst t) n)) ta
            error.A <- 1.
            Trace.ReverseSweep()
            for l in n.l do
                for n in l.n do
                    n.b <- n.b - eta * n.b.A // Update neuron bias
                    n.w <- Vector.map (fun (w:Adj) -> w - eta * w.A) n.w // Update neuron weights
            if i = timeout then printfn "Failed to converge within %i steps" timeout
            yield primal error}
    |> Seq.takeWhile (fun x -> x > 0.005)

(**

Using reverse AD here has two big advantages: it makes the backpropagation code succint and straightforward to write and maintain; and it allows us to freely choose activation functions without the burden of coding their derivatives or modifying the backpropagation code accordingly.

We can now test the algorithm by training some networks. 

It is known that [linearly separable](http://en.wikipedia.org/wiki/Linear_separability) rules such as [logical disjunction](http://en.wikipedia.org/wiki/Logical_disjunction) can be learned by a single neuron.

*)
open FSharp.Charting

let trainOR = [|vector [0.; 0.], vector [0.]
                vector [0.; 1.], vector [1.]
                vector [1.; 0.], vector [1.]
                vector [1.; 1.], vector [1.]|]

// 2 inputs, one layer with one neuron
let net2 = createNetwork 2 [|1|]

// Train
let train2 = backprop trainOR 0.9 10000 net2

// Plot the error during training
Chart.Line train2

(*** hide, define-output: o ***)
printf "val net2 : Network =
  {l =
    [|{n =
        [|{w =
            Vector
              [|Adj(0.3039949223, -0.120172164);
                Adj(-0.1498002706, -0.1234536468)|];
           b = Adj(0.1627550189, -0.120581615);}|];}|];}
val train2 : seq<float>"
(*** include-output: o ***)

(** 

<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-neuralnetworks-chart1.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Linearly inseparable problems such as [exclusive or](http://en.wikipedia.org/wiki/Exclusive_or) require one or more hidden layers to learn.
    
*)

let trainXOR = [|vector [0.; 0.], vector [0.]
                 vector [0.; 1.], vector [1.]
                 vector [1.; 0.], vector [1.]
                 vector [1.; 1.], vector [0.]|]

// 2 inputs, 3 neurons in a hidden layer, 1 neuron in the output layer
let net3 = createNetwork 2 [|3; 1|]

// Train
let train3 = backprop trainXOR 0.9 10000 net3

// Plot the error during training
Chart.Line train3

(*** hide, define-output: o2 ***)
printf "val net3 : Network =
  {l =
    [|{n =
        [|{w =
            Vector
              [|Adj(-0.3990952149, 7.481450298e-05);
                Adj(0.2626295973, -0.0005625556545)|];
           b = Adj(0.4077099938, -0.0002455469757);};
          {w =
            Vector
              [|Adj(0.3472105762, -0.0003902540939);
                Adj(0.2698220153, -0.0004052317731)|];
           b = Adj(0.03246956809, -0.000286118247);};
          {w =
            Vector
              [|Adj(0.1914881005, -0.0001046784245);
                Adj(-0.1030110692, -7.688368233e-05)|];
           b = Adj(0.05589360816, -5.863152837e-05);}|];};
      {n =
        [|{w =
            Vector
              [|Adj(-0.3930620788, 0.0002184686632);
                Adj(0.4657231793, -0.0001747499928);
                Adj(-0.4974639057, -3.725300124e-05)|];
           b = Adj(-0.4166501578, -8.108279605e-05);}|];}|];}
val train3 : seq<float>"
(*** include-output: o2 ***)

(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-neuralnetworks-chart2.png" alt="Chart" style="width:550px"/>
    </div>
</div>

*)