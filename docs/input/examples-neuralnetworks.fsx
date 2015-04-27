(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.10/FSharp.Charting.fsx"

(**
Neural Networks
===============

[Artificial neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) are computational architectures based on the properties of biological neural systems, capable of learning and pattern recognition.

Let us create a [feedforward neural network](http://en.wikipedia.org/wiki/Feedforward_neural_network) model and use the DiffSharp library for implementing the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) algorithm for training it. 

We start by defining our neural network structure.

*)

open DiffSharp.AD
open FsAlg.Generic

// A layer of neurons
type Layer =
    {W:Matrix<D>  // Weigth matrix
     b:Vector<D>} // Bias vector

// A feedforward network of neuron layers
type Network =
    {layers:Layer[]} // The layers forming this network

(** 

The network will consist of several layers of neurons. Each neuron works by taking inputs $\{x_1, \dots, x_n\}$ and calculating the activation (output)

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

A pleasant way of implementing network evaluation is to use linear algebra, where we have a weight matrix $\mathbf{W}^l$ holding the weights of all neurons in layer $l$. The elements of this matrix $w_{ij}$ represent the weight between the $j$-th neuron in layer $l - 1$ and the $i$-th neuron in layer $l$. We also have a bias vector $\mathbf{b}^l$ holding the biases of all neurons in layer $l$ (one bias per neuron). Then network evaluation is just a matter of computing the activation

$$$
  \mathbf{a}^l = \mathbf{W}^l \mathbf{a}^{l - 1} + \mathbf{b}^l \; ,

for each layer and passing the activation vector (output) of each layer as the input to the next layer, until we have the network output as the output vector of the last layer.
*)

let sigmoid (x:D) = 1. / (1. + exp -x)

let runLayer (x:Vector<D>) (l:Layer) =
    l.W * x + l.b
    |> Vector.map sigmoid

let runNetwork (x:Vector<D>) (n:Network) =
    Array.fold runLayer x n.layers


let rnd = System.Random()

// Initialize a fully connected feedforward neural network
// Weights and biases between -0.5 and 0.5
// l : number of inputs and number of neurons in each subsequent layer
let createNetwork (l:int[]) =
    {layers = Array.init (l.Length - 1) (fun i ->
        {W = Matrix.init l.[i + 1] l.[i] (fun _ _ -> D (-0.5 + rnd.NextDouble()))
         b = Vector.init l.[i + 1] (fun _ -> D (-0.5 + rnd.NextDouble()))})}
(**

This gives us a highly scalable feedforward network architecture capable of expressing any number of inputs, outputs, and hidden layers. The network is fully connected, meaning that each neuron in a layer receives the outputs of all the neurons in the previous layer as its input.

For example, using the code

*)

let net1 = createNetwork [|3; 4; 2|]

(**

would give us the following network with 3 input nodes, a hidden layer with 4 neurons, and an output layer with 2 neurons:

<div class="row">
    <div class="span6 offset2">
        <img src="img/examples-neuralnetworks-network.png" alt="Chart" style="width:400px;"/>
    </div>
</div>

We can also have more than one hidden layer.

For training networks, we will make use of reverse mode AD for propagating the error at the output $E$ backwards through the network synapse weights. This will give us the partial derivative of the error at the output with respect to each weight $w_i$ and bias $b_i$ in the network, which we will then use in an update rule

$$$
 \begin{eqnarray*}
 \Delta w_i &=& -\eta \frac{\partial E}{\partial w_i} \; ,\\
 \Delta b_i &=& -\eta \frac{\partial E}{\partial b_i} \; ,\\
 \end{eqnarray*}

where $\eta$ is the learning rate.

It is important to note that the backpropagation algorithm is just a special case of reverse mode AD, with which it shares a common history. Please see the [Nested AD](gettingstarted-nestedad.html) page for an explanation of the usage of adjoints and their backwards propagation.

*)

// The backpropagation algorithm
// n: network to be trained
// eta: learning rate
// epsilon: error threshold
// timeout: maximum number of iterations
// t: training set consisting of input and output vectors
let backprop (n:Network) eta epsilon timeout (t:(Vector<_>*Vector<_>)[]) =
    let i = DiffSharp.Util.GlobalTagger.Next
    seq {for j in 0 .. timeout do
            for l in n.layers do
                l.W |> Matrix.replace (makeDR i)
                l.b |> Vector.replace (makeDR i) 

            let error = t |> Array.sumBy (fun (x, y) -> Vector.normSq (y - runNetwork x n))
            error |> reverseProp (D 1.) // Propagate adjoint value 1 backward

            for l in n.layers do
                l.W |> Matrix.replace (fun (x:D) -> x.P - eta * x.A)
                l.b |> Vector.replace (fun (x:D) -> x.P - eta * x.A)

            if j = timeout then printfn "Failed to converge within %i steps." timeout
            yield float error}
    |> Seq.takeWhile ((<) epsilon)

(**

Using reverse mode AD here has two big advantages: it makes the backpropagation code succinct and straightforward to write and maintain; and it allows us to freely choose activation functions without the burden of coding their derivatives or modifying the backpropagation code accordingly.

We can now test the algorithm by training some networks. 

It is known that [linearly separable](http://en.wikipedia.org/wiki/Linear_separability) rules such as [logical disjunction](http://en.wikipedia.org/wiki/Logical_disjunction) can be learned by a single neuron.

*)
open FSharp.Charting

let trainOR = [|vector [D 0.; D 0.], vector [D 0.]
                vector [D 0.; D 1.], vector [D 1.]
                vector [D 1.; D 0.], vector [D 1.]
                vector [D 1.; D 1.], vector [D 1.]|]

// 2 inputs, one layer with one neuron
let net2 = createNetwork [|2; 1|]

// Train
let train2 = backprop net2 0.9 0.005 10000 trainOR

// Plot the error during training
Chart.Line train2

(*** hide, define-output: o ***)
printf "val net2 : Network =
  {l = [|{n = [|{w = Vector [|D -0.3042126283; D -0.2509630955|];
                 b = D 0.4165584179;}|];}|];}
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

let trainXOR = [|vector [D 0.; D 0.], vector [D 0.]
                 vector [D 0.; D 1.], vector [D 1.]
                 vector [D 1.; D 0.], vector [D 1.]
                 vector [D 1.; D 1.], vector [D 0.]|]

// 2 inputs, 3 neurons in a hidden layer, 1 neuron in the output layer
let net3 = createNetwork [|2; 3; 1|]

// Train
let train3 = backprop net3 0.9 0.005 10000 trainXOR

// Plot the error during training
Chart.Line train3

(*** hide, define-output: o2 ***)
printf "val net3 : Network =
  {layers =
    [|{W = Matrix [[D 0.3691323418; D -0.4268625504]
                   [D 0.2538574085; D 0.4656410399]
                   [D 0.3023036475; D -0.09005093509]];
       b = Vector [|D -0.1326141556; D 0.1238703284; D 0.461187453|];};
      {W = Matrix [[D 0.1193747351; D 0.4290782972; D -0.1465413457]];
       b = Vector [|D -0.4840164538|];}|];}
val train3 : seq<float>"
(*** include-output: o2 ***)

(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-neuralnetworks-chart2.png" alt="Chart" style="width:550px"/>
    </div>
</div>

*)