(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"
#load "../../packages/FSharp.Charting.0.90.12/FSharp.Charting.fsx"

(**

<div class="row">
    <div class="span9">
    <div class="well well-small" id="nuget" style="background-color:#E0EBEB">
        <b>Please note:</b> the code in this example is provided for illustrating the basics and therefore kept simple and unoptimized. It is not intended for production use. It will be replaced with a better version supporting more complex cases.
    </div>
    </div>
</div>

Neural Networks
===============

[Artificial neural networks](http://en.wikipedia.org/wiki/Artificial_neural_network) are computational models inspired by biological nervous systems, capable of approximating functions depending on a large number of inputs. A network is defined by a connectivity structure and a set of weights between interconnected processing units ("neurons"). Optimizing, or tuning, the set of weights for a given task makes neural networks capable of learning.

Let's create a [feedforward neural network](http://en.wikipedia.org/wiki/Feedforward_neural_network) and use DiffSharp for implementing the [backpropagation](http://en.wikipedia.org/wiki/Backpropagation) algorithm for training it. As mentioned before, backpropagation is just a special case of reverse mode AD.

We start by defining our neural network structure.

*)

open DiffSharp.AD.Float64
open DiffSharp.Util

// A layer of neurons
type Layer =
    {mutable W:DM  // Weight matrix
     mutable b:DV} // Bias vector

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

A conventional choice for the activation function had been the [sigmoid](http://en.wikipedia.org/wiki/Sigmoid_function) $\sigma (z) = 1 / (1 + e^{-z})$ for a long period because of its simple derivative and gain control properties. Recently the hyperbolic tangent $\tanh$ and the [rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) $\textrm{ReLU}(z) = \max(0, z)$ have been more popular choices due to their convergence and training performance characteristics.

Now let's write the network evaluation code and a function for creating a given network configuration and initializing the weights and biases with small random values. In practice, proper weight initialization would depend on the network structure and the type of activation functions used, as it has been demonstrated to have an important effect on training convergence.

Network evaluation is implemented using linear algebra, where we have a weight matrix $\mathbf{W}^l$ holding the weights of all neurons in layer $l$. The elements of this matrix $w_{ij}$ represent the weight between the $j$-th neuron in layer $l - 1$ and the $i$-th neuron in layer $l$. We also have a bias vector $\mathbf{b}^l$ holding the biases of all neurons in layer $l$ (one bias per neuron). Thus evaluating the network is just a matter of computing the activation

$$$
  \mathbf{a}^l = \mathbf{W}^l \mathbf{a}^{l - 1} + \mathbf{b}^l \; ,

for each layer and passing the activation vector (output) of each layer as the input to the next layer, until we have the network output as the output vector of the last layer.
*)

let runLayer (x:DV) (l:Layer) =
    l.W * x + l.b |> sigmoid

let runNetwork (x:DV) (n:Network) =
    Array.fold runLayer x n.layers


let rnd = System.Random()

// Initialize a fully connected feedforward neural network
// Weights and biases between -0.5 and 0.5
// l : number of inputs and number of neurons in each subsequent layer
let createNetwork (l:int[]) =
    {layers = Array.init (l.Length - 1) (fun i ->
        {W = DM.init l.[i + 1] l.[i] (fun _ _ -> D (-0.5 + rnd.NextDouble()))
         b = DV.init l.[i + 1] (fun _ -> D (-0.5 + rnd.NextDouble()))})}
(**

This gives us an easily scalable feedforward network architecture capable of expressing any number of inputs, outputs, and hidden layers. The network is fully connected, meaning that each neuron in a layer receives the outputs of all the neurons in the previous layer as its input.

For example,

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

For training networks, we will make use of reverse mode AD for propagating an error $E$ at the output backwards through the network weights. This will give us the partial derivative of the error at the output with respect to each weight $w_i$ and bias $b_i$ in the network, which we will then use in an update rule

$$$
 \begin{eqnarray*}
 \Delta w_i &=& -\eta \frac{\partial E}{\partial w_i} \; ,\\
 \Delta b_i &=& -\eta \frac{\partial E}{\partial b_i} \; ,\\
 \end{eqnarray*}

where $\eta$ is the learning rate.

It is important to note that the backpropagation algorithm is just a special case of reverse mode AD, with which it shares a common history. Please see the [Nested AD](gettingstarted-nestedad.html) page for an explanation of the low-level usage of adjoints and their backwards propagation.

*)

// The backpropagation algorithm
// n: network to be trained
// eta: learning rate
// epsilon: error threshold
// timeout: maximum number of iterations
// t: training set consisting of input and output vectors
let backprop (n:Network) eta epsilon timeout (t:(DV*DV)[]) =
    let i = DiffSharp.Util.GlobalTagger.Next
    seq {for j in 0 .. timeout do
            for l in n.layers do
                l.W <- l.W |> makeReverse i
                l.b <- l.b |> makeReverse i

            let error = t |> Array.sumBy (fun (x, y) -> DV.l2normSq (y - runNetwork x n))
            error |> reverseProp (D 1.) // Propagate adjoint value 1 backward

            for l in n.layers do
                l.W <- l.W.P - eta * l.W.A
                l.b <- l.b.P - eta * l.b.A

            printfn "Iteration %i, error %f" j (float error)
            if j = timeout then printfn "Failed to converge within %i steps." timeout
            yield float error}
    |> Seq.takeWhile ((<) epsilon)

(**

Using reverse mode AD here has two big advantages: (1) it makes the backpropagation code succinct and straightforward to write and maintain; and (2) it allows us to freely choose activation functions without the burden of coding their derivatives or modifying the backpropagation code accordingly.

We can now test the algorithm by training some networks.

It is known that [linearly separable](http://en.wikipedia.org/wiki/Linear_separability) rules such as [logical disjunction](http://en.wikipedia.org/wiki/Logical_disjunction) can be learned by a single neuron.

*)
open FSharp.Charting

let trainOR = [|toDV [0.; 0.], toDV [0.]
                toDV [0.; 1.], toDV [1.]
                toDV [1.; 0.], toDV [1.]
                toDV [1.; 1.], toDV [1.]|]

// 2 inputs, one layer with one neuron
let net2 = createNetwork [|2; 1|]

// Train
let train2 = backprop net2 0.9 0.005 10000 trainOR

// Plot the error during training
Chart.Line train2

(*** hide, define-output: o ***)
printf "val net2 : Network = {layers = [|{W = DM [[0.230677625; 0.1414874814]];
                                  b = DV [|0.4233988253|];}|];}
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

let trainXOR = [|toDV [0.; 0.], toDV [0.]
                 toDV [0.; 1.], toDV [1.]
                 toDV [1.; 0.], toDV [1.]
                 toDV [1.; 1.], toDV [0.]|]

// 2 inputs, 3 neurons in a hidden layer, 1 neuron in the output layer
let net3 = createNetwork [|2; 3; 1|]

// Train
let train3 = backprop net3 0.9 0.005 10000 trainXOR

// Plot the error during training
Chart.Line train3

(*** hide, define-output: o2 ***)
printf "val net3 : Network =
  {layers =
    [|{W = DM [[-0.04536837132; -0.3447727025]
               [-0.07626016418; 0.06522091877]
               [0.2581558948; 0.1597980939]];
       b = DV [|-0.3051199176; 0.2980325892; 0.4621827649|];};
      {W = DM [[-0.347911722; 0.2696812725; 0.2704776571]];
       b = DV [|-0.1477482923|];}|];}
val train3 : seq<float>"
(*** include-output: o2 ***)

(**
<div class="row">
    <div class="span6 offset1">
        <img src="img/examples-neuralnetworks-chart2.png" alt="Chart" style="width:550px"/>
    </div>
</div>

Some Performance Tricks
-----------------------

High performance neural network implementations propagate matrices, instead of vectors, through the network. In other words, instead of treating the traning data as a set of input vectors $\mathbf{x}_i \in \mathbb{R}^n$ and target vectors $\mathbf{y}_i \in \mathbb{R}^m$, $i = 1 \dots d$, we can have one input matrix $\mathbf{X} \in \mathbb{R}^{n\times d}$ and a target matrix $\mathbf{Y} \in \mathbb{R}^{m\times d}$, where $d$ is the number of examples in the training set. In DiffSharp, as in other linear algebra libraries, computing matrix-matrix multiplications are significantly faster than computing a series of matrix-vector multiplications.

For simplifying network evaluation further, and therefore making things even faster, we can implement the bias of each neuron as just another weight of an input that is constantly $1$. For accomplishing this, we just have to add an extra row of $1$s to our input matrix, giving $\mathbf{X} \in \mathbb{R}^{(n+1)\times d}$. This is sometimes known as the "bias trick".
*)

// A layer of neurons
type Layer =
    {mutable W:DM  // Weight matrix
     mutable b:DV} // Bias vector

// A feedforward network of neuron layers
type Network =
    {layers:Layer[]} // The layers forming this network

(**
The Obligatory MNIST Example
----------------------------

*)