(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
API Overview
============

The following table gives an overview of the differentiation API provided by DiffSharp.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-a60z{font-size:10px;background-color:#ecf4ff}
.tg .tg-1r2d{font-size:10px;background-color:#ecf4ff;text-align:center}
.tg .tg-glis{font-size:10px}
.tg .tg-wcxf{font-size:10px;background-color:#ffffc7;text-align:center}
.tg .tg-aycn{font-size:10px;background-color:#e4ffb3;text-align:center}
.tg .tg-wklz{font-size:10px;background-color:#ecf4ff;color:#000000;text-align:center}
.tg .tg-7dqz{font-weight:bold;font-size:10px}
</style>
<table class="tg">
  <tr>
    <th class="tg-glis"></th>
    <th class="tg-wcxf">diff</th>
    <th class="tg-wcxf">diff2</th>
    <th class="tg-wcxf">diffn</th>
    <th class="tg-aycn">grad</th>
    <th class="tg-aycn">gradv</th>
    <th class="tg-aycn">hessian</th>
    <th class="tg-aycn">hessianv</th>
    <th class="tg-aycn">gradhessian</th>
    <th class="tg-aycn">gradhessianv</th>
    <th class="tg-aycn">laplacian</th>
    <th class="tg-wklz">jacobian</th>
    <th class="tg-1r2d">jacobianv</th>
    <th class="tg-1r2d">jacobianT</th>
    <th class="tg-1r2d">jacobianTv</th>
    <th class="tg-a60z">curl</th>
    <th class="tg-a60z">div</th>
    <th class="tg-a60z">curldiv</th>
  </tr>
  <tr>
    <td class="tg-7dqz">AD</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">Numerical</td>
    <td class="tg-wcxf">A</td>
    <td class="tg-wcxf">A</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">A</td>
    <td class="tg-aycn">A</td>
    <td class="tg-aycn">A</td>
    <td class="tg-aycn">A</td>
    <td class="tg-aycn">A</td>
    <td class="tg-aycn">A</td>
    <td class="tg-aycn">A</td>
    <td class="tg-wklz">A</td>
    <td class="tg-1r2d">A</td>
    <td class="tg-1r2d">A</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">A</td>
    <td class="tg-1r2d">A</td>
    <td class="tg-1r2d">A</td>
  </tr>
  <tr>
    <td class="tg-7dqz">Symbolic</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-wklz">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
</table>

**Yellow**: For scalar-to-scalar functions; **Green**: For vector-to-scalar functions; **Blue**: For vector-to-vector functions

**X**: Exact value; **A**: Numerical approximation

The main functionality in DiffSharp is provided by the **DiffSharp.AD** namespace, which supports nesting and can combine forward and reverse AD as needed. We also implement symbolic and numerical differentiation. Currently the library provides the following namespaces:

- **DiffSharp.AD**: Nested AD
- **DiffSharp.Numerical**: Numerical differentiation with finite differences
- **DiffSharp.Symbolic**: Symbolic differentiation using code quotations
- **DiffSharp.Interop**: Nested AD wrapper for easier [use from C#](csharp.html) and other non-functional languages

All these namespaces have **Float32** and **Float64** modules, giving you 32- or 64-bit floating point precision. (E.g. **DiffSharp.AD.Float32**, **DiffSharp.Interop.Float64**. )

For brief explanations of these implementations, please refer to the [Nested AD](gettingstarted-nestedad.html), [Numerical Differentiation](gettingstarted-numericaldifferentiation.html), and [Symbolic Differentiation](gettingstarted-symbolicdifferentiation.html) pages.

Differentiation Operations and Variants
---------------------------------------

The operations summarized in the above table have _prime-suffixed_ variants (e.g. **diff** and **diff'** ) that return tuples containing the value of the original function together with the value of the desired operation. This provides a performance advantage in the majority of AD operations, since the original function value is almost always computed during the same execution of the code, as a by-product of AD computations.
*)

open DiffSharp.AD.Float64

// Derivative of Sin(Sqrt(x)) at x = 2
let a = diff (fun x -> sin (sqrt x)) (D 2.)

// (Original value, derivative) of Sin(Sqrt(x)) at x = 2
let b, c = diff' (fun x -> sin (sqrt x)) (D 2.)

(*** hide, define-output: o ***)
printf "val a : D = D 0.05513442203
val c : D = D 0.05513442203
val b : D = D 0.987765946"
(*** include-output: o ***)

(**

The library provides the following differentiation operations:

##### diff : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$
*)

diff f x

(**
The first derivative of a scalar-to-scalar function `f`, at the point `x`.

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the derivative evaluated at $x$

$$$
  \left. \frac{d}{da} f(a) \right|_{a\; =\; x} \; .

----------------------

##### diff' : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$
*)

diff' f x

(**
The original value and the first derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### diff2 : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$
*)

diff2 f x

(**
The second derivative of a scalar-to-scalar function `f`, at the point `x`.

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the second derivative evaluated at $x$

$$$
  \left. \frac{d^2}{da^2} f(a) \right|_{a\; =\; x} \; .

----------------------

##### diff2' : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$
*)

diff2' f x

(**
The original value and the second derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### diff2'' : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R} \times \mathbb{R})}$
*)

diff2'' f x

(**
The original value, the first derivative, and the second derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### diffn : $\color{red}{\mathbb{R} \to (\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$
*)

diffn n f x

(**
The `n`-th derivative of a scalar-to-scalar function `f`, at the point `x`.

For $n \in \mathbb{N}$, a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the n-th derivative evaluated at $x$

$$$
  \left. \frac{d^n}{da^n} f(a) \right|_{a\; =\; x} \; .

----------------------

##### diffn' : $\color{red}{\mathbb{R} \to (\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$
*)

diffn' n f x

(**
The original value and the `n`-th derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### grad : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^n}$
*)

grad f x

(**
The [gradient](http://en.wikipedia.org/wiki/Gradient) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the gradient evaluated at $\mathbf{x}$

$$$
  \left( \nabla f \right)_\mathbf{x} = \left. \left[ \frac{\partial f}{{\partial a}_1}, \dots, \frac{\partial f}{{\partial a}_n} \right] \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

----------------------

##### grad' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n)}$
*)

grad' f x

(**
The original value and the gradient of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$
*)

gradv f x v

(**
The [gradient-vector product](http://en.wikipedia.org/wiki/Directional_derivative) (directional derivative) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the dot product of the gradient of $f$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \nabla f \right)_\mathbf{x} \cdot \mathbf{v} \; .

This value can be computed efficiently by the **DiffSharp.AD.Forward** module in one forward evaluation of the function, without computing the full gradient.

----------------------

##### gradv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$
*)

gradv' f x v

(**
The original value and the gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### hessian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{n \times n}}$
*)

hessian f x

(**
The [Hessian](http://en.wikipedia.org/wiki/Hessian_matrix) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the Hessian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{H}_f \right)_\mathbf{x} = \left. \begin{bmatrix}
                                           \frac{\partial ^2 f}{\partial a_1^2} & \frac{\partial ^2 f}{\partial a_1 \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_1 \partial a_n} \\
                                           \frac{\partial ^2 f}{\partial a_2 \partial a_1} & \frac{\partial ^2 f}{\partial a_2^2} & \cdots & \frac{\partial ^2 f}{\partial a_2 \partial a_n} \\
                                           \vdots  & \vdots  & \ddots & \vdots  \\
                                           \frac{\partial ^2 f}{\partial a_n \partial a_1} & \frac{\partial ^2 f}{\partial a_n \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_n^2}
                                          \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

----------------------

##### hessian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^{n \times n})}$
*)

hessian' f x

(**
The original value and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### hessianv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^n}$
*)

hessianv f x v

(**
The [Hessian-vector product](http://en.wikipedia.org/wiki/Hessian_automatic_differentiation) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the multiplication of the Hessian matrix of $f$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{H}_f \right)_\mathbf{x} \; \mathbf{v} \; .

This value can be computed efficiently by the **DiffSharp.AD.ForwardReverse** module using one forward and one reverse evaluation of the function, in a matrix-free way (without computing the full Hessian matrix).

----------------------

##### hessianv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n)}$
*)

hessianv' f x v

(**
The original value and the Hessian-vector product of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### gradhessian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^n \times \mathbb{R}^{n \times n})}$
*)

gradhessian f x

(**
The gradient and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradhessian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n \times \mathbb{R}^{n \times n})}$
*)

gradhessian' f x

(**
The original value, the gradient, and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradhessianv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n)}$
*)

gradhessianv f x v

(**
The gradient-vector product (directional derivative) and the Hessian-vector product of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### gradhessianv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R} \times \mathbb{R}^n)}$
*)

gradhessianv' f x v

(**
The original value, the gradient-vector product (directional derivative), and the Hessian-vector product of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### laplacian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$
*)

laplacian f x

(**
The [Laplacian](http://en.wikipedia.org/wiki/Laplace_operator#Laplace.E2.80.93Beltrami_operator) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the sum of second derivatives evaluated at $\mathbf{x}$

$$$
  \mathrm{tr}\left(\mathbf{H}_f \right)_\mathbf{x} = \left. \left(\frac{\partial ^2 f}{\partial a_1^2} + \dots + \frac{\partial ^2 f}{\partial a_n^2}\right) \right|_{\mathbf{a} \; = \; \mathbf{x}} \; ,

which is the trace of the Hessian matrix.

This value can be computed efficiently by the **DiffSharp.AD.Forward2** module, without computing the full Hessian matrix.

----------------------

##### laplacian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$
*)

laplacian' f x

(**
The original value and the Laplacian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### jacobian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{m \times n}}$
*)

jacobian f x

(**
The [Jacobian](http://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $m$-by-$n$ Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_1}{\partial a_n} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_m}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

----------------------

##### jacobian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^{m \times n})}$
*)

jacobian' f x

(**
The original value and the Jacobian of a vector-to-vector function `f`, at the point `x`.

----------------------

##### jacobianv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^m}$
*)

jacobianv f x v

(**
The Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives matrix product of the Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} \mathbf{v} \; .
  
This value can be computed efficiently by the **DiffSharp.AD.Forward** module in one forward evaluation of the function, in a matrix-free way (without computing the full Jacobian matrix).

----------------------

##### jacobianv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^m)}$
*)

jacobianv' f x v

(**
The original value and the Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

----------------------

##### jacobianT : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{n \times m}}$
*)

jacobianT f x

(**
The transposed Jacobian of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $n$-by-$m$ transposed Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_1} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_1}{\partial a_n} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

----------------------

##### jacobianT' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^{n \times m})}$
*)

jacobianT' f x

(**
The original value and the transposed Jacobian of a vector-to-vector function `f`, at the point `x`.

----------------------

##### jacobianTv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^m} \to \color{blue}{\mathbb{R}^n}$
*)

jacobianTv f x v

(**
The transposed Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$, and $\mathbf{v} \in \mathbb{R}^m$, this gives the matrix product of the transposed Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} \mathbf{v} \; .
  
This value can be computed efficiently by the **DiffSharp.AD.Reverse** module in one forward and one reverse evaluation of the function, in a matrix-free way (without computing the full Jacobian matrix).

----------------------

##### jacobianTv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^m} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^n)}$
*)

jacobianTv' f x v

(**
The original value and the transposed Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

----------------------

##### jacobianTv'' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times (\mathbb{R}^m \to \mathbb{R}^n))}$
*)

jacobianTv'' f x

(**
The original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. 

Of the returned pair, the first is the original value of `f` at the point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can be used to compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of the reverse mode AD, with the given vector, without repeating the forward pass).

This can be computed efficiently by the **DiffSharp.AD.Reverse** module in a matrix-free way (without computing the full Jacobian matrix).

##### curl : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{\mathbb{R}^3}$
*)

curl f x

(**
The [curl](http://en.wikipedia.org/wiki/Curl_(mathematics)) of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^3 \to \mathbb{R}^3$ with components $F_1(a_1, a_2, a_3),\; F_2(a_1, a_2, a_3),\; F_3(a_1, a_2, a_3)$, and $\mathbf{x} \in \mathbb{R}^3$, this gives

$$$
  \left( \textrm{curl} \, \mathbf{F} \right)_{\mathbf{x}} = \left( \nabla \times \mathbf{F} \right)_{\mathbf{x}}= \left. \left[ \frac{\partial F_3}{\partial a_2} - \frac{\partial F_2}{\partial a_3}, \; \frac{\partial F_1}{\partial a_3} - \frac{\partial F_3}{\partial a_1}, \; \frac{\partial F_2}{\partial a_1} - \frac{\partial F_1}{\partial a_2} \right] \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

##### curl' : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{(\mathbb{R}^3 \times \mathbb{R}^3)}$
*)

curl' f x

(**
The original value and the curl of a vector-to-vector function `f`, at the point `x`.

##### div : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^n) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$
*)

div f x

(**
The [divergence](http://en.wikipedia.org/wiki/Divergence) of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ with components $F_1(a_1, \dots, a_n),\; \dots, \; F_n(a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives

$$$
  \left( \textrm{div} \, \mathbf{F} \right)_{\mathbf{x}} = \left( \nabla \cdot \mathbf{F} \right)_{\mathbf{x}} = \textrm{tr}\left( \mathbf{J}_{\mathbf{F}} \right)_{\mathbf{x}} = \left. \left( \frac{\partial F_1}{\partial a_1} + \dots + \frac{\partial F_n}{\partial a_n}\right) \right|_{\mathbf{a}\; = \; \mathbf{x}} \; ,

which is the trace of the Jacobian matrix evaluated at $\mathbf{x}$.

##### div' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^n) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^n \times \mathbb{R})}$
*)

div' f x

(**
The original value and the divergence of a vector-to-vector function `f`, at the point `x`.

##### curldiv : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{(\mathbb{R}^3 \times \mathbb{R})}$
*)

curldiv f x

(**
The curl and the divergence of a vector-to-vector function `f`, at the point `x`.

##### curldiv' : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{(\mathbb{R}^3 \times \mathbb{R}^3 \times \mathbb{R})}$
*)

curldiv' f x

(**
The original value, the curl, and the divergence of a vector-to-vector function `f`, at the point `x`.

Linear Algebra Operations
-------------------------

The library provides the **D**, **DV**, and **DM** types for scalar, vector, and matrix values. The underlying linear algebra operations when using these types are performed by

The following sections show only a small selection of operations commonly used with these types. For a full list of supported operations, please see the [API Reference](reference/index.html).

### Vector Operations

#### Creating Vectors
*)

open DiffSharp.AD.Float64
open DiffSharp.Util

let v1 = toDV [1.; 2.; 3.] // Create DV from sequence of floats
let v2 = DV.create 3 1.    // Create DV of length 3, each element with value 1.
let v3 = DV.init 3 (fun i -> exp (float i)) // Create DV of length 3, compute elements by a function

(**
#### Basic Operations
*)

let v4  = v1 + v2  // Vector addition
let v5  = v1 - v2  // Vector subtraction
let v6  = v1 * v2  // Vector inner (dot, scalar) product
let v8  = v1 &* v2 // Vector outer (dyadic, tensor) product
let v7  = v1 .* v2 // Element-wise (Hadamard) product
let v9  = v1 ./ v2 // Element-wise division
let v10 = v1 ** v2 // Element-wise exponentiation
let v11 = atan2 v1 v2 // Element-wise atan2
let v12 = v1 + 2.  // Add scalar to vector
let v13 = v1 - 2.  // Subtract scalar from vector
let v14 = 2. - v1  // Subtract each element of vector from scalar
let v15 = v1 * 2.  // Vector-scalar multiplication
let v16 = v1 / 2.  // Vector-scalar division
let v17 = 2. / v1  // Divide each element of vector by scalar
let v18 = -v1      // Unary negation
let v19 = log v1
let v20 = log10 v1
let v21 = exp v1
let v22 = sin v1
let v23 = cos v1
let v24 = tan v1
let v25 = sqrt v1
let v26 = sinh v1
let v27 = cosh v1
let v28 = tanh v1
let v29 = asin v1
let v30 = acos v1
let v31 = atan v1
let v32 = abs v1
let v33 = signum v1
let v34 = floor v1
let v35 = ceil v1
let v36 = round v1
let v37 = softmax v1
let v38 = softplus v1
let v39 = softsign v1
let v40 = logsumexp v1
let v41 = sigmoid v1
let v42 = reLU v1

(**
#### Vector Norms
*)

let n1 = DV.l1norm v1   // L1 norm
let n2 = DV.l2norm v1   // L2 norm
let n3 = DV.l2normSq v1 // Squared L2 norm

(**
#### Accessing Elements & Conversions
*)

let e1 = v1.[0]    // 1st element of v1
let e2 = v1.[..1]  // Slice of v1, until 2nd element
let e3 = v1.[1..2] // Slice of v2, between 2nd and 3rd elements
let v43 = DV [|1.; 2.; 3.|]       // Create DV from float[]
let v44 = toDV [|1.; 2.; 3.|]     // Create DV from sequence of floats
let v45 = toDV [1.; 2.; 3.]       // Create DV from sequence of floats
let v46 = toDV [D 1.; D 2.; D 3.] // Create DV from sequence of Ds
let a1:float[] = convert v1       // Convert DV to array of floats

(**
#### Splitting and Concatenating
*)

let ss1 = DV.splitEqual 3 (toDV [1.; 2.; 3.; 4.; 5.; 6.]) // Split DV into 3 vectors of equal length
let ss2 = DV.split [2; 4] (toDV [1.; 2.; 3.; 4.; 5.; 6.]) // Split DV into vectors of given lengths
let cc1 = DV.concat ss1 // Concatenate sequence of DVs into one

(**
#### Mathematica and MATLAB Strings
*)

let s1 = v1.ToMathematicaString()
let s2 = v1.ToMatlabString()

(**
#### Other Operations
*)

let len = DV.length v1 // Length of DV
let min = DV.min v1    // Minimum element of DV
let max = DV.max v1    // Maximum element of DV
let sum = DV.sum v1    // Sum of elements of DV
let v47 = DV.unitVector v1 // Unit vector codirectional with v1
let v48 = DV.normalize v1  // Normalize vector to have zero mean and unit variance

(**
### Matrix Operations

#### Creating Matrices
*)

open DiffSharp.AD.Float64

let m1 = toDM [[1.; 2.]; [3.; 4.]]
let m2 = DM.create 2 2 1.
let m3 = DM.init 2 2 (fun i j -> exp (float (i + j)))

(**
#### Basic Operations
*)

let m4 = m1 + m2