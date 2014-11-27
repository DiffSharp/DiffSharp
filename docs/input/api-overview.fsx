(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
API Overview
============

The following table gives an overview of the differentiation API provided by the DiffSharp library.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-5nhg{font-size:11px;background-color:#ecf4ff;text-align:center}
.tg .tg-0e45{font-size:11px}
.tg .tg-sgic{font-size:11px;background-color:#ffffc7;text-align:center}
.tg .tg-71xk{font-size:11px;background-color:#e4ffb3;text-align:center}
.tg .tg-sfug{font-size:11px;background-color:#ecf4ff;color:#000000;text-align:center}
.tg .tg-nl5m{font-weight:bold;font-size:11px}
</style>
<table class="tg">
  <tr>
    <th class="tg-0e45"></th>
    <th class="tg-sgic">diff</th>
    <th class="tg-sgic">diff2</th>
    <th class="tg-sgic">diffn</th>
    <th class="tg-71xk">grad</th>
    <th class="tg-71xk">gradv</th>
    <th class="tg-71xk">hessian</th>
    <th class="tg-71xk">gradhessian</th>
    <th class="tg-71xk">laplacian</th>
    <th class="tg-sfug">jacobian</th>
    <th class="tg-5nhg">jacobianv</th>
    <th class="tg-5nhg">jacobianT</th>
    <th class="tg-5nhg">jacobianTv</th>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-sfug">X</td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Forward2</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-sfug">X</td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardG</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-sfug">X</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardGH</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-sfug">X</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.ForwardN</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.AD.Reverse</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic"></td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">A</td>
    <td class="tg-71xk">XA</td>
    <td class="tg-71xk">A</td>
    <td class="tg-sfug">X</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg">X</td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Numerical</td>
    <td class="tg-sgic">A</td>
    <td class="tg-sgic">A</td>
    <td class="tg-sgic"></td>
    <td class="tg-71xk">A</td>
    <td class="tg-71xk">A</td>
    <td class="tg-71xk">A</td>
    <td class="tg-71xk">A</td>
    <td class="tg-71xk">A</td>
    <td class="tg-sfug">A</td>
    <td class="tg-5nhg">A</td>
    <td class="tg-5nhg">A</td>
    <td class="tg-5nhg"></td>
  </tr>
  <tr>
    <td class="tg-nl5m">DiffSharp.Symbolic</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic">X</td>
    <td class="tg-sgic">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk"></td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-71xk">X</td>
    <td class="tg-sfug">X</td>
    <td class="tg-5nhg"></td>
    <td class="tg-5nhg">X</td>
    <td class="tg-5nhg"></td>
  </tr>
</table>

**Yellow**: For scalar-to-scalar functions; **Green**: For vector-to-scalar functions; **Blue**: For vector-to-vector functions

**X**: Exact value; **A**: Numerical approximation; **XA**: Exact gradient, approximated Hessian

Differentiation Operations and Variants
---------------------------------------

The operations summarized in the above table have _prime-suffixed_ variants (e.g. **diff** and **diff'** ) that return tuples of (_the value of the original function_, _the value of the desired operation_). This is advantageous in the majority of AD operations, since the original function value would have been already computed as a by-product of AD computations, providing a performance advantage. 
*)

// Use forward AD
open DiffSharp.AD.Forward

// Derivative of Sin(Sqrt(x)) at x = 2
let a = diff (fun x -> sin (sqrt x)) 2.

// (Original value, derivative) of Sin(Sqrt(x)) at x = 2
let b, c = diff' (fun x -> sin (sqrt x)) 2.

(**

Currently, the library provides the following operations:

##### diff : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$

**`diff f x`** returns the first derivative of a scalar-to-scalar function `f`, at the point `x`.

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the derivative evaluated at $x$

$$$
  \left. \frac{d}{da} f(a) \right|_{a\; =\; x} .

----------------------

##### diff' : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`diff' f x`** returns the original value and the first derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### diff2 : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$

**`diff2 f x`** returns the second derivative of a scalar-to-scalar function `f`, at the point `x`.

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the second derivative evaluated at $x$

$$$
  \left. \frac{d^2}{da^2} f(a) \right|_{a\; =\; x} .

----------------------

##### diff2' : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`diff2' f x`** returns the original value and the second derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### diff2'' : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R} \times \mathbb{R})}$

**`diff2'' f x`** returns the original value, the first derivative, and the second derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### diffn : $\color{red}{\mathbb{R} \to (\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$

**`diffn n f x`** returns the `n`-th derivative of a scalar-to-scalar function `f`, at the point `x`.

For $n \in \mathbb{N}$, a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the n-th derivative evaluated at $x$

$$$
  \left. \frac{d^n}{da^n} f(a) \right|_{a\; =\; x} .

----------------------

##### diffn' : $\color{red}{\mathbb{R} \to (\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`diffn' n f x`** returns the original value and the `n`-th derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### grad : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^n}$

**`grad f x`** returns the [gradient](http://en.wikipedia.org/wiki/Gradient) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, a_2, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the gradient evaluated at $\mathbf{x}$

$$$
  \left( \nabla f \right)_\mathbf{x} = \left. \left[ \frac{\partial f}{{\partial a}_1}, \frac{\partial f}{{\partial a}_2}, \dots, \frac{\partial f}{{\partial a}_n} \right] \right|_{\mathbf{a}\; = \; \mathbf{x}} .

----------------------

##### grad' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n)}$

**`grad' f x`** returns the original value and the gradient of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$

**`gradv f x v`** returns the [gradient-vector product](http://en.wikipedia.org/wiki/Directional_derivative) (directional derivative) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the dot product of the gradient of $f$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \nabla f \right)_\mathbf{x} \cdot \mathbf{v} \; .

This value can be computed by the **DiffSharp.AD.Forward** module in an efficient way.

----------------------

##### gradv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`gradv' f x v`** returns the original value and the gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### hessian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{n \times n}}$

**`hessian f x`** returns the [Hessian](http://en.wikipedia.org/wiki/Hessian_matrix) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, a_2, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the Hessian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{H}_f \right)_\mathbf{x} = \left. \begin{bmatrix}
                                           \frac{\partial ^2 f}{\partial a_1^2} & \frac{\partial ^2 f}{\partial a_1 \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_1 \partial a_n} \\
                                           \frac{\partial ^2 f}{\partial a_2 \partial a_1} & \frac{\partial ^2 f}{\partial a_2^2} & \cdots & \frac{\partial ^2 f}{\partial a_2 \partial a_n} \\
                                           \vdots  & \vdots  & \ddots & \vdots  \\
                                           \frac{\partial ^2 f}{\partial a_n \partial a_1} & \frac{\partial ^2 f}{\partial a_n \partial a_2} & \cdots & \frac{\partial ^2 f}{\partial a_n^2}
                                          \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} .

----------------------

##### hessian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^{n \times n})}$

**`hessian' f x`** returns the original value and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradhessian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^n \times \mathbb{R}^{n \times n})}$

**`gradhessian f x`** returns the gradient and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradhessian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n \times \mathbb{R}^{n \times n})}$

**`gradhessian' f x`** returns the original value, the gradient, and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### laplacian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$

**`laplacian f x`** returns the [Laplacian](http://en.wikipedia.org/wiki/Laplace_operator#Laplace.E2.80.93Beltrami_operator) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, a_2, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the sum of second derivatives evaluated at $\mathbf{x}$

$$$
  \mathrm{tr}\left(\mathbf{H}_f \right)_\mathbf{x} = \left. \left(\frac{\partial ^2 f}{\partial a_1^2} + \frac{\partial ^2 f}{\partial a_2^2} + \dots + \frac{\partial ^2 f}{\partial a_n^2}\right) \right|_{\mathbf{a} \; = \; \mathbf{x}} ,

which is the trace of the Hessian matrix.

This value can be computed by the **DiffSharp.AD.Forward2** module in a matrix-free and efficient way.

----------------------

##### laplacian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`laplacian' f x`** returns the original value and the Laplacian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### jacobian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{m \times n}}$

**`jacobian f x`** returns the [Jacobian](http://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, a_2, \dots, a_n), \dots, F_m (a_1, a_2, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $m$-by-$n$ Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_1}{\partial a_n} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_m}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} .

----------------------

##### jacobian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^{m \times n})}$

**`jacobian' f x`** returns the original value and the Jacobian of a vector-to-vector function `f`, at the point `x`.

----------------------

##### jacobianv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^m}$

**`jacobianv f x v`** returns the Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives matrix product of the Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} \mathbf{v} \; .
  
This value can be computed by the **DiffSharp.AD.Forward** module in a matrix-free and efficient way.

----------------------

##### jacobianv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^m)}$

**`jacobianv' f x v`** returns the original value and the Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

----------------------

##### jacobianT : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{n \times m}}$

**`jacobianT f x`** returns the transposed Jacobian of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, a_2, \dots, a_n), \dots, F_m (a_1, a_2, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $n$-by-$m$ transposed Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_1} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_1}{\partial a_n} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} .

----------------------

##### jacobianT' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^{n \times m})}$

**`jacobianT' f x`** returns the original value and the transposed Jacobian of a vector-to-vector function `f`, at the point `x`.

----------------------

##### jacobianTv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^m} \to \color{blue}{\mathbb{R}^n}$

**`jacobianTv f x v`** returns the transposed Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$, and $\mathbf{v} \in \mathbb{R}^m$, this gives the matrix product of the transposed Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} \mathbf{v} \; .
  
This value can be computed by the **DiffSharp.AD.Reverse** module in a matrix-free and efficient way.

----------------------

##### jacobianTv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^m} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^n)}$

**`jacobianTv' f x v`** returns the original value and the transposed Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

This can be computed by the **DiffSharp.AD.Reverse** module in a matrix-free and efficient way.

----------------------

##### jacobianTv'' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times (\mathbb{R}^m \to \mathbb{R}^n))}$

**`jacobianTv'' f x`** returns the original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. 

Of the returned pair, the first is the original value of function `f` at point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of the reverse mode AD, with the given vector, without repeating the forward pass).

This can be computed by the **DiffSharp.AD.Reverse** module in a matrix-free and efficient way.

Implemented Differentiation Techniques
----------------------

The main focus of the DiffSharp library is AD, but we also implement symbolic and numerical differentiation.

Currently, the library provides the following implementations in separate modules:

- **DiffSharp.AD.Forward**: Forward AD 
- **DiffSharp.AD.Forward2**: Forward AD, 2nd order
- **DiffSharp.AD.ForwardG**: Forward AD, keeping vectors of gradient components
- **DiffSharp.AD.ForwardGH**: Forward AD, keeping vectors of gradient components and matrices of Hessian components
- **DiffSharp.AD.ForwardN**: Forward AD, lazy higher-order
- **DiffSharp.AD.Reverse**: Reverse AD
- **DiffSharp.Numerical**: Numerical differentiation
- **DiffSharp.Symbolic**: Symbolic differentiation

For brief explanations of these implementations, please refer to the [Forward AD](gettingstarted-forwardad.html), [Reverse AD](gettingstarted-reversead.html), [Numerical Differentiation](gettingstarted-numericaldifferentiation.html), and [Symbolic Differentiation](gettingstarted-symbolicdifferentiation.html) pages.

Vector and Matrix versus float[] and float[,]
---------------------------------------------

When a differentiation module such as **DiffSharp.AD.Forward** is opened, the default operations involving vectors or matrices handle these via **float[]** and **float[,]** arrays.

*)

open DiffSharp.AD.Forward

// Gradient of a vector-to-scalar function
// g1: float[] -> float[]
// Inner lambda expression: Dual[] -> Dual
// i.e. take the function arguments as a float[] and return the gradient as a float[]
let g1 = grad (fun x -> sin (x.[0] * x.[1]))

(**

In addition to this, every module provides a **Vector** submodule containing versions of the same differentiation operators using the **Vector** and **Matrix** types instead of **float[]** and **float[,]**.

*)

open DiffSharp.AD.Forward.Vector

// Gradient of a vector-to-scalar function
// g2: Vector<float> -> Vector<float>
// Inner lambda expression: Vector<Dual> -> Dual
// i.e. take the function arguments as a Vector<float> and return the gradient as a Vector<float>
let g2 = grad (fun x -> sin (x.[0] * x.[1]))
