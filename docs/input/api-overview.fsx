(*** hide ***)
#r "../../src/DiffSharp/bin/Debug/FsAlg.dll"
#r "../../src/DiffSharp/bin/Debug/DiffSharp.dll"

(**
API Overview
============

The following table gives an overview of the differentiation API provided by the DiffSharp library.

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
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
    <td class="tg-7dqz">ADF</td>
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
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">ADR</td>
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
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">SADF1</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-wklz">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">SADF2</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-wklz">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">SADFG</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-wklz">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">SADFGH</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
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
  <tr>
    <td class="tg-7dqz">SADFN</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">SADR1</td>
    <td class="tg-wcxf">X</td>
    <td class="tg-wcxf"></td>
    <td class="tg-wcxf"></td>
    <td class="tg-aycn">X</td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-aycn"></td>
    <td class="tg-wklz">X</td>
    <td class="tg-1r2d"></td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
    <td class="tg-1r2d">X</td>
  </tr>
  <tr>
    <td class="tg-7dqz">N</td>
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
    <td class="tg-7dqz">S</td>
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

The main functionality in DiffSharp is provided by the **DiffSharp.AD** namespace, which supports nesting and can combine forward and reverse AD as needed. The modules under **DiffSharp.AD.Specialized** provide several non-nested implementations of AD, for situations where you know beforehand that nesting will not be needed. This can give better performance for some specific non-nested tasks.

The main focus of the DiffSharp library is AD, but we also implement symbolic and numerical differentiation.

Currently, the library provides the following implementations:

- **AD | DiffSharp.AD**: Nested AD, forward and reverse mode
- **ADF | DiffSharp.AD.Forward**: Nested AD, only forward mode
- **ADR | DiffSharp.AD.Reverse**: Nested AD, only reverse mode
- **SADF1 | DiffSharp.AD.Specialized.Forward1**: Non-nested AD, forward mode, 1st order
- **SADF2 | DiffSharp.AD.Specialized.Forward2**: Non-nested AD, forward mode, 2nd order
- **SADFG | DiffSharp.AD.Specialized.ForwardG**: Non-nested AD, forward mode, keeping vectors of gradient components
- **SADFGH | DiffSharp.AD.Specialized.ForwardGH**: Non-nested AD, forward mode, keeping vectors of gradient components and matrices of Hessian components
- **SADFN | DiffSharp.AD.Specialized.ForwardN**: Non-nested AD, forward mode, lazy higher-order
- **SADR1 | DiffSharp.AD.Specialized.Reverse1**: Non-nested AD, reverse mode, 1st order
- **N | DiffSharp.Numerical**: Numerical differentiation
- **S | DiffSharp.Symbolic**: Symbolic differentiation

For brief explanations of these implementations, please refer to the [Nested AD](gettingstarted-nestedad.html), [Non-nested AD](gettingstarted-nonnestedad.html), [Numerical Differentiation](gettingstarted-numericaldifferentiation.html), and [Symbolic Differentiation](gettingstarted-symbolicdifferentiation.html) pages.

Differentiation Operations and Variants
---------------------------------------

The operations summarized in the above table have _prime-suffixed_ variants (e.g. **diff** and **diff'** ) that return tuples containing the value of the original function together with the value of the desired operation. This provides a performance advantage in the majority of AD operations, since the original function value is almost always computed during the same execution of the code, as a by-product of AD computations.
*)

open DiffSharp.AD

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

Currently, the library provides the following operations:

##### diff : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$

**`diff f x`** returns the first derivative of a scalar-to-scalar function `f`, at the point `x`.

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the derivative evaluated at $x$

$$$
  \left. \frac{d}{da} f(a) \right|_{a\; =\; x} \; .

----------------------

##### diff' : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`diff' f x`** returns the original value and the first derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### diff2 : $\color{red}{(\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{\mathbb{R}}$

**`diff2 f x`** returns the second derivative of a scalar-to-scalar function `f`, at the point `x`.

For a function $f(a): \mathbb{R} \to \mathbb{R}$, and $x \in \mathbb{R}$, this gives the second derivative evaluated at $x$

$$$
  \left. \frac{d^2}{da^2} f(a) \right|_{a\; =\; x} \; .

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
  \left. \frac{d^n}{da^n} f(a) \right|_{a\; =\; x} \; .

----------------------

##### diffn' : $\color{red}{\mathbb{R} \to (\mathbb{R} \to \mathbb{R}) \to \mathbb{R}} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`diffn' n f x`** returns the original value and the `n`-th derivative of a scalar-to-scalar function `f`, at the point `x`.

----------------------

##### grad : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^n}$

**`grad f x`** returns the [gradient](http://en.wikipedia.org/wiki/Gradient) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the gradient evaluated at $\mathbf{x}$

$$$
  \left( \nabla f \right)_\mathbf{x} = \left. \left[ \frac{\partial f}{{\partial a}_1}, \dots, \frac{\partial f}{{\partial a}_n} \right] \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

----------------------

##### grad' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n)}$

**`grad' f x`** returns the original value and the gradient of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$

**`gradv f x v`** returns the [gradient-vector product](http://en.wikipedia.org/wiki/Directional_derivative) (directional derivative) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the dot product of the gradient of $f$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \nabla f \right)_\mathbf{x} \cdot \mathbf{v} \; .

This value can be computed efficiently by the **DiffSharp.AD.Forward** module in one forward evaluation of the function, without computing the full gradient.

----------------------

##### gradv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`gradv' f x v`** returns the original value and the gradient-vector product (directional derivative) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### hessian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{n \times n}}$

**`hessian f x`** returns the [Hessian](http://en.wikipedia.org/wiki/Hessian_matrix) of a vector-to-scalar function `f`, at the point `x`.

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

**`hessian' f x`** returns the original value and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### hessianv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^n}$

**`hessianv f x v`** returns the [Hessian-vector product](http://en.wikipedia.org/wiki/Hessian_automatic_differentiation) of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

For a function $f: \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives the multiplication of the Hessian matrix of $f$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{H}_f \right)_\mathbf{x} \; \mathbf{v} \; .

This value can be computed efficiently by the **DiffSharp.AD.ForwardReverse** module using one forward and one reverse evaluation of the function, in a matrix-free way (without computing the full Hessian matrix).

----------------------

##### hessianv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n)}$

**`hessianv' f x v`** returns the original value and the Hessian-vector product of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### gradhessian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^n \times \mathbb{R}^{n \times n})}$

**`gradhessian f x`** returns the gradient and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradhessian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n \times \mathbb{R}^{n \times n})}$

**`gradhessian' f x`** returns the original value, the gradient, and the Hessian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### gradhessianv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R}^n)}$

**`gradhessianv f x v`** returns the gradient-vector product (directional derivative) and the Hessian-vector product of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### gradhessianv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R} \times \mathbb{R}^n)}$

**`gradhessianv' f x v`** returns the original value, the gradient-vector product (directional derivative), and the Hessian-vector product of a vector-to-scalar function `f`, at the point `x`, along the vector `v`.

----------------------

##### laplacian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$

**`laplacian f x`** returns the [Laplacian](http://en.wikipedia.org/wiki/Laplace_operator#Laplace.E2.80.93Beltrami_operator) of a vector-to-scalar function `f`, at the point `x`.

For a function $f(a_1, \dots, a_n): \mathbb{R}^n \to \mathbb{R}$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the sum of second derivatives evaluated at $\mathbf{x}$

$$$
  \mathrm{tr}\left(\mathbf{H}_f \right)_\mathbf{x} = \left. \left(\frac{\partial ^2 f}{\partial a_1^2} + \dots + \frac{\partial ^2 f}{\partial a_n^2}\right) \right|_{\mathbf{a} \; = \; \mathbf{x}} \; ,

which is the trace of the Hessian matrix.

This value can be computed efficiently by the **DiffSharp.AD.Forward2** module, without computing the full Hessian matrix.

----------------------

##### laplacian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R} \times \mathbb{R})}$

**`laplacian' f x`** returns the original value and the Laplacian of a vector-to-scalar function `f`, at the point `x`.

----------------------

##### jacobian : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{m \times n}}$

**`jacobian f x`** returns the [Jacobian](http://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $m$-by-$n$ Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_1}{\partial a_n} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_m}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

----------------------

##### jacobian' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^{m \times n})}$

**`jacobian' f x`** returns the original value and the Jacobian of a vector-to-vector function `f`, at the point `x`.

----------------------

##### jacobianv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^m}$

**`jacobianv f x v`** returns the Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, and $\mathbf{x}, \mathbf{v} \in \mathbb{R}^n$, this gives matrix product of the Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F} \right)_\mathbf{x} \mathbf{v} \; .
  
This value can be computed efficiently by the **DiffSharp.AD.Forward** module in one forward evaluation of the function, in a matrix-free way (without computing the full Jacobian matrix).

----------------------

##### jacobianv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^m)}$

**`jacobianv' f x v`** returns the original value and the Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

----------------------

##### jacobianT : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}^{n \times m}}$

**`jacobianT f x`** returns the transposed Jacobian of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$ with components $F_1 (a_1, \dots, a_n), \dots, F_m (a_1, \dots, a_n)$, and $\mathbf{x} \in \mathbb{R}^n$, this gives the $n$-by-$m$ transposed Jacobian matrix evaluated at $\mathbf{x}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} = \left. \begin{bmatrix}
                                                            \frac{\partial F_1}{\partial a_1} & \cdots & \frac{\partial F_m}{\partial a_1} \\
                                                            \vdots & \ddots & \vdots  \\
                                                            \frac{\partial F_1}{\partial a_n} & \cdots & \frac{\partial F_m}{\partial a_n}
                                                           \end{bmatrix} \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

----------------------

##### jacobianT' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^{n \times m})}$

**`jacobianT' f x`** returns the original value and the transposed Jacobian of a vector-to-vector function `f`, at the point `x`.

----------------------

##### jacobianTv : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^m} \to \color{blue}{\mathbb{R}^n}$

**`jacobianTv f x v`** returns the transposed Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$, $\mathbf{x} \in \mathbb{R}^n$, and $\mathbf{v} \in \mathbb{R}^m$, this gives the matrix product of the transposed Jacobian of $\mathbf{F}$ at $\mathbf{x}$ with $\mathbf{v}$

$$$
  \left( \mathbf{J}_\mathbf{F}^\textrm{T} \right)_\mathbf{x} \mathbf{v} \; .
  
This value can be computed efficiently by the **DiffSharp.AD.Reverse** module in one forward and one reverse evaluation of the function, in a matrix-free way (without computing the full Jacobian matrix).

----------------------

##### jacobianTv' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n \to \mathbb{R}^m} \to \color{blue}{(\mathbb{R}^m \times \mathbb{R}^n)}$

**`jacobianTv' f x v`** returns the original value and the transposed Jacobian-vector product of a vector-to-vector function `f`, at the point `x`, along the vector `v`.

----------------------

##### jacobianTv'' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^m) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^m \times (\mathbb{R}^m \to \mathbb{R}^n))}$

**`jacobianTv'' f x`** returns the original value and a function for evaluating the transposed Jacobian-vector product of a vector-to-vector function `f`, at point `x`. 

Of the returned pair, the first is the original value of `f` at the point `x` (the result of the forward pass of the reverse mode AD) and the second is a function (the reverse evaluator) that can be used to compute the transposed Jacobian-vector product many times along many different vectors (performing a new reverse pass of the reverse mode AD, with the given vector, without repeating the forward pass).

This can be computed efficiently by the **DiffSharp.AD.Reverse** module in a matrix-free way (without computing the full Jacobian matrix).

##### curl : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{\mathbb{R}^3}$

**`curl f x`** returns the [curl](http://en.wikipedia.org/wiki/Curl_(mathematics)) of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^3 \to \mathbb{R}^3$ with components $F_1(a_1, a_2, a_3),\; F_2(a_1, a_2, a_3),\; F_3(a_1, a_2, a_3)$ this gives

$$$
  \left( \textrm{curl} \, \mathbf{F} \right)_{\mathbf{x}} = \left( \nabla \times \mathbf{F} \right)_{\mathbf{x}}= \left. \left[ \frac{\partial F_3}{\partial a_2} - \frac{\partial F_2}{\partial a_3}, \; \frac{\partial F_1}{\partial a_3} - \frac{\partial F_3}{\partial a_1}, \; \frac{\partial F_2}{\partial a_1} - \frac{\partial F_1}{\partial a_2} \right] \right|_{\mathbf{a}\; = \; \mathbf{x}} \; .

##### curl' : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{(\mathbb{R}^3 \times \mathbb{R}^3)}$

**`curl' f x`** returns the original value and the curl of a vector-to-vector function `f`, at the point `x`.

##### div : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^n) \to \mathbb{R}^n} \to \color{blue}{\mathbb{R}}$

**`div f x`** returns the [divergence](http://en.wikipedia.org/wiki/Divergence) of a vector-to-vector function `f`, at the point `x`.

For a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ with components $F_1(a_1, \dots, a_n),\; \dots, \; F_n(a_1, \dots, a_n)$ this gives

$$$
  \left( \textrm{div} \, \mathbf{F} \right)_{\mathbf{x}} = \left( \nabla \cdot \mathbf{F} \right)_{\mathbf{x}} = \textrm{tr}\left( \mathbf{J}_{\mathbf{F}} \right)_{\mathbf{x}} = \left. \left( \frac{\partial F_1}{\partial a_1} + \dots + \frac{\partial F_n}{\partial a_n}\right) \right|_{\mathbf{a}\; = \; \mathbf{x}} \; ,

which is the trace of the Jacobian matrix.

##### div' : $\color{red}{(\mathbb{R}^n \to \mathbb{R}^n) \to \mathbb{R}^n} \to \color{blue}{(\mathbb{R}^n \times \mathbb{R})}$

**`div' f x`** returns the original value and the divergence of a vector-to-vector function `f`, at the point `x`.

##### curldiv : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{(\mathbb{R}^3 \times \mathbb{R})}$

**`curldiv f x`** returns the curl and the divergence of a vector-to-vector function `f`, at the point `x`.

##### curldiv' : $\color{red}{(\mathbb{R}^3 \to \mathbb{R}^3) \to \mathbb{R}^3} \to \color{blue}{(\mathbb{R}^3 \times \mathbb{R}^3 \times \mathbb{R})}$

**`curldiv' f x`** returns the original value, the curl, and the divergence of a vector-to-vector function `f`, at the point `x`.

Vector and Matrix versus float[] and float[,]
---------------------------------------------

When a differentiation module such as **DiffSharp.AD** is opened, the default operations involving vectors or matrices handle these via **float[]** and **float[,]** arrays.

*)

open DiffSharp.AD

// Gradient of a vector-to-scalar function
// g1: D[] -> D[]
// i.e., take the function arguments as a D[] and return the gradient as a D[]
// Inner lambda expression: D[] -> D
let g1 = grad (fun x -> sin (x.[0] * x.[1]))

// Compute the gradient at (1, 2)
// g1val: D[]
let g1val = g1 [|D 1.; D 2.|]

// Jacobian of a vector-to-vector function
// j1: D[] -> D[,]
// i.e., take the function arguments as a D[] and return the Jacobian as a D[,]
// Inner lambda expression: D[] -> D[]
let j1 = jacobian (fun x -> [|sin x.[0]; cos x.[1]|])

// Compute the Jacobian at (1, 2)
let j1val = j1 [|D 1.; D 2.|]

(*** hide, define-output: o2 ***)
printf "val g1 : (D [] -> D [])
val g1val : D [] = [|D -0.8322936731; D -0.4161468365|]
val j1 : (D [] -> D [,])
val j1val : D [,] = [[D 0.5403023059; D 0.0]
                     [D 0.0; D -0.9092974268]]"
(*** include-output: o2 ***)

(**

In addition to this, every module provides a **Vector** submodule containing versions of the same differentiation operators using the **Vector** and **Matrix** types instead of **float[]** and **float[,]**. This is advantageous in situations where you have to manipulate vectors in the rest of your algorithm. For instance, see the example on [Gradient Descent](examples-gradientdescent.html).

Vector and Matrix operations are handled through the [FsAlg Linear Algebra Library](http://gbaydin.github.io/FsAlg/), which supports most of the commonly used linear algebra operations, including matrix–vector operations, matrix inverse, determinants, eigenvalues, LU and QR decompositions. Please see the API reference of FsAlg for a complete list of supported linear algebra operations.

*)

open DiffSharp.AD
open DiffSharp.AD.Vector
open FsAlg.Generic

// Gradient of a vector-to-scalar function
// g2: Vector<D> -> Vector<D>
// i.e., take the function arguments as a Vector<D> and return the gradient as a Vector<D>
// Inner lambda expression: Vector<D> -> D
let g2 = grad (fun x -> sin (x.[0] * x.[1]))

// Compute the gradient at (1, 2)
// g2val: Vector<D>
let g2val = g2 (vector [D 1.; D 2.])

// Compute the negative of the gradient vector
let g2valb = -g2val

// Scale the gradient vector
let g2valc = D 0.2 * g2val

// Get the norm of the gradient vector
let g2vald = Vector.norm g2val

// Jacobian of a vector-to-vector function
// j2: Vector<D> -> Matrix<D>
// i.e., take the function arguments as a Vector<D> and return the Jacobian as a Matrix<D>
// Inner lambda expression: Vector<D> -> Vector<D>
let j2 = jacobian (fun x -> vector [sin x.[0]; cos x.[1]])

// Compute the Jacobian at (1, 2)
let j2val = j2 (vector [D 1.; D 2.])

(*** hide, define-output: o3 ***)
printf "val g2 : (Vector<D> -> Vector<D>)
val g2val : Vector<D> = Vector [|D -0.8322936731; D -0.4161468365|]
val g2valb : Vector<D> = Vector [|D 0.8322936731; D 0.4161468365|]
val g2valc : Vector<D> = Vector [|D -0.1664587346; D -0.08322936731|]
val g2vald : D = D 0.9305326151
val j2 : (Vector<D> -> Matrix<D>)
val j2val : Matrix<D> = Matrix [[D 0.5403023059; D 0.0]
                                [D 0.0; D -0.9092974268]]"
(*** include-output: o3 ***)