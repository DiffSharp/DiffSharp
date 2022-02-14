# 0.8.4
24 Aug 2019

By @cgravill

* **Fix:** correction in array slicing when used in variable updates #59 

# 0.8.3
4 Jul 2019

By @cgravill

* **Improvement:** put the unmanaged/native dependencies in /runtime directory in the NuGet for better .NET Core support

# 0.8.2
25 Jun 2019

By @cgravill

* **Fix:** initialisation using a constant matrix on the left of multiplication #56 

# 0.8.1
20 Jun 2019

By @cgravill

* **Fix:** single-precision computeAdjoints with non-scalar values #55 

# 0.8.0
11 Jun 2019

By @cgravill

* **Improvement:** Moved to .NET Standard to allow targeting .NET Core as well as .NET Standard
* **Note:** There are some breaking API changes in this pre-release. Conversion examples are welcome.

# 0.7.7
25 Dec 2015

- **Fixed**: Bug fix in forward AD implementation of `Sigmoid` and `ReLU` for `D`, `DV`, and `DM` (fixes #16, thank you @mrakgr )
- **Improvement**: Performance improvement by removing several more `Parallel.For` and `Array.Parallel.map` operations, working better with OpenBLAS multithreading
- **Added**: Operations involving incompatible dimensions of `DV` and `DM` will now throw exceptions for warning the user

# 0.7.6
15 Dec 2015

- **Fixed**: Bug fix in LAPACK wrappers `ssysv` and `dsysv` in the OpenBLAS backend that caused incorrect solution for linear systems described by a symmetric matrix (fixes #11, thank you @grek142)
- **Added**: Added unit tests covering the whole backend interface

# 0.7.5
6 Dec 2015

- **Improved**: Performance improvement thanks to faster `Array2D.copy` operations (thank you Don Syme @dsyme)
- **Improved**: Significantly faster matrix transposition using extended BLAS operations `cblas_?omatcopy` provided by OpenBLAS
- **Improved**: Performance improvement by disabling parts of the OpenBLAS backend using `System.Threading.Tasks`, which was interfering with OpenBLAS multithreading. Pending further tests.
- **Update**: Updated the Win64 binaries of OpenBLAS to version 0.2.15 (27-10-2015), which has bug fixes and optimizations. Change log [here](http://www.openblas.net/Changelog.txt)
- **Fixed**: Bug fixes in reverse AD operations `Sub_D_DV` and `Sub_D_DM` (fixes #8, thank you @mrakgr)
- **Fixed**: Fixed bug in the benchmarking module causing incorrect reporting of the overhead factor of the AD `grad` operation
- **Improved**: Documentation updates

# 0.7.4
13 Oct 2015

- **Improved**: Overall performance improvements with parallelization and memory reshaping in OpenBLAS backend
- **Fixed**: Bug fixes in reverse AD `Make_DM_ofDV` and `DV.Append`
- **Fixed**: Bug fixes in `DM` operations `map2Cols`, `map2Rows`, `mapi2Cols`, `mapi2Rows`
- **Added**: New operation `primalDeep` for the deepest primal value in nested AD values

# 0.7.3
6 Oct 2015

- **Fixed**: Bug fix in `DM.Min`
- **Added**: `Mean`, `Variance`, `StandardDev`, `Normalize`, and `Standardize` functions
- **Added**: Support for visualizations with configurable Unicode/ASCII palette and contrast

# 0.7.2
4 Oct 2015

- **Added**: Fast reshape operations `ReshapeCopy_DV_DM` and `ReshapeCopy_DM_DV`

# 0.7.1
4 Oct 2015

- **Fixed**: Bug fixes for reverse AD `Abs`, `Sign`, `Floor`, `Ceil`, `Round`, `DV.AddSubVector`, `Make_DM_ofDs`, `Mul_Out_V_V`, `Mul_DVCons_D`
- **Added**: New methods `DV.isEmpty` and `DM.isEmpty`

# 0.7.0
29 Sep 2015

Version 0.7.0 is a reimplementation of the library with support for **linear algebra primitives**, **BLAS/LAPACK**, **32- and 64-bit precision** and different **CPU/GPU backends**
- **Changed**: Namespaces have been reorganized and simplified. This is a breaking change. There is now just one AD implementation, under `DiffSharp.AD` (with `DiffSharp.AD.Float32` and `DiffSharp.AD.Float64` variants, see below). This internally makes use of forward or reverse AD as needed.
- **Added**: Support for 32 bit (single precision) and 64 bit (double precision) floating point operations. All modules have `Float32` and `Float64` versions providing the same functionality with the specified precision. 32 bit floating point operations are significantly faster (as much as twice as fast) on many current systems.
- **Added**: DiffSharp now uses the OpenBLAS library by default for linear algebra operations. The AD operations with the types `D` for scalars, `DV` for vectors, and `DM` for matrices use the underlying linear algebra backend for highly optimized native BLAS and LAPACK operations. For non-BLAS operations (such as Hadamard products and matrix transpose), parallel implementations in managed code are used. All operations with the `D`, `DV`, and `DM` types support forward and reverse nested AD up to any level. This also paves the way for GPU backends (CUDA/CuBLAS) which will be introduced in following releases. Please see the documentation and API reference for information about how to use the `D`, `DV`, and `DM` types. (**Deprecated**: The FsAlg generic linear algebra library and the `Vector<'T>` and `Matrix<'T>` types are no longer used.)
- **Fixed**: Reverse mode AD has been reimplemented in a tail-recursive way for better performance and preventing StackOverflow exceptions encountered in previous versions.
- **Changed**: The library now uses F# 4.0 (FSharp.Core 4.4.0.0).
- **Changed**: The library is now 64 bit only, meaning that users should set "x64" as the platform target for all build configurations.
- **Fixed**: Overall bug fixes.

# 0.6.3
18 Jul 2015

- **Fixed:** Bug fix in `DiffSharp.AD` subtraction operation between `D` and `DF`

# 0.6.2
6 Jun 2015

- **Changed**: Update FsAlg to 0.5.8

# 0.6.1
3 Jun 2015

- **Added**: Support for C#, through the new `DiffSharp.Interop` namespace
- **Added**: Support for casting AD types to `int`
- **Changed**: Update FsAlg to 0.5.6
- **Improved**: Documentation updates

# 0.6.0
27 Apr 2015

- **Changed:** DiffSharp is now released under the LGPL license, allowing use (as a dynamically linked library) in closed-source projects and open-source projects under non-GPL licenses
- **Added:** Nesting support. The modules `DiffSharp.AD`, `DiffSharp.AD.Forward` and `DiffSharp.AD.Reverse` are now the main components of the library, providing support for nested AD operations.
- **Changed:** The library now uses the FsAlg linear algebra library for handling vector and matrix operations and interfaces
- **Changed:** All AD-enabled numeric types in the library are now called `D`
- **Changed:** The non-nested modules `DiffSharp.AD.Forward`, `DiffSharp.AD.Forward2`, `DiffSharp.AD.ForwardG`, `DiffSharp.AD.ForwardGH`, `DiffSharp.AD.ForwardN`, `DiffSharp.AD.Reverse` are now called `DiffSharp.AD.Specialized.Forward1`, `DiffSharp.AD.Specialized.Forward2`, `DiffSharp.AD.Specialized.ForwardG`, `DiffSharp.AD.Specialized.ForwardGH`, `DiffSharp.AD.Specialized.ForwardN`, `DiffSharp.AD.Specialized.Reverse1`
- **Improved:** The non-nested `DiffSharp.AD.Specialized.Reverse1` module is reimplemented from scratch, not requiring a stack
- **Removed:** The non-nested `DiffSharp.AD.ForwardReverse` module is removed. This functionality is now handled by the nested modules.
- **Improved:** Major rewrite of documentation and examples, to reflect changed library structure
- **Improved:** Updated benchmarks

# 0.5.10
27 Mar 2015

- **Improved:** Improvements in the `DiffSharp.Util.LinearAlgebra` module
- **Changed:** Minor changes in the internal API, such as `dualSet` -> `dualPT`, `dualAct` -> `dualP1`

# 0.5.9
26 Feb 2015

- **Added:** New operations `curl`, `div`, and `curldiv` for the curl and divergence of vector-to-vector functions
- **Improved:** Improvements in the `DiffSharp.Util.LinearAlgebra` module

# 0.5.8
23 Feb 2015

- **Fixed:** Bug fixes in all AD modules, where some unhandled cases of undefined derivatives caused `infinity` or `nan` values (such as `log(0)`)
- **Removed:** `Q` and `R` numeric literals are no longer used by `DiffSharp.Forward.AD`. Reserving these for future nested AD use.
- **Added:** New vector norms: `l1norm`, `l2norm`, `l2normSq`, `lpnorm`
- **Improved:** Documentation updates

# 0.5.7
17 Feb 2015

- **Improved:** Major performance increase due to optimized binary
- **Improved:** Overhaul of `DiffSharp.Symbolic` module, symbolic derivatives are simplified before compilation, giving better performance
- **Improved:** `DiffSharp.AD.Reverse` module now handles binary operations of `Adj` and constants better
- **Fixed:** Fixed a minor bug in several operation signatures affecting backward compatibility

# 0.5.6
13 Feb 2015

- **Added:** New module `DiffSharp.AD.ForwardReverse` implementing reverse-on-forward AD
- **Added:** New operation `hessianv`, returning the Hessian-vector product
- **Added:** New operation `gradhessianv`, returning the gradient- and Hessian-vector product
- **Added:** New operation `jacobianvTv`, returning the Jacobian-vector product and the transposed Jacobian-vector product
- **Improved:** `DiffSharp.Symbolic` module now allows compiling derivatives of functions. The returned derivative functions run many times faster than the initial one-off compilation step.
- **Improved:** Benchmarks now show compilation and use times of `DiffSharp.Symbolic` operations separately.
- **Improved:** Overall revision of documentation

# 0.5.5
15 Dec 2014

- **Added:** Support for `abs`, `atan2`, `ceil`, `floor`, `log10`, `round` operations
- **Added:** Comparison support for the `Adj` type
- **Added:** Improved `DiffSharp.Util.LinearAlgebra` module, with support for LU and QR decompositions, matrix inverse, determinant, eigenvalues
- **Improved:** `DiffSharp.Symbolic` module reimplemented to use `FSharp.Quotations.Evaluator`
- **Improved:** Updated documentation with better explanations and more examples
- **Fixed:** Overall bug fixes

# 0.5.4
23 Nov 2014

- **Added:** New operation `jacobianTv''`, returning the original value of a function and a reverse mode AD evaluator for the transposed Jacobian-vector product
- **Added:** Generic vector and matrix support, e.g. `Vector<float>`, `Vector<Dual>`, `Vector<Adj>`
- **Fixed:** Fixed minor bugs in `DiffSharp.Util.LinearAlgebra`
- **Changed:** Vector mode differentiation operations now require the original function to be defined using vectors instead of arrays, e.g. `f:Vector<Dual> -> Dual` instead of `f:Dual[] -> Dual` for `grad f`
- **Improved:** Updates in API documentation

# 0.5.3
7 Nov 2014

- **Fixed:** Fixed bug in `DiffSharp.AD.ForwardG` and `DiffSharp.AD.ForwardGH` modules, causing error with `Array.sum` operations
- **Added:** New operation `jacobianTv`, computing the transposed Jacobian-vector product in a matrix-free and highly efficient way
- **Added:** Support for comparisons of dual number types, enabling uses with `Array.sort` etc.
- **Added:** Command line benchmarking tool
- **Improved:** Updates in API documentation

# 0.5.2
4 Nov 2014

- **Added:** New operation `jacobianv`, computing the Jacobian-vector product in a matrix-free and highly efficient way
- **Changed:** The `diffdir` operation is now called `gradv`, for "gradient-vector product"
- **Improved:** Updates in API documentation

# 0.5.1
2 Nov 2014

- **Added:** New operation `diff2''`, returning the _original value_, _first derivative_, and _second derivative_ of a scalar-to-scalar function
- **Added:** Support for operations involving `int` type
- **Fixed:** Bug fixes in `DiffSharp.AD.ForwardN`
- **Improved:** Overall performance improvements

# 0.5.0
2 Nov 2014

* Initial release

