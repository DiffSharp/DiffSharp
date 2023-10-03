<div align="left">
  <a href="https://diffsharp.github.io"> <img height="80px" src="docs/img/diffsharp-logo-text.png"></a>
</div>

-----------------------------------------

[![Build Status](https://github.com/DiffSharp/DiffSharp/workflows/Build/test/docs/publish/badge.svg)](https://github.com/DiffSharp/DiffSharp/actions)
[![Coverage Status](https://coveralls.io/repos/github/DiffSharp/DiffSharp/badge.svg?branch=)](https://coveralls.io/github/DiffSharp/DiffSharp?branch=)

This is the development branch of DiffSharp 1.0.

> **NOTE: This branch is undergoing development. It has incomplete code, functionality, and design that are likely to change without notice; when using TorchSharp backend, only x64 platform is currently supported out of the box, see [DEVGUIDE.md] for more details.**

DiffSharp is a tensor library with support for [differentiable programming](https://en.wikipedia.org/wiki/Differentiable_programming). It is designed for use in machine learning, probabilistic programming, optimization and other domains.

**Key features**

* Nested and mixed-mode differentiation
* Common optimizers, model elements, differentiable probability distributions
* F# for robust functional programming
* PyTorch familiar naming and idioms, efficient LibTorch CUDA/C++ tensors with GPU support
* Linux, macOS, Windows supported
* Use interactive notebooks in Jupyter and Visual Studio Code
* 100% open source

## Documentation

You can find the documentation [here](https://diffsharp.github.io/), including information on installation and getting started.

Release notes can be found [here](https://github.com/DiffSharp/DiffSharp/blob/dev/RELEASE_NOTES.md).

## Communication

Please use [GitHub issues](https://github.com/DiffSharp/DiffSharp/issues) to share bug reports, feature requests, installation issues, suggestions etc.

## Contributing

We welcome all contributions.

* Bug fixes: if you encounter a bug, please open an [issue](https://github.com/DiffSharp/DiffSharp/issues) describing the bug. If you are planning to contribute a bug fix, please feel free to do so in a pull request.
* New features: if you plan to contribute new features, please first open an [issue](https://github.com/DiffSharp/DiffSharp/issues) to discuss the feature before creating a pull request.

## The Team

DiffSharp is developed by [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/), [Don Syme](https://www.microsoft.com/en-us/research/people/dsyme/) and other contributors, having started as a project supervised by the automatic differentiation wizards [Barak Pearlmutter](https://scholar.google.com/citations?user=AxFrw0sAAAAJ&hl=en) and [Jeffrey Siskind](https://scholar.google.com/citations?user=CgSBtPYAAAAJ&hl=en).

## License

DiffSharp is licensed under the BSD 2-Clause "Simplified" License, which you can find in the [LICENSE](https://github.com/DiffSharp/DiffSharp/blob/dev/LICENSE) file in this repository. 
