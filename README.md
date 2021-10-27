# PLoM

The **PLoM** is an open source python package that implements the algorithm of **Probabilistic Learning on Manifolds** with and without constraints ([Soize and Ghanem, 2016](https://doi.org/10.1016/j.jcp.2016.05.044); [Soize and Ghanem, 2019](https://doi.org/10.1002/nme.6202)) for ***generating realizations of a random vector in a finite Euclidean space that are statistically consistent with a given dataset of that vector***. The package mainly consists of python modules and invokes a dynamic library for more efficiently computing the gradient of the potential, which could be imported and run on Linux, macOS, and Windows platform. This repository also archives the unit/integration tests and examples of applying the algorithm to practical engineering problems.

## Documentation
### General
* [About](doc/about.md)
* [Calculation workflow](doc/calculation-workflow.md)
* [Modules and API](doc/modules.md)
* [Continuous integration and testing](doc/testing.md)
### Installation
* [Dependencies and requirements](doc/requirements.md)
* [Compilation](doc/compilation.md)
### Examples
* [Example 0: Simple example in 20 dimensions](example/example0/ExampleScript_20D.ipynb)
* [Example 1: Simple example in 2 dimensions with constraints](example/example1/ExampleScript_2D.ipynb)
* [Example 2: Surrogating MSA of a 12-story RC frame](example/example2/ExampleScript_FullMSA.ipynb)
* [Example 3: Surrogating IDA of a 12-story RC frame](example/example3/ExampleScript_IDA.ipynb)
* [Example 4: Application in damage and loss assessment](example/example4/ExampleScript_DL.ipynb)

## Acknowledgement
