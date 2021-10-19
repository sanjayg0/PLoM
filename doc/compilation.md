## Compilation

The package invokes a dynamic library that computes the gradient of potential in constructing the Ito Stochastic Differential Equation (ISDE) to 
generate realizations. The dynamic library is compiled from [PLoM_C_library.c](../PLoM_C_library.c). Windows users could directly run the [make.bat](../make.bat) from the **PLoM** root (i.e., ```./make.bat```). Linux and macOS users could either run ```make all``` from the **PLoM** root or run the shell commands in [Makefile](../Makefile):

```shell
gcc -fPIC -O2 -c PLoM_C_library.c
gcc PLoM_C_library.o -shared -o PLoM_C_library.so
```

Alternatively, the pre-built dynamic library for three platforms are also archieved in the lib folder from which users could copy the .so library to the main root directory.

Note: please refer to the [official page](https://gcc.gnu.org/install/binaries.html) for installing **gcc** compiler if needed.
