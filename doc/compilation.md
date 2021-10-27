## Compilation

The package invokes a dynamic library that computes the gradient of the potential used in constructing the Ito Stochastic Differential Equation (ISDE) to 
generate realizations. The dynamic library is compiled from [PLoM_C_library.c](../PLoM_C_library.c). Users can directly run the [install.py](../install.py) script to configure it:
```shell
python install.py
```
or
```shell
python3 install.py
```

Alternatively, Windows users can run the [make.bat](../make.bat) from the **PLoM** root (i.e., ```./make.bat```) and copy the
resulting .o and .so files into the "./lib/win" folder. Linux and macOS users can either run ```make all``` from the **PLoM** root or
run the shell commands in [Makefile](../Makefile) and copy the .o and .so files into the "./lib/linux" or "./lib/macOS":

```shell
gcc -fPIC -O2 -c PLoM_C_library.c
gcc PLoM_C_library.o -shared -o PLoM_C_library.so
```

Pre-built dynamic libraries for three platforms are also archieved in the lib folder; for macOS this is available for M1 machines and
thus depending on machine architecture it may be required to run python install.py on older machines.

Note: please refer to the [official page](https://gcc.gnu.org/install/binaries.html) for installing **gcc** compiler if needed.
