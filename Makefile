all: PLoM_C_library

PLoM_C_library: PLoM_C_library.o
	gcc PLoM_C_library.o -shared -o PLoM_C_library.so

PLoM_C_library.o:
	gcc -fPIC -O2 -c PLoM_C_library.c

clean:
	rm -f *.o *.so *.out