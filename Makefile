# system dependent flags for CUDA
# CUDA sdk path
CUDA_SDK?=/usr/local/cuda/sdk
# CUDA path
CUDA_PATH?=/usr/local/cuda

# C compiler 
CC = gcc
CFLAGS  	= -lm -Wall -O3 -fomit-frame-pointer -funroll-loops

# CUDA compiler
NVCC = nvcc
NVCCFLAGS = -ccbin /usr/bin/g++ \
	    -I$(CUDA_SDK)/common/inc \
	    -I$(CUDA_PATH)/include -arch=sm_20 -m64

# linker and linker options
LD = $(NVCC)
LFLAGS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart

PROJ = neville
.PHONY: clean

all: $(PROJ)
# CUDA source
%.o: %.cu
	@echo CUDA compiling $@
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

%.o: %.c
	@echo C compiling $@
	$(CC) -c $(CFLAGS) -o $@ $<
# linking
neville:neville.o
	@echo linking $@
	$(LD) -o $@ $^ $(LFLAGS)

clean:
	rm -f *.o $(PROJ)

# DEPENDENCIES
neville: neville.o 
neville.o: neville.cu	

