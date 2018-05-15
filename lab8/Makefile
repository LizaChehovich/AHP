# Variable check
ifeq ($(COMPUTE_VERSION),)
COMPUTE_VERSION=30
endif
ifeq ($(N),)
N=1
endif

# Folder config
ObjectDir=obj/
SourceDir=lab8/
BinDir=bin/

# Project config
SOURCES=main.cpp
CUDA_SOURCES=kernel.cu
EXECUTABLE=lab8

INCLUDE =
INCLUDE += /usr/lib/openmpi/include 

# Tools
MKDIR=mkdir
CC=gcc
MPICXX=mpicxx
CFLAGS=-std=c++11 -Wall -pedantic 
CUDACC=nvcc
CUDACC_SPEC=$(CUDACC) -ccbin $(MPICXX)
CUDAFLAGS=-std=c++11 -arch=sm_$(COMPUTE_VERSION) -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__

# Automatic features
OBJECTS=$(SOURCES:.cpp=.o) 
CSources=$(addprefix $(SourceDir),$(SOURCES))
CObjects=$(addprefix $(ObjectDir),$(OBJECTS))

CUDA_OBJECTS=$(CUDA_SOURCES:.cu=.o)
CudaSources=$(addprefix $(SourceDir),$(CUDA_SOURCES))
CudaObjects=$(addprefix $(ObjectDir),$(CUDA_OBJECTS))

Include_temp=$(addprefix $(SourceDir),$(INLCUDE_ADDITIONAL)) $(INCLUDE)
Include=$(addprefix -I,$(Include_temp))
CExecutable=$(addprefix $(BinDir),$(EXECUTABLE))

.DEFAULT: all_local
.PHONY: all, dir_init, clean, all_local, all_cluster, run, run_local, run_cluster

all: $(CSources) $(CudaSources) dir_init $(CExecutable)

$(CExecutable): $(CObjects) $(CudaObjects)
	$(CUDACC_SPEC) $(LDFLAGS) $(CObjects) $(CudaObjects) -o $@

$(ObjectDir)%.o: $(SourceDir)%.cpp
	$(CUDACC_SPEC) $(CUDAFLAGS) $(Include) $< -c -o $@

$(ObjectDir)%.o: $(SourceDir)%.cu
	$(CUDACC_SPEC) $(CUDAFLAGS) $(Include) $< -c -o $@

dir_init:
	@$(MKDIR) -p $(ObjectDir)
	@$(MKDIR) -p $(BinDir)

clean:
	@rm -rf $(ObjectDir)
	@rm -rf $(BinDir)

all_local:
	$(MAKE) all COMPUTE_VERSION=50

all_cluster:
	$(MAKE) all COMPUTE_VERSION=20

run:
	mpiexec -n $(N) -wdir $(SourceDir) ../$(CExecutable)
	
run_local:
	$(MAKE) run N=4

run_cluster:
	@qsub run_torque_mpi.pbs
