CC := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_PATH ?= /usr/local/cuda

CCFLAGS := -O3 -I$(CUDA_PATH)/include
CCFLAGS_DEBUG := -g -O0 -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_61,code=compute_61
NVCCFLAGS_DEBUG := -g -O0 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_61,code=compute_61
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -pthread

CPU_SRC := RCKangaroo.cpp GpuKang.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET)

debug: $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS_DEBUG) -o $(TARGET) $^ $(LDFLAGS)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS_DEBUG) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS_DEBUG) -c $< -o $@

clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS)
