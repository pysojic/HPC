# Makefile for Black–Scholes CUDA pricer

# CUDA compiler
NVCC       := nvcc

# Compiler flags: C++17, optimize
CXXFLAGS   := -std=c++17 -O3

# Source and target
SRC        := main.cu
TARGET     := bs_simple

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

