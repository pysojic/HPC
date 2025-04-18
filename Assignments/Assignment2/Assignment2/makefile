
# Compiler and flags
CXX      = g++
CXXFLAGS = -O3 -fopenmp -Wall

# Target executable name
TARGET   = option_pricer

# Build target
all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o $(TARGET)

# Clean up build files
clean:
	rm -f $(TARGET)
