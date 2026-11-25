CXX ?= clang++
CXXFLAGS ?= -O3 -std=c++17 -mavx2 -mfma -Wall -Wextra
all: test bench
test: src/test.cpp src/gemm.cpp src/gemm.hpp
	$(CXX) $(CXXFLAGS) src/test.cpp src/gemm.cpp -o test -pthread
bench: src/bench.cpp src/gemm.cpp src/gemm.hpp
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK src/bench.cpp src/gemm.cpp -o bench -pthread -framework Accelerate
clean:
	rm -f test bench
.PHONY: all clean
