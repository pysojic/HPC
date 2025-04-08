#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>

std::mutex coutMutex;

// Map Pattern
// The map pattern applies a given function to each element in a collection
// In this example, each element is doubled and printed out
void mapFunction(int element)
{
    std::lock_guard<std::mutex> lock(coutMutex);
    std::cout << "[Map] Element: " << element 
              << " -> Doubled: " << (element * 2) << std::endl;
}

// Reduction Pattern
// The reduction pattern computes a result by reducing (combining) parts of a data set
// Here, each thread computes the sum for a subrange of the vector, and then
// the main thread aggregates these partial sums
void reductionFunction(const std::vector<int>& data, int start, int end, int &partialSum) 
{
    int sum = 0;
    for (int i = start; i < end; ++i) 
    {
        sum += data[i];
    }
    partialSum = sum;
}

int main() {
    // --------------Map Pattern--------------
    std::cout << "=== Map Pattern Demonstration ===" << std::endl;
    std::vector<int> mapData = { 1, 2, 3, 4, 5 };
    std::vector<std::thread> mapThreads;
    
    // Launch one thread per element, applying the mapFunction
    for (int element : mapData) 
    {
        mapThreads.emplace_back(mapFunction, element);
    }
    for (auto &t : mapThreads) 
    {
        t.join();
    }
    
    // --------------Reduction Pattern--------------
    std::cout << "\n=== Reduction Pattern Demonstration ===" << std::endl;
    std::vector<int> reductionData = { 1, 2, 3, 4, 5 };
    
    int numReductionThreads = 3;
    std::vector<std::thread> reductionThreads;
    std::vector<int> partialSums(numReductionThreads, 0);
    
    int dataSize = reductionData.size();
    int chunkSize = dataSize / numReductionThreads;
    int remainder = dataSize % numReductionThreads;
    
    int startIndex = 0;
    for (int i = 0; i < numReductionThreads; ++i) 
    {
        int endIndex = startIndex + chunkSize + (i < remainder ? 1 : 0);
        reductionThreads.emplace_back(reductionFunction,
                                      std::cref(reductionData),
                                      startIndex, 
                                      endIndex,
                                      std::ref(partialSums[i]));
        startIndex = endIndex;
    }
    
    for (auto &t : reductionThreads) 
    {
        t.join();
    }
    
    int totalSum = 0;
    for (int sum : partialSums) 
    {
        totalSum += sum;
    }
    
    {
        std::lock_guard<std::mutex> lock(coutMutex);
        std::cout << "[Reduction] Total sum of elements: " << totalSum << std::endl;
    }
    
    return 0;
}
