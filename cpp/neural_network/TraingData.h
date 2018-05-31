//
// Created by karen on 5/31/18.
//

#ifndef ML_ALGORITHMS_FROM_SCRATCH_TRAINGDATA_H
#define ML_ALGORITHMS_FROM_SCRATCH_TRAINGDATA_H

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof() { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};


#endif //ML_ALGORITHMS_FROM_SCRATCH_TRAINGDATA_H
