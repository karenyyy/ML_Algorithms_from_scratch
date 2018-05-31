//
// Created by karen on 5/31/18.
//

#ifndef ML_ALGORITHMS_FROM_SCRATCH_NET_H
#define ML_ALGORITHMS_FROM_SCRATCH_NET_H

#include <vector>
#include "Neuron.h"

using namespace std;

class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


#endif //ML_ALGORITHMS_FROM_SCRATCH_NET_H
