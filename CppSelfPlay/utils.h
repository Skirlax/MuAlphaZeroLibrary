//
// Created by skyr on 3/4/24.
//

#ifndef CPPSELFPLAY_UTILS_H
#define CPPSELFPLAY_UTILS_H
#include "torch/torch.h"
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

class utils {
public:
    static double scaleAction(int action, int num_actions);
    static torch::Tensor scaleRewardValue(torch::Tensor reward);
    static double scaleReward(double reward);
    static torch::Tensor scaleState(torch::Tensor state);
    static torch::Tensor addActionToObs(torch::Tensor observations,torch::Tensor actions,int dim);
    static torch::Tensor matchActionWithObs(torch::Tensor observations,int action);
    static torch::Tensor numpyToPytorch(py::array_t<float> inputArray);
    static torch::Tensor resizeObs(torch::Tensor obs,std::vector<int64_t> size);
    static std::map<int,double> tensorProbabilitiesToMap(torch::Tensor probabilities);
    static std::vector<double> tensorToVector(torch::Tensor tensor);
    static int sampleMove(std::map<int,double> action_probabilities);
};


#endif //CPPSELFPLAY_UTILS_H
