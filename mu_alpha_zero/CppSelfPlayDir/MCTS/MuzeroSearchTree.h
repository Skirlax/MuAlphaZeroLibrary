//
// Created by skyr on 3/3/24.
//

#ifndef CPPSELFPLAY_MUZEROSEARCHTREE_H
#define CPPSELFPLAY_MUZEROSEARCHTREE_H
#include "Node.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/functional.h"
#include "../Buffers/MuZeroFrameBuffer.h"
#include "../Network/MuZeroDefaultNet.h"
#include "torch/torch.h"

namespace py = pybind11;

struct PlayeOneStepReturn {
    std::map<int,double> probabilities;
    double v;
    std::tuple<double,int,double> info;
    std::string framePath;
};

struct SearchReturn {
    std::map<int,double> probabilities;
    std::tuple<double,torch::Tensor> rootValLatent;
};

class MuzeroSearchTree {
private:
    py::object game_manager;
    std::map<std::string,py::object> config_args;
    MuZeroFrameBuffer* frame_buffer;
    std::vector<double> minMaxQ;

public:
    MuzeroSearchTree(py::object game_manager, std::map<std::string,py::object> config_args);
    std::tuple<PlayeOneStepReturn> playOneGame(MuZeroDefaultNet networkWrapper);
    SearchReturn search(MuZeroDefaultNet networkWrapper, torch::Tensor state,double tau);
    void backprop(std::vector<Node*> path,double value);
    void updateMinMaxQ(double Q);


};


#endif //CPPSELFPLAY_MUZEROSEARCHTREE_H
