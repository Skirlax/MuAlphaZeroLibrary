//
// Created by skyr on 3/5/24.
//

#ifndef CPPSELFPLAY_MUZERODEFAULTNET_H
#define CPPSELFPLAY_MUZERODEFAULTNET_H
#include "torch/torch.h"


class MuZeroDefaultNet {
public:
    MuZeroDefaultNet(std::string modelPath);
    std::shared_ptr<torch::jit::script::Module> model;

    std::tuple<torch::Tensor,torch::Tensor> dynamicsForward(torch::Tensor stateWithAction,bool predict);
    std::tuple<torch::Tensor,torch::Tensor> predictionForward(torch::Tensor state,bool predict);
    torch::Tensor representationForward(torch::Tensor state);
};


#endif //CPPSELFPLAY_MUZERODEFAULTNET_H
