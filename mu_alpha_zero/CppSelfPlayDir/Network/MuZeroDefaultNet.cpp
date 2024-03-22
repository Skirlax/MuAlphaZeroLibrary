//
// Created by skyr on 3/5/24.
//

#include "MuZeroDefaultNet.h"
#include <torch/torch.h>
#include "torch/script.h"


MuZeroDefaultNet::MuZeroDefaultNet(std::string modelPath) {
    this->model = std::make_shared<torch::jit::script::Module>(torch::jit::load(modelPath));
}

std::tuple<torch::Tensor,torch::Tensor> MuZeroDefaultNet::dynamicsForward(torch::Tensor stateWithAction, bool predict) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(stateWithAction);
    auto state_rew = this->model->get_method("dynamics_forward")(inputs);
    std::tuple<torch::Tensor,torch::Tensor> state_rew_tuple = std::make_tuple(state_rew.toTuple()->elements()[0].toTensor(), state_rew.toTuple()->elements()[1].toTensor());
    return state_rew_tuple;
}

std::tuple<torch::Tensor,torch::Tensor> MuZeroDefaultNet::predictionForward(torch::Tensor state, bool predict) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state);
    inputs.push_back(torch::jit::IValue(predict));
    try {
        auto pi_v = this->model->get_method("prediction_forward")(inputs);
        std::tuple<torch::Tensor,torch::Tensor> pi_v_tuple = std::make_tuple(pi_v.toTuple()->elements()[0].toTensor(), pi_v.toTuple()->elements()[1].toTensor());
        return pi_v_tuple;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return std::make_tuple(torch::zeros({1,1,1}), torch::zeros({1,1,1}));
}

torch::Tensor MuZeroDefaultNet::representationForward(torch::Tensor state) {
    std::vector<torch::jit::IValue> inputs = std::vector<torch::jit::IValue>();
    inputs.push_back(state);
    try {
        auto result =  this->model->get_method("representation_forward")(inputs);
        return result.toTensor();
    }
    catch (const c10::Error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return torch::zeros({1,1,1});

}
