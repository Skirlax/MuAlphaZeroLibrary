//
// Created by skyr on 3/5/24.
//

#include "MuZeroDefaultNet.h"
#include "torch/script.h"

MuZeroDefaultNet::MuZeroDefaultNet(std::string modelPath) {
    this->model = torch::jit::load(modelPath);
}

std::tuple<torch::Tensor,torch::Tensor> MuZeroDefaultNet::dynamicsForward(torch::Tensor stateWithAction, bool predict) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(stateWithAction);
    inputs.push_back(predict);
    torch::IValue state_rew = this->model.get_method("dynamics_forward")(inputs).toTensor();
    std::tuple<torch::Tensor,torch::Tensor> state_rew_tuple = std::make_tuple(state_rew.toTuple()->elements()[0].toTensor(), state_rew.toTuple()->elements()[1].toTensor());
    return state_rew_tuple;
}

std::tuple<torch::Tensor,torch::Tensor> MuZeroDefaultNet::predictionForward(torch::Tensor state, bool predict) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state);
    inputs.push_back(predict);
    torch::IValue pi_v = this->model.get_method("prediction_forward")(inputs).toTensor();
    std::tuple<torch::Tensor,torch::Tensor> pi_v_tuple = std::make_tuple(pi_v.toTuple()->elements()[0].toTensor(), pi_v.toTuple()->elements()[1].toTensor());
    return pi_v_tuple;
}

torch::Tensor MuZeroDefaultNet::representationForward(torch::Tensor state) {
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(state);
    return this->model.get_method("representation_forward")(inputs).toTensor();
}
