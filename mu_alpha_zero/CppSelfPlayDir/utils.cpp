//
// Created by skyr on 3/4/24.
//

#include "utils.h"
#include "torch/torch.h"


double utils::scaleAction(int action, int num_actions) {
    return (double) action / (double)(num_actions - 1);
}

torch::Tensor utils::scaleRewardValue(torch::Tensor reward) {
    return torch::sign(reward) * torch::sqrt(torch::abs(reward) + 1) - 1  + reward * 0.001;
}

double utils::scaleReward(double reward) {
    return std::log(reward + 1) / std::log(5);
}

torch::Tensor utils::scaleState(torch::Tensor state) {
    return state / 255;
}

torch::Tensor utils::addActionToObs(torch::Tensor observations, torch::Tensor actions, int dim) {
    return torch::cat((observations,actions),dim);
}
torch::Tensor utils::matchActionWithObs(torch::Tensor observations, int action) {
    return addActionToObs(observations, torch::full((1, observations.size(1), observations.size(2)), action), 0);
}

torch::Tensor utils::numpyToPytorch(py::array_t<float> inputArray) {
    py::buffer_info buf = inputArray.request();
    return torch::from_blob(buf.ptr, {buf.shape[0], buf.shape[1],buf.shape[2]}, torch::kFloat32).clone();
}

torch::Tensor utils::resizeObs(torch::Tensor obs, std::vector<int64_t> size) {
    return torch::nn::functional::interpolate(obs,torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kNearest));
}

std::map<int, double> utils::tensorProbabilitiesToMap(torch::Tensor probabilities) {
    std::map<int,double> probMap;
	for (int i = 0; i < probabilities.size(0); i++)
	{
				probMap[i] = probabilities[i].item<double>();
	}
	return probMap;
}

std::vector<double> utils::tensorToVector(torch::Tensor tensor)
{
	std::vector<double> vec;
	for (int i = 0; i < tensor.size(0); i++)
	{
		vec.push_back(tensor[i].item<double>());
	}
	return vec;
}

int utils::sampleMove(std::map<int, double> action_probabilities) {
    torch::Tensor probabilities = torch::zeros(action_probabilities.size());
    for (auto [action,prob] : action_probabilities) {
        probabilities[action] = prob;
    }
    return torch::multinomial(probabilities,1).item<int>();
}


