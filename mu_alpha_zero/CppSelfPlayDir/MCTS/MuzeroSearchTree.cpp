//
// Created by skyr on 3/3/24.
//

#include "MuzeroSearchTree.h"
#include "../utils.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;
#include "cmath"
#include "../MCTS/Node.h"
#include "../uuid_v4/endianness.h"
#include "../uuid_v4/uuid_v4.h"
#include <iostream>

MuzeroSearchTree::MuzeroSearchTree(py::object game_manager, std::map<std::string, py::object> config_args) {
    this->game_manager = game_manager;
    this->config_args = config_args;
    std::cout << "Args and game manager created." << std::endl;
    this->frame_buffer = new MuZeroFrameBuffer(config_args["frameBufferSize"].cast<int>(),config_args["noopAction"].cast<int>(),config_args["actionSpaceSize"].cast<int>());
    this->minMaxQ = {-INFINITY,INFINITY};
    std::cout << "MuzeroSearchTree created." << std::endl;
}


std::tuple<PlayeOneStepReturn> MuzeroSearchTree::playOneGame(MuZeroDefaultNet networkWrapper) {
    int numSteps = this->config_args["num_steps"].cast<int>();
    int frameSkip = this->config_args["frame_skip"].cast<int>();
    torch::Tensor state = utils::numpyToPytorch(this->game_manager.attr("reset")());
    state = utils::resizeObs(state,{96,96});
    state = utils::scaleState(state);
    this->frame_buffer->initBuffer(state);
    std::vector<PlayeOneStepReturn> data;
    for (int step = 0;step < numSteps; step++) {
        SearchReturn searchReturn = this->search(networkWrapper,state,this->config_args["tau"].cast<double>());
        int move = utils::sampleMove(searchReturn.probabilities);
        std::tuple<torch::Tensor,torch::Tensor> pi_v = networkWrapper.predictionForward(std::get<1>(searchReturn.rootValLatent).unsqueeze(0),true);
        auto pred_v = std::get<1>(pi_v).item<double>();
        auto stateRewDone = this->game_manager.attr("frame_skip_step")(move,py::none(),frameSkip).cast<py::tuple>();
        bool done = stateRewDone[2].cast<bool>();
        auto rew = stateRewDone[1].cast<double>();
        state = utils::numpyToPytorch(stateRewDone[0].cast<py::array_t<float>>());
        state = utils::resizeObs(state,{96,96});
        state = utils::scaleState(state);
        if (done) {
            break;
        }
        double move2 = utils::scaleAction(move,this->config_args["net_action_size"].cast<int>());
        UUIDv4::UUIDGenerator<std::mt19937_64> uuidGenerator;
        UUIDv4::UUID uuid = uuidGenerator.getUUID();
        std::string framePath = this->config_args["pickle_dir"].cast<std::string>() + "/array_" + uuid.bytes() + ".pth";
        torch::save(this->frame_buffer->concatFrames(),framePath);
        PlayeOneStepReturn gameReturn = {searchReturn.probabilities,get<0>(searchReturn.rootValLatent),std::make_tuple(rew,move2,pred_v),framePath};
        data.push_back(gameReturn);
        this->frame_buffer->addFrame(state,move);


    }

}

SearchReturn MuzeroSearchTree::search(MuZeroDefaultNet networkWrapper, torch::Tensor state, double tau) {
    if (this->frame_buffer->size() == 0) {
        this->frame_buffer->initBuffer(state);
    }
    int num_simulations = this->config_args["num_simulations"].cast<int>();
    auto c = this->config_args["c"].cast<double>();
    auto c2 = this->config_args["c2"].cast<double>();
    unique_ptr<Node> rootNode = make_unique<Node>(0, nullptr);
    torch::Tensor state_ = networkWrapper.representationForward(this->frame_buffer->concatFrames().permute({2,0,1}).unsqueeze(0)).squeeze(0);
    std::tuple<torch::Tensor, torch::Tensor> pi_v = networkWrapper.predictionForward(state_.unsqueeze(0), true);
    std::vector<double> pi = utils::tensorToVector(std::get<0>(pi_v));
    rootNode->expand(state_,pi,0.0);
    for (int simulation = 0; simulation < num_simulations; simulation++)
    {
	    Node* currentNode = rootNode.get();
        std::vector<Node*> path;
        double action = -INFINITY;
        while (currentNode->was_visited())
        {
            std::tuple<Node*,int> actionNode = currentNode->get_best_child(c, c2);
            path.push_back(get<0>(actionNode));
            action = std::get<1>(actionNode);
        }
        action = utils::scaleAction(action,this->config_args["net_action_size"].cast<int>());
        torch::Tensor current_node_state_with_action = utils::matchActionWithObs(currentNode->parent->state,action);
        std::tuple<torch::Tensor, torch::Tensor> stateRew = networkWrapper.dynamicsForward(current_node_state_with_action.unsqueeze(0),true);
        auto reward = get<1>(stateRew).item<double>();
        py::object v = this->game_manager.attr("game_result")(currentNode->current_player);
        double value;
        if (v.is_none() || !v.cast<bool>())
        {
            std::tuple<torch::Tensor, torch::Tensor> pi_v = networkWrapper.predictionForward(std::get<0>(stateRew).unsqueeze(0),true);
            std::vector<double> pi = utils::tensorToVector(std::get<0>(pi_v));
            value = std::get<1>(pi_v).item<double>();
        }
		else
		{
            value = v.cast<double>();
		}
        this->backprop(move(path), value);
        std::map<int,double> action_probabilities = currentNode->get_self_action_probabilities(tau,true);
        std::tuple<double,torch::Tensor> rootValLatent = std::make_tuple(rootNode->get_self_value(),rootNode->get_latent());
        return SearchReturn{action_probabilities,rootValLatent};
    }

}

void MuzeroSearchTree::backprop(std::vector<Node*> path, double value)
{
    double G = 0;
    auto gamma = this->config_args["gamma"].cast<double>();
    std::reverse(path.begin(),path.end());
    for (Node* node : path) {
	    if (G == 0)
	    {
            G = value + node->reward;
	    }
        node->total_value += G;
        node->update_Q(G);
        this->updateMinMaxQ(node->q);
        node->scale_q(this->minMaxQ[0], this->minMaxQ[1]);
        node->times_visited += 1;
        G = node->reward + gamma * G;
    }
}
void MuzeroSearchTree::updateMinMaxQ(double Q)
{
    this->minMaxQ[0] = min(this->minMaxQ[0], Q);
    this->minMaxQ[1] = max(this->minMaxQ[1], Q);
}



