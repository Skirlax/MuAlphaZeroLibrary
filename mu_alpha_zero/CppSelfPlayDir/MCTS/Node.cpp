//
// Created by skyr on 3/3/24.
//

#include "Node.h"

Node::Node(double select_probability, Node* parent) {
    this->select_probability = select_probability;
    this->parent = parent;
    this->times_visited = 0;
    this->total_value = 0;
    this->q = -1;
    this->reward = 0;

}

void Node::expand(torch::Tensor state,std::vector<double> action_probabilities,double im_reward) {
    this->state = state;
    this->reward = im_reward;
    for (int i = 0; i < action_probabilities.size(); i++) {
        this->children[i] = std::make_unique<Node>(action_probabilities[i], this);
    }

}
std::tuple<Node*,int> Node::get_best_child(double c, double c2) {
    double best_uct = -1;
    Node* best_child = nullptr;
    int best_action = -1;
    for (auto& [key, value] : this->children) {
        double uct = value->calculate_uct_score(c, c2);
        if (uct > best_uct) {
            best_uct = uct;
            best_child = value.get();
            best_action = key;
        }
    }
    return {best_child,best_action};
}
double Node::calculate_uct_score(double c, double c2) const {
    if (this->q == -1) {
        return c * this->select_probability * sqrt(this->parent->times_visited + 1e-8);
    }
    else {
        double utc1 = this->q + this->select_probability * ((sqrt(parent->times_visited)) / (1 + this->times_visited));
        double utc2 = c + log((this->parent->times_visited + c2 + 1) / c2);
        return utc1 * utc2;
    }
}
void Node::scale_q(double minQ, double maxQ) {
    if (minQ == maxQ) {
        return;
    }
    this->q = (this->q - minQ) / (maxQ - minQ);
}

void Node::update_Q(double value) {
    if (this->q == -1) {
        this->q = value;
    }
    else {
        this->q = (this->q * this->times_visited + value) / (this->times_visited + 1);
    }
}
bool Node::was_visited() {
    return this->children.size() > 0;
}
std::map<int,double> Node::get_self_action_probabilities(double tau,bool adjust) {
    std::map<int,double> action_probabilities = std::map<int,double>();
    for (auto& [action,child] : this->children) {
        action_probabilities.insert({action, static_cast<double>(child->times_visited) / static_cast<double>(this->times_visited)});
    }
    if (adjust) {
        return this->adjustProbabilities(action_probabilities,tau);
    }
    return action_probabilities;


}
std::map<int,double> Node::adjustProbabilities(std::map<int,double> action_probabilities,double inner_tau) {
    if (inner_tau == 0) {
        int max_val = -1;
        int max_key = -1;
        for (auto [key, val] : action_probabilities) {
            if (val > max_val) {
                max_val = val;
                max_key = key;
            }
        }
        action_probabilities.insert({max_key,1});
        for (auto& [key, val] : action_probabilities) {
            if (key != max_key) {
                action_probabilities.insert({key,0});
            }
        }
        return action_probabilities;
    }

    std::vector<double> values = std::vector<double>();
    for (auto [key, val] : action_probabilities) {
        values.push_back(val);
    }
    torch::Tensor value_tensor = torch::tensor(values);
    value_tensor = value_tensor.pow(1 / inner_tau);
    value_tensor = value_tensor / value_tensor.sum();
    std::map<int,double> new_action_probabilities = std::map<int,double>();
    for (int i = 0; i < values.size(); i++) {
        new_action_probabilities.insert({i,value_tensor[i].item<double>()});
    }
    return new_action_probabilities;

}

torch::Tensor Node::get_latent() {
    return this->state;
}

double Node::get_self_value() {
    if (this-> times_visited > 0) {
        return this->total_value / this->times_visited;
    }
    return 0;
}
