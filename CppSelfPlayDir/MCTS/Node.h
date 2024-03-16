//
// Created by skyr on 3/3/24.
//

#ifndef CPPSELFPLAY_NODE_H
#define CPPSELFPLAY_NODE_H
#include "torch/torch.h"

class Node {
public:
    int current_player{};
    double select_probability{};
    torch::Tensor state;
    int times_visited{};
    double total_value{};
    double q{};
    double reward{};
    Node* parent{};
    std::map<int, std::unique_ptr<Node>> children{};

public:
    Node(double select_probability, Node* parent);
    void expand(torch::Tensor state,std::vector<double> action_probabilities,double im_reward);
    std::tuple<Node*,int> get_best_child(double c,double c2);
    double calculate_uct_score(double c, double c2) const;
    void scale_q(double minQ, double maxQ);
    bool was_visited();
    void update_Q(double value);
    double get_self_value();
    std::map<int,double> get_self_action_probabilities(double tau,bool adjust);
    std::map<int,double> adjustProbabilities(std::map<int,double> action_probabilities,double inner_tau);

    torch::Tensor get_latent();
};



#endif //CPPSELFPLAY_NODE_H
