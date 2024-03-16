//
// Created by skyr on 3/3/24.
//

#ifndef CPPSELFPLAY_MUZEROFRAMEBUFFER_H
#define CPPSELFPLAY_MUZEROFRAMEBUFFER_H
#include "torch/torch.h"

using namespace std;

class MuZeroFrameBuffer {
private:
    int frameBufferSize;
    int noopAction;
    int actionSpaceSize;
    unique_ptr<deque<tuple<torch::Tensor,int>>> buffer;

public:
    MuZeroFrameBuffer(int frameBufferSize, int noopAction,int actionSpaceSize);
    void addFrame(torch::Tensor, int action);
    torch::Tensor concatFrames();
    void initBuffer(torch::Tensor initState);
    int size();


};


#endif //CPPSELFPLAY_MUZEROFRAMEBUFFER_H
