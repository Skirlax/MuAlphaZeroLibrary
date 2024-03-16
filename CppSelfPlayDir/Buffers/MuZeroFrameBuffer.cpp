//
// Created by skyr on 3/3/24.
//

#include "MuZeroFrameBuffer.h"

#include "torch/torch.h"
using namespace std;

MuZeroFrameBuffer::MuZeroFrameBuffer(int frameBufferSize, int noopAction, int actionSpaceSize) {
    this->frameBufferSize = frameBufferSize;
    this->noopAction = noopAction;
    this->actionSpaceSize = actionSpaceSize;
    this->buffer = make_unique<deque<tuple<torch::Tensor, int>>>();

}

void MuZeroFrameBuffer::addFrame(torch::Tensor frame, int action) {
    if (this->buffer.get()->size() < this->frameBufferSize) {
        this->buffer.get()->push_back(make_tuple(frame, action));
    }
    else {
        this->buffer->pop_front();
        this->buffer->push_back(make_tuple(frame, action));
    }
}

torch::Tensor MuZeroFrameBuffer::concatFrames() {
    unique_ptr<vector<torch::Tensor>> framesWithActions(new vector<torch::Tensor>);
    for (auto [frame,action] : *this->buffer) {
        framesWithActions->push_back(torch::cat((frame,torch::full((frame.size(0), frame.size(1),1),action)),2));


    }

    return torch::cat(*framesWithActions,2);
}

void MuZeroFrameBuffer::initBuffer(torch::Tensor initState) {
    for (int i = 0; i < this->frameBufferSize,i++;) {
        this->addFrame(initState,this->noopAction);
    }
}

int MuZeroFrameBuffer::size() {
    return this->buffer->size();
}

