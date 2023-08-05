//
//  FedBabuSGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "FedBabuSGD.hpp"
#include "OpGrad.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
FedBabuSGD::FedBabuSGD(std::shared_ptr<Module> module) : ParameterOptimizer(module) {
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

void FedBabuSGD::setLearningRate(float rate) {
    mLearningRate = rate;
}

void FedBabuSGD::setMomentum(float momentum) {
    mMomentum = momentum;
}

void FedBabuSGD::setWeightDecay(float decay) {
    mWeightDecay = decay;
}

void FedBabuSGD::setRegularizationMethod(RegularizationMethod method) {
    mRegularizationMethod = method;
}

float FedBabuSGD::currentLearningRate() {
    return mLearningRate;
}

float FedBabuSGD::getMomentum() {
    return mMomentum;
}

float FedBabuSGD::getWeightDecay() {
    return mWeightDecay;
}

FedBabuSGD::RegularizationMethod FedBabuSGD::getRegularizationMethod() {
    return mRegularizationMethod;
}

Express::VARP FedBabuSGD::regularizeParameters(Express::VARP param, Express::VARP grad) {
    VARP addWeightDecayGrad;
    if (mRegularizationMethod == L1) {
        auto temp          = _Sign(param);
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * temp + grad;
    } else if (mRegularizationMethod == L2) {
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * param + grad;
    } else if (mRegularizationMethod == L1L2) {
        auto temp          = _Sign(param);
        auto L1 = _Const(mWeightDecay, {}, NCHW) * temp;
        auto L2 = _Const(mWeightDecay, {}, NCHW) * param;
        addWeightDecayGrad = L1 + L2 + grad;
    }

    return addWeightDecayGrad;
}

Express::VARP FedBabuSGD::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
    // Filter the output layer
    auto dim = param->getInfo()->dim;
    VARP lr;
    if (dim[0] == 62) {
        lr         = _Const(0.0f, {}, NCHW);
    } else {
        lr         = _Const(mLearningRate, {}, NCHW);
    }
    mHistory[param] = lr * grad + _Const(mMomentum, {}, NCHW) * mHistory[param];
    mHistory[param].fix(Express::VARP::CONSTANT);
    //FUNC_PRINT_ALL(_ReduceMax(grad)->readMap<float>()[0], f);
    return mHistory[param];
}

std::map<Express::VARP, Express::VARP> FedBabuSGD::onGetNextParameter(Express::VARP loss) {
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
    auto parameters = module()->parameters();
    std::vector<VARP> prepareCompute;
    for (auto iter : parameters) {
        if (iter->expr().first->get() != nullptr) {
            prepareCompute.emplace_back(iter);
        }
    }
    for (auto& iter : grad) {
        prepareCompute.emplace_back(iter.second);
    }
    Variable::prepareCompute(prepareCompute);
    std::vector<VARP> replaceOp(prepareCompute.size());
    for (int i=0; i<prepareCompute.size(); ++i) {
        auto info = prepareCompute[i]->getInfo();
        auto ptr = prepareCompute[i]->readMap<void>();
        if (nullptr == ptr) {
            MNN_ERROR("Compute error in SGD\n");
            return {};
        }
        auto newVar = _Const(ptr, info->dim, info->order, info->type);
        replaceOp[i]= newVar;
    }
    for (int i=0; i<prepareCompute.size(); ++i) {
        Variable::replace(prepareCompute[i], replaceOp[i]);
    }

    for (auto& iter : grad) {
        // apply regularization
        auto addWeightDecayGrad = regularizeParameters(iter.first, iter.second);
        addWeightDecayGrad.fix(Express::VARP::CONSTANT);
        // apply momentum, etc.
        auto updateValue = this->onComputeUpdateValue(iter.first, addWeightDecayGrad);
        // apply update
        auto newParameter = iter.first - updateValue;
        iter.second       = newParameter;
    }
    return grad;
}

} // namespace Train
} // namespace MNN
