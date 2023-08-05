//
//  pictureRecognition_module.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN::CV;
using namespace MNN;

#define DUMP_NUM_DATA(type)                          \
    auto data = tensor->host<type>();                \
    for (int z = 0; z < outside; ++z) {              \
        for (int x = 0; x < width; ++x) {            \
            outputOs << data[x + z * width] << "\t"; \
        }                                            \
        outputOs << "\n";                            \
    }

#define DUMP_CHAR_DATA(type)                                           \
    auto data = tensor->host<type>();                                  \
    for (int z = 0; z < outside; ++z) {                                \
        for (int x = 0; x < width; ++x) {                              \
            outputOs << static_cast<int>(data[x + z * width]) << "\t"; \
        }                                                              \
        outputOs << "\n";                                              \
    }

static void dumpTensor2File(const Tensor* tensor, const char* file) {
    std::ofstream outputOs(file);
    auto type = tensor->getType();

    int dimension = tensor->buffer().dimensions;
    int width     = 1;
    if (dimension > 1) {
        width = tensor->length(dimension - 1);
    }

    const int outside = tensor->elementSize() / width;

    const auto dataType  = type.code;
    const auto dataBytes = type.bytes();

    if (dataType == halide_type_float) {
        DUMP_NUM_DATA(float);
    }
    if (dataType == halide_type_int && dataBytes == 4) {
        DUMP_NUM_DATA(int32_t);
    }
    if (dataType == halide_type_uint && dataBytes == 1) {
        DUMP_CHAR_DATA(uint8_t);
    }
    if (dataType == halide_type_int && dataBytes == 1) {
#ifdef MNN_USE_SSE
        auto data = tensor->host<uint8_t>();
        for (int z = 0; z < outside; ++z) {
            for (int x = 0; x < width; ++x) {
                outputOs << (static_cast<int>(data[x + z * width]) - 128) << "\t";
            }
            outputOs << "\n";
        }
#else
        DUMP_CHAR_DATA(int8_t);
#endif
    }
}
static void _initDebug() {
    MNN::TensorCallBackWithInfo beforeCallBack = [&](const std::vector<MNN::Tensor*>& ntensors, const OperatorInfo*) {
        return true;
    };
    MNN::TensorCallBackWithInfo callBack = [&](const std::vector<MNN::Tensor*>& ntensors,  const OperatorInfo* info) {
        auto opName = info->name();
        for (int i = 0; i < ntensors.size(); ++i) {
            auto ntensor    = ntensors[i];
            auto outDimType = ntensor->getDimensionType();
            auto expectTensor = new MNN::Tensor(ntensor, outDimType);
            ntensor->copyToHostTensor(expectTensor);

            auto tensor = expectTensor;

            std::ostringstream outputFileName;
            auto opCopyName = opName;
            for (int j = 0; j < opCopyName.size(); ++j) {
                if (opCopyName[j] == '/') {
                    opCopyName[j] = '_';
                }
            }
            if (tensor->dimensions() == 4) {
                MNN_PRINT("Dimensions: 4, W,H,C,B: %d X %d X %d X %d, OP name %s : %d\n",
                        tensor->width(), tensor->height(), tensor->channel(), tensor->batch(), opName.c_str(), i);
            } else {
                std::ostringstream oss;
                for (int i = 0; i < tensor->dimensions(); i++) {
                    oss << (i ? " X " : "") << tensor->length(i);
                }

                MNN_PRINT("Dimensions: %d, %s, OP name %s : %d\n", tensor->dimensions(), oss.str().c_str(), opName.c_str(), i);
            }

            outputFileName << "output/" << opCopyName << "_" << i;
            dumpTensor2File(expectTensor, outputFileName.str().c_str());
            delete expectTensor;
        }
        return true;
    };
    Express::Executor::getGlobalExecutor()->setCallBack(std::move(beforeCallBack), std::move(callBack));
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./pictureRecognition_module.out model.mnn input0.jpg input1.jpg input2.jpg ... \n");
        return 0;
    }
    if (false) {
        _initDebug();
    }
    // Load module with Config
    /*
    MNN::Express::Module::BackendInfo bnInfo;
    bnInfo.type = MNN_FORWARD_CPU;
    MNN::Express::Module::Config configs;
    configs.backend = &bnInfo;
    std::shared_ptr<MNN::Express::Module> net(MNN::Express::Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], &configs));
    */
    
    // Load module with Runtime
    MNN::ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_AUTO;
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> rtmgr = std::shared_ptr<MNN::Express::Executor::RuntimeManager>(MNN::Express::Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }
    
    // Give cache full path which must be Readable and writable
    rtmgr->setCache(".cachefile");
    
    std::shared_ptr<MNN::Express::Module> net(MNN::Express::Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
    
    // Create Input
    int batchSize = argc - 2;
    auto input = MNN::Express::_Input({batchSize, 3, 224, 224}, MNN::Express::NC4HW4);
    for (int batch = 0; batch < batchSize; ++batch) {
        int size_w   = 224;
        int size_h   = 224;
        int bpp      = 3;

        auto inputPatch = argv[batch + 2];
        int width, height, channel;
        auto inputImage = stbi_load(inputPatch, &width, &height, &channel, 4);
        if (nullptr == inputImage) {
            MNN_ERROR("Can't open %s\n", inputPatch);
            return 0;
        }
        MNN_PRINT("origin size: %d, %d\n", width, height);
        Matrix trans;
        // Set transform, from dst scale to src, the ways below are both ok
        trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
        ImageProcess::Config config;
        config.filterType = BILINEAR;
        float mean[3]     = {103.94f, 116.78f, 123.68f};
        float normals[3] = {0.017f, 0.017f, 0.017f};
        // float mean[3]     = {127.5f, 127.5f, 127.5f};
        // float normals[3] = {0.00785f, 0.00785f, 0.00785f};
        ::memcpy(config.mean, mean, sizeof(mean));
        ::memcpy(config.normal, normals, sizeof(normals));
        config.sourceFormat = RGBA;
        config.destFormat   = BGR;

        std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(config));
        pretreat->setMatrix(trans);
        // for NC4HW4, UP_DIV(3, 4) * 4 = 4
        pretreat->convert((uint8_t*)inputImage, width, height, 0, input->writeMap<float>() + batch * 4 * 224 * 224, 224, 224, 4, 0,  halide_type_of<float>());
        stbi_image_free(inputImage);
    }
    auto outputs = net->onForward({input});
    auto output = MNN::Express::_Convert(outputs[0], MNN::Express::NHWC);
    output = MNN::Express::_Reshape(output, {0, -1});
    int topK = 10;
    auto topKV = MNN::Express::_TopKV2(output, MNN::Express::_Scalar<int>(topK));
    auto value = topKV[0]->readMap<float>();
    auto indice = topKV[1]->readMap<int>();
    for (int batch = 0; batch < batchSize; ++batch) {
        MNN_PRINT("For Input: %s \n", argv[batch+2]);
        for (int i=0; i<topK; ++i) {
            MNN_PRINT("%d, %f\n", indice[batch * topK + i], value[batch * topK + i]);
        }
    }
    rtmgr->updateCache();

    return 0;
}
