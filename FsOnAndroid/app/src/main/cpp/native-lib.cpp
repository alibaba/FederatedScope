#include <jni.h>
#include <string>
#include <MNN/expr/Executor.hpp>
#include <PipelineModule.hpp>
#include <cmath>
#include <MnistDataset.hpp>
#include <FEMnistDataset.hpp>
#include <CelebADataset.hpp>
#include <TextDataset.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <SGD.hpp>
#include <LearningRateScheduler.hpp>
#include <Loss.hpp>
#include <Lenet.hpp>
#include "NN.hpp"
#define MNN_OPEN_TIME_TRACE
#include <AutoTime.hpp>
#include "RandomGenerator.hpp"
#include "Transformer.hpp"
#include <android/log.h>
#include <unistd.h>
#include <map>
#include <sys/stat.h>
#include <unistd.h>
#include "MNN_generated.h"
#include <cli.hpp>
#include <fstream>
#include <flatbuffers/minireflect.h>
#include "OpGrad.hpp"
#include "FedBabuSGD.hpp"
// For quantization
#include "half.hpp"
#include "addBizCode.hpp"
#include "writeFb.hpp"
#include "PostConverter.hpp"
#include "ConvolutionCommon.hpp"

#define FSLOG(...) __android_log_print(ANDROID_LOG_VERBOSE, "FS-DEVICE", __VA_ARGS__);

using namespace MNN;
using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

bool IsFileExist(const char* path)
{
    return !access(path, F_OK);
}

void CastFp16toFp32(std::unique_ptr<MNN::OpT> &op) {
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_ConvolutionDepthwise: {
            // the faltbuffer object before serialization
            auto paramT = op->main.AsConvolution2D();
            flatbuffers::FlatBufferBuilder builder;
            auto param_offset = Convolution2D::Pack(builder, paramT);
            builder.Finish(param_offset);
            auto param = flatbuffers::GetRoot<Convolution2D>(builder.GetBufferPointer());

            const float *fp32Weight = nullptr;
            int fp32WeightSize = 0;
            std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
            ConvolutionCommon::getConvParameters(&quanCommon,
                                                 param,
                                                 &fp32Weight,
                                                 &fp32WeightSize);

            paramT->quanParameter.reset();
            paramT->weight.reserve(fp32WeightSize);
            for (int i = 0; i < fp32WeightSize; ++i) {
                paramT->weight.emplace_back(*(fp32Weight + i));
            }
            break;
        }
        case MNN::OpType_Const: {
            auto blob = op->main.AsBlob();
            if (blob->dataType == MNN::DataType_DT_HALF) {
                blob->dataType = MNN::DataType_DT_FLOAT;
                uint32_t weightLength = blob->uint8s.size() / sizeof(half_float::half);
                blob->float32s.resize(weightLength);

                std::vector<int8_t> tempHalfWeight(blob->uint8s.begin(), blob->uint8s.end());
                auto halfWeight = reinterpret_cast<half_float::half *>(tempHalfWeight.data());
                blob->float32s.reserve(weightLength);
                std::transform(halfWeight, halfWeight + weightLength, blob->float32s.begin(),
                               [](half_float::half h) { return float(h); });
                blob->uint8s.clear();
            }
            break;
        }
        default:
            break;
    }

}

void CastInt2sim8toFp32(std::unique_ptr<MNN::OpT> &op) {
    const auto opType = op->type;
    switch (opType) {
        case MNN::OpType_Convolution:
        case MNN::OpType_ConvolutionDepthwise: {
            // the faltbuffer object before serialization
            auto paramT = op->main.AsConvolution2D();
            flatbuffers::FlatBufferBuilder builder;
            auto param_offset = Convolution2D::Pack(builder, paramT);
            builder.Finish(param_offset);
            auto param = flatbuffers::GetRoot<Convolution2D>(builder.GetBufferPointer());

            const float *fp32Weight = nullptr;
            int fp32WeightSize = 0;
            std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
            ConvolutionCommon::getConvParameters(&quanCommon,
                                                 param,
                                                 &fp32Weight,
                                                 &fp32WeightSize);

            paramT->quanParameter.reset();
            paramT->weight.reserve(fp32WeightSize);
            for (int i = 0; i < fp32WeightSize; ++i) {
                paramT->weight.emplace_back(*(fp32Weight + i));
            }
            break;
        }
        default:
            break;
    }

}

bool revertModelToFp32(const std::string &src_path, const std::string &tgt_path,
                       int src_quant_bits) {
    modelConfig modelPath;
    modelPath.MNNModel = tgt_path;
    modelPath.model = modelConfig::MNN;
    if (IsFileExist(src_path.c_str())) {
        modelPath.modelFile = src_path;
    } else {
        FSLOG("Model File Does Not Exist! ==> ");
        return false;
    }

    // convert a fp16 or int2~8 mnn model to a mnn model with fp32 format
    std::cout << "Start to Convert fp16 or int2~8 MNN Model Format To fp32 MNN Model..."
              << std::endl;
    std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
    int parseRes = 1;
    int error = 0;
    modelPath.bizCode = "revert_fp32";
    modelPath.forTraining = true;
    parseRes = addBizCode(modelPath.modelFile, modelPath.bizCode, netT);

    // convert the src model to fp32
    for (auto &op: netT->oplists) {
        if (src_quant_bits == 16) {
            CastFp16toFp32(op);
        } else {
            CastInt2sim8toFp32(op);
        }
    }
    for (auto &subgraph: netT->subgraphs) {
        for (auto &op: subgraph->nodes) {
            if (src_quant_bits == 16) {
                CastFp16toFp32(op);
            } else {
                CastInt2sim8toFp32(op);
            }
        }
    }

    if (modelPath.model != modelConfig::MNN || modelPath.optimizeLevel >= 2) {
        std::cout << "Start to Optimize the MNN Net..." << std::endl;
        std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining, modelPath);
        error = writeFb(newNet, modelPath.MNNModel, modelPath);
    } else {
        error = writeFb(netT, modelPath.MNNModel, modelPath);
    }
    if (0 == error) {
        std::cout << "Converted Success!" << std::endl;
        return true;
    } else {
        std::cout << "Converted Failed!" << std::endl;
        return false;
    }
}

DatasetPtr getDatasetPtr(std::string type, std::string path, std::string mode) {
    if (type == "FEMNIST" || type == "femnist") {
        return FEMnistDataset::create(path, mode);
    } else if (type == "CELEBA" || type == "celeba") {
        return CelebADataset::create(path, mode);
    } else if (type == "TWITTER" || type == "twitter") {
        return TextDataset::create(path, mode);
    } else {
        if (mode == "train") {
            return MnistDataset::create(path, MnistDataset::Mode::TRAIN);
        } else {
            return MnistDataset::create(path, MnistDataset::Mode::TEST);
        }
    }
}

VARP getMean(std::string type) {
    if (type == "FEMNIST" || type == "femnist") {
        return _Const(0.9637f);
    } else if (type == "CELEBA" || type == "celeba") {
        const float means[] = {0.5063479, 0.42581648, 0.3832074};
        return _Const(means, {3}, NHWC, halide_type_of<float>());
    } else {
        return _Const(0.f);
    }
}

VARP getStd(std::string type){
    if (type == "FEMNIST" || type == "femnist") {
        return _Const(0.1597f);
    } else if (type == "CELEBA" || type == "celeba") {
        float stds[] = {0.26549563, 0.24475749, 0.24083897};
        return _Const(stds, {3}, NHWC, halide_type_of<float>());
    } else {
        return _Const(1.f);
    }
}

jobject getJavaAttribute(JNIEnv* env, jobject parentObj, char* attrName, char* javaClass){
    jclass parentClass = env->GetObjectClass(parentObj);
    jfieldID id_attr = env->GetFieldID(parentClass, attrName, javaClass);
    jobject attrObject = env->GetObjectField(parentObj, id_attr);
    return attrObject;
}

jint getIntAttribute(JNIEnv* env, jobject parentObj, char* attrName){
    jclass parentClass = env->GetObjectClass(parentObj);
    jfieldID id_attr = env->GetFieldID(parentClass, attrName, "I");
    jint attr = env->GetIntField(parentObj, id_attr);
    return attr;
}

jboolean getBoolAttribute(JNIEnv* env, jobject parentObj, char* attrName){
    jclass parentClass = env->GetObjectClass(parentObj);
    jfieldID id_attr = env->GetFieldID(parentClass, attrName, "Z");
    jboolean attr = env->GetBooleanField(parentObj, id_attr);
    return attr;
}

jfloat getFloatAttribute(JNIEnv* env, jobject parentObj, char* attrName){
    jclass parentClass = env->GetObjectClass(parentObj);
    jfieldID id_attr = env->GetFieldID(parentClass, attrName, "F");
    jfloat attr = env->GetFloatField(parentObj, id_attr);
    return attr;
}

std::string getStrAttribute(JNIEnv* env, jobject parentObj, char* attrName){
    jclass parentClass = env->GetObjectClass(parentObj);
    jfieldID id_attr = env->GetFieldID(parentClass, attrName, "Ljava/lang/String;");
    jstring attr = (jstring) env->GetObjectField(parentObj, id_attr);
    const char *attr_char = env->GetStringUTFChars(attr, 0);
    std::string attr_str = attr_char;
    return attr_str;
}

/**
 * @brief test model in image classification task
 * @param model test model
 * @param testDataLoader test dataloader
 * @return accuracy
 */
std::map<std::string, float> test(std::shared_ptr<Module> &model,
                                  std::shared_ptr<DataLoader> &testDataLoader,
                                  std::string type,
                                  int category, 
                                  std::string mode,
                                  VARP meansVarp, 
                                  VARP stdsVarp){
    int correct = 0;
    size_t testIterations = testDataLoader->iterNumber();
    testDataLoader->reset();
    model->setIsTraining(false);
    int moveBatchSize = 0;
    for (int i = 0; i < testIterations; i++) {
        auto data       = testDataLoader->next();
        auto example    = data[0];
        moveBatchSize += example.first[0]->getInfo()->dim[0];
        auto cast       = _Cast<float>(example.first[0]);
        if (type == "twitter" || type == "TWITTER") {
            // forward
        } else {
            example.first[0] = (cast * _Const(1.0f / 255.0f) - meansVarp) / stdsVarp;
        }
        if (example.first[0]->getInfo()->order == NHWC) {
            example.first[0] = _Convert(example.first[0], NCHW);
        }
        auto predict    = model->forward(example.first[0]);
        predict         = _ArgMax(predict, 1);
        auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
        correct += accu->readMap<int32_t>()[0];

        auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(category), _Scalar<float>(1.0f),
                                 _Scalar<float>(0.0f));
        auto loss    = _CrossEntropy(predict, newTarget);
    }
    auto acc = (float)correct / (float)testDataLoader->size();

    // wrap the evaluation results
    std::map<std::string, float> metrics;
    metrics[mode+"_total"] = (float) moveBatchSize;
    metrics[mode+"_correct"] = (float) correct;
    metrics[mode+"_acc"] = acc;
    return metrics;
}

std::shared_ptr<Module> loadMnnModel(const char *cPath) {
    auto varMap = Variable::loadMap(cPath);
    auto inputOutputs = Variable::getInputAndOutput(varMap);
    auto inputs       = Variable::mapToSequence(inputOutputs.first);
    auto outputs      = Variable::mapToSequence(inputOutputs.second);
    // 将转换得到的模型转换为可训练模型(将推理模型中的卷积，BatchNorm，Dropout抽取出来，转换成可训练模块)
    auto pipelineModule = NN::extract(inputs, outputs, true);
    std::shared_ptr<Module> model(pipelineModule);
    return model;
}

std::shared_ptr<Module> loadMnnModel(const char * cPath, int src_quant_bits) {
    std::string src_model_path(cPath);
    std::string dst_model_path = src_model_path+".fp32";
    revertModelToFp32(src_model_path, dst_model_path, src_quant_bits);

    return loadMnnModel(dst_model_path.c_str());
}

extern "C" JNIEXPORT int JNICALL
Java_com_example_fsandroid_MainActivity_generalTrainModelInMNN(
        JNIEnv* env,
        jobject thiz, jstring pathModel, jobject pConfig, jstring pPathData) {

    // recall java function
    jclass main_activity = (*env).FindClass("com/example/fsandroid/MainActivity");
    jmethodID uptFedTrainInfo = (*env).GetMethodID(main_activity, "updateTrainInfoFromCpp", "(IF)V");
    // jmethodID uptFedTestInfo = (*env).GetMethodID(main_activity, "updateTestInfoFromCpp", "(IF)V");
    jmethodID uptProgressBar = (*env).GetMethodID(main_activity, "uptProgressBar", "(I)V");

    // extract hyper-parameters
    jobject config_data = getJavaAttribute(env, pConfig, "data",  "Lcom/example/fsandroid/configs/Data;");
    jobject config_train = getJavaAttribute(env, pConfig, "train", "Lcom/example/fsandroid/configs/Train;");
    jobject config_opt = getJavaAttribute(env, pConfig, "optimizer", "Lcom/example/fsandroid/configs/Optimizer;");
    jobject config_task = getJavaAttribute(env, pConfig, "task", "Lcom/example/fsandroid/configs/Task;");

    float baseLr = (float) getFloatAttribute(env, config_opt, "lr");
    float weightDecay = (float) getFloatAttribute(env, config_opt, "weight_decay");
    float momentum = (float) getFloatAttribute(env, config_opt, "momentum");
    std::string root = getStrAttribute(env, config_data, "root");
    std::string type = getStrAttribute(env, config_data, "type");
    int batchSize = (int) getIntAttribute(env, config_data, "batch_size");
    int localUpdateSteps = (int) getIntAttribute(env, config_train, "local_update_steps");
    int category = (int) getIntAttribute(env, config_task, "category");
    int mnn_quantization = (int) getIntAttribute(env, config_train, "mnn_quantization");

    // obtain mean and std varp
    VARP meansVarp = getMean(type);
    VARP stdsVarp = getStd(type);

    // training dataset
    const char *path_data = env->GetStringUTFChars(pPathData, 0);
    if (!IsFileExist(path_data)) {
        FSLOG("Data in %s doesn't exist", path_data);
    }

    // path
    const char *cPath = env->GetStringUTFChars(pathModel, 0);
    std::shared_ptr<Module> model;
    if (mnn_quantization == 8 || mnn_quantization == 16) {
        model = loadMnnModel(cPath, mnn_quantization);
    } else {
        model = loadMnnModel(cPath);
    }

    // init Executor
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);

    // init optimizer
    std::shared_ptr<SGD> sgd(new SGD(model));
    sgd->setMomentum(momentum);
    sgd->setWeightDecay(weightDecay);

    auto dataset = getDatasetPtr(type,path_data, "train");

    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    /**
     * main loop for training
     */
    int nSamples = 0;
    for (int epoch = 0; epoch < localUpdateSteps; ++epoch) {
        FSLOG("Begin %d epoch training in C++", epoch);
        model->clearCache();
        /**
         * 手动回收内存，当在循环中调用MNN表达式求值时，常量部分数据不会在每次循环结束释放，当执行次数增加时会有内存增长现象，可以在每次循环结束时调用该函数回收常量内存
         * 参数：
         *   - `full:bool` 是否全部回收，*目前回收方式`True`和`False`没有区别*
         */
        exe->gc(Executor::FULL);
        // TODO: looks like to calculate FLOPs and time
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;

            // update progress bar
            (*env).CallVoidMethod(thiz, uptProgressBar, 0);

            for (int i = 0; i < iterations; i++) {
                auto lastPara = model->parameters();
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);
                if (type == "twitter" || type == "TWITTER") {
                    // forward

                } else {
                    example.first[0] = (cast * _Const(1.0f / 255.0f) - meansVarp) / stdsVarp;
                }
                if (example.first[0]->getInfo()->order == NHWC) {
                    example.first[0] = _Convert(example.first[0], NCHW);
                }
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Compute One-Hot
                auto predict = model->forward(example.first[0]);
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(category), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));


                auto logits = _Softmax(predict);
                auto loss    = _CrossEntropy(logits, newTarget);

                // Learning rate decays
                float rate   = LrScheduler::inv(baseLr, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);

                // Update UI and print log
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    (*env).CallVoidMethod(thiz, uptFedTrainInfo, epoch+1, loss->readMap<float>()[0]);
                    // update progress bar
                    (*env).CallVoidMethod(thiz, uptProgressBar, int(i * 100./ iterations));
                    _100Time.reset();
                    lastIndex = i;
                }
                sgd->step(loss);
            }
            nSamples += moveBatchSize;
            // update progress bar
            (*env).CallVoidMethod(thiz, uptProgressBar, 100);
        }
        FSLOG("Finish %d epoch training in C++", epoch);

        exe->dumpProfile();
    }
    // save model
    Variable::save(model->parameters(), "/data/user/0/com.example.fsandroid/files/localTrainedModel.mnn");
    return nSamples;
}

std::shared_ptr<Module> loadFedBabuModel(JNIEnv *env, jstring pPathWholeModel, jstring pPathBodyModel) {
    // Loading computational graph
    auto model = loadMnnModel(env->GetStringUTFChars(pPathWholeModel, 0));

    // Loading parameters if it exists
    auto cBodyPath = env->GetStringUTFChars(pPathBodyModel, 0);
    if (IsFileExist(cBodyPath)) {
        auto body = loadMnnModel(cBodyPath);
        // replace parameters
        auto allParameters = model->parameters();
        auto bodyParameters = body->parameters();
        std::vector<std::string> replaceNames;
        for(int i=0; i<allParameters.size(); ++i) {
            auto dstInfo = allParameters[i]->getInfo();
            // find the parameter with the same shape
            for (int j=0; j<bodyParameters.size(); ++j) {
                auto srcInfo = bodyParameters[j]->getInfo();
                if (dstInfo->dim.size() == srcInfo->dim.size() && dstInfo->order == srcInfo->order && dstInfo->size == srcInfo->size) {
                    Variable::replace(allParameters[i], bodyParameters[j]);
                    replaceNames.push_back(allParameters[i]->name());
                    break;
                }
            }
        }
        std::string logging = "Replace params named";
        for (auto &name: replaceNames) {
            logging += " " + name;
        }
        FSLOG("%s", logging.c_str());
    } else {
        FSLOG("Body model doesn't exists");
    }
    return model;
}

extern "C"
JNIEXPORT int JNICALL
Java_com_example_fsandroid_MainActivity_fedBabuTrainModelInMNN(JNIEnv *env, jobject thiz,
                                                               jstring pPathWholeModel,
                                                               jstring pPathBodyModel,
                                                               jobject pConfig,
                                                               jstring pPathData) {

    auto model = loadFedBabuModel(env, pPathWholeModel, pPathBodyModel);

    // recall java function
    jclass main_activity = (*env).FindClass("com/example/fsandroid/MainActivity");
    jmethodID uptFedTrainInfo = (*env).GetMethodID(main_activity, "updateTrainInfoFromCpp", "(IF)V");
    // jmethodID uptFedTestInfo = (*env).GetMethodID(main_activity, "updateTestInfoFromCpp", "(IF)V");
    jmethodID uptProgressBar = (*env).GetMethodID(main_activity, "uptProgressBar", "(I)V");

    // extract hyper-parameters
    jobject config_data = getJavaAttribute(env, pConfig, "data",  "Lcom/example/fsandroid/configs/Data;");
    jobject config_train = getJavaAttribute(env, pConfig, "train", "Lcom/example/fsandroid/configs/Train;");
    jobject config_opt = getJavaAttribute(env, pConfig, "optimizer", "Lcom/example/fsandroid/configs/Optimizer;");
    jobject config_task = getJavaAttribute(env, pConfig, "task", "Lcom/example/fsandroid/configs/Task;");

    float baseLr = (float) getFloatAttribute(env, config_opt, "lr");
    float weightDecay = (float) getFloatAttribute(env, config_opt, "weight_decay");
    float momentum = (float) getFloatAttribute(env, config_opt, "momentum");
    std::string root = getStrAttribute(env, config_data, "root");
    std::string type = getStrAttribute(env, config_data, "type");
    int batchSize = (int) getIntAttribute(env, config_data, "batch_size");
    int localUpdateSteps = (int) getIntAttribute(env, config_train, "local_update_steps");
    int category = (int) getIntAttribute(env, config_task, "category");

    // obtain mean and var varp
    VARP meansVarp = getMean(type);
    VARP stdsVarp = getStd(type);

    // training dataset
    const char *path_data = env->GetStringUTFChars(pPathData, 0);
    if (!IsFileExist(path_data)) {
        FSLOG("Data in %s doesn't exist", path_data);
    }

    // init Executor
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);

    // init optimizer
    std::shared_ptr<FedBabuSGD> sgd(new FedBabuSGD(model));
    sgd->setMomentum(momentum);
    sgd->setWeightDecay(weightDecay);

    auto dataset = getDatasetPtr(type,path_data, "train");

    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    /**
     * main loop for training
     */
    int nSamples = 0;
    for (int epoch = 0; epoch < localUpdateSteps; ++epoch) {
        FSLOG("Begin %d epoch training in C++", epoch);
        model->clearCache();
        /**
         * 手动回收内存，当在循环中调用MNN表达式求值时，常量部分数据不会在每次循环结束释放，当执行次数增加时会有内存增长现象，可以在每次循环结束时调用该函数回收常量内存
         * 参数：
         *   - `full:bool` 是否全部回收，*目前回收方式`True`和`False`没有区别*
         */
        exe->gc(Executor::FULL);
        // TODO: looks like to calculate FLOPs and time
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;
            int lastIndex = 0;
            int moveBatchSize = 0;

            // update progress bar
            (*env).CallVoidMethod(thiz, uptProgressBar, 0);

            for (int i = 0; i < iterations; i++) {
                auto lastPara = model->parameters();
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto cast       = _Cast<float>(example.first[0]);
                if (type == "twitter" || type == "TWITTER") {
                    // forward
                } else {
                    example.first[0] = (cast * _Const(1.0f / 255.0f) - meansVarp) / stdsVarp;
                }
                if (example.first[0]->getInfo()->order == NHWC) {
                    example.first[0] = _Convert(example.first[0], NCHW);
                }
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Compute One-Hot
                auto predict = model->forward(example.first[0]);
                auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(category), _Scalar<float>(1.0f),
                                         _Scalar<float>(0.0f));


                auto logits = _Softmax(predict);
                auto loss    = _CrossEntropy(logits, newTarget);

                // Learning rate decays
                float rate   = LrScheduler::inv(baseLr, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);

                // Update UI and print log
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    (*env).CallVoidMethod(thiz, uptFedTrainInfo, epoch+1, loss->readMap<float>()[0]);
                    // update progress bar
                    (*env).CallVoidMethod(thiz, uptProgressBar, int(i * 100./ iterations));
                    _100Time.reset();
                    lastIndex = i;
                }
                sgd->step(loss);
            }
            nSamples += moveBatchSize;
            // update progress bar
            (*env).CallVoidMethod(thiz, uptProgressBar, 100);
        }
        FSLOG("Finish %d epoch training in C++", epoch);

        exe->dumpProfile();
    }
    // save model
    auto parameters = model->parameters();
    std::vector<VARP> parameters2Save;
    // only save body parameters here
    for (auto iter : parameters) {
        auto dim = iter->getInfo()->dim;
        if (dim[0] == category) {
            continue;
        } else {
            parameters2Save.push_back(iter);
        }
    }
    Variable::save(parameters2Save, "/data/user/0/com.example.fsandroid/files/localTrainedModel.mnn");
    return nSamples;
}

jobject finetuneModel(JNIEnv *env, jobject thiz, std::shared_ptr<Module> model, jobject pConfig, jstring pPathData) {
    // extract hyper-parameters
    jobject config_data = getJavaAttribute(env, pConfig, "data",  "Lcom/example/fsandroid/configs/Data;");
    jobject config_task = getJavaAttribute(env, pConfig, "task", "Lcom/example/fsandroid/configs/Task;");
    jobject config_finetune = getJavaAttribute(env, pConfig, "finetune", "Lcom/example/fsandroid/configs/Finetune;");

    std::string root = getStrAttribute(env, config_data, "root");
    std::string type = getStrAttribute(env, config_data, "type");
    int batchSize = (int) getIntAttribute(env, config_data, "batch_size");
    int category = (int) getIntAttribute(env, config_task, "category");

    // obtain mean and variance
    VARP meansVarp = getMean(type);
    VARP stdsVarp = getStd(type);

    // sign for finetuning
    bool finetune = (bool) getBoolAttribute(env, config_finetune, "use");

    // Create hashMap object
    jclass hashMap = (*env).FindClass("java/util/HashMap");
    // create hashmap constructor
    jmethodID hashMap_init = (*env).GetMethodID(hashMap, "<init>", "()V");
    // create hashmap object
    jobject jMetrics = (*env).NewObject(hashMap, hashMap_init);

    // training dataset
    const char *path_data = env->GetStringUTFChars(pPathData, 0);
    if (IsFileExist(path_data)) {
        // init Executor
        auto exe = Executor::getGlobalExecutor();
        BackendConfig config;
        exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);

        auto testDataset = getDatasetPtr(type, path_data, "test");
        auto valDataset = getDatasetPtr(type, path_data, "val");

        const size_t testNumWorkers = 0;
        bool shuffle                = false;
        auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(batchSize, true, shuffle, testNumWorkers));
        auto valDataLoader = std::shared_ptr<DataLoader>(valDataset.createLoader(batchSize, true, shuffle, testNumWorkers));

        // Fine tune
        if (finetune) {
            // Obtain hyper-parameters
            int localUpdateSteps = (int) getIntAttribute(env, config_finetune, "local_update_steps");
            float baseLr = (float) getFloatAttribute(env, config_finetune, "lr");
            float weightDecay = (float) getFloatAttribute(env, config_finetune, "weight_decay");
            float momentum = (float) getFloatAttribute(env, config_finetune, "momentum");

            // init optimizer
            std::shared_ptr<SGD> sgd(new SGD(model));
            sgd->setMomentum(momentum);
            sgd->setWeightDecay(weightDecay);

            const size_t trainNumWorkers = 0;
            bool shuffle = true;
            auto trainDataset = getDatasetPtr(type, path_data, "train");
            auto trainDataLoader = std::shared_ptr<DataLoader>(trainDataset.createLoader(batchSize, true, shuffle, trainNumWorkers));

            size_t iterations = trainDataLoader->iterNumber();

            FSLOG("Begin fine-tuning in C++ for %d epochs", localUpdateSteps);
            for (int epoch = 0; epoch < localUpdateSteps; ++epoch) {
                model->clearCache();
                exe->gc(Executor::FULL);
                exe->resetProfile();
                {
                    AUTOTIME;
                    trainDataLoader->reset();
                    model->setIsTraining(true);
                    Timer _100Time;

                    for (int i = 0; i < iterations; i++) {
                        auto trainData  = trainDataLoader->next();
                        auto example    = trainData[0];
                        auto cast       = _Cast<float>(example.first[0]);
                        if (type == "twitter" || type == "TWITTER") {
                            // forward
                        } else {
                            example.first[0] = (cast * _Const(1.0f / 255.0f) - meansVarp) / stdsVarp;
                        }
                        if (example.first[0]->getInfo()->order == NHWC) {
                            example.first[0] = _Convert(example.first[0], NCHW);
                        }

                        // Compute One-Hot
                        auto predict = model->forward(example.first[0]);
                        auto newTarget = _OneHot(_Cast<int32_t>(example.second[0]), _Scalar<int>(category), _Scalar<float>(1.0f),
                                                 _Scalar<float>(0.0f));


                        auto logits = _Softmax(predict);
                        auto loss    = _CrossEntropy(logits, newTarget);

                        // Learning rate decays
                        float rate   = LrScheduler::inv(baseLr, epoch * iterations + i, 0.0001, 0.75);
                        sgd->setLearningRate(rate);
                        sgd->step(loss);
                    }
                }
                exe->dumpProfile();
            }
        }
        FSLOG("Finish fine-tuning in C++");
        // Start evaluateion
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        FSLOG("Begin evaluation in C++")
        // test model
        auto testMetrics = test(model, testDataLoader, type, category, "test", meansVarp, stdsVarp);
        auto valMetrics = test(model, valDataLoader, type, category, "val", meansVarp, stdsVarp);
        FSLOG("Finish evaluation in C++")
        // TODO: Update UI
        exe->dumpProfile();

        // find put method
        jmethodID hashMap_put = (*env).GetMethodID(hashMap, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

        // Merge two metrics
        std::map<std::string, float> metrics;
        metrics.insert(testMetrics.begin(), testMetrics.end());
        metrics.insert(valMetrics.begin(), valMetrics.end());
        std::map<std::string, float> ::iterator it = metrics.begin();
        std::map<std::string, float> ::iterator itEnd = metrics.end();
        //
        jclass classFloat = env->FindClass("java/lang/Float");
        jmethodID initInteger =env->GetMethodID(classFloat,"<init>", "(F)V");
        while (it != itEnd) {
            jstring key = (*env).NewStringUTF(it->first.c_str());
            jobject value = env->NewObject(classFloat,initInteger, it->second);
            (*env).CallObjectMethod(jMetrics, hashMap_put, key, value);
            it++;
        }
        return jMetrics;
    } else {
        // return empty map
        return jMetrics;
    }
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_fsandroid_MainActivity_fedBabuTestModelInMNN(JNIEnv *env, jobject thiz,
                                                              jstring pPathWholeModel,
                                                              jstring pPathBodyModel,
                                                              jobject pConfig,
                                                              jstring pPathData) {
    // Load the computation graph and head
    // trainhead is true
    auto model = loadFedBabuModel(env, pPathWholeModel, pPathBodyModel);

    auto metrics = finetuneModel(env, thiz, model, pConfig, pPathData);
    return metrics;
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_fsandroid_MainActivity_generalTestModelInMNN(JNIEnv *env, jobject thiz, jstring path, jobject pConfig, jstring pPathData) {
    jobject config_train = getJavaAttribute(env, pConfig, "train", "Lcom/example/fsandroid/configs/Train;");
    int mnn_quantization = (int) getIntAttribute(env, config_train, "mnn_quantization");

    const char *cPath = env->GetStringUTFChars(path, 0);
    std::shared_ptr<Module> model;
    if (mnn_quantization == 8 || mnn_quantization == 16) {
        model = loadMnnModel(cPath, mnn_quantization);
    } else {
        model = loadMnnModel(cPath);
    }

    // finetune and test
    return finetuneModel(env, thiz, model, pConfig, pPathData);
}
