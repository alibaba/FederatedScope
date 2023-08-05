//
// Created by David on 2023/3/16.
//

#ifndef TextDataset_hpp
#define TextDataset_hpp

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"
#include "map"

using namespace std;

namespace MNN {
    namespace Train {
        class MNN_PUBLIC TextDataset : public Dataset {
        public:

        Example get(size_t index) override;

        size_t size() override;

        const VARP texts();

        const VARP labels();

        static DatasetPtr create(const std::string path, std::string mode);
        private:
        explicit TextDataset(const std::string path, std::string mode);
        uint32_t mDataCnt = 0;
        VARP mText,  mLabels;
        const float * mTextPtr  = nullptr;
        const uint8_t* mLabelsPtr = nullptr;

        std::map<int, std::vector<float>> mEmbeding;
    };
}
}


#endif