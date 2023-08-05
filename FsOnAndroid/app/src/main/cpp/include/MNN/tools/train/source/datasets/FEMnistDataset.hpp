
#ifndef FEMnistDataset_hpp
#define FEMnistDataset_hpp

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"

namespace MNN {
namespace Train {
class MNN_PUBLIC FEMnistDataset : public Dataset {
public:

    Example get(size_t index) override;

    size_t size() override;

    const VARP images();

    const VARP labels();

    static DatasetPtr create(const std::string path, std::string mode);
private:
    explicit FEMnistDataset(const std::string path, std::string mode);
    VARP mImages, mLabels;
    const uint8_t* mImagePtr  = nullptr;
    const uint8_t* mLabelsPtr = nullptr;
};
}
}


#endif // FEMnistDataset_hpp
