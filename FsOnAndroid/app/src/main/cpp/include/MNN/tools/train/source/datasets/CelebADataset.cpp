#include "CelebADataset.hpp"
#include <string.h>
#include <fstream>
#include <string>
#include <android/log.h>
#define CELEBALOG(...) __android_log_print(ANDROID_LOG_VERBOSE, "FS-DEVICE", __VA_ARGS__);
namespace MNN {
    namespace Train {

// referenced from pytorch C++ frontend mnist.cpp
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/src/data/datasets/mnist.cpp
        const int32_t kImageMagicNumber   = 2052;
        const int32_t kTargetMagicNumber  = 2049;
        const int32_t kImageRows          = 84;
        const int32_t kImageColumns       = 84;
        const int32_t kImageChannels      = 3;


        inline bool check_is_little_endian() {
            const uint32_t word = 1;
            return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
        }

//bool IS_LITTLE_ENDIAN = check_is_little_endian();

        inline constexpr uint32_t flip_endianness(uint32_t value) {
        return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) | ((value & 0xff0000u) >> 8u) |
        ((value & 0xff000000u) >> 24u);
    }

    inline uint32_t read_int32(std::ifstream& stream) {
        static const bool is_little_endian = check_is_little_endian();
        uint32_t value;
        stream.read(reinterpret_cast<char*>(&value), sizeof value);
        return is_little_endian ? flip_endianness(value) : value;
    }

    inline uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
        const auto value = read_int32(stream);
        // clang-format off
        MNN_ASSERT(value == expected);
        // clang-format on
        return value;
    }

    inline std::string join_paths(std::string head, const std::string& tail) {
        if (head.back() != '/') {
            head.push_back('/');
        }
        head += tail;
        return head;
    }

    VARP read_celeba_images(const std::string& path, std::string& mode) {
        CELEBALOG("*** Reading CelebA ***")
        auto f_name = join_paths(path, mode + ".data.idx3");
        std::ifstream images(f_name, std::ios::binary);
        if (!images.is_open()) {
            MNN_PRINT("Error opening images file at %s", f_name.c_str());
            MNN_ASSERT(false);
        }

        // From http://yann.lecun.com/exdb/mnist/
        auto read_magic = expect_int32(images, kImageMagicNumber);
        const auto count = read_int32(images);
        auto read_rows = expect_int32(images, kImageRows);
        auto read_cols = expect_int32(images, kImageColumns);
        auto read_channels = expect_int32(images, kImageChannels);
        CELEBALOG("read headers: %d, %d, %d, %d, %d", read_magic, count, read_rows, read_cols, read_channels)


        std::vector<int> dims = {static_cast<int>(count), kImageRows, kImageColumns, kImageChannels};
        int length            = 1;
        for (int i = 0; i < dims.size(); ++i) {
            length *= dims[i];
        }
        auto data = _Input(dims, NHWC, halide_type_of<uint8_t>());
        images.read(reinterpret_cast<char*>(data->writeMap<uint8_t>()), length);
        return data;
    }

    VARP read_celeba_targets(const std::string& path, std::string& mode) {
        auto f_name = join_paths(path, mode + ".label.idx1");
        std::ifstream targets(f_name, std::ios::binary);

        if (!targets.is_open()) {
            MNN_PRINT("Error opening images file at %s", f_name.c_str());
            MNN_ASSERT(false);
        }


        expect_int32(targets, kTargetMagicNumber);
        const auto count = read_int32(targets);
        CELEBALOG("read target headers: %d", count)

        std::vector<int> dims = {static_cast<int>(count)};
        int length            = 1;
        for (int i = 0; i < dims.size(); ++i) {
            length *= dims[i];
        }
        auto labels = _Input(dims, NHWC, halide_type_of<uint8_t>());
        targets.read(reinterpret_cast<char*>(labels->writeMap<uint8_t>()), length);

        return labels;
    }

    CelebADataset::CelebADataset(const std::string path, std::string mode)
            : mImages(read_celeba_images(path, mode)), mLabels(read_celeba_targets(path, mode)) {
        CELEBALOG("Read CelebA data: %s", path.c_str())
        mImagePtr  = mImages->readMap<uint8_t>();
        mLabelsPtr = mLabels->readMap<uint8_t>();
    }

    Example CelebADataset::get(size_t index) {
        auto data  = _Input({kImageRows, kImageColumns, kImageChannels}, NHWC, halide_type_of<uint8_t>());
        auto label = _Input({}, NHWC, halide_type_of<uint8_t>());

        auto dataPtr = mImagePtr + index * kImageRows * kImageColumns * kImageChannels;
        ::memcpy(data->writeMap<uint8_t>(), dataPtr,  kImageRows * kImageColumns * kImageChannels );

        auto labelPtr = mLabelsPtr + index;
        ::memcpy(label->writeMap<uint8_t>(), labelPtr, 1);

        auto returnIndex = _Const(index);
        // return the index for test
        return {{data, returnIndex}, {label}};
    }

    size_t CelebADataset::size() {
        return mImages->getInfo()->dim[0];
    }

    const VARP CelebADataset::images() {
        return mImages;
    }

    const VARP CelebADataset::labels() {
        return mLabels;
    }

    DatasetPtr CelebADataset::create(const std::string path, std::string mode) {
        DatasetPtr res;
        res.mDataset.reset(new CelebADataset(path, mode));
        return res;
    }
}
}