#include "TextDataset.hpp"
#include <string.h>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <android/log.h>


namespace MNN {
    namespace Train {

// referenced from pytorch C++ frontend mnist.cpp
        const int32_t kTextMagicNumber   = 2025;  // customized magic number
        const int32_t kTargetMagicNumber  = 2049;
        const int32_t kTextRows          = 140;  // max text length. 140 is adopted for twitter
//  TODO: for simplicity, the current version assumes pre-padding.
        // In the future, we will support non-padding format
        const int32_t kTextColumns       = 50;   // feature length, for Glove emb, we adopt 50

        inline bool check_is_little_endian() {
            const uint32_t word = 1;
            return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
        }

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

        inline uint32_t expect_int32(std::ifstream& stream, int32_t expected) {
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

        int read_emb_binary(const std::string& path, std::map<int, std::vector<float>> &res){

            // load embedding file
            auto f_name = join_paths(path, "../glove.6B.50d.idx1");
            std::ifstream emb_f(f_name, std::ios::binary);
            if (!emb_f.is_open()) {
                MNN_PRINT("Error opening speeches file at %s", f_name.c_str());
                MNN_ASSERT(false);
                return -1;
            }

            const auto word_count = read_int32(emb_f);
            const auto emb_dim = read_int32(emb_f);

            std::vector<uint32_t> dims = {word_count, emb_dim};
            for (int word_idx = 0; word_idx < word_count; ++word_idx) {
                std::vector<float> emb_val;
                emb_val.reserve(emb_dim);

                // method 1: read each float one by one
                float f;
                for (int i = 0; i < emb_dim; i++){
                    emb_f.read(reinterpret_cast<char*>(&f), sizeof(float));
                    emb_val.emplace_back(f);
                }

                res.insert(std::pair<int, std::vector<float>>(word_idx, emb_val));
            }

            return 0;
        }


        VARP read_texts(const std::string& path, std::string& mode, std::map<int, std::vector<float>> &emb) {
            auto f_name = join_paths(path, mode + ".data.idx2");
            std::ifstream texts(f_name, std::ios::binary);
            if (!texts.is_open()) {
                MNN_PRINT("Error opening text file at %s", f_name.c_str());
                MNN_ASSERT(false);
            }
            if (emb.empty()){
                MNN_PRINT("Error loading embedding file");
                MNN_ASSERT(false);
            }

            expect_int32(texts, kTextMagicNumber);
            const auto count = read_int32(texts);

            expect_int32(texts, kTextRows);

            std::vector<int> dims = {static_cast<int>(count), kTextRows, kTextColumns};
            int length            = 1;
            for (int i = 0; i < dims.size(); ++i) {
                length *= dims[i];
            }
            auto data = _Input(dims, NCHW, halide_type_of<float>());
            auto dataPtr = data->writeMap<float>();
            ::memset(dataPtr, 0, length);
            for (int i=0; i <length; i += kTextColumns){
                int token_idx = int(read_int32(texts));
                std::vector<float> values (kTextColumns);
                if (token_idx != -1){
                    values = emb[token_idx];
                }
                for (int j=0; j < kTextColumns; j++){
                    dataPtr[i+j] = values[j];
                }
            }

            return data;
        }

        inline VARP read_lengths(const std::string& path, std::string& mode) {
            auto f_name = join_paths(path, mode + ".length.idx1");
            std::ifstream lengths(f_name, std::ios::binary);

            if (!lengths.is_open()) {
                MNN_PRINT("Error opening length file at %s", f_name.c_str());
                MNN_ASSERT(false);
            }


            expect_int32(lengths, kTargetMagicNumber);
            const auto count = read_int32(lengths);

            std::vector<int> dims = {static_cast<int>(count)};
            int length            = 1;
            for (int i = 0; i < dims.size(); ++i) {
                length *= dims[i];
            }
            auto labels = _Input(dims, NCHW, halide_type_of<uint8_t>());
            lengths.read(reinterpret_cast<char*>(labels->writeMap<uint8_t>()), length);

            return labels;
        }

        inline VARP read_targets(const std::string& path, std::string& mode) {
            auto f_name = join_paths(path, mode + ".label.idx1");
            std::ifstream targets(f_name, std::ios::binary);

            if (!targets.is_open()) {
                MNN_PRINT("Error opening label file at %s", f_name.c_str());
                MNN_ASSERT(false);
            }


            expect_int32(targets, kTargetMagicNumber);
            const auto count = read_int32(targets);

            std::vector<int> dims = {static_cast<int>(count)};
            int length            = 1;
            for (int i = 0; i < dims.size(); ++i) {
                length *= dims[i];
            }
            auto labels = _Input(dims, NCHW, halide_type_of<uint8_t>());
            targets.read(reinterpret_cast<char*>(labels->writeMap<uint8_t>()), length);

            return labels;
        }



        TextDataset::TextDataset(const std::string path, std::string mode)
        {

            auto start = std::chrono::high_resolution_clock::now();
            read_emb_binary(path, mEmbeding);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            mText = read_texts(path, mode, mEmbeding);
            mLabels = read_targets(path, mode);

            mTextPtr  = mText->readMap<float>();
            mLabelsPtr = mLabels->readMap<uint8_t>();
        }

        Example TextDataset::get(size_t index) {
            auto data  = _Input({kTextRows*kTextColumns, 1, 1}, NCHW, halide_type_of<float>());
            auto label = _Input({}, NCHW, halide_type_of<uint8_t>());

            // TODO: add embedding lookup operation

            auto dataPtr = mTextPtr + index * kTextRows * kTextColumns;
            ::memcpy(data->writeMap<float>(), dataPtr, kTextRows * kTextColumns);

            auto labelPtr = mLabelsPtr + index;
            ::memcpy(label->writeMap<uint8_t>(), labelPtr, 1);

            auto returnIndex = _Const(index);
            return {{data, returnIndex}, {label}};
        }

        size_t TextDataset::size() {
            return mText->getInfo()->dim[0];
        }

        const VARP TextDataset::texts() {
            return mText;
        }

        const VARP TextDataset::labels() {
            return mLabels;
        }

        DatasetPtr TextDataset::create(const std::string path, std::string mode) {
            DatasetPtr res;
            res.mDataset.reset(new TextDataset(path, mode));
            return res;
        }
    }
}
