#include <iostream>
#include <sstream>
#include <cassert>
#include <stdfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <bit>
#include <cstdint>
#include <vector>
#include <thread>
#include <cinttypes>

#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "safetensors.hh"
#define USE_MMAP

// TODO: provide printer for each dtype for efficiency.
std::string to_string(safetensors::dtype dtype, const uint8_t *data)
{
    switch (dtype)
    {
    case safetensors::dtype::kBOOL:
    {
        return std::to_string(data[0] ? 1 : 0);
    }
    case safetensors::dtype::kUINT8:
    {
        return std::to_string(data[0]);
    }
    case safetensors::dtype::kINT8:
    {
        return std::to_string(*reinterpret_cast<const int8_t *>(data));
    }
    case safetensors::dtype::kUINT16:
    {
        return std::to_string(*reinterpret_cast<const uint16_t *>(data));
    }
    case safetensors::dtype::kINT16:
    {
        return std::to_string(*reinterpret_cast<const int16_t *>(data));
    }
    case safetensors::dtype::kUINT32:
    {
        return std::to_string(*reinterpret_cast<const uint32_t *>(data));
    }
    case safetensors::dtype::kINT32:
    {
        return std::to_string(*reinterpret_cast<const int32_t *>(data));
    }
    case safetensors::dtype::kUINT64:
    {
        return std::to_string(*reinterpret_cast<const uint64_t *>(data));
    }
    case safetensors::dtype::kINT64:
    {
        return std::to_string(*reinterpret_cast<const int64_t *>(data));
    }
    case safetensors::dtype::kFLOAT16:
    {

        std::float16_t data_val = *reinterpret_cast<const std::float16_t *>(data);
        return std::to_string(static_cast<float>(data_val));
    }
    case safetensors::dtype::kBFLOAT16:
    {
        std::bfloat16_t data_val = *reinterpret_cast<const std::bfloat16_t *>(data);
        return std::to_string(static_cast<float>(data_val));
    }
    case safetensors::dtype::kFLOAT32:
    {
        return std::to_string(*reinterpret_cast<const float *>(data));
    }
    case safetensors::dtype::kFLOAT64:
    {
        return std::to_string(*reinterpret_cast<const double *>(data));
    }
    }

    return std::string("???");
}

//
// print tensor in linearized 1D array
// In safetensors, data is not strided(tightly packed)
//
std::string to_string_snipped(const safetensors::tensor_t &t,
                              const uint8_t *databuffer, size_t N = 8)
{
    std::stringstream ss;
    size_t nitems = safetensors::get_shape_size(t);
    size_t itembytes = safetensors::get_dtype_bytes(t.dtype);

    if ((N == 0) || ((N * 2) >= nitems))
    {
        ss << "[";
        for (size_t i = 0; i < nitems; i++)
        {
            if (i > 0)
            {
                ss << ", ";
            }
            ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
        }
        ss << "]";
    }
    else
    {
        ss << "[";
        size_t head_end = (std::min)(N, nitems);
        size_t tail_start = (std::max)(nitems - N, head_end);

        for (size_t i = 0; i < head_end; i++)
        {
            if (i > 0)
            {
                ss << ", ";
            }
            ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
        }

        ss << ", ..., ";

        for (size_t i = tail_start; i < nitems; i++)
        {
            if (i > tail_start)
            {
                ss << ", ";
            }
            ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
        }

        ss << "]";
    }

    return ss.str();
}

int safetensor_sanity_checks(safetensors::safetensors_t &st, std::string &warn, std::string &err, bool open_ret)
{

    if (warn.size())
    {
        std::cout << "WARN: " << warn << "\n";
    }

    if (!open_ret)
    {
        std::cerr << "  ERR: " << err << "\n";
        return EXIT_FAILURE;
    }

    // Check if data_offsets are valid.
    if (!safetensors::validate_data_offsets(st, err))
    {
        std::cerr << "Invalid data_offsets\n";
        std::cerr << err << "\n";

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{ // The main function, where program execution begins

    // format file
    if (argc != 2)
    {
        std::cout << "Please pass in a tensor file" << std::endl;
        return -1;
    }
    std::string filename = argv[1];

    std::string warn, err;
    safetensors::safetensors_t st;

#if defined(USE_MMAP)
    printf("USE mmap\n");
    bool ret = safetensors::mmap_from_file(filename, &st, &warn, &err);
#else
    bool ret = safetensors::load_from_file(filename, &st, &warn, &err);
#endif

    int sanity_res = safetensor_sanity_checks(st, warn, err, ret);

    const uint8_t *databuffer{nullptr};
    if (st.mmaped)
    {
        databuffer = st.databuffer_addr;
    }
    else
    {
        databuffer = st.storage.data();
    }

    // Print Tensor info & value.
    for (size_t i = 0; i < st.tensors.size(); i++)
    {
        std::string key = st.tensors.keys()[i];
        safetensors::tensor_t tensor;
        st.tensors.at(i, &tensor);

        std::cout << key << ": "
                  << safetensors::get_dtype_str(tensor.dtype) << " ";
        std::cout << "[";
        for (size_t i = 0; i < tensor.shape.size(); i++)
        {
            if (i > 0)
            {
                std::cout << ", ";
            }
            std::cout << std::to_string(tensor.shape[i]);
        }
        std::cout << "]\n";

        std::cout << "  data_offsets["
                  << std::to_string(tensor.data_offsets[0]) << ", "
                  << std::to_string(tensor.data_offsets[1]) << "]\n";
        std::cout << "  " << to_string_snipped(tensor, databuffer) << "\n";
    }

    // Print metadata
    if (st.metadata.size())
    {
        std::cout << "\n";
        std::cout << "__metadata__\n";
        for (size_t i = 0; i < st.metadata.size(); i++)
        {
            std::string key = st.metadata.keys()[i];
            std::string value;
            st.metadata.at(i, &value);

            std::cout << "  " << key << ":" << value << "\n";
        }
    }

    return 0; // Indicates successful program execution
}