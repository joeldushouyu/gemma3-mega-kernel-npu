
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
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gemma3_common.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu.h"

#include <cassert>
#if !defined(SAFETENSORS_CPP_NO_IMPLEMENTATION)
#define SAFETENSORS_CPP_IMPLEMENTATION
#endif
#include "safetensors.hh"


#include <fstream>
#include <nlohmann/json.hpp>



void add_tensor_to_safetensor(ggml_tensor* cur_ggml_tensor, const std::string& tensor_name,  safetensors::safetensors_t&st, bool convert_FP_16 ){


    char num_non_one_dimeison = 0;
    size_t data_size = 1;
    for(size_t i = 0; i < 4; i++){
        if(cur_ggml_tensor->ne[i] > 1){ 
            num_non_one_dimeison++; 
            data_size *= cur_ggml_tensor->ne[i];
        }
    }

    size_t dst_offset = st.storage.size();
    safetensors::tensor_t tensor;
    if(cur_ggml_tensor->type ==GGML_TYPE_F32 ){
        tensor.dtype = safetensors::dtype::kFLOAT32;
        data_size *=sizeof(float);

    }else if(cur_ggml_tensor->type == GGML_TYPE_BF16){
        tensor.dtype = safetensors::dtype::kBFLOAT16;
        data_size *= sizeof(std::bfloat16_t);

    }else if(cur_ggml_tensor->type == GGML_TYPE_F16){
        if(convert_FP_16){
            tensor.dtype = safetensors::dtype::kFLOAT32; 
            data_size *=sizeof(float);
        }
        else{
            tensor.dtype = safetensors::dtype::kFLOAT16; 
            data_size *=sizeof(int16_t);
        }

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q4_0){
        // going to convert to bfloat16 through dequantization
        tensor.dtype = safetensors::dtype::kBFLOAT16;
        data_size *= sizeof(std::bfloat16_t);

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q4_K){
        tensor.dtype = safetensors::dtype::kBFLOAT16;
        data_size *= sizeof(std::bfloat16_t);

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q6_K){
        tensor.dtype = safetensors::dtype::kBFLOAT16;
        data_size *= sizeof(std::bfloat16_t);
    }else{
        std::cout << "Unexpected datatype" <<  cur_ggml_tensor->type<< std::endl;
        assert(1 == 2);

    }


    tensor.data_offsets[0] = dst_offset;
    tensor.data_offsets[1] = dst_offset + data_size;
    tensor.shape.resize(num_non_one_dimeison);

    if(num_non_one_dimeison == 1){
        tensor.shape[0] = cur_ggml_tensor->ne[0]; 
    }else if(num_non_one_dimeison == 2){
        // the shape order is different in safetensor
        // although both are row-major order
        // safetensor is row_dimxcolum_dim (more mathematically version)
        // ggml is column_dim(number of element in a row) x row_dim
        tensor.shape[0] = cur_ggml_tensor->ne[1];
        tensor.shape[1] = cur_ggml_tensor->ne[0];
    }else if (num_non_one_dimeison == 4){
        tensor.shape[0] = cur_ggml_tensor->ne[3];
        tensor.shape[1] = cur_ggml_tensor->ne[2];
        tensor.shape[2] = cur_ggml_tensor->ne[1];
        tensor.shape[3] = cur_ggml_tensor->ne[0];
    }else{
        // num_non_one_dimeison == 3
        tensor.shape[0] = cur_ggml_tensor->ne[2];
        tensor.shape[1] = cur_ggml_tensor->ne[1];
        tensor.shape[2] = cur_ggml_tensor->ne[0];
    }


    st.storage.resize(dst_offset + data_size);

    if(cur_ggml_tensor->type == GGML_TYPE_Q4_0){
        std::vector<std::bfloat16_t> res = dequant_whole_q4_0_tensor( cur_ggml_tensor );
        assert(res.size() ==   data_size/sizeof(std::bfloat16_t) );
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q4_K){
        std::vector<std::bfloat16_t> res = dequant_whole_q4_k_tensor(cur_ggml_tensor);
        assert(res.size() ==   data_size/sizeof(std::bfloat16_t) );
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q6_K){
        std::vector<std::bfloat16_t> res = dequant_whole_q6_k_tensor(cur_ggml_tensor);
        assert(res.size() == data_size/sizeof(std::bfloat16_t) );
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);


    }else if( cur_ggml_tensor->type == GGML_TYPE_F16 && convert_FP_16 ){
        std::vector<float> res;
        const std::float16_t* src = static_cast<const std::float16_t*>(cur_ggml_tensor->data);

        for(size_t i = 0; i < data_size/sizeof(float)   ; i++){
            res.push_back(static_cast<float>(src[i]));
        }
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);
    }else{
        memcpy(st.storage.data() + dst_offset, cur_ggml_tensor->data, data_size);
    }

    st.tensors.insert(tensor_name, tensor);
}


gemma3_model random_init_from_file(const std::string &fname_language_model){

    gemma3_model model("CPU");
    

    // a local scope for the gguf context
    {
        struct gguf_init_params language_params = {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ &model.language_ctx_gguf,
        };
        model.language_gguf_metadata_ctx = gguf_init_from_file(fname_language_model.c_str(), language_params);
        
        struct gguf_init_params vision_params = {
            /*.no_alloc   =*/ true,
            /*.ctx        =*/ &model.vision_ctx_gguf,
        };

    }


    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.language_buf_gguf = ggml_backend_alloc_ctx_tensors(model.language_ctx_gguf, model.backends[0]);

    model.output_norm_weight = checked_get_tensor(model.language_ctx_gguf,  "output_norm.weight"); // for example
    model.token_embd_weight = checked_get_tensor(model.language_ctx_gguf,  "output_embd.weight"); // for example
    load_tensor_to_backend_memory(fname_language_model, 2, model.language_gguf_metadata_ctx, model.language_ctx_gguf);



    return model;


}

int main(int argc, char ** argv){
    // format -m <language_model_file>  <vision_model_file> <output_directory> decode_mode
    if (argc != 4) {
        fprintf(stderr, "Usage: %s -m <language_model_file> <vision_model_file> \n", argv[0]);
        return 1;
    }   
    // Get the Q4_K_M quantization file and then dequantization it
    std::string language_model_file = argv[2];

    std::string output_directory = argv[3];

    auto model = random_init_from_file(language_model_file);
    

    fprintf(stdout, "Model initialized from file: %s\n", language_model_file.c_str());



    safetensors::safetensors_t st;
    //language model part
    add_tensor_to_safetensor(model.output_norm_weight, "language_model.model.norm.weight", st, true);
    add_tensor_to_safetensor(model.token_embd_weight, "language_model.model.embd.weight", st, true);

    std::string filename = "model-00001-of-00001.safetensors";
    std::string warn;
    std::string err;

 bool ret = safetensors::save_to_file(st,output_directory + "/" + filename, &warn, &err);
    if (warn.size()) {
        std::cout << "WARN: " << warn << "\n";
    }
    if (!ret) {
        std::cerr << "Failed to write safetensor data to " << filename << "\n";
        if (err.size()) {
        std::cout << "ERR: " << err << "\n";
        }
        return EXIT_FAILURE;
    }
    std::cout <<"SUCCESS" <<std::endl;
    return EXIT_SUCCESS;

}