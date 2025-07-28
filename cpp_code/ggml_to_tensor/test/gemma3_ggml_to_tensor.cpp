
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

bool almost_equal(float a, float b, float abs_tol = 1e-3f, float rel_tol = 1e-3f) {
    return std::abs(a - b) <= std::max(abs_tol, rel_tol * std::max(std::abs(a), std::abs(b)));
}





// // Convert float to bfloat16_t by truncating the lower 16 bits
// std::bfloat16_t float_to_bfloat16(float f) {
//     uint32_t as_int = std::bit_cast<uint32_t>(f);
//     uint16_t upper = static_cast<uint16_t>(as_int >> 16); // keep high 16 bits
//     return std::bit_cast<std::bfloat16_t>(upper);
// }

// void dequantize_row_q4_0_bfloat_at_end(const block_q4_0 * GGML_RESTRICT x, std::bfloat16_t * GGML_RESTRICT y, int64_t k) {
//     static const int qk = QK4_0;

//     assert(k % qk == 0);

//     const int nb = k / qk;

//     for (int i = 0; i < nb; i++) {
//         const float d = GGML_FP16_TO_FP32(x[i].d);

//         for (int j = 0; j < qk/2; ++j) {
//             const int x0 = (x[i].qs[j] & 0x0F) - 8;
//             const int x1 = (x[i].qs[j] >>   4) - 8;

//             y[i*qk + j + 0   ] = static_cast<std::bfloat16_t>( x0*d);
//             y[i*qk + j + qk/2] = static_cast<std::bfloat16_t>( x1*d);
//         }
//     }
// }
//copy from ggml for now

template< typename T>
void minus_one_on_vector(std::vector<T> & vec ){
    for(size_t i = 0; i < vec.size(); i++){
        vec[i] -= static_cast<T>(1);
    }
}

uint32_t add_tensor_to_safetensor(ggml_tensor* cur_ggml_tensor, const std::string& tensor_name,  safetensors::safetensors_t&st, bool convert_FP_16, bool apply_weight_minus_one_offset=false){

    // apply_weight_minus_one_offset due to
    //Note: Because of the special RMS norm, in llama.cpp, it adds 1.0 to the weight normalization https://github.com/ggml-org/llama.cpp/blob/0a5a3b5cdfd887cf0f8e09d9ff89dee130cfcdde/convert_hf_to_gguf.py#L4327
    // But in order to convert to a safetensor that runs in python for reference, need to minus 1 for all nrom weight

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

    }
    else if(cur_ggml_tensor->type == GGML_TYPE_F16){
        if(convert_FP_16){
            tensor.dtype = safetensors::dtype::kFLOAT32; 
            data_size *=sizeof(float);
        }
        else{
            tensor.dtype = safetensors::dtype::kFLOAT16; 
            data_size *=sizeof(int16_t);
        }

     }
    else if(cur_ggml_tensor->type == GGML_TYPE_Q4_0){
        // going to convert to bfloat16 through dequantization
        tensor.dtype = safetensors::dtype::kBFLOAT16;
        data_size *= sizeof(std::bfloat16_t);

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q4_K){
        tensor.dtype = safetensors::dtype::kBFLOAT16;
        data_size *= sizeof(std::bfloat16_t);

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q6_K){
        tensor.dtype = safetensors::dtype::kBFLOAT16;
        data_size *= sizeof(std::bfloat16_t);
    }
    else{
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
    // void * data_res_ptr = nullptr;
    if(cur_ggml_tensor->type == GGML_TYPE_Q4_0){
        std::vector<std::bfloat16_t> res = dequant_whole_q4_0_tensor( cur_ggml_tensor );
        assert(res.size() ==   data_size/sizeof(std::bfloat16_t) );
        
        if(apply_weight_minus_one_offset){
            for(size_t i = 0; i < res.size(); i++){
                res[i] -= static_cast<std::bfloat16_t>(1);
            }
        }
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);

    }
    else if(cur_ggml_tensor->type == GGML_TYPE_Q4_K){
        std::vector<std::bfloat16_t> res = dequant_whole_q4_k_tensor(cur_ggml_tensor);
        assert(res.size() ==   data_size/sizeof(std::bfloat16_t) );
        
        if(apply_weight_minus_one_offset){
            for(size_t i = 0; i < res.size(); i++){
                res[i] -= static_cast<std::bfloat16_t>(1);
            }
        }
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);

    }else if(cur_ggml_tensor->type == GGML_TYPE_Q6_K){
        std::vector<std::bfloat16_t> res = dequant_whole_q6_k_tensor(cur_ggml_tensor);
        assert(res.size() == data_size/sizeof(std::bfloat16_t) );

        if(apply_weight_minus_one_offset){
            for(size_t i = 0; i < res.size(); i++){
                res[i] -= static_cast<std::bfloat16_t>(1);
            }
        }
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);
    }
    else if( cur_ggml_tensor->type == GGML_TYPE_F16 && convert_FP_16 ){
        std::vector<float> res;
        const std::float16_t* src = static_cast<const std::float16_t*>(cur_ggml_tensor->data);
        for(size_t i = 0; i < data_size/sizeof(float); i++){
            res.push_back(static_cast<float>(src[i]));
        }

        
        if(apply_weight_minus_one_offset){
            for(size_t i = 0; i < res.size(); i++){
                res[i] -= static_cast<std::bfloat16_t>(1);
            }
        }
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);

    }else if(cur_ggml_tensor->type == GGML_TYPE_F16 && !convert_FP_16){
        std::float16_t* data_ptr = static_cast<std::float16_t*>(cur_ggml_tensor->data);
        std::vector<std::float16_t> res( data_ptr, data_ptr+  data_size/sizeof(std::float16_t)  );
   
        if(apply_weight_minus_one_offset){
            for(size_t i = 0; i < res.size(); i++){
                res[i] -= static_cast<std::float16_t>(1);
            }
        }
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);

    }
    else if(cur_ggml_tensor->type == GGML_TYPE_F32){
        float * data_ptr = static_cast<float*>(cur_ggml_tensor->data);
        std::vector<float> res( data_ptr, data_ptr+  data_size/sizeof(float)  );
   
        if(apply_weight_minus_one_offset){
            for(size_t i = 0; i < res.size(); i++){
                res[i] -= static_cast<float>(1);
            }
        }
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);
  
    }
    else if(cur_ggml_tensor->type == GGML_TYPE_BF16){
        std::bfloat16_t* data_ptr = static_cast<std::bfloat16_t*>(cur_ggml_tensor->data);
        std::vector<std::bfloat16_t> res( data_ptr, data_ptr+  data_size/sizeof(std::bfloat16_t)  );
   
        if(apply_weight_minus_one_offset){
            for(size_t i = 0; i < res.size(); i++){
                res[i] -= static_cast<std::bfloat16_t>(1);
            }
        }
 
        memcpy(st.storage.data() + dst_offset, res.data(), data_size);
    }
    else{
        std::cout << "unhanle type" << cur_ggml_tensor->type  << std::endl;
        assert(1==2);
    }

    st.tensors.insert(tensor_name, tensor);

    return data_size;
}



int main(int argc, char ** argv){
    // format -m <language_model_file>  <vision_model_file> <output_directory> decode_mode
    if (argc != 6) {
        fprintf(stderr, "Usage: %s -m <language_model_file> <vision_model_file> \n", argv[0]);
        return 1;
    }   
    // Get the Q4_K_M quantization file and then dequantization it
    std::string language_model_file = argv[2];
    std::string vision_model_file = argv[3];
    std::string output_directory = argv[4];
    std::string decode_mode = argv[5];
    auto model = gemma3_mode_init_from_file(language_model_file, vision_model_file);
    
    bool convert_fp_16 = true;
    if(decode_mode =="No_FP16"){
        convert_fp_16 = false;
    }
    fprintf(stdout, "Model initialized from file: %s\n", language_model_file.c_str());



    uint32_t total_byte_size = 0;

    safetensors::safetensors_t st;
    //language model part
    total_byte_size += add_tensor_to_safetensor(model.token_embd_weight, "language_model.model.embed_tokens.weight", st,convert_fp_16);
    const std::string common_language_header = "language_model.model.layers.";
    for(size_t i = 0; i < model.attn_k_weight.size(); i++){


        std::string att_norm_weight_header = common_language_header + std::to_string(i) + ".input_layernorm.weight";
        std::string ffn_norm_weight_header = common_language_header + std::to_string(i) + ".pre_feedforward_layernorm.weight";
        std::string post_attention_norm_header = common_language_header + std::to_string(i)  + ".post_attention_layernorm.weight";
        std::string post_ffw_norm_weight_header = common_language_header + std::to_string(i) + ".post_feedforward_layernorm.weight";
        std::string self_attn_q_norm_header = common_language_header + std::to_string(i) + ".self_attn.q_norm.weight";
        std::string self_attn_k_norm_header = common_language_header + std::to_string(i) + ".self_attn.k_norm.weight";


        std::string ffn_down_weight_header = common_language_header + std::to_string(i) + ".mlp.down_proj.weight";
        std::string ffn_gate_weight_header = common_language_header + std::to_string(i) + ".mlp.gate_proj.weight";
        std::string ffn_up_weight_header = common_language_header + std::to_string(i) + ".mlp.up_proj.weight";



        std::string self_attn_k_weight_header = common_language_header + std::to_string(i) + ".self_attn.k_proj.weight";
        std::string self_attn_output_weight_header = common_language_header + std::to_string(i) + ".self_attn.o_proj.weight";

        std::string self_attn_q_weight_header = common_language_header + std::to_string(i) + ".self_attn.q_proj.weight";
        std::string self_attn_v_weight_header = common_language_header + std::to_string(i) +  ".self_attn.v_proj.weight";



        //Note: Because of the special RMS norm, in llama.cpp, it adds 1.0 to the weight normalization https://github.com/ggml-org/llama.cpp/blob/0a5a3b5cdfd887cf0f8e09d9ff89dee130cfcdde/convert_hf_to_gguf.py#L4327
        // But in order to convert to a safetensor that runs in python for reference, need to minus 1 for all nrom weight


        total_byte_size += add_tensor_to_safetensor(model.attn_norm_weight.at(i), att_norm_weight_header, st,convert_fp_16,true);
        total_byte_size += add_tensor_to_safetensor(model.ffn_norm_weight.at(i), ffn_norm_weight_header, st,convert_fp_16, true);
        total_byte_size += add_tensor_to_safetensor(model.post_attention_norm_weight.at(i), post_attention_norm_header, st,convert_fp_16,true);
        total_byte_size += add_tensor_to_safetensor(model.post_ffw_norm_weight.at(i), post_ffw_norm_weight_header, st,convert_fp_16,true);
        total_byte_size += add_tensor_to_safetensor(model.attn_q_norm_weight.at(i), self_attn_q_norm_header, st,convert_fp_16,true);
        total_byte_size += add_tensor_to_safetensor(model.attn_k_norm_weight.at(i), self_attn_k_norm_header, st,convert_fp_16,true);


        total_byte_size += add_tensor_to_safetensor(model.ffn_down_weight.at(i), ffn_down_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.ffn_gate_weight.at(i), ffn_gate_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.ffn_up_weight.at(i), ffn_up_weight_header, st,convert_fp_16);

        total_byte_size += add_tensor_to_safetensor(model.attn_k_weight.at(i), self_attn_k_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.attn_output_weight.at(i), self_attn_output_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.attn_q_weight.at(i), self_attn_q_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.attn_v_weight.at(i), self_attn_v_weight_header, st,convert_fp_16);





    }
    total_byte_size += add_tensor_to_safetensor(model.output_norm_weight, "language_model.model.norm.weight", st, convert_fp_16);

    total_byte_size += add_tensor_to_safetensor(model.mm_input_projection_weight,"multi_modal_projector.mm_input_projection_weight", st,convert_fp_16);
    //NOTEL becayuse of specuial gemma rms norm weight, also apply -1 on soft_emb_norm weight
    //https://github.com/ggml-org/llama.cpp/blob/00fa15fedc79263fa0285e6a3bbb0cfb3e3878a2/convert_hf_to_gguf.py#L4797    
    total_byte_size += add_tensor_to_safetensor(model.mm_soft_emb_norm_weight,"multi_modal_projector.mm_soft_emb_norm.weight", st,convert_fp_16, true);
    total_byte_size += add_tensor_to_safetensor(model.v_patch_embd_bias,"vision_tower.vision_model.embeddings.patch_embedding.bias",st,convert_fp_16);
    total_byte_size += add_tensor_to_safetensor(model.v_patch_embd_weight, "vision_tower.vision_model.embeddings.patch_embedding.weight", st,convert_fp_16);
    total_byte_size += add_tensor_to_safetensor(model.v_position_embd_weight, "vision_tower.vision_model.embeddings.position_embedding.weight", st,convert_fp_16);
    total_byte_size += add_tensor_to_safetensor(model.v_post_ln_biase, "vision_tower.vision_model.post_layernorm.bias", st,convert_fp_16);
    total_byte_size += add_tensor_to_safetensor(model.v_post_ln_weight,"vision_tower.vision_model.post_layernorm.weight", st,convert_fp_16);

    const std::string common_vision_encoder_header = "vision_tower.vision_model.encoder.layers.";
    for(size_t i = 0; i < model.v_blk_ln1_bias.size(); i++){

        std::string v_blk_ln1_bias_header = common_vision_encoder_header + std::to_string(i) + ".layer_norm1.bias";
        std::string v_blk_ln1_weight_header = common_vision_encoder_header + std::to_string(i) + ".layer_norm1.weight";
        std::string v_blk_ln2_bias_header = common_vision_encoder_header + std::to_string(i) + ".layer_norm2.bias";
        std::string v_blk_ln2_weight_header = common_vision_encoder_header + std::to_string(i) + ".layer_norm2.weight";
        std::string v_blk_ffn_down_bias_header = common_vision_encoder_header + std::to_string(i)  + ".mlp.fc2.bias";
        std::string v_blk_ffn_down_weight_header = common_vision_encoder_header + std::to_string(i) + ".mlp.fc2.weight";
        std::string v_blk_ffn_up_bias_header = common_vision_encoder_header + std::to_string(i) + ".mlp.fc1.bias";
        std::string v_blk_ffn_up_weight_header = common_vision_encoder_header + std::to_string(i)  + ".mlp.fc1.weight";
        std::string v_blk_attn_k_bias_header = common_vision_encoder_header + std::to_string(i) + ".self_attn.k_proj.bias";
        std::string v_blk_attn_k_weight_header = common_vision_encoder_header + std::to_string(i) + ".self_attn.k_proj.weight";
        std::string v_blk_attn_out_bias_header = common_vision_encoder_header + std::to_string(i) + ".self_attn.out_proj.bias";
        std::string v_blk_attn_out_weight_header = common_vision_encoder_header + std::to_string(i) + ".self_attn.out_proj.weight";
        std::string v_blk_attn_q_bias_header = common_vision_encoder_header + std::to_string(i) + ".self_attn.q_proj.bias";
        std::string v_blk_attn_q_weight_header = common_vision_encoder_header + std::to_string(i)  + ".self_attn.q_proj.weight";
        std::string v_blk_attn_v_bias_header = common_vision_encoder_header + std::to_string(i) + ".self_attn.v_proj.bias";
        std::string v_blk_attn_v_weight_header = common_vision_encoder_header + std::to_string(i) + ".self_attn.v_proj.weight";
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ln1_bias.at(i), v_blk_ln1_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ln1_weight.at(i), v_blk_ln1_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ln2_bias.at(i), v_blk_ln2_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ln2_weight.at(i), v_blk_ln2_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ffn_down_bias.at(i), v_blk_ffn_down_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ffn_down_weight.at(i), v_blk_ffn_down_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ffn_up_bias.at(i), v_blk_ffn_up_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_ffn_up_weight.at(i), v_blk_ffn_up_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_k_bias.at(i), v_blk_attn_k_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_k_weight.at(i), v_blk_attn_k_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_out_bias.at(i), v_blk_attn_out_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_out_weight.at(i), v_blk_attn_out_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_q_bias.at(i), v_blk_attn_q_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_q_weight.at(i), v_blk_attn_q_weight_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_v_bias.at(i), v_blk_attn_v_bias_header, st,convert_fp_16);
        total_byte_size += add_tensor_to_safetensor(model.v_blk_attn_v_weight.at(i), v_blk_attn_v_weight_header, st,convert_fp_16);  

        

    }



    // __metadata__
    {
        // st.metadata.insert("creator", "safetensors-cpp");
    }

    std::string filename = "model-00001-of-00001.safetensors";
    std::string warn;
    std::string err;
    
    // also generate the model.safetensors.index.json file

    nlohmann::json safetensor_index_j;
    safetensor_index_j["metadata"]["total_size"] = st.storage.size();
    // saying all tensor is in on file
    std::map<std::string, std::string> safetensor_config_json;
    for(auto ten_name : st.tensors.keys()){
        safetensor_config_json[ten_name] = filename;
    }
    safetensor_index_j["weight_map"] = safetensor_config_json;


    
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
    
    // also save the json file
    std::ofstream json_file(output_directory + "/model.safetensors.index.json");
    if (!json_file.is_open()) {
        std::cerr << "Failed to open json file for writing: " << output_directory + "model.safetensors.index.json\n";
        return EXIT_FAILURE;
    }
    json_file << safetensor_index_j.dump(4); // pretty print with 4 spaces
    json_file.close();

    std::cout <<"SUCCESS" <<std::endl;
    return EXIT_SUCCESS;

}