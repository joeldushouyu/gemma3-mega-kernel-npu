
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



#include <fstream>
#include <nlohmann/json.hpp>
#include <map>


void load_tensor_to_backend_memory( const std::string fname, const int num_tensors, struct gguf_context * gguf_metadata_ctx,
    struct ggml_context * ggml_tensor_ctx
){

    // now, load model tensor from file
    FILE * f = fopen(fname.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen() failed\n", __func__);
    }

    for (int i = 0; i < num_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_metadata_ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ggml_tensor_ctx, name);
        size_t offs = gguf_get_data_offset(gguf_metadata_ctx) + gguf_get_tensor_offset(gguf_metadata_ctx, i);

        //printf("%-30s: [%3ld, %3ld, %3ld, %3ld] %s\n",
        //    name,
        //    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        //    ggml_type_name(tensor->type));
        std::vector<uint8_t> buf(ggml_nbytes(tensor));
        if (fseek(f, offs, SEEK_SET) != 0) {
            fprintf(stderr, "%s: fseek() failed\n", __func__);
            fclose(f);
 
        }

        if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
            fprintf(stderr, "%s: fread() failed\n", __func__);
            fclose(f);
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());
    }

    fclose(f);


}
struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name) {
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
    assert(tensor != nullptr);

    return tensor;
}


gemma3_model gemma3_mode_init_from_file(const std::string &fname_language_model, const std::string &fname_vision_model){

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
        model.vision_gguf_metadata_ctx = gguf_init_from_file(fname_vision_model.c_str(), vision_params);

        if (!model.language_gguf_metadata_ctx) {
            fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
            exit(1);
        }
        if(!model.vision_gguf_metadata_ctx) {
            fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
            exit(1);
        }   
    }

    // at this point, ctx is initialized with the GGUF file data

    model.arch = gguf_get_val_str(model.language_gguf_metadata_ctx, gguf_find_key(model.language_gguf_metadata_ctx, "general.architecture"));

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.language_buf_gguf = ggml_backend_alloc_ctx_tensors(model.language_ctx_gguf, model.backends[0]);
    model.vision_buf_gguf = ggml_backend_alloc_ctx_tensors(model.vision_ctx_gguf, model.backends[0]);



    const int n_language_tensors = gguf_get_n_tensors(model.language_gguf_metadata_ctx);
    const int n_vision_tensor = gguf_get_n_tensors(model.vision_gguf_metadata_ctx);
    // At this point, the model.ctx_gguf should has all the tensor
    // now retrive the corresponding tensor ptr in model.ctx_gguf
    
    model.output_norm_weight = checked_get_tensor(model.language_ctx_gguf,  "output_norm.weight"); // for example
    model.token_embd_weight =  checked_get_tensor(model.language_ctx_gguf,  "token_embd.weight");
    size_t layer = 0;
    for(size_t cur_cout = 2; cur_cout < n_language_tensors; cur_cout+=13,layer++ ){  // 2 for token_embd.weight and output_norm.weight, 13 for each block

        auto share_str_header = "blk." + std::to_string(layer) +".";
        auto attn_k_weight_name =share_str_header + "attn_k.weight";
        auto attn_k_norm_weight_name = share_str_header + "attn_k_norm.weight";
        auto attn_norm_weight_name = share_str_header + "attn_norm.weight";
        auto attn_output_weight_name = share_str_header + "attn_output.weight";
        auto attn_q_weight_name = share_str_header + "attn_q.weight";
        auto attn_q_norm_weight_name = share_str_header + "attn_q_norm.weight";
        auto attn_v_weight_name = share_str_header + "attn_v.weight";
        auto ffn_down_weight_name = share_str_header +"ffn_down.weight";
        auto ffn_gate_weight_name = share_str_header + "ffn_gate.weight";
        auto ffn_norm_weight_name = share_str_header+ "ffn_norm.weight";
        auto ffn_up_weight_name = share_str_header + "ffn_up.weight";
        auto post_attention_norm_weight_name = share_str_header + "post_attention_norm.weight";
        auto post_ffw_norm_weight_name = share_str_header + "post_ffw_norm.weight"; 

        model.attn_k_weight.push_back(  checked_get_tensor(model.language_ctx_gguf, attn_k_weight_name.data())  );
        model.attn_k_norm_weight.push_back(checked_get_tensor(model.language_ctx_gguf, attn_k_norm_weight_name.data()));
        model.attn_norm_weight.push_back(checked_get_tensor(model.language_ctx_gguf, attn_norm_weight_name.data()));
        model.attn_output_weight.push_back(checked_get_tensor(model.language_ctx_gguf, attn_output_weight_name.data()));
        model.attn_q_weight.push_back(checked_get_tensor(model.language_ctx_gguf, attn_q_weight_name.data()));
        model.attn_q_norm_weight.push_back(checked_get_tensor(model.language_ctx_gguf, attn_q_norm_weight_name.data()));
        model.attn_v_weight.push_back(checked_get_tensor(model.language_ctx_gguf, attn_v_weight_name.data()));      
        model.ffn_down_weight.push_back(checked_get_tensor(model.language_ctx_gguf, ffn_down_weight_name.data()));
        model.ffn_gate_weight.push_back(checked_get_tensor(model.language_ctx_gguf, ffn_gate_weight_name.data()));
        model.ffn_norm_weight.push_back(checked_get_tensor(model.language_ctx_gguf, ffn_norm_weight_name.data()));
        model.ffn_up_weight.push_back(checked_get_tensor(model.language_ctx_gguf, ffn_up_weight_name.data()));
        model.post_attention_norm_weight.push_back(checked_get_tensor(model.language_ctx_gguf, post_attention_norm_weight_name.data()));
        model.post_ffw_norm_weight.push_back(checked_get_tensor(model.language_ctx_gguf, post_ffw_norm_weight_name.data()));
        

    }
  

    model.v_post_ln_biase = checked_get_tensor(model.vision_ctx_gguf, "v.post_ln.bias");
    model.v_post_ln_weight = checked_get_tensor(model.vision_ctx_gguf, "v.post_ln.weight");
        
    model.mm_input_projection_weight = checked_get_tensor(model.vision_ctx_gguf, "mm.input_projection.weight");
    model.mm_soft_emb_norm_weight = checked_get_tensor(model.vision_ctx_gguf, "mm.soft_emb_norm.weight");
    model.v_patch_embd_bias = checked_get_tensor(model.vision_ctx_gguf, "v.patch_embd.bias");
    model.v_patch_embd_weight = checked_get_tensor(model.vision_ctx_gguf, "v.patch_embd.weight");
    model.v_position_embd_weight = checked_get_tensor(model.vision_ctx_gguf, "v.position_embd.weight");
    layer = 0;
    for(size_t cur_cout = 7; cur_cout < n_vision_tensor; cur_cout+=16, layer++){ // 7 for v.post_ln.bias and v.post_ln.weight, 12 for each block
    auto share_str_header = "v.blk." + std::to_string(layer) +".";
        auto ln1_bias_name = share_str_header + "ln1.bias";
        auto ln1_weight_name = share_str_header + "ln1.weight";
        auto ln2_bias_name = share_str_header + "ln2.bias";
        auto ln2_weight_name = share_str_header + "ln2.weight";
        auto ffn_down_bias_name = share_str_header + "ffn_down.bias";
        auto ffn_down_weight_name = share_str_header + "ffn_down.weight";
        auto ffn_up_bias_name = share_str_header + "ffn_up.bias";
        auto ffn_up_weight_name = share_str_header + "ffn_up.weight";
        auto attn_k_bias_name = share_str_header + "attn_k.bias";
        auto attn_k_weight_name = share_str_header + "attn_k.weight";
        auto attn_out_bias_name = share_str_header + "attn_out.bias";
        auto attn_out_weight_name = share_str_header + "attn_out.weight";
        auto attn_q_bias_name = share_str_header + "attn_q.bias";
        auto attn_q_weight_name = share_str_header + "attn_q.weight";
        auto attn_v_bias_name = share_str_header + "attn_v.bias";
        auto attn_v_weight_name = share_str_header + "attn_v.weight";
        model.v_blk_ln1_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, ln1_bias_name.data()));
        model.v_blk_ln1_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, ln1_weight_name.data()));
        model.v_blk_ln2_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, ln2_bias_name.data()));
        model.v_blk_ln2_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, ln2_weight_name.data()));
        model.v_blk_ffn_down_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, ffn_down_bias_name.data()));
        model.v_blk_ffn_down_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, ffn_down_weight_name.data()));
        model.v_blk_ffn_up_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, ffn_up_bias_name.data()));
        model.v_blk_ffn_up_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, ffn_up_weight_name.data()));
        model.v_blk_attn_k_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_k_bias_name.data()));
        model.v_blk_attn_k_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_k_weight_name.data()));
        model.v_blk_attn_out_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_out_bias_name.data()));
        model.v_blk_attn_out_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_out_weight_name.data()));
        model.v_blk_attn_q_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_q_bias_name.data()));
        model.v_blk_attn_q_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_q_weight_name.data()));
        model.v_blk_attn_v_bias.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_v_bias_name.data()));
        model.v_blk_attn_v_weight.push_back(checked_get_tensor(model.vision_ctx_gguf, attn_v_weight_name.data()));      

    }


    load_tensor_to_backend_memory(fname_language_model, n_language_tensors, model.language_gguf_metadata_ctx, model.language_ctx_gguf);
    load_tensor_to_backend_memory(fname_vision_model,n_vision_tensor, model.vision_gguf_metadata_ctx, model.vision_ctx_gguf );
    /*
    // now, load model tensor from file
    FILE * f = fopen(fname_language_model.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "%s: fopen() failed\n", __func__);
    }

    for (int i = 0; i < n_language_tensors; i++) {
        const char * name = gguf_get_tensor_name(model.language_gguf_metadata_ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model.language_ctx_gguf, name);
        size_t offs = gguf_get_data_offset(model.language_gguf_metadata_ctx) + gguf_get_tensor_offset(model.language_gguf_metadata_ctx, i);

        //printf("%-30s: [%3ld, %3ld, %3ld, %3ld] %s\n",
        //    name,
        //    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
        //    ggml_type_name(tensor->type));
        std::vector<uint8_t> buf(ggml_nbytes(tensor));
        if (fseek(f, offs, SEEK_SET) != 0) {
            fprintf(stderr, "%s: fseek() failed\n", __func__);
            fclose(f);
 
        }

        if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
            fprintf(stderr, "%s: fread() failed\n", __func__);
            fclose(f);
        }

        ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());
    }

    fclose(f);
    */



    uint language_kv_size = gguf_get_n_kv(model.language_gguf_metadata_ctx);
    uint vision_kv_size = gguf_get_n_kv(model.vision_gguf_metadata_ctx);
    uint language_tensor_info_size = gguf_get_n_tensors(model.language_gguf_metadata_ctx);
    uint vision_tensor_info_size = gguf_get_n_tensors(model.vision_gguf_metadata_ctx);

    for(uint i = 0; i < language_kv_size; i++){
       std::string name = gguf_get_key(model.language_gguf_metadata_ctx, i);
        model.language_gguf_kv_map[name] =i;
    }
    for(uint i = 0; i < vision_kv_size; i++){
        std::string name = gguf_get_key(model.vision_gguf_metadata_ctx, i);
        model.vision_gguf_kv_map[name] =i;
    }
    for(uint i = 0; i < language_tensor_info_size; i++){
        std::string name = gguf_get_tensor_name(model.language_gguf_metadata_ctx,   i); 
        model.language_gguf_tensor_info_map[name] = i;  
    }
    for(uint i = 0; i < vision_tensor_info_size; i++){
        std::string name = gguf_get_tensor_name(model.vision_gguf_metadata_ctx, i);
        model.vision_gguf_tensor_info_map[name] = i;
    }



    return model;


}




static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

void dequantize_row_q4_0_bfloat_dur_process(const block_q4_0 * GGML_RESTRICT x,  std::bfloat16_t * GGML_RESTRICT y, int64_t k) {
    // This assume preprocess of turning d to bfloat16_t for npu implementation
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const std::bfloat16_t d = static_cast<std::bfloat16_t>(GGML_FP16_TO_FP32(x[i].d));

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = static_cast<std::bfloat16_t>(x0*d);
            y[i*qk + j + qk/2] = static_cast<std::bfloat16_t>(x1*d);
        }
    }
}

void dequantize_row_q4_K_bfloat_dur_process(const block_q4_K * GGML_RESTRICT x, std::bfloat16_t * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;
        //NOTE: assume preprocess of fp16 to bfloat16
        const std::bfloat16_t d   =  static_cast<std::bfloat16_t>(GGML_FP16_TO_FP32(x[i].d));
        const std::bfloat16_t min =  static_cast<std::bfloat16_t>(GGML_FP16_TO_FP32(x[i].dmin));

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const std::bfloat16_t d1 = static_cast<std::bfloat16_t>( d * sc); 
            const std::bfloat16_t m1 = static_cast<std::bfloat16_t>(min * m);
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const std::bfloat16_t d2 = static_cast<std::bfloat16_t>( d * sc); 
            const std::bfloat16_t m2 = static_cast<std::bfloat16_t>(min * m);
            for (int l = 0; l < 32; ++l) *y++ = static_cast<std::bfloat16_t>(d1 * (q[l] & 0xF) - m1);
            for (int l = 0; l < 32; ++l) *y++ = static_cast<std::bfloat16_t>(d2 * (q[l]  >> 4) - m2);
            q += 32; is += 2;
        }
    }
}

void dequantize_row_q6_K_bfloat_dur_process(const block_q6_K * GGML_RESTRICT x, std::bfloat16_t  * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const std::bfloat16_t  d = static_cast<std::bfloat16_t>( GGML_FP16_TO_FP32(x[i].d));

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = static_cast<std::bfloat16_t>( d * sc[is + 0] * q1);
                y[l + 32] = static_cast<std::bfloat16_t>( d * sc[is + 2] * q2);
                y[l + 64] = static_cast<std::bfloat16_t>( d * sc[is + 4] * q3);
                y[l + 96] = static_cast<std::bfloat16_t>( d * sc[is + 6] * q4);
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

std::vector<std::bfloat16_t> dequant_whole_q4_0_tensor(ggml_tensor* tensor) {
    
    // This function assumes the tensor is of type GGML_TYPE_Q4_0
    assert(tensor->type == GGML_TYPE_Q4_0);
    // assert dimension <= 2
    char valid_dim = 0;
    for(int i = 0; i < 4; i++){
        if(tensor->ne[i] > 1){
            valid_dim++;
        }
    }
    assert(valid_dim <=2);
    // note ggml is num_col(number of element in a row) x num_row 
    int64_t nrows = tensor->ne[1];
    int64_t ncols = tensor->ne[0];


    std::vector<std::bfloat16_t> res_vector(ncols*nrows);
    
    for(int64_t row = 0; row<nrows; row++){
        block_q4_0 *qrow = reinterpret_cast<block_q4_0*>(
             (uint8_t *)tensor->data + row*tensor->nb[1]
        );
        std::bfloat16_t *data_out = res_vector.data() + row*ncols;
        dequantize_row_q4_0_bfloat_dur_process(  qrow, data_out, ncols );
    }
    
    return res_vector;
}


std::vector<std::bfloat16_t> dequant_whole_q4_k_tensor(ggml_tensor* tensor) {
    
    // This function assumes the tensor is of type GGML_TYPE_Q4_0
    assert(tensor->type == GGML_TYPE_Q4_K);
    // assert dimension <= 2
    char valid_dim = 0;
    for(int i = 0; i < 4; i++){
        if(tensor->ne[i] > 1){
            valid_dim++;
        }
    }
    assert(valid_dim <=2);
    // note ggml is num_col(number of element in a row) x num_row 
    int64_t nrows = tensor->ne[1];
    int64_t ncols = tensor->ne[0];


    std::vector<std::bfloat16_t> res_vector(ncols*nrows);
    
    for(int64_t row = 0; row<nrows; row++){
        block_q4_K *qrow = reinterpret_cast<block_q4_K*>(
             (uint8_t *)tensor->data + row*tensor->nb[1]
        );
        std::bfloat16_t *data_out = res_vector.data() + row*ncols;
        dequantize_row_q4_K_bfloat_dur_process(  qrow, data_out, ncols );
    }
    
    return res_vector;
}



std::vector<std::bfloat16_t> dequant_whole_q6_k_tensor(ggml_tensor* tensor) {
    
    // This function assumes the tensor is of type GGML_TYPE_Q4_0
    assert(tensor->type == GGML_TYPE_Q6_K);
    // assert dimension <= 2
    char valid_dim = 0;
    for(int i = 0; i < 4; i++){
        if(tensor->ne[i] > 1){
            valid_dim++;
        }
    }
    assert(valid_dim <=2);
    // note ggml is num_col(number of element in a row) x num_row 
    int64_t nrows = tensor->ne[1];
    int64_t ncols = tensor->ne[0];


    std::vector<std::bfloat16_t> res_vector(ncols*nrows);
    
    for(int64_t row = 0; row<nrows; row++){
        block_q6_K *qrow = reinterpret_cast<block_q6_K*>(
             (uint8_t *)tensor->data + row*tensor->nb[1]
        );
        std::bfloat16_t *data_out = res_vector.data() + row*ncols;
        dequantize_row_q6_K_bfloat_dur_process(  qrow, data_out, ncols );
    }
    
    return res_vector;
}
