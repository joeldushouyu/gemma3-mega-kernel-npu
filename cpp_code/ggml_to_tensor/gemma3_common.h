#ifndef GEMMA3_COMMON_H
#define GEMMA3_COMMON_H

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



#include <fstream>
#include <nlohmann/json.hpp>
struct gemma3_model{
    std::string arch;
    ggml_backend_sched_t backend_sched;
    std::vector<ggml_backend_t> backends;    

    struct gguf_context * vision_gguf_metadata_ctx = nullptr; //Only intended to store the metata of model, not the actual tensor
    struct ggml_context * vision_ctx_gguf    = nullptr; //hold tensor from gguf file for ggml library
    ggml_backend_buffer_t vision_buf_gguf    = nullptr;

    struct gguf_context * language_gguf_metadata_ctx = nullptr; //Only intended to store the metata of model, not the actual tensor
    struct ggml_context * language_ctx_gguf    = nullptr; //hold tensor from gguf file for ggml library
    ggml_backend_buffer_t language_buf_gguf    = nullptr;

    std::map <std::string, uint> language_gguf_kv_map; // key-value pairs for the language model
    std::map <std::string, uint > vision_gguf_kv_map; // key-value pairs for the vision model
    std::map <std::string, uint> language_gguf_tensor_info_map; // tensor info for the language model
    std::map <std::string, uint> vision_gguf_tensor_info_map; // tensor info for the vision model

    // language part
    struct ggml_tensor * output_norm_weight = nullptr;
    struct ggml_tensor *token_embd_weight = nullptr;

    std::vector<struct ggml_tensor*> attn_k_weight;
    std::vector<struct ggml_tensor*> attn_k_norm_weight;
    std::vector<struct ggml_tensor*> attn_norm_weight;
    std::vector<struct ggml_tensor*> attn_output_weight;
    std::vector<struct ggml_tensor*> attn_q_weight;
    std::vector<struct ggml_tensor*> attn_q_norm_weight;
    std::vector<struct ggml_tensor*> attn_v_weight;
    std::vector<struct ggml_tensor*>ffn_down_weight;
    std::vector<struct ggml_tensor*>ffn_gate_weight;
    std::vector<struct ggml_tensor*>ffn_norm_weight;
    std::vector<struct ggml_tensor*>ffn_up_weight;
    std::vector<struct ggml_tensor*>post_attention_norm_weight;
    std::vector<struct ggml_tensor*>post_ffw_norm_weight;


    //vision part
    struct ggml_tensor * v_post_ln_biase = nullptr;
    struct ggml_tensor * v_post_ln_weight = nullptr;
    struct ggml_tensor * mm_input_projection_weight = nullptr;
    struct ggml_tensor * mm_soft_emb_norm_weight = nullptr;
    struct ggml_tensor * v_patch_embd_bias = nullptr;
    struct ggml_tensor * v_patch_embd_weight = nullptr;     
    struct ggml_tensor * v_position_embd_weight = nullptr;
    std::vector<struct ggml_tensor*> v_blk_ln1_bias;
    std::vector<struct ggml_tensor*> v_blk_ln1_weight;
    std::vector<struct ggml_tensor*> v_blk_ln2_bias;
    std::vector<struct ggml_tensor*> v_blk_ln2_weight;
    std::vector<struct ggml_tensor*> v_blk_ffn_down_bias;
    std::vector<struct ggml_tensor*> v_blk_ffn_down_weight;
    std::vector<struct ggml_tensor*> v_blk_ffn_up_bias;
    std::vector<struct ggml_tensor*> v_blk_ffn_up_weight;
    std::vector<struct ggml_tensor*> v_blk_attn_k_bias;
    std::vector<struct ggml_tensor*> v_blk_attn_k_weight;
    std::vector<struct ggml_tensor*> v_blk_attn_out_bias;   
    std::vector<struct ggml_tensor*> v_blk_attn_out_weight;
    std::vector<struct ggml_tensor*> v_blk_attn_q_bias;
    std::vector<struct ggml_tensor*> v_blk_attn_q_weight;
    std::vector<struct ggml_tensor*> v_blk_attn_v_bias;
    std::vector<struct ggml_tensor*> v_blk_attn_v_weight;


    gemma3_model( const std::string &backend_name){
        std::vector<ggml_backend_dev_t> devices;
        const int ncores_logical = std::thread::hardware_concurrency(); // an estimate number of physical cores
        const int nthreads = std::min(ncores_logical,   (ncores_logical + 4) / 2);


        // add primary backend
        if(!backend_name.empty()){
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
            if (dev == nullptr) {
                fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev_i = ggml_backend_dev_get(i);
                    fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(dev_i), ggml_backend_dev_description(dev_i));
                }
                exit(1);
            }
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }


        // Add all available backends as fallback.
        // A "backend" is a stream on a physical device so there is no problem with adding multiple backends for the same device.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }


        // The order of the backends passed to ggml_backend_sched_new determines which backend is given priority.
        backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
        fprintf(stderr, "%s: using %s (%s) as primary backend\n",
                __func__, ggml_backend_name(backends[0]), ggml_backend_dev_description(devices[0]));
        if (backends.size() >= 2) {
            fprintf(stderr, "%s: unsupported operations will be executed on the following fallback backends (in order of priority):\n", __func__);
            for (size_t i = 1; i < backends.size(); ++i) {
                fprintf(stderr, "%s:  - %s (%s)\n", __func__, ggml_backend_name(backends[i]), ggml_backend_dev_description(devices[i]));
            }
        }




    }

    ~gemma3_model(){
        ggml_free(language_ctx_gguf);
        ggml_free(vision_ctx_gguf);
        gguf_free(language_gguf_metadata_ctx);
        gguf_free(vision_gguf_metadata_ctx);

        ggml_backend_buffer_free(language_buf_gguf);
        ggml_backend_buffer_free(vision_buf_gguf);
        ggml_backend_sched_free(backend_sched);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }        
  

    }

};

gemma3_model gemma3_mode_init_from_file(const std::string &fname_language_model, const std::string &fname_vision_model);

void load_tensor_to_backend_memory( const std::string fname, const int num_tensors, struct gguf_context * gguf_metadata_ctx,
    struct ggml_context * ggml_tensor_ctx
);
struct ggml_tensor * checked_get_tensor(struct ggml_context * ctx, const char * name);

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m);
void dequantize_row_q4_0_bfloat_dur_process(const block_q4_0 * GGML_RESTRICT x,  std::bfloat16_t * GGML_RESTRICT y, int64_t k) ;
void dequantize_row_q4_K_bfloat_dur_process(const block_q4_K * GGML_RESTRICT x, std::bfloat16_t * GGML_RESTRICT y, int64_t k);
void dequantize_row_q6_K_bfloat_dur_process(const block_q6_K * GGML_RESTRICT x, std::bfloat16_t  * GGML_RESTRICT y, int64_t k);
std::vector<std::bfloat16_t> dequant_whole_q4_0_tensor(ggml_tensor* tensor);
std::vector<std::bfloat16_t> dequant_whole_q4_k_tensor(ggml_tensor* tensor) ;
std::vector<std::bfloat16_t> dequant_whole_q6_k_tensor(ggml_tensor* tensor) ;
#endif // GEMMA3_COMMON_H