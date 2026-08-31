
#ifndef BAI_C_API_H_
#define BAI_C_API_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to BaiEngine instance.
 */
namespace bai { class BaiEngine; }

struct BaiEngineHandle_t {
    bai::BaiEngine* engine;
};

typedef BaiEngineHandle_t* BaiEngineHandle;

/**
 * Status codes for C API operations.
 */
typedef enum {
    BAI_SUCCESS = 0,
    BAI_ERROR_INIT_FAILED = 1,
    BAI_ERROR_INFERENCE_FAILED = 2,
    BAI_ERROR_INVALID_PARAM = 3,
    BAI_ERROR_MODEL_NOT_FOUND = 4,
    BAI_ERROR_MEMORY_ALLOC_FAILED = 5,
    BAI_ERROR_INVALID_HANDLE = 6,
    BAI_ERROR_NOT_INITIALIZED = 7,
} BaiStatusCode;

/**
 * Engine configuration structure for C API.
 */
typedef struct {
    const char* model_path;
    int num_threads;
    int device_id;
    int use_gpu;               // 0 = false, 1 = true
    int enable_fp16;           // 0 = false, 1 = true
    size_t max_seq_length;
    int graph_optimization_level;
} BaiEngineConfig;

/**
 * Inference result structure for C API.
 */
typedef struct {
    float category_logits[5];  // [INBOX_PINNED, INBOX, BAIT, BAIS, BAIADS]
    float otp_logit;
    float confidence;
} BaiInferenceResult;

/**
 * Create a new BaiEngine instance.
 * 
 * @param config Pointer to EngineConfig structure
 * @param out_handle Pointer to receive engine handle (allocated internally)
 * @return BAI_SUCCESS on success, error code otherwise
 */
BaiStatusCode bai_engine_create(
    const BaiEngineConfig* config,
    BaiEngineHandle* out_handle
);

/**
 * Run inference on tokenized input.
 * 
 * @param handle Engine handle from bai_engine_create
 * @param input_ids Array of token IDs (size = seq_len)
 * @param attention_mask Array of attention mask (size = seq_len)
 * @param seq_len Sequence length (must be <= max_seq_length)
 * @param out_result Pointer to receive inference result
 * @return BAI_SUCCESS on success, error code otherwise
 */
BaiStatusCode bai_engine_infer(
    BaiEngineHandle handle,
    const int64_t* input_ids,
    const int64_t* attention_mask,
    size_t seq_len,
    BaiInferenceResult* out_result
);

/**
 * Destroy engine instance and free resources.
 * 
 * @param handle Engine handle to destroy
 * @return BAI_SUCCESS on success, error code otherwise
 */
BaiStatusCode bai_engine_destroy(BaiEngineHandle handle);

/**
 * Get human-readable error message for status code.
 * 
 * @param status Status code
 * @return Error message string (static, do not free)
 */
const char* bai_get_error_message(BaiStatusCode status);

/**
 * Get library version string.
 * 
 * @return Version string (e.g., "1.0.0")
 */
const char* bai_get_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // BAI_C_API_H_
