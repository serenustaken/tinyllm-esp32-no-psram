#ifndef CLASSIC_ESP32_LLM_ENGINE_H
#define CLASSIC_ESP32_LLM_ENGINE_H

// Minimal Arduino interface for the flash-resident Q8_0 inference engine.
// Derived from llama2.c concepts; see LICENSE and NOTICE.md in the repository.

#include <stddef.h>
#include <stdint.h>

typedef void (*LlmTokenCallback)(const char *piece, void *user_data);

// Model and tokenizer byte arrays must remain valid for the engine lifetime.
bool llm_begin(const uint8_t *model, size_t model_len,
               const uint8_t *tokenizer, size_t tokenizer_len,
               int context_tokens);

// Returns the number of processed positions, or -1 on error.
int llm_generate(const char *prompt, int steps, float temperature, float topp,
                 uint64_t seed, LlmTokenCallback callback, void *user_data);

const char *llm_last_error(void);
size_t llm_estimated_runtime_bytes(void);
int llm_context_tokens(void);
void llm_end(void);

#endif
