#include <Arduino.h>
#include "esp_heap_caps.h"

#include "llm_engine.h"
#include "model_data.h"
#include "tokenizer_data.h"

// TinyLLM for a classic ESP32 without PSRAM.
// Model and tokenizer data remain in flash; only runtime buffers use RAM.

static constexpr int CONTEXT_TOKENS = 64;
static constexpr int GENERATION_STEPS = 64;
static constexpr float TEMPERATURE = 1.0f;
static constexpr float TOP_P = 0.9f;

static void print_piece(const char *piece, void *) {
  Serial.print(piece);
}

static void print_memory(const char *label) {
  Serial.printf("[%s] free heap: %u bytes, largest block: %u bytes\n",
                label,
                (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
                (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
}

static void run_prompt(const String &prompt) {
  Serial.println();
  Serial.println("--- output ---");
  const uint32_t started = millis();
  const int positions = llm_generate(prompt.c_str(), GENERATION_STEPS,
                                     TEMPERATURE, TOP_P,
                                     (uint64_t)esp_random(), print_piece, nullptr);
  const uint32_t elapsed = millis() - started;
  Serial.println();
  Serial.println("--- done ---");
  if (positions < 0) {
    Serial.printf("Generation error: %s\n", llm_last_error());
  } else {
    const float speed = elapsed ? positions * 1000.0f / elapsed : 0.0f;
    Serial.printf("Positions: %d, time: %.2f s, average: %.2f tok/s\n",
                  positions, elapsed / 1000.0f, speed);
  }
  print_memory("after generation");
  Serial.println("\nType a new English story prompt and press Enter:");
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\nTinyLLM on classic ESP32 (no PSRAM)");
  Serial.printf("Model in flash: %u bytes; tokenizer: %u bytes\n",
                model_data_len, tokenizer_data_len);
  print_memory("before init");

  if (!llm_begin(model_data, model_data_len,
                 tokenizer_data, tokenizer_data_len,
                 CONTEXT_TOKENS)) {
    Serial.printf("Initialization error: %s\n", llm_last_error());
    return;
  }

  Serial.printf("Runtime buffers: about %u bytes; context: %d tokens\n",
                (unsigned)llm_estimated_runtime_bytes(), llm_context_tokens());
  print_memory("after init");
  run_prompt("Once upon a time");
}

void loop() {
  if (!Serial.available()) {
    delay(20);
    return;
  }
  String prompt = Serial.readStringUntil('\n');
  prompt.trim();
  if (prompt.length() == 0) return;
  run_prompt(prompt);
}
