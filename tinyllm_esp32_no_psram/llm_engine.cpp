/*
 * TinyLLM classic-ESP32 inference engine.
 *
 * Derived from the model format and minimal inference approach of llama2.c:
 * https://github.com/karpathy/llama2.c
 * Copyright (c) 2023 Andrej — MIT License.
 *
 * Adapted for an Arduino-based classic ESP32 target with flash-resident Q8_0
 * weights, an internal-RAM runtime state, and a configurable reduced context.
 * See the repository LICENSE and NOTICE.md files for complete attribution.
 */

#include "llm_engine.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(ARDUINO_ARCH_ESP32)
#include <Arduino.h>
#include "esp_heap_caps.h"
#endif

namespace {

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct QuantizedTensor {
    const int8_t *q;
    const float *s;
};

struct TransformerWeights {
    QuantizedTensor *q_tokens;
    const float *rms_att_weight;
    const float *rms_ffn_weight;
    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;
    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;
    const float *rms_final_weight;
    QuantizedTensor *wcls;
};

struct MutableQuantizedTensor {
    int8_t *q;
    float *s;
};

struct RunState {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    MutableQuantizedTensor xq;
    MutableQuantizedTensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
};

struct Transformer {
    Config config;
    TransformerWeights weights;
    RunState state;
    int group_size;
    bool shared_classifier;
};

struct TokenIndex {
    char *str;
    int id;
};

struct Tokenizer {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    char *storage;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
};

struct ProbIndex {
    float prob;
    int index;
};

struct Sampler {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    uint64_t rng_state;
};

Transformer g_transformer{};
Tokenizer g_tokenizer{};
bool g_ready = false;
char g_error[192] = "not initialized";
size_t g_runtime_bytes = 0;

void set_error(const char *message) {
    snprintf(g_error, sizeof(g_error), "%s", message);
}

void *llm_calloc(size_t count, size_t size) {
#if defined(ARDUINO_ARCH_ESP32)
    return heap_caps_calloc(count, size, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
    return calloc(count, size);
#endif
}

void llm_yield() {
#if defined(ARDUINO_ARCH_ESP32)
    yield();
#endif
}

uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

int32_t read_i32(const uint8_t *p) {
    return (int32_t)read_u32(p);
}

float read_f32(const uint8_t *p) {
    const uint32_t bits = read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

bool take_i32(const uint8_t *&p, const uint8_t *end, int32_t &value) {
    if ((size_t)(end - p) < 4) return false;
    value = read_i32(p);
    p += 4;
    return true;
}

bool take_f32(const uint8_t *&p, const uint8_t *end, float &value) {
    if ((size_t)(end - p) < 4) return false;
    value = read_f32(p);
    p += 4;
    return true;
}

void free_qtensor_array(QuantizedTensor *p) {
    free(p);
}

void free_state(RunState &s) {
    free(s.x); free(s.xb); free(s.xb2); free(s.hb); free(s.hb2);
    free(s.xq.q); free(s.xq.s); free(s.hq.q); free(s.hq.s);
    free(s.q); free(s.k); free(s.v); free(s.att); free(s.logits);
    free(s.key_cache); free(s.value_cache);
    memset(&s, 0, sizeof(s));
}

void free_weights(TransformerWeights &w) {
    free_qtensor_array(w.q_tokens);
    free_qtensor_array(w.wq); free_qtensor_array(w.wk);
    free_qtensor_array(w.wv); free_qtensor_array(w.wo);
    free_qtensor_array(w.w1); free_qtensor_array(w.w2);
    free_qtensor_array(w.w3);
    if (w.wcls != w.q_tokens) free_qtensor_array(w.wcls);
    memset(&w, 0, sizeof(w));
}

void free_tokenizer(Tokenizer &t) {
    free(t.vocab);
    free(t.vocab_scores);
    free(t.sorted_vocab);
    free(t.storage);
    memset(&t, 0, sizeof(t));
}

QuantizedTensor *map_qtensors(const uint8_t *&p, const uint8_t *end,
                              int count, size_t elements_each, int group_size) {
    QuantizedTensor *result = (QuantizedTensor *)llm_calloc(count, sizeof(QuantizedTensor));
    if (!result) return nullptr;
    for (int i = 0; i < count; ++i) {
        const size_t q_bytes = elements_each;
        const size_t scale_bytes = (elements_each / group_size) * sizeof(float);
        if ((size_t)(end - p) < q_bytes + scale_bytes) {
            free(result);
            return nullptr;
        }
        result[i].q = (const int8_t *)p;
        p += q_bytes;
        result[i].s = (const float *)p;
        p += scale_bytes;
    }
    return result;
}

bool allocate_state(Transformer &t) {
    Config &p = t.config;
    RunState &s = t.state;
    const int gs = t.group_size;
    const int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;

    s.x = (float *)llm_calloc(p.dim, sizeof(float));
    s.xb = (float *)llm_calloc(p.dim, sizeof(float));
    s.xb2 = (float *)llm_calloc(p.dim, sizeof(float));
    s.hb = (float *)llm_calloc(p.hidden_dim, sizeof(float));
    s.hb2 = (float *)llm_calloc(p.hidden_dim, sizeof(float));
    s.xq.q = (int8_t *)llm_calloc(p.dim, sizeof(int8_t));
    s.xq.s = (float *)llm_calloc(p.dim / gs, sizeof(float));
    s.hq.q = (int8_t *)llm_calloc(p.hidden_dim, sizeof(int8_t));
    s.hq.s = (float *)llm_calloc(p.hidden_dim / gs, sizeof(float));
    s.q = (float *)llm_calloc(p.dim, sizeof(float));
    s.k = (float *)llm_calloc(kv_dim, sizeof(float));
    s.v = (float *)llm_calloc(kv_dim, sizeof(float));
    s.att = (float *)llm_calloc(p.n_heads * p.seq_len, sizeof(float));
    s.logits = (float *)llm_calloc(p.vocab_size, sizeof(float));
    s.key_cache = (float *)llm_calloc((size_t)p.n_layers * p.seq_len * kv_dim, sizeof(float));
    s.value_cache = (float *)llm_calloc((size_t)p.n_layers * p.seq_len * kv_dim, sizeof(float));

    if (!s.x || !s.xb || !s.xb2 || !s.hb || !s.hb2 ||
        !s.xq.q || !s.xq.s || !s.hq.q || !s.hq.s || !s.q || !s.k ||
        !s.v || !s.att || !s.logits || !s.key_cache || !s.value_cache) {
        set_error("not enough internal RAM for runtime buffers");
        free_state(s);
        return false;
    }

    g_runtime_bytes =
        (size_t)(p.dim * 4 + p.hidden_dim * 2 + p.dim + kv_dim * 2 +
                 p.n_heads * p.seq_len + p.vocab_size +
                 2 * p.n_layers * p.seq_len * kv_dim) * sizeof(float) +
        (size_t)p.dim + (size_t)(p.dim / gs) * sizeof(float) +
        (size_t)p.hidden_dim + (size_t)(p.hidden_dim / gs) * sizeof(float);
    return true;
}

bool map_model(Transformer &t, const uint8_t *data, size_t len, int context_tokens) {
    if (!data || len < 256) {
        set_error("model is missing or shorter than its v2 header");
        return false;
    }
    if (read_u32(data) != 0x616b3432u) {
        set_error("bad model magic; expected llama2.c ak42 v2");
        return false;
    }
    if (read_i32(data + 4) != 2) {
        set_error("bad model version; expected v2 Q8_0");
        return false;
    }

    t.config.dim = read_i32(data + 8);
    t.config.hidden_dim = read_i32(data + 12);
    t.config.n_layers = read_i32(data + 16);
    t.config.n_heads = read_i32(data + 20);
    t.config.n_kv_heads = read_i32(data + 24);
    t.config.vocab_size = read_i32(data + 28);
    const int model_seq_len = read_i32(data + 32);
    t.shared_classifier = data[36] != 0;
    t.group_size = read_i32(data + 37);

    if (t.config.dim <= 0 || t.config.hidden_dim <= 0 || t.config.n_layers <= 0 ||
        t.config.n_heads <= 0 || t.config.n_kv_heads <= 0 ||
        t.config.vocab_size <= 0 || model_seq_len <= 0 || t.group_size <= 0) {
        set_error("model header contains invalid dimensions");
        return false;
    }
    if (t.config.dim % t.group_size != 0 ||
        t.config.hidden_dim % t.group_size != 0) {
        set_error("group size must divide both dim and hidden_dim");
        return false;
    }
    if (context_tokens < 8) context_tokens = 8;
    if (context_tokens > model_seq_len) context_tokens = model_seq_len;
    t.config.seq_len = context_tokens;

    const uint8_t *p = data + 256;
    const uint8_t *end = data + len;
    const size_t norm_count = (size_t)t.config.n_layers * t.config.dim;
    const size_t norm_bytes = norm_count * sizeof(float);
    if ((size_t)(end - p) < norm_bytes * 2 + (size_t)t.config.dim * sizeof(float)) {
        set_error("model ended inside fp32 normalization weights");
        return false;
    }
    t.weights.rms_att_weight = (const float *)p; p += norm_bytes;
    t.weights.rms_ffn_weight = (const float *)p; p += norm_bytes;
    t.weights.rms_final_weight = (const float *)p; p += (size_t)t.config.dim * sizeof(float);

    const int head_size = t.config.dim / t.config.n_heads;
    const int kv_dim = t.config.n_kv_heads * head_size;
    t.weights.q_tokens = map_qtensors(p, end, 1,
        (size_t)t.config.vocab_size * t.config.dim, t.group_size);
    t.weights.wq = map_qtensors(p, end, t.config.n_layers,
        (size_t)t.config.dim * t.config.dim, t.group_size);
    t.weights.wk = map_qtensors(p, end, t.config.n_layers,
        (size_t)t.config.dim * kv_dim, t.group_size);
    t.weights.wv = map_qtensors(p, end, t.config.n_layers,
        (size_t)t.config.dim * kv_dim, t.group_size);
    t.weights.wo = map_qtensors(p, end, t.config.n_layers,
        (size_t)t.config.dim * t.config.dim, t.group_size);
    t.weights.w1 = map_qtensors(p, end, t.config.n_layers,
        (size_t)t.config.dim * t.config.hidden_dim, t.group_size);
    t.weights.w2 = map_qtensors(p, end, t.config.n_layers,
        (size_t)t.config.hidden_dim * t.config.dim, t.group_size);
    t.weights.w3 = map_qtensors(p, end, t.config.n_layers,
        (size_t)t.config.dim * t.config.hidden_dim, t.group_size);
    if (t.shared_classifier) {
        t.weights.wcls = t.weights.q_tokens;
    } else {
        t.weights.wcls = map_qtensors(p, end, 1,
            (size_t)t.config.dim * t.config.vocab_size, t.group_size);
    }

    if (!t.weights.q_tokens || !t.weights.wq || !t.weights.wk ||
        !t.weights.wv || !t.weights.wo || !t.weights.w1 || !t.weights.w2 ||
        !t.weights.w3 || !t.weights.wcls) {
        set_error("model ended while mapping quantized weights");
        return false;
    }
    if (p != end) {
        set_error("model byte count does not match its header/weights");
        return false;
    }
    return allocate_state(t);
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

bool build_tokenizer(Tokenizer &t, const uint8_t *data, size_t len, int vocab_size) {
    if (!data || len < 4) {
        set_error("tokenizer is missing or truncated");
        return false;
    }
    const uint8_t *end = data + len;
    const uint8_t *p = data;
    int32_t max_len = 0;
    if (!take_i32(p, end, max_len) || max_len <= 0 || max_len > 1024) {
        set_error("invalid tokenizer header");
        return false;
    }

    const uint8_t *scan = p;
    size_t storage_bytes = 0;
    for (int i = 0; i < vocab_size; ++i) {
        float score;
        int32_t token_len;
        if (!take_f32(scan, end, score) || !take_i32(scan, end, token_len) ||
            token_len < 0 || (size_t)(end - scan) < (size_t)token_len) {
            set_error("tokenizer ended inside vocabulary");
            return false;
        }
        storage_bytes += (size_t)token_len + 1;
        scan += token_len;
    }
    if (scan != end) {
        set_error("tokenizer has unexpected trailing bytes");
        return false;
    }

    t.vocab = (char **)llm_calloc(vocab_size, sizeof(char *));
    t.vocab_scores = (float *)llm_calloc(vocab_size, sizeof(float));
    t.storage = (char *)llm_calloc(storage_bytes, 1);
    if (!t.vocab || !t.vocab_scores || !t.storage) {
        set_error("not enough RAM for tokenizer");
        return false;
    }
    t.vocab_size = vocab_size;
    t.max_token_length = (unsigned int)max_len;
    for (int i = 0; i < 256; ++i) {
        t.byte_pieces[i * 2] = (unsigned char)i;
        t.byte_pieces[i * 2 + 1] = '\0';
    }

    char *storage_at = t.storage;
    for (int i = 0; i < vocab_size; ++i) {
        int32_t token_len;
        take_f32(p, end, t.vocab_scores[i]);
        take_i32(p, end, token_len);
        t.vocab[i] = storage_at;
        memcpy(storage_at, p, token_len);
        storage_at[token_len] = '\0';
        storage_at += token_len + 1;
        p += token_len;
    }
    return true;
}

void dequantize_slice(const QuantizedTensor &qx, float *x, int start, int n, int gs) {
    for (int i = 0; i < n; ++i) {
        const int index = start + i;
        x[i] = (float)qx.q[index] * qx.s[index / gs];
    }
}

void quantize(MutableQuantizedTensor &qx, const float *x, int n, int gs) {
    const int groups = n / gs;
    for (int group = 0; group < groups; ++group) {
        float wmax = 0.0f;
        for (int i = 0; i < gs; ++i) {
            const float value = fabsf(x[group * gs + i]);
            if (value > wmax) wmax = value;
        }
        float scale = wmax / 127.0f;
        if (scale < 1e-12f) scale = 1e-12f;
        qx.s[group] = scale;
        for (int i = 0; i < gs; ++i) {
            int q = (int)roundf(x[group * gs + i] / scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qx.q[group * gs + i] = (int8_t)q;
        }
    }
}

void matmul(float *out, const MutableQuantizedTensor &x,
            const QuantizedTensor &w, int n, int d, int gs) {
    for (int row = 0; row < d; ++row) {
        float value = 0.0f;
        const int base = row * n;
        for (int j = 0; j < n; j += gs) {
            int32_t integer_sum = 0;
            for (int k = 0; k < gs; ++k) {
                integer_sum += (int32_t)x.q[j + k] * (int32_t)w.q[base + j + k];
            }
            value += (float)integer_sum * w.s[(base + j) / gs] * x.s[j / gs];
        }
        out[row] = value;
        if ((row & 31) == 31) llm_yield();
    }
}

void rmsnorm(float *out, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; ++i) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / size + 1e-5f);
    for (int i = 0; i < size; ++i) out[i] = weight[i] * (ss * x[i]);
}

void softmax(float *x, int size) {
    float max_value = x[0];
    for (int i = 1; i < size; ++i) if (x[i] > max_value) max_value = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = expf(x[i] - max_value);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) x[i] /= sum;
}

float *forward(Transformer &t, int token, int pos) {
    Config &p = t.config;
    TransformerWeights &w = t.weights;
    RunState &s = t.state;
    const int dim = p.dim;
    const int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    const int kv_mul = p.n_heads / p.n_kv_heads;
    const int hidden_dim = p.hidden_dim;
    const int head_size = dim / p.n_heads;
    const int gs = t.group_size;

    dequantize_slice(*w.q_tokens, s.x, token * dim, dim, gs);

    for (int layer = 0; layer < p.n_layers; ++layer) {
        rmsnorm(s.xb, s.x, w.rms_att_weight + layer * dim, dim);
        quantize(s.xq, s.xb, dim, gs);
        matmul(s.q, s.xq, w.wq[layer], dim, dim, gs);
        matmul(s.k, s.xq, w.wk[layer], dim, kv_dim, gs);
        matmul(s.v, s.xq, w.wv[layer], dim, kv_dim, gs);

        for (int i = 0; i < dim; i += 2) {
            const int head_dim = i % head_size;
            const float frequency = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            const float angle = pos * frequency;
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            const int rotations = i < kv_dim ? 2 : 1;
            for (int r = 0; r < rotations; ++r) {
                float *vector = r == 0 ? s.q : s.k;
                const float v0 = vector[i];
                const float v1 = vector[i + 1];
                vector[i] = v0 * cosine - v1 * sine;
                vector[i + 1] = v0 * sine + v1 * cosine;
            }
        }

        const int layer_offset = layer * p.seq_len * kv_dim;
        float *key_row = s.key_cache + layer_offset + pos * kv_dim;
        float *value_row = s.value_cache + layer_offset + pos * kv_dim;
        memcpy(key_row, s.k, kv_dim * sizeof(float));
        memcpy(value_row, s.v, kv_dim * sizeof(float));

        for (int head = 0; head < p.n_heads; ++head) {
            float *query = s.q + head * head_size;
            float *attention = s.att + head * p.seq_len;
            for (int time = 0; time <= pos; ++time) {
                float *key = s.key_cache + layer_offset + time * kv_dim +
                             (head / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; ++i) score += query[i] * key[i];
                attention[time] = score / sqrtf((float)head_size);
            }
            softmax(attention, pos + 1);
            float *xb = s.xb + head * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int time = 0; time <= pos; ++time) {
                float *value = s.value_cache + layer_offset + time * kv_dim +
                               (head / kv_mul) * head_size;
                const float a = attention[time];
                for (int i = 0; i < head_size; ++i) xb[i] += a * value[i];
            }
        }

        quantize(s.xq, s.xb, dim, gs);
        matmul(s.xb2, s.xq, w.wo[layer], dim, dim, gs);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb2[i];

        rmsnorm(s.xb, s.x, w.rms_ffn_weight + layer * dim, dim);
        quantize(s.xq, s.xb, dim, gs);
        matmul(s.hb, s.xq, w.w1[layer], dim, hidden_dim, gs);
        matmul(s.hb2, s.xq, w.w3[layer], dim, hidden_dim, gs);
        for (int i = 0; i < hidden_dim; ++i) {
            float value = s.hb[i];
            value *= 1.0f / (1.0f + expf(-value));
            s.hb[i] = value * s.hb2[i];
        }
        quantize(s.hq, s.hb, hidden_dim, gs);
        matmul(s.xb, s.hq, w.w2[layer], hidden_dim, dim, gs);
        for (int i = 0; i < dim; ++i) s.x[i] += s.xb[i];
        llm_yield();
    }

    rmsnorm(s.x, s.x, w.rms_final_weight, dim);
    quantize(s.xq, s.x, dim, gs);
    matmul(s.logits, s.xq, *w.wcls, dim, p.vocab_size, gs);
    return s.logits;
}

int lookup_token(char *text, TokenIndex *sorted, int vocab_size) {
    TokenIndex key{text, 0};
    TokenIndex *found = (TokenIndex *)bsearch(&key, sorted, vocab_size,
                                               sizeof(TokenIndex), compare_tokens);
    return found ? found->id : -1;
}

bool encode(Tokenizer &t, const char *text, bool bos, int *tokens,
            int capacity, int &count) {
    if (!text || capacity < 1) return false;
    if (!t.sorted_vocab) {
        t.sorted_vocab = (TokenIndex *)llm_calloc(t.vocab_size, sizeof(TokenIndex));
        if (!t.sorted_vocab) return false;
        for (int i = 0; i < t.vocab_size; ++i) {
            t.sorted_vocab[i].str = t.vocab[i];
            t.sorted_vocab[i].id = i;
        }
        qsort(t.sorted_vocab, t.vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    const size_t buffer_size = t.max_token_length * 2 + 3;
    char *buffer = (char *)llm_calloc(buffer_size, 1);
    if (!buffer) return false;
    count = 0;
    if (bos) tokens[count++] = 1;
    if (text[0] != '\0') {
        const int prefix = lookup_token((char *)" ", t.sorted_vocab, t.vocab_size);
        if (prefix >= 0 && count < capacity) tokens[count++] = prefix;
    }

    size_t char_len = 0;
    for (const char *c = text; *c != '\0'; ++c) {
        if (((unsigned char)*c & 0xC0) != 0x80) char_len = 0;
        if (char_len + 1 >= buffer_size) { free(buffer); return false; }
        buffer[char_len++] = *c;
        buffer[char_len] = '\0';
        if ((((unsigned char)*(c + 1)) & 0xC0) == 0x80 && char_len < 4) continue;
        int id = lookup_token(buffer, t.sorted_vocab, t.vocab_size);
        if (id >= 0) {
            if (count >= capacity) { free(buffer); return false; }
            tokens[count++] = id;
        } else {
            for (size_t i = 0; i < char_len; ++i) {
                if (count >= capacity) { free(buffer); return false; }
                tokens[count++] = (unsigned char)buffer[i] + 3;
            }
        }
    }

    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_index = -1;
        for (int i = 0; i < count - 1; ++i) {
            snprintf(buffer, buffer_size, "%s%s", t.vocab[tokens[i]], t.vocab[tokens[i + 1]]);
            const int id = lookup_token(buffer, t.sorted_vocab, t.vocab_size);
            if (id >= 0 && t.vocab_scores[id] > best_score) {
                best_score = t.vocab_scores[id];
                best_id = id;
                best_index = i;
            }
        }
        if (best_index < 0) break;
        tokens[best_index] = best_id;
        for (int i = best_index + 1; i < count - 1; ++i) tokens[i] = tokens[i + 1];
        --count;
    }
    free(buffer);
    return true;
}

const char *decode(Tokenizer &t, int previous, int token) {
    if (token < 0 || token >= t.vocab_size) return "";
    char *piece = t.vocab[token];
    if (previous == 1 && piece[0] == ' ') ++piece;
    unsigned char byte_value = 0;
    if (sscanf(piece, "<0x%02hhX>", &byte_value) == 1) {
        piece = (char *)t.byte_pieces + byte_value * 2;
    }
    return piece;
}

bool safe_piece(const char *piece) {
    if (!piece || piece[0] == '\0') return false;
    if (piece[1] == '\0') {
        const unsigned char value = (unsigned char)piece[0];
        return isprint(value) || isspace(value);
    }
    return true;
}

int compare_prob(const void *a, const void *b) {
    const ProbIndex *pa = (const ProbIndex *)a;
    const ProbIndex *pb = (const ProbIndex *)b;
    return pa->prob > pb->prob ? -1 : pa->prob < pb->prob ? 1 : 0;
}

uint32_t random_u32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (uint32_t)((state * 0x2545F4914F6CDD1DULL) >> 32);
}

float random_f32(uint64_t &state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler &sampler, float *logits) {
    if (sampler.temperature == 0.0f) {
        int best = 0;
        for (int i = 1; i < sampler.vocab_size; ++i)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < sampler.vocab_size; ++i) logits[i] /= sampler.temperature;
    softmax(logits, sampler.vocab_size);
    const float coin = random_f32(sampler.rng_state);
    if (sampler.topp <= 0.0f || sampler.topp >= 1.0f) {
        float cdf = 0.0f;
        for (int i = 0; i < sampler.vocab_size; ++i) {
            cdf += logits[i];
            if (coin < cdf) return i;
        }
        return sampler.vocab_size - 1;
    }

    const float cutoff = (1.0f - sampler.topp) / (sampler.vocab_size - 1);
    int count = 0;
    for (int i = 0; i < sampler.vocab_size; ++i) {
        if (logits[i] >= cutoff) {
            sampler.probindex[count++] = ProbIndex{logits[i], i};
        }
    }
    qsort(sampler.probindex, count, sizeof(ProbIndex), compare_prob);
    float cumulative = 0.0f;
    int last = count - 1;
    for (int i = 0; i < count; ++i) {
        cumulative += sampler.probindex[i].prob;
        if (cumulative > sampler.topp) { last = i; break; }
    }
    const float target = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= last; ++i) {
        cdf += sampler.probindex[i].prob;
        if (target < cdf) return sampler.probindex[i].index;
    }
    return sampler.probindex[last].index;
}

} // namespace

bool llm_begin(const uint8_t *model, size_t model_len,
               const uint8_t *tokenizer, size_t tokenizer_len,
               int context_tokens) {
    llm_end();
    if (!map_model(g_transformer, model, model_len, context_tokens)) {
        llm_end();
        return false;
    }
    if (!build_tokenizer(g_tokenizer, tokenizer, tokenizer_len,
                         g_transformer.config.vocab_size)) {
        llm_end();
        return false;
    }
    g_ready = true;
    set_error("ok");
    return true;
}

int llm_generate(const char *prompt, int steps, float temperature, float topp,
                 uint64_t seed, LlmTokenCallback callback, void *user_data) {
    if (!g_ready) { set_error("engine is not initialized"); return -1; }
    if (!prompt) prompt = "";
    if (steps < 1) steps = 1;
    if (steps > g_transformer.config.seq_len) steps = g_transformer.config.seq_len;

    const int prompt_capacity = (int)strlen(prompt) + 3;
    int *prompt_tokens = (int *)llm_calloc(prompt_capacity, sizeof(int));
    if (!prompt_tokens) { set_error("not enough RAM for prompt tokens"); return -1; }
    int prompt_count = 0;
    if (!encode(g_tokenizer, prompt, true, prompt_tokens, prompt_capacity, prompt_count) ||
        prompt_count < 1) {
        free(prompt_tokens);
        set_error("prompt tokenization failed or prompt is too long");
        return -1;
    }
    if (prompt_count >= g_transformer.config.seq_len) {
        free(prompt_tokens);
        set_error("prompt consumes the whole 64-token context");
        return -1;
    }

    Sampler sampler{};
    sampler.vocab_size = g_transformer.config.vocab_size;
    sampler.temperature = temperature;
    sampler.topp = topp;
    sampler.rng_state = seed ? seed : 1;
    sampler.probindex = (ProbIndex *)llm_calloc(sampler.vocab_size, sizeof(ProbIndex));
    if (!sampler.probindex) {
        free(prompt_tokens);
        set_error("not enough RAM for sampler");
        return -1;
    }

    int token = prompt_tokens[0];
    int position = 0;
    while (position < steps) {
        float *logits = forward(g_transformer, token, position);
        const int next = position < prompt_count - 1
            ? prompt_tokens[position + 1]
            : sample(sampler, logits);
        ++position;
        if (next == 1) break;
        const char *piece = decode(g_tokenizer, token, next);
        if (callback && safe_piece(piece)) callback(piece, user_data);
        token = next;
        llm_yield();
    }

    free(sampler.probindex);
    free(prompt_tokens);
    set_error("ok");
    return position;
}

const char *llm_last_error(void) {
    return g_error;
}

size_t llm_estimated_runtime_bytes(void) {
    return g_runtime_bytes;
}

int llm_context_tokens(void) {
    return g_ready ? g_transformer.config.seq_len : 0;
}

void llm_end(void) {
    free_tokenizer(g_tokenizer);
    free_state(g_transformer.state);
    free_weights(g_transformer.weights);
    memset(&g_transformer, 0, sizeof(g_transformer));
    g_ready = false;
    g_runtime_bytes = 0;
}
