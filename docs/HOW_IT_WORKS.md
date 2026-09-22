# How it works — without assuming an AI background

## The short version

The ESP32 stores a tiny language model in flash, turns an English prompt into
numbers, repeatedly predicts the next number, and translates those numbers back
into text. Every calculation happens on the microcontroller.

## Flash is the cupboard; RAM is the desk

A classic ESP32 has roughly 4 MB of flash storage but only about 520 KB of total
internal SRAM, not all of which is available to an application. Think of flash
as a cupboard and RAM as a desk. The complete model can stay in the cupboard;
only the information needed for the current calculation must be on the desk.

The engine therefore points directly into the model array stored in flash. It
does not unpack the entire model into RAM.

## Why quantization matters

The original checkpoint stores each weight as a 32-bit floating-point number.
At roughly 260,000 parameters, that is about one megabyte. Q8_0 quantization
stores each weight as an 8-bit integer plus a scale shared by a small group.

Group size 4 is used here. The model dimensions 64 and 172 are both divisible
by 4, which keeps every matrix group correctly aligned. A group size of 64
would produce a smaller file but is not valid for the 172-wide matrices.

## Why context is limited to 64 tokens

During generation, a Transformer keeps a temporary memory called a KV cache.
With the model's original 512-token context, that cache alone would require
about 655 KB—more than the target can provide. Limiting context to 64 tokens
reduces the cache to about 82 KB.

A token is a word or part of a word. The prompt and generated continuation
share the same 64-token window.

## What happens after Enter is pressed

1. The tokenizer converts the prompt into token numbers.
2. The model processes one position through five Transformer layers.
3. It produces a probability for each of the 512 possible next tokens.
4. Temperature and top-p sampling select one token.
5. The tokenizer converts that token back into text and prints it over USB.
6. The selected token is fed back into the model and the cycle repeats.

Generation stops when the end token is produced, the model fills the context,
or the configured generation-step limit is reached.

## What each sketch file does

- `tinyllm_esp32_no_psram.ino`: startup, serial input/output, configuration.
- `llm_engine.cpp`: model loading, tokenizer, Transformer math, and sampling.
- `llm_engine.h`: the small interface between the sketch and engine.
- `model_data.h`: the Q8_0 model embedded as flash-resident bytes.
- `tokenizer_data.h`: the 512-token vocabulary embedded as bytes.

## What “offline” means here

The USB connection supplies power and provides a text terminal. It does not
perform the language-model calculation. Wi-Fi and Bluetooth are unused; no
prompt or generated text is sent to a server.
