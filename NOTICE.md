# Notices and provenance

This repository combines an original hardware integration and documentation
with ideas and code structures derived from prior open-source work.

## llama2.c and TinyLlamas

The Transformer inference approach, tokenizer behavior, sampling logic, Q8_0
model layout, and the TinyStories 260K checkpoint originate from:

- Project: `karpathy/llama2.c`
- URL: https://github.com/karpathy/llama2.c
- Model: https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K
- Copyright: Copyright (c) 2023 Andrej
- License: MIT

The full MIT license text is included in `LICENSE`.

## Related ESP32 work

The following projects were consulted as prior art and implementation
references:

- DaveBben/esp32-llm: https://github.com/DaveBben/esp32-llm
- doryiii/esp32-llm: https://github.com/doryiii/esp32-llm

Those projects target ESP32-S3 hardware with PSRAM. This repository's Arduino
integration, classic-ESP32 memory budget, flash-resident embedded model, reduced
context configuration, conversion workflow, and documentation were assembled
for a no-PSRAM classic ESP32 target.

## AI assistance

AI tools assisted with research, code adaptation, conversion utilities,
debugging, testing, and documentation. The result was compiled and run on the
physical target board. This statement is provided for transparency and does not
change the licensing terms.

The classic-ESP32 integration is maintained by GitHub user `serenustaken`.
