# TinyLLM on a Classic ESP32 — No PSRAM

I run a real, fully offline 260K-parameter TinyStories Transformer on a classic
ESP32/ESP-32S development board with **no external PSRAM**.

This is a small educational text generator, not a general-purpose chatbot. It
continues short English story prompts directly on the microcontroller. No Wi-Fi,
cloud API, SD card, or external server is required.

> I started this project with a practical question: can a real language model
> run locally on a widely available classic ESP32 without PSRAM? I am publishing
> the working result so other curious makers can reproduce it and build on it.
> See [Project story](#project-story) and [Credits](#credits) for full attribution.

I maintain this project as [@serenustaken](https://github.com/serenustaken).

## Why this is interesting

Existing ESP32 LLM demonstrations commonly target an ESP32-S3 with external
PSRAM. I took a different route for the widely available classic
ESP32:

- model weights stay in flash instead of being copied to RAM;
- Q8_0 quantization reduces the weight size;
- the runtime context is limited to 64 tokens;
- runtime allocations use internal RAM;
- the complete sketch builds in the Arduino IDE.

## Verified configuration

| Item | Verified value |
|---|---:|
| Target board | Classic ESP32 / ESP-32S |
| External PSRAM | Not required |
| Model | TinyStories 260K |
| Architecture | dim 64, hidden dim 172, 5 layers, 8 heads, 4 KV heads |
| Weight format | llama2.c v2 Q8_0, group size 4 |
| Embedded model size | 521,728 bytes |
| Embedded tokenizer size | 6,227 bytes |
| Runtime context | 64 tokens |
| Estimated runtime buffers | about 89,400 bytes |
| Verified compiled sketch | 829,828 bytes flash, 22,996 bytes static RAM |

The exact compile totals can vary slightly with the ESP32 Arduino core version.

## What you need

- a classic ESP32/ESP-32S development board with 4 MB flash;
- a data-capable USB cable;
- Arduino IDE with the Espressif ESP32 board package installed;
- this repository.

I tested the project on the common 30-pin ESP-32S development board. ESP32-S3
is not required.

## Quick start

1. Download or clone this repository.
2. Open
   `tinyllm_esp32_no_psram/tinyllm_esp32_no_psram.ino` in Arduino IDE.
3. Select these settings under **Tools**:

   | Setting | Value |
   |---|---|
   | Board | ESP32 Dev Module |
   | CPU Frequency | 240MHz (WiFi/BT) |
   | Flash Frequency | 80MHz |
   | Flash Mode | QIO |
   | Flash Size | 4MB (32Mb) |
   | Partition Scheme | Huge APP (3MB No OTA/1MB SPIFFS) |
   | PSRAM | Disabled |
   | Upload Speed | 921600; use 115200 if upload is unreliable |

4. Select the board's USB port and click **Upload**.
5. Open **Serial Monitor** at `115200 baud` and choose `New Line`.
6. After the automatic first story, enter a short English prompt and press
   Enter.

Example prompts:

```text
Once there was a little dog
Lily went to the forest
Tom found a red ball
```

Generation on a classic ESP32 is slow compared with a computer. Let the board
finish before sending another prompt.

## Expected serial output

```text
TinyLLM on classic ESP32 (no PSRAM)
Model in flash: 521728 bytes; tokenizer: 6227 bytes
Runtime buffers: about 89400 bytes; context: 64 tokens

--- output ---
Once upon a time, there was a little girl named Lily...
--- done ---
Positions: 64, time: ... s, average: ... tok/s
```

## Important limitations

- This is **not ChatGPT** and not a question-answering assistant.
- It is trained for short English children's stories.
- Prompt plus generated text must fit in the 64-token context.
- Output quality is limited by the extremely small 260K-parameter model.
- Wi-Fi and Bluetooth are intentionally unused to preserve RAM.

## How it fits without PSRAM

The original float32 checkpoint is about 1.06 MB. The included conversion tool
quantizes it to Q8_0. The resulting 521,728-byte model is compiled into the
firmware and read directly from flash.

Only temporary activations and a reduced 64-token KV cache are allocated in
internal RAM. The model uses quantization group size 4 because both model
dimensions `64` and `172` must be divisible by the group size.

For a plain-language explanation, read
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). A Turkish guide is available at
[README_TR.md](README_TR.md).

## Rebuilding the embedded model

The repository already contains the generated headers, so this is optional.
To reproduce them, install NumPy, download `stories260K.bin` and `tok512.bin`
from the upstream TinyLlamas model, then run:

```bash
python3 -m pip install -r tools/requirements.txt
python3 tools/quantize_legacy_to_q80.py stories260K.bin stories260K_q80.bin --group_size 4
python3 tools/bin_to_c_array.py stories260K_q80.bin tinyllm_esp32_no_psram/model_data.h model_data
python3 tools/bin_to_c_array.py tok512.bin tinyllm_esp32_no_psram/tokenizer_data.h tokenizer_data
```

The original checkpoint and tokenizer are available from the
[karpathy/tinyllamas stories260K model](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K).

Known SHA-256 hashes used for this build:

```text
stories260K.bin      b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696
stories260K_q80.bin  c8c032d04f8a48b2c8f9b708cc571a895d45fe23521f47f542b897d64ad77c50
tok512.bin           037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312
```

## Troubleshooting

- **`llm_engine.h: No such file or directory`:** open the `.ino` inside the
  supplied sketch folder; do not move only the `.ino` file elsewhere.
- **Upload stops at `Connecting...`:** hold the board's `BOOT` button until
  writing begins.
- **`resource busy`:** close every other Serial Monitor, `screen`, or serial
  terminal. Only one program can use the port at a time.
- **Initialization says there is not enough RAM:** keep Wi-Fi/Bluetooth code
  disabled, reboot, or reduce `CONTEXT_TOKENS` to 48 or 32.
- **Output is nonsense:** confirm the startup line reports a 521,728-byte model.
  A 278,608-byte header is the older incompatible group-size-64 conversion.
- **Watchdog resets:** do not remove the `yield()` calls in the engine.
- **Flash errors:** try DIO, 40MHz flash, and 115200 upload speed.

## Project story

I started this project with a practical question: can a real language model run
locally on a widely available classic ESP32 without PSRAM? Building on existing
open-source work, I adapted, tested, and documented a memory-conscious Arduino
implementation that others can reproduce.

I used AI tools to support parts of the research, implementation, debugging,
and documentation process. I compiled the complete system, flashed it, and
verified it on physical hardware. My focus is practical integration,
reproducibility, and making embedded AI more approachable.

## Credits

- [Andrej Karpathy's llama2.c](https://github.com/karpathy/llama2.c) — original
  minimal Transformer inference code, Q8_0 format, TinyStories checkpoint, and
  tokenizer; MIT licensed.
- [DaveBben/esp32-llm](https://github.com/DaveBben/esp32-llm) — earlier ESP32-S3
  demonstration and inspiration for running TinyStories on ESP32.
- [doryiii/esp32-llm](https://github.com/doryiii/esp32-llm) — ESP32-S3 Q8
  implementation and useful reference for on-the-fly dequantization and memory
  layout.
- [TinyStories](https://arxiv.org/abs/2305.07759) — the narrow story-generation
  dataset that makes such a tiny model possible.

See [NOTICE.md](NOTICE.md) for provenance details.

## License

Code in this repository is available under the MIT License. The included model
repository is also marked MIT by its publisher. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md).
