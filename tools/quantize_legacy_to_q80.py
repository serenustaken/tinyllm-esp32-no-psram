"""Convert a legacy llama2.c float32 checkpoint to the v2 Q8_0 format.

This small conversion utility requires NumPy but not PyTorch. The output byte
layout matches the Q8_0 format read by llama2.c/runq.c.

For the TinyStories 260K model, group size 4 is required because it divides
both dim=64 and hidden_dim=172.

Usage:
    python3 quantize_legacy_to_q80.py stories260K.bin stories260K_q80.bin --group_size 4
"""
import struct
import argparse
import numpy as np


def read_legacy(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len = \
        struct.unpack_from('iiiiiii', data, 0)
    offset = 28  # 7 * 4 byte

    shared_classifier = vocab_size > 0
    vocab_size = abs(vocab_size)

    kv_dim = (dim * n_kv_heads) // n_heads
    head_size = dim // n_heads

    def read_f32(count):
        nonlocal offset
        arr = np.frombuffer(data, dtype=np.float32, count=count, offset=offset).copy()
        offset += count * 4
        return arr

    tok_embeddings = read_f32(vocab_size * dim).reshape(vocab_size, dim)

    attention_norm = [read_f32(dim) for _ in range(n_layers)]
    wq = [read_f32(dim * dim).reshape(dim, dim) for _ in range(n_layers)]
    wk = [read_f32(kv_dim * dim).reshape(kv_dim, dim) for _ in range(n_layers)]
    wv = [read_f32(kv_dim * dim).reshape(kv_dim, dim) for _ in range(n_layers)]
    wo = [read_f32(dim * dim).reshape(dim, dim) for _ in range(n_layers)]
    ffn_norm = [read_f32(dim) for _ in range(n_layers)]
    w1 = [read_f32(hidden_dim * dim).reshape(hidden_dim, dim) for _ in range(n_layers)]
    w2 = [read_f32(dim * hidden_dim).reshape(dim, hidden_dim) for _ in range(n_layers)]
    w3 = [read_f32(hidden_dim * dim).reshape(hidden_dim, dim) for _ in range(n_layers)]
    final_norm = read_f32(dim)

    freq_len = max_seq_len * (head_size // 2)
    freqs_cos = read_f32(freq_len)  # okuyoruz ama v2 formatinda kullanilmiyor
    freqs_sin = read_f32(freq_len)

    output_weight = None
    if not shared_classifier:
        output_weight = read_f32(vocab_size * dim).reshape(vocab_size, dim)

    assert offset == len(data), f"Dosyada beklenmeyen fazla/eksik veri var: {offset} okundu, {len(data)} mevcut"

    cfg = dict(dim=dim, hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
               n_kv_heads=n_kv_heads, vocab_size=vocab_size, max_seq_len=max_seq_len,
               shared_classifier=shared_classifier)

    return cfg, dict(
        tok_embeddings=tok_embeddings,
        attention_norm=attention_norm, wq=wq, wk=wk, wv=wv, wo=wo,
        ffn_norm=ffn_norm, w1=w1, w2=w2, w3=w3,
        final_norm=final_norm, output_weight=output_weight,
    )


def quantize_q80(w_flat, group_size):
    """ w_flat: 1D numpy float32 array. Returns (int8 vals, fp32 scales, max_err) """
    assert w_flat.size % group_size == 0
    groups = w_flat.reshape(-1, group_size)
    wmax = np.abs(groups).max(axis=1)
    wmax[wmax == 0] = 1e-10  # sifira bolmeyi onle
    scale = wmax / 127.0
    quant = groups / scale[:, None]
    int8val = np.round(quant).clip(-127, 127).astype(np.int8)
    dequant = int8val.astype(np.float32) * scale[:, None]
    max_err = float(np.abs(dequant - groups).max())
    return int8val.reshape(-1), scale.astype(np.float32), max_err


def write_q80(filepath, cfg, weights, group_size):
    dim = cfg['dim']
    hidden_dim = cfg['hidden_dim']
    if group_size <= 0:
        raise ValueError("group_size must be greater than zero")
    if dim % group_size != 0 or hidden_dim % group_size != 0:
        raise ValueError(
            f"group_size={group_size} must divide both dim={dim} "
            f"and hidden_dim={hidden_dim}; use --group_size 4 for stories260K"
        )

    quant_list = [weights['tok_embeddings']]
    quant_list += weights['wq'] + weights['wk'] + weights['wv'] + weights['wo']
    quant_list += weights['w1'] + weights['w2'] + weights['w3']
    if not cfg['shared_classifier']:
        quant_list.append(weights['output_weight'])

    for i, mat in enumerate(quant_list):
        if mat.size % group_size != 0:
            raise ValueError(
                f"matrix {i} ({mat.shape}, {mat.size} elements) is not divisible "
                f"by group_size={group_size}"
            )

    with open(filepath, "wb") as f:
        f.write(struct.pack('I', 0x616b3432))   # magic "ak42"
        f.write(struct.pack('i', 2))             # version 2
        f.write(struct.pack('iiiiiii', dim, cfg['hidden_dim'], cfg['n_layers'],
                             cfg['n_heads'], cfg['n_kv_heads'], cfg['vocab_size'],
                             cfg['max_seq_len']))
        f.write(struct.pack('B', int(cfg['shared_classifier'])))
        f.write(struct.pack('i', group_size))
        pad = 256 - f.tell()
        assert pad >= 0
        f.write(b'\0' * pad)

        for x in weights['attention_norm']:
            f.write(x.astype(np.float32).tobytes())
        for x in weights['ffn_norm']:
            f.write(x.astype(np.float32).tobytes())
        f.write(weights['final_norm'].astype(np.float32).tobytes())

        max_err_overall = 0.0
        for i, mat in enumerate(quant_list):
            flat = mat.reshape(-1).astype(np.float32)
            q, s, err = quantize_q80(flat, group_size)
            f.write(q.tobytes())
            f.write(s.tobytes())
            max_err_overall = max(max_err_overall, err)
            print(f"  [{i+1}/{len(quant_list)}] shape={mat.shape} max_error={err:.5f}")

    print(f"\nWrote: {filepath}")
    print(f"Group size: {group_size}")
    print(f"Largest quantization error: {max_err_overall:.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="legacy v0 float32 .bin checkpoint")
    ap.add_argument("output", help="output v2 Q8_0 .bin checkpoint")
    ap.add_argument("--group_size", type=int, default=4)
    args = ap.parse_args()

    cfg, weights = read_legacy(args.input)
    print("Model configuration:", cfg)
    write_q80(args.output, cfg, weights, args.group_size)


if __name__ == "__main__":
    main()
