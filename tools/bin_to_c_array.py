"""Convert a binary file into a flash-resident Arduino C header.

Usage:
    python3 bin_to_c_array.py stories260K_q80.bin model_data.h model_data
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input binary file")
    ap.add_argument("output", help="output C header")
    ap.add_argument("varname", help="C array name, for example model_data")
    ap.add_argument("--bytes_per_line", type=int, default=20)
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        data = f.read()

    guard = args.varname.upper() + "_H"

    with open(args.output, "w") as f:
        f.write(f"#ifndef {guard}\n#define {guard}\n\n")
        f.write("#include <Arduino.h>\n\n")
        # The inference engine reads some regions as float pointers, so keep
        # the embedded byte array four-byte aligned on ESP32.
        f.write(f"alignas(4) const unsigned char {args.varname}[] PROGMEM = {{\n")
        for i in range(0, len(data), args.bytes_per_line):
            chunk = data[i:i + args.bytes_per_line]
            line = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write("  " + line + ",\n")
        f.write("};\n\n")
        f.write(f"const unsigned int {args.varname}_len = {len(data)};\n\n")
        f.write(f"#endif // {guard}\n")

    print(f"Wrote: {args.output}")
    print(f"Array: {args.varname}; {args.varname}_len = {len(data)}")
    print(f"Header text is about {len(data) * 6 // 1024} KB; embedded data is {len(data)} bytes")


if __name__ == "__main__":
    main()
