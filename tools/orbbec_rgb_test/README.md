# Orbbec RGB Test

Tiny C++ Orbbec SDK probe for macOS. It captures the color stream only, converts the SDK frame to RGB when needed, and writes PPM files that can be inspected or used by the next VR integration step without Python or OpenCV.

## Build

```sh
cmake -S tools/orbbec_rgb_test -B build/orbbec_rgb_test -DORBBEC_SDK_ROOT="$HOME/orbbec/sdk"
cmake --build build/orbbec_rgb_test
```

## Run

```sh
./build/orbbec_rgb_test/orbbec_rgb_test --frames 30 --output-dir artifacts/orbbec_rgb
```

The newest frame is written to `artifacts/orbbec_rgb/latest.ppm`, with metadata in `artifacts/orbbec_rgb/latest.json`.

Use `--frames 0 --latest-only` for continuous capture while the VR backend is running.
