# Orbbec RGB/RGB-D Test

Tiny C++ Orbbec SDK probe for macOS. It captures the color stream, converts the SDK frame to RGB when needed, and writes PPM files that can be inspected or used by the next VR integration step without Python or OpenCV. With `--enable-depth`, it also writes a 16-bit millimetre depth PGM for the real ROS bridge.

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

For the real exploration ROS bridge, run the sidecar with depth enabled and point the bridge at the same directory:

```sh
./build/orbbec_rgb_test/orbbec_rgb_test --frames 0 --latest-only --enable-depth --output-dir artifacts/orbbec_rgbd
```

This writes `latest.ppm`, `latest_depth.pgm`, and `latest.json`.

Depth defaults to the camera's first matching Y16 profile. If you need to force a specific depth mode, pass `--depth-width`, `--depth-height`, and `--depth-fps`.
