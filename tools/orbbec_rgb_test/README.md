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

For the real exploration ROS bridge, start `xlerobot_playground.robot_brain_agent` first, then run the sidecar with depth enabled and paired RGB-D HTTP streaming:

```sh
./build/orbbec_rgb_test/orbbec_rgb_test \
  --frames 0 \
  --no-file-output \
  --enable-depth \
  --enable-imu \
  --imu-udp-host 127.0.0.1 \
  --imu-udp-port 8766 \
  --camera-http-enable \
  --camera-http-host 127.0.0.1 \
  --camera-http-port 8765 \
  --camera-http-path /camera/rgbd
```

This sends each RGB-D pair to the robot brain as a single in-memory frame at the camera capture rate. With `--enable-imu`, the sidecar also pushes each IMU callback as a non-blocking UDP datagram to the robot brain, which then serves `/imu` as an in-memory debug snapshot and `/ws/imu` as the high-rate stream.

Depth defaults to the camera's first matching Y16 profile at the selected RGB FPS. If you need to force a specific depth mode, pass `--depth-width`, `--depth-height`, and `--depth-fps`.
