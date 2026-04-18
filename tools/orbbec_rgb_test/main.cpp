#include <libobsensor/ObSensor.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;

struct Options {
    fs::path output_dir = "artifacts/orbbec_rgb";
    int frames = 30;
    int warmup_frames = 10;
    int timeout_ms = 1000;
    int width = 640;
    int height = 480;
    int fps = 30;
    int log_every = 1;
    bool latest_only = false;
};

int parse_int(const std::string &value, const std::string &name) {
    try {
        size_t consumed = 0;
        int parsed = std::stoi(value, &consumed);
        if(consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    }
    catch(const std::exception &) {
        throw std::runtime_error("Invalid integer for " + name + ": " + value);
    }
}

Options parse_args(int argc, char **argv) {
    Options options;
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string &name) -> std::string {
            if(i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if(arg == "--output-dir") {
            options.output_dir = require_value(arg);
        }
        else if(arg == "--frames") {
            options.frames = parse_int(require_value(arg), arg);
        }
        else if(arg == "--warmup-frames") {
            options.warmup_frames = parse_int(require_value(arg), arg);
        }
        else if(arg == "--timeout-ms") {
            options.timeout_ms = parse_int(require_value(arg), arg);
        }
        else if(arg == "--width") {
            options.width = parse_int(require_value(arg), arg);
        }
        else if(arg == "--height") {
            options.height = parse_int(require_value(arg), arg);
        }
        else if(arg == "--fps") {
            options.fps = parse_int(require_value(arg), arg);
        }
        else if(arg == "--log-every") {
            options.log_every = parse_int(require_value(arg), arg);
        }
        else if(arg == "--latest-only") {
            options.latest_only = true;
        }
        else if(arg == "--help" || arg == "-h") {
            std::cout << "Usage: orbbec_rgb_test [--output-dir DIR] [--frames N]\n"
                      << "                       [--warmup-frames N] [--timeout-ms MS]\n"
                      << "                       [--width PX] [--height PX] [--fps FPS]\n"
                      << "                       [--log-every N] [--latest-only]\n";
            std::exit(EXIT_SUCCESS);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if(options.frames < 0) {
        throw std::runtime_error("--frames must be 0 for continuous capture, or a positive frame count");
    }
    if(options.width < 1 || options.height < 1 || options.fps < 1) {
        throw std::runtime_error("--width, --height, and --fps must be positive");
    }
    if(options.log_every < 0) {
        throw std::runtime_error("--log-every must be 0 or positive");
    }
    return options;
}

std::string format_name(OBFormat format) {
    return ob::TypeHelper::convertOBFormatTypeToString(format);
}

std::shared_ptr<ob::ColorFrame> to_rgb_frame(
    const std::shared_ptr<ob::ColorFrame> &color_frame,
    const std::shared_ptr<ob::FormatConvertFilter> &converter
) {
    if(color_frame->getFormat() == OB_FORMAT_RGB) {
        return color_frame;
    }
    if(color_frame->getFormat() == OB_FORMAT_MJPG) {
        converter->setFormatConvertType(FORMAT_MJPG_TO_RGB);
    }
    else if(color_frame->getFormat() == OB_FORMAT_UYVY) {
        converter->setFormatConvertType(FORMAT_UYVY_TO_RGB);
    }
    else if(color_frame->getFormat() == OB_FORMAT_YUYV) {
        converter->setFormatConvertType(FORMAT_YUYV_TO_RGB);
    }
    else {
        throw std::runtime_error(std::string("Unsupported color format: ") + format_name(color_frame->getFormat()));
    }

    return converter->process(color_frame)->as<ob::ColorFrame>();
}

void write_ppm(const fs::path &path, const std::shared_ptr<ob::ColorFrame> &rgb_frame) {
    const auto width = rgb_frame->getWidth();
    const auto height = rgb_frame->getHeight();
    const auto expected_size = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 3;
    if(rgb_frame->getFormat() != OB_FORMAT_RGB) {
        throw std::runtime_error("write_ppm received a non-RGB frame");
    }
    if(rgb_frame->getDataSize() < expected_size) {
        throw std::runtime_error("RGB frame data is smaller than width * height * 3");
    }

    std::ofstream out(path, std::ios::binary);
    if(!out) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char *>(rgb_frame->getData()), static_cast<std::streamsize>(expected_size));
}

void write_ppm_atomic(const fs::path &path, const std::shared_ptr<ob::ColorFrame> &rgb_frame) {
    const auto tmp_path = path.string() + ".tmp";
    write_ppm(tmp_path, rgb_frame);
    fs::rename(tmp_path, path);
}

void write_latest_metadata(
    const fs::path &path,
    const std::shared_ptr<ob::ColorFrame> &frame,
    int captured_frames,
    const std::string &source_format
) {
    const auto tmp_path = path.string() + ".tmp";
    std::ofstream out(tmp_path);
    if(!out) {
        throw std::runtime_error("Could not open metadata file: " + tmp_path);
    }
    out << "{\n"
        << "  \"captured_frames\": " << captured_frames << ",\n"
        << "  \"width\": " << frame->getWidth() << ",\n"
        << "  \"height\": " << frame->getHeight() << ",\n"
        << "  \"format\": \"RGB\",\n"
        << "  \"source_format\": \"" << source_format << "\",\n"
        << "  \"frame_index\": " << frame->getIndex() << ",\n"
        << "  \"device_timestamp_us\": " << frame->getTimeStampUs() << ",\n"
        << "  \"system_timestamp_us\": " << frame->getSystemTimeStampUs() << ",\n"
        << "  \"latest_frame\": \"latest.ppm\"\n"
        << "}\n";
    out.close();
    fs::rename(tmp_path, path);
}

int main(int argc, char **argv) try {
    const Options options = parse_args(argc, argv);
    fs::create_directories(options.output_dir);

    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();
    config->enableVideoStream(
        OB_STREAM_COLOR,
        static_cast<uint32_t>(options.width),
        static_cast<uint32_t>(options.height),
        static_cast<uint32_t>(options.fps),
        OB_FORMAT_ANY
    );

    auto device = pipeline.getDevice();
    auto info = device->getDeviceInfo();
    std::cout << "Orbbec device: " << info->getName()
              << " pid=0x" << std::hex << info->getPid() << std::dec
              << " sn=" << info->getSerialNumber() << "\n";

    auto converter = std::make_shared<ob::FormatConvertFilter>();
    pipeline.start(config);

    for(int i = 0; i < options.warmup_frames; ++i) {
        pipeline.waitForFrameset(options.timeout_ms);
    }

    int captured = 0;
    const auto started_at = std::chrono::steady_clock::now();
    while(options.frames == 0 || captured < options.frames) {
        auto frame_set = pipeline.waitForFrameset(options.timeout_ms);
        if(!frame_set) {
            std::cout << "No color frames received within " << options.timeout_ms << "ms\n";
            continue;
        }

        auto frame = frame_set->getFrame(OB_FRAME_COLOR);
        if(!frame) {
            continue;
        }

        auto color_frame = frame->as<ob::ColorFrame>();
        const std::string source_format = format_name(color_frame->getFormat());
        auto rgb_frame = to_rgb_frame(color_frame, converter);

        ++captured;
        const auto latest_path = options.output_dir / "latest.ppm";
        if(!options.latest_only) {
            const auto numbered_path = options.output_dir / ("frame_" + std::to_string(captured) + ".ppm");
            write_ppm_atomic(numbered_path, rgb_frame);
        }
        write_ppm_atomic(latest_path, rgb_frame);
        write_latest_metadata(options.output_dir / "latest.json", rgb_frame, captured, source_format);

        if(options.log_every > 0 && captured % options.log_every == 0) {
            std::cout << "Captured RGB frame " << captured
                      << " " << rgb_frame->getWidth() << "x" << rgb_frame->getHeight()
                      << " source=" << source_format
                      << " saved=" << latest_path << "\n";
        }
    }

    pipeline.stop();

    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    std::cout << "Captured " << captured << " frames in " << elapsed << "s\n";
    return EXIT_SUCCESS;
}
catch(const ob::Error &e) {
    std::cerr << "Orbbec SDK error\n"
              << "function: " << e.getFunction() << "\n"
              << "args: " << e.getArgs() << "\n"
              << "message: " << e.what() << "\n"
              << "type: " << e.getExceptionType() << "\n";
    return EXIT_FAILURE;
}
catch(const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
}
