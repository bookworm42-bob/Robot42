#include <libobsensor/ObSensor.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace fs = std::filesystem;

struct Options {
    fs::path output_dir = "artifacts/orbbec_rgb";
    int frames = 30;
    int warmup_frames = 10;
    int timeout_ms = 1000;
    int width = 640;
    int height = 480;
    int fps = 30;
    int depth_width = 0;
    int depth_height = 0;
    int depth_fps = 0;
    int log_every = 1;
    int imu_log_every = 200;
    bool latest_only = false;
    bool enable_depth = false;
    bool enable_imu = false;
    std::string imu_udp_host = "127.0.0.1";
    int imu_udp_port = 8766;
};

struct LatestImuSample {
    bool has_accel = false;
    bool has_gyro = false;
    uint64_t accel_frame_index = 0;
    uint64_t gyro_frame_index = 0;
    uint64_t accel_timestamp_us = 0;
    uint64_t gyro_timestamp_us = 0;
    float accel_temperature = 0.0f;
    float gyro_temperature = 0.0f;
    OBAccelValue accel_value{};
    OBGyroValue gyro_value{};
};

std::string imu_sample_to_json(const LatestImuSample &imu_sample) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    const uint64_t latest_imu_timestamp_us = std::max(imu_sample.accel_timestamp_us, imu_sample.gyro_timestamp_us);
    out << "{"
        << "\"timestamp_s\":" << (static_cast<double>(latest_imu_timestamp_us) / 1'000'000.0)
        << ",\"system_timestamp_us\":" << latest_imu_timestamp_us
        << ",\"has_accel\":" << (imu_sample.has_accel ? "true" : "false")
        << ",\"has_gyro\":" << (imu_sample.has_gyro ? "true" : "false");
    if(imu_sample.has_accel) {
        out << ",\"linear_acceleration_m_s2\":{"
            << "\"x\":" << imu_sample.accel_value.x
            << ",\"y\":" << imu_sample.accel_value.y
            << ",\"z\":" << imu_sample.accel_value.z
            << "}"
            << ",\"accel_frame_index\":" << imu_sample.accel_frame_index
            << ",\"accel_timestamp_us\":" << imu_sample.accel_timestamp_us
            << ",\"accel_temperature_c\":" << imu_sample.accel_temperature;
    }
    if(imu_sample.has_gyro) {
        out << ",\"angular_velocity_rad_s\":{"
            << "\"x\":" << imu_sample.gyro_value.x
            << ",\"y\":" << imu_sample.gyro_value.y
            << ",\"z\":" << imu_sample.gyro_value.z
            << "}"
            << ",\"gyro_frame_index\":" << imu_sample.gyro_frame_index
            << ",\"gyro_timestamp_us\":" << imu_sample.gyro_timestamp_us
            << ",\"gyro_temperature_c\":" << imu_sample.gyro_temperature;
    }
    out << "}";
    return out.str();
}

class ImuDatagramPublisher {
public:
    ImuDatagramPublisher(const std::string &host, int port) : socket_fd_(-1) {
        socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if(socket_fd_ < 0) {
            throw std::runtime_error("Failed to create IMU UDP socket");
        }
        const int flags = fcntl(socket_fd_, F_GETFL, 0);
        if(flags < 0 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
            ::close(socket_fd_);
            socket_fd_ = -1;
            throw std::runtime_error("Failed to mark IMU UDP socket non-blocking");
        }
        std::memset(&destination_, 0, sizeof(destination_));
        destination_.sin_family = AF_INET;
        destination_.sin_port = htons(static_cast<uint16_t>(port));
        if(::inet_pton(AF_INET, host.c_str(), &destination_.sin_addr) != 1) {
            ::close(socket_fd_);
            socket_fd_ = -1;
            throw std::runtime_error("Invalid IMU UDP host: " + host);
        }
    }

    ImuDatagramPublisher(const ImuDatagramPublisher &) = delete;
    ImuDatagramPublisher &operator=(const ImuDatagramPublisher &) = delete;

    ~ImuDatagramPublisher() {
        if(socket_fd_ >= 0) {
            ::close(socket_fd_);
        }
    }

    bool publish(const LatestImuSample &imu_sample) {
        last_error_message_.clear();
        const std::string payload = imu_sample_to_json(imu_sample);
        const ssize_t sent = ::sendto(
            socket_fd_,
            payload.data(),
            payload.size(),
            0,
            reinterpret_cast<const sockaddr *>(&destination_),
            sizeof(destination_)
        );
        if(sent >= 0) {
            return true;
        }
        if(errno == EWOULDBLOCK || errno == EAGAIN) {
            return false;
        }
        last_error_message_ = std::strerror(errno);
        return false;
    }

    const std::string &last_error_message() const {
        return last_error_message_;
    }

private:
    int socket_fd_;
    sockaddr_in destination_{};
    std::string last_error_message_;
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
        else if(arg == "--depth-width") {
            options.depth_width = parse_int(require_value(arg), arg);
        }
        else if(arg == "--depth-height") {
            options.depth_height = parse_int(require_value(arg), arg);
        }
        else if(arg == "--depth-fps") {
            options.depth_fps = parse_int(require_value(arg), arg);
        }
        else if(arg == "--log-every") {
            options.log_every = parse_int(require_value(arg), arg);
        }
        else if(arg == "--imu-log-every") {
            options.imu_log_every = parse_int(require_value(arg), arg);
        }
        else if(arg == "--latest-only") {
            options.latest_only = true;
        }
        else if(arg == "--enable-depth") {
            options.enable_depth = true;
        }
        else if(arg == "--enable-imu") {
            options.enable_imu = true;
        }
        else if(arg == "--imu-udp-host") {
            options.imu_udp_host = require_value(arg);
        }
        else if(arg == "--imu-udp-port") {
            options.imu_udp_port = parse_int(require_value(arg), arg);
        }
        else if(arg == "--help" || arg == "-h") {
            std::cout << "Usage: orbbec_rgb_test [--output-dir DIR] [--frames N]\n"
                      << "                       [--warmup-frames N] [--timeout-ms MS]\n"
                      << "                       [--width PX] [--height PX] [--fps FPS]\n"
                      << "                       [--depth-width PX] [--depth-height PX] [--depth-fps FPS]\n"
                      << "                       [--log-every N] [--imu-log-every N]\n"
                      << "                       [--imu-udp-host HOST] [--imu-udp-port PORT]\n"
                      << "                       [--latest-only] [--enable-depth] [--enable-imu]\n";
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
    if(options.depth_width < 0 || options.depth_height < 0 || options.depth_fps < 0) {
        throw std::runtime_error("--depth-width, --depth-height, and --depth-fps must be zero/auto or positive");
    }
    if(options.log_every < 0) {
        throw std::runtime_error("--log-every must be 0 or positive");
    }
    if(options.imu_log_every < 0) {
        throw std::runtime_error("--imu-log-every must be 0 or positive");
    }
    if(options.imu_udp_port < 1 || options.imu_udp_port > 65535) {
        throw std::runtime_error("--imu-udp-port must be between 1 and 65535");
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

void write_depth_pgm_mm(const fs::path &path, const std::shared_ptr<ob::DepthFrame> &depth_frame) {
    const auto width = depth_frame->getWidth();
    const auto height = depth_frame->getHeight();
    const auto pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    const auto expected_size = pixel_count * sizeof(uint16_t);
    if(depth_frame->getDataSize() < expected_size) {
        throw std::runtime_error("Depth frame data is smaller than width * height * uint16");
    }

    const auto *raw = reinterpret_cast<const uint16_t *>(depth_frame->getData());
    const float scale_to_mm = depth_frame->getValueScale();
    std::ofstream out(path, std::ios::binary);
    if(!out) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    out << "P5\n" << width << " " << height << "\n65535\n";
    for(uint64_t index = 0; index < pixel_count; ++index) {
        const float scaled = static_cast<float>(raw[index]) * scale_to_mm;
        const auto clamped = static_cast<uint16_t>(std::max(0.0f, std::min(65535.0f, scaled)));
        const char bytes[2] = {
            static_cast<char>((clamped >> 8) & 0xff),
            static_cast<char>(clamped & 0xff),
        };
        out.write(bytes, 2);
    }
}

void write_depth_pgm_atomic(const fs::path &path, const std::shared_ptr<ob::DepthFrame> &depth_frame) {
    const auto tmp_path = path.string() + ".tmp";
    write_depth_pgm_mm(tmp_path, depth_frame);
    fs::rename(tmp_path, path);
}

void write_latest_metadata(
    const fs::path &path,
    const std::shared_ptr<ob::ColorFrame> &frame,
    const std::shared_ptr<ob::DepthFrame> &depth_frame,
    const LatestImuSample *imu_sample,
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
        << "  \"latest_frame\": \"latest.ppm\"";
    if(depth_frame) {
        out << ",\n"
            << "  \"depth_width\": " << depth_frame->getWidth() << ",\n"
            << "  \"depth_height\": " << depth_frame->getHeight() << ",\n"
            << "  \"depth_format\": \"PGM_U16_MM\",\n"
            << "  \"depth_value_scale\": " << depth_frame->getValueScale() << ",\n"
            << "  \"latest_depth_frame\": \"latest_depth.pgm\"\n";
    }
    else {
        out << "\n";
    }
    if(imu_sample && (imu_sample->has_accel || imu_sample->has_gyro)) {
        out << ",\n"
            << "  \"imu\": {\n"
            << "    \"has_accel\": " << (imu_sample->has_accel ? "true" : "false") << ",\n"
            << "    \"has_gyro\": " << (imu_sample->has_gyro ? "true" : "false") << ",\n";
        if(imu_sample->has_accel) {
            out << "    \"linear_acceleration_m_s2\": {\n"
                << "      \"x\": " << imu_sample->accel_value.x << ",\n"
                << "      \"y\": " << imu_sample->accel_value.y << ",\n"
                << "      \"z\": " << imu_sample->accel_value.z << "\n"
                << "    },\n"
                << "    \"accel_frame_index\": " << imu_sample->accel_frame_index << ",\n"
                << "    \"accel_timestamp_us\": " << imu_sample->accel_timestamp_us << ",\n"
                << "    \"accel_temperature_c\": " << imu_sample->accel_temperature << ",\n";
        }
        if(imu_sample->has_gyro) {
            out << "    \"angular_velocity_rad_s\": {\n"
                << "      \"x\": " << imu_sample->gyro_value.x << ",\n"
                << "      \"y\": " << imu_sample->gyro_value.y << ",\n"
                << "      \"z\": " << imu_sample->gyro_value.z << "\n"
                << "    },\n"
                << "    \"gyro_frame_index\": " << imu_sample->gyro_frame_index << ",\n"
                << "    \"gyro_timestamp_us\": " << imu_sample->gyro_timestamp_us << ",\n"
                << "    \"gyro_temperature_c\": " << imu_sample->gyro_temperature << ",\n";
        }
        const uint64_t latest_imu_timestamp_us = std::max(imu_sample->accel_timestamp_us, imu_sample->gyro_timestamp_us);
        out << "    \"system_timestamp_us\": " << latest_imu_timestamp_us << "\n"
            << "  }\n";
    }
    out
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
    if(options.enable_depth) {
        const uint32_t depth_width = options.depth_width > 0 ? static_cast<uint32_t>(options.depth_width) : OB_WIDTH_ANY;
        const uint32_t depth_height = options.depth_height > 0 ? static_cast<uint32_t>(options.depth_height) : OB_HEIGHT_ANY;
        const uint32_t depth_fps = options.depth_fps > 0 ? static_cast<uint32_t>(options.depth_fps) : OB_FPS_ANY;
        config->enableVideoStream(
            OB_STREAM_DEPTH,
            depth_width,
            depth_height,
            depth_fps,
            OB_FORMAT_Y16
        );
    }

    auto device = pipeline.getDevice();
    auto info = device->getDeviceInfo();
    std::cout << "Orbbec device: " << info->getName()
              << " pid=0x" << std::hex << info->getPid() << std::dec
              << " sn=" << info->getSerialNumber() << "\n";
    if(options.enable_depth) {
        std::cout << "Depth stream requested: "
                  << (options.depth_width > 0 ? std::to_string(options.depth_width) : "any")
                  << "x"
                  << (options.depth_height > 0 ? std::to_string(options.depth_height) : "any")
                  << "@"
                  << (options.depth_fps > 0 ? std::to_string(options.depth_fps) : "any")
                  << " Y16\n";
    }

    auto converter = std::make_shared<ob::FormatConvertFilter>();

    std::shared_ptr<ob::Pipeline> imu_pipeline = nullptr;
    std::unique_ptr<ImuDatagramPublisher> imu_publisher;
    std::mutex imu_mutex;
    LatestImuSample latest_imu_sample;
    bool imu_running = false;
    uint64_t imu_callback_count = 0;
    uint64_t imu_udp_drop_count = 0;
    uint64_t imu_udp_error_count = 0;
    auto imu_log_started_at = std::chrono::steady_clock::now();
    if(options.enable_imu) {
        try {
            imu_publisher = std::make_unique<ImuDatagramPublisher>(options.imu_udp_host, options.imu_udp_port);
            std::cout << "IMU UDP publisher: " << options.imu_udp_host << ":" << options.imu_udp_port << "\n";
            if(options.frames != 0) {
                std::cerr << "WARNING: --enable-imu with finite --frames=" << options.frames
                          << " will stop the IMU UDP stream when RGB capture exits. "
                          << "Use --frames 0 for continuous robot runs.\n";
            }
            imu_pipeline = std::make_shared<ob::Pipeline>(device);
            auto imu_config = std::make_shared<ob::Config>();
            imu_config->enableGyroStream();
            imu_config->enableAccelStream();
            imu_config->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);
            imu_pipeline->start(imu_config, [&](std::shared_ptr<ob::FrameSet> frame_set) {
                auto accel_frame_raw = frame_set->getFrame(OB_FRAME_ACCEL);
                auto gyro_frame_raw = frame_set->getFrame(OB_FRAME_GYRO);
                LatestImuSample sample_to_publish;
                bool has_sample = false;
                {
                    std::lock_guard<std::mutex> lock(imu_mutex);
                    if(accel_frame_raw) {
                        auto accel_frame = accel_frame_raw->as<ob::AccelFrame>();
                        latest_imu_sample.has_accel = true;
                        latest_imu_sample.accel_frame_index = accel_frame->getIndex();
                        latest_imu_sample.accel_timestamp_us = accel_frame->getTimeStampUs();
                        latest_imu_sample.accel_temperature = accel_frame->getTemperature();
                        latest_imu_sample.accel_value = accel_frame->getValue();
                    }
                    if(gyro_frame_raw) {
                        auto gyro_frame = gyro_frame_raw->as<ob::GyroFrame>();
                        latest_imu_sample.has_gyro = true;
                        latest_imu_sample.gyro_frame_index = gyro_frame->getIndex();
                        latest_imu_sample.gyro_timestamp_us = gyro_frame->getTimeStampUs();
                        latest_imu_sample.gyro_temperature = gyro_frame->getTemperature();
                        latest_imu_sample.gyro_value = gyro_frame->getValue();
                    }
                    if(latest_imu_sample.has_accel || latest_imu_sample.has_gyro) {
                        sample_to_publish = latest_imu_sample;
                        has_sample = true;
                    }
                }
                if(!has_sample) {
                    return;
                }
                ++imu_callback_count;
                if(imu_publisher && !imu_publisher->publish(sample_to_publish)) {
                    if(imu_publisher->last_error_message().empty()) {
                        ++imu_udp_drop_count;
                    }
                    else {
                        ++imu_udp_error_count;
                    }
                }
                if(options.imu_log_every > 0 && imu_callback_count % static_cast<uint64_t>(options.imu_log_every) == 0) {
                    const double elapsed_s =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - imu_log_started_at).count();
                    const double rate_hz = elapsed_s > 0.0 ? static_cast<double>(imu_callback_count) / elapsed_s : 0.0;
                    std::cout << "IMU callback rate ~= " << rate_hz << " Hz"
                              << " callbacks=" << imu_callback_count
                              << " udp_drops=" << imu_udp_drop_count
                              << " udp_errors=" << imu_udp_error_count
                              << "\n";
                    imu_log_started_at = std::chrono::steady_clock::now();
                    imu_callback_count = 0;
                    imu_udp_drop_count = 0;
                    imu_udp_error_count = 0;
                    if(imu_publisher && !imu_publisher->last_error_message().empty()) {
                        std::cerr << "Latest IMU UDP error: " << imu_publisher->last_error_message() << "\n";
                    }
                }
            });
            imu_running = true;
        }
        catch(const std::exception &e) {
            std::cerr << "IMU stream unavailable: " << e.what() << "\n";
        }
    }
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
        std::shared_ptr<ob::DepthFrame> depth_frame = nullptr;
        if(options.enable_depth) {
            auto depth = frame_set->getFrame(OB_FRAME_DEPTH);
            if(!depth) {
                continue;
            }
            depth_frame = depth->as<ob::DepthFrame>();
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
        if(depth_frame) {
            const auto latest_depth_path = options.output_dir / "latest_depth.pgm";
            if(!options.latest_only) {
                const auto numbered_depth_path =
                    options.output_dir / ("frame_" + std::to_string(captured) + "_depth.pgm");
                write_depth_pgm_atomic(numbered_depth_path, depth_frame);
            }
            write_depth_pgm_atomic(latest_depth_path, depth_frame);
        }
        LatestImuSample imu_snapshot;
        const LatestImuSample *imu_ptr = nullptr;
        if(imu_running) {
            std::lock_guard<std::mutex> lock(imu_mutex);
            imu_snapshot = latest_imu_sample;
            if(imu_snapshot.has_accel || imu_snapshot.has_gyro) {
                imu_ptr = &imu_snapshot;
            }
        }
        write_latest_metadata(options.output_dir / "latest.json", rgb_frame, depth_frame, imu_ptr, captured, source_format);

        if(options.log_every > 0 && captured % options.log_every == 0) {
            std::cout << "Captured RGB frame " << captured
                      << " " << rgb_frame->getWidth() << "x" << rgb_frame->getHeight()
                      << " source=" << source_format
                      << " saved=" << latest_path;
            if(depth_frame) {
                std::cout << " depth=" << depth_frame->getWidth() << "x" << depth_frame->getHeight()
                          << " saved=" << (options.output_dir / "latest_depth.pgm");
            }
            std::cout << "\n";
        }
    }

    pipeline.stop();
    if(imu_running && imu_pipeline) {
        imu_pipeline->stop();
    }

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
