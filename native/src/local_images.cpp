#include "local_images.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cwctype>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <Windows.h>
#include <bcrypt.h>
#include <opencv2/imgcodecs.hpp>

namespace visual {
namespace fs = std::filesystem;
namespace {

void requireLocalDrive(const fs::path& path) {
    const auto root = path.root_path().native();
    if (root.size() != 3 || root[1] != L':' || root[2] != L'\\') {
        throw std::runtime_error("Visual Analysis only accepts local drive paths.");
    }
    const auto type = GetDriveTypeW(root.c_str());
    if (type != DRIVE_FIXED && type != DRIVE_REMOVABLE && type != DRIVE_CDROM && type != DRIVE_RAMDISK) {
        throw std::runtime_error("The requested path is not on a supported local drive.");
    }
}

// Check each component before traversing it because a junction could lead to a share.
void rejectRemoteLinks(const fs::path& path) {
    auto current = path.root_path();
    for (const auto& component : path.relative_path()) {
        if (component.native().find(L':') != std::wstring::npos) {
            throw std::runtime_error("Alternate data streams are unsupported.");
        }
        current /= component;
        const auto attributes = GetFileAttributesW(current.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
            throw std::runtime_error("The selected path contains a junction or symbolic link.");
        }
    }
}

std::string jsonString(std::string_view input) {
    std::ostringstream json;
    json << '"';
    for (unsigned char value : input) {
        if (value == '"' || value == '\\') {
            json << '\\' << value;
        } else if (value < ' ') {
            json << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(value);
        } else {
            json << value;
        }
    }
    json << '"';
    return json.str();
}

std::string pathUtf8(const fs::path& path) {
    const auto utf8 = path.u8string();
    return {utf8.begin(), utf8.end()};
}

std::string sha256(const fs::path& path) {
    BCRYPT_ALG_HANDLE algorithm{};
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0) {
        throw std::runtime_error("Could not initialize Windows SHA-256.");
    }
    struct Guard {
        BCRYPT_ALG_HANDLE algorithm;
        BCRYPT_HASH_HANDLE hash{};
        ~Guard() {
            if (hash) BCryptDestroyHash(hash);
            BCryptCloseAlgorithmProvider(algorithm, 0);
        }
    } handle{algorithm};
    if (BCryptCreateHash(algorithm, &handle.hash, nullptr, 0, nullptr, 0, 0) < 0) {
        throw std::runtime_error("Could not create a SHA-256 hash.");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open the source image for hashing.");
    }
    std::array<UCHAR, 65536> buffer{};
    while (input.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || input.gcount()) {
        if (BCryptHashData(handle.hash, buffer.data(), static_cast<ULONG>(input.gcount()), 0) < 0) {
            throw std::runtime_error("Could not hash the source image.");
        }
    }
    std::array<UCHAR, 32> digest{};
    if (input.bad() || BCryptFinishHash(handle.hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0) {
        throw std::runtime_error("Could not finish the source image hash.");
    }
    std::ostringstream result;
    for (const auto byte : digest) {
        result << std::hex << std::setw(2) << std::setfill('0') << int(byte);
    }
    return result.str();
}

bool supported(const fs::path& path) {
    auto extension = path.extension().wstring();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](wchar_t c) {
        return static_cast<wchar_t>(std::towlower(c));
    });
    constexpr std::array<std::wstring_view, 9> formats{
        L".jpg", L".jpeg", L".png", L".bmp", L".webp", L".tif", L".tiff", L".gif", L".avif"
    };
    return std::find(formats.begin(), formats.end(), extension) != formats.end();
}

fs::path newRunDirectory(const fs::path& root) {
    fs::create_directories(root);
    const auto tick = std::chrono::system_clock::now().time_since_epoch().count();
    for (int nonce = 0; nonce < 100; ++nonce) {
        const auto path = root / (L"run-native-" + std::to_wstring(tick) + L"-" + std::to_wstring(nonce));
        if (fs::create_directory(path)) {
            return path;
        }
    }
    throw std::runtime_error("Could not reserve a case output directory.");
}

void writeReport(const fs::path& path, const std::string& content) {
    const auto target = localPath(path);
    fs::create_directories(target.parent_path());
    auto temporary = target;
    temporary += L"." + std::to_wstring(GetCurrentProcessId()) + L".part";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    output << content;
    output.close();
    if (!output || !MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        fs::remove(temporary);
        throw std::runtime_error("Could not store the case report.");
    }
}

}

std::filesystem::path localPath(const fs::path& path) {
    const auto absolute = fs::absolute(path).lexically_normal();
    requireLocalDrive(absolute);
    rejectRemoteLinks(absolute);
    const auto resolved = fs::weakly_canonical(absolute);
    requireLocalDrive(resolved);
    return resolved;
}

cv::Mat readImage(const fs::path& path) {
    const auto source = localPath(path);
    constexpr std::uintmax_t maxBytes = 160u * 1024 * 1024;
    if (!fs::is_regular_file(source) || fs::file_size(source) > maxBytes) {
        throw std::runtime_error("The local image is missing or exceeds 160 MB.");
    }
    std::ifstream input(source, std::ios::binary);
    const std::vector<uchar> encoded{std::istreambuf_iterator<char>(input), {}};
    auto image = encoded.empty() ? cv::Mat{} : cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("The selected image format could not be decoded.");
    }
    return image;
}

void writePng(const fs::path& path, const cv::Mat& pixels) {
    const auto target = localPath(path);
    std::vector<uchar> encoded;
    if (!cv::imencode(".png", pixels, encoded)) {
        throw std::runtime_error("Could not encode the restoration as PNG.");
    }
    HANDLE file = CreateFileW(target.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Could not create a new output PNG.");
    }
    DWORD written{};
    const BOOL success = WriteFile(file, encoded.data(), static_cast<DWORD>(encoded.size()), &written, nullptr);
    CloseHandle(file);
    if (!success || written != encoded.size()) {
        DeleteFileW(target.c_str());
        throw std::runtime_error("Could not write the output PNG.");
    }
}

std::wstring wide(const std::string& utf8) {
    if (utf8.empty()) {
        return {};
    }
    const int size = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), result.data(), size);
    return result;
}

int runCases(const fs::path& inputs, const fs::path& outputs, const fs::path& report, const fs::path& models,
             bool preferRecall) {
    try {
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(localPath(inputs))) {
            if (supported(entry.path())) {
                const auto source = localPath(entry.path());
                if (fs::is_regular_file(source)) {
                    files.push_back(source);
                }
            }
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            throw std::runtime_error("The case input directory contains no supported images.");
        }
        const auto run = newRunDirectory(localPath(outputs));
        bool complete = true;
        std::ostringstream cases;
        cases << std::fixed << std::setprecision(2);
        for (size_t start = 0; start < files.size(); start += 3) {
            const auto end = std::min(start + 3, files.size());
            std::vector<cv::Mat> images;
            for (size_t index = start; index < end; ++index) {
                images.push_back(readImage(files[index]));
            }
            const auto results = restorePortraits(images, models, std::nullopt, preferRecall);
            for (size_t index = start; index < end; ++index) {
                const auto& options = results[index - start];
                complete &= !options.empty();
                if (index) {
                    cases << ",\n";
                }
                cases << "    {\"source\":" << jsonString(pathUtf8(files[index].filename()))
                      << ",\"source_sha256\":" << jsonString(sha256(files[index]))
                      << ",\"candidates\":[";
                const auto directory = run / std::to_wstring(index + 1);
                fs::create_directory(directory);
                for (size_t rank = 0; rank < options.size(); ++rank) {
                    const auto& option = options[rank];
                    const auto target = directory / (L"candidate-" + std::to_wstring(rank + 1) + L".png");
                    writePng(target, option.pixels);
                    if (rank) {
                        cases << ',';
                    }
                    cases << "{\"model\":" << jsonString(option.model)
                          << ",\"score\":" << option.score << ",\"agreement\":" << option.agreement
                          << ",\"detail\":" << option.detail
                          << ",\"confidence\":\"" << (option.moderateConfidence ? "中" : "低")
                          << "\",\"file\":" << jsonString(pathUtf8(target)) << '}';
                }
                cases << "]}";
            }
        }
        writeReport(report, std::string("{\"ok\":") + (complete ? "true" : "false")
            + ",\"runtime\":\"native-cpp-onnxruntime\",\"prefer_recall\":" + (preferRecall ? "true" : "false")
            + ",\"output_directory\":" + jsonString(pathUtf8(run))
            + ",\"cases\":[\n" + cases.str() + "\n]}\n");
        return complete ? 0 : 2;
    } catch (const std::exception& error) {
        writeReport(report, "{\"ok\":false,\"error\":" + jsonString(error.what()) + "}\n");
        return 2;
    }
}

}
