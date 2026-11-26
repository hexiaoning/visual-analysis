#include "storage.h"
#include "local_images.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include <Windows.h>
#include <bcrypt.h>
#include <winsqlite/winsqlite3.h>

namespace visual {
namespace fs = std::filesystem;

std::string utf8(std::wstring_view text) {
    if (text.empty()) return {};
    const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                                            static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
    if (!length) throw std::runtime_error("无效的 Unicode 文本。");
    std::string content(length, '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
                        content.data(), length, nullptr, nullptr);
    return content;
}

std::string pathText(const fs::path& path) { return utf8(path.native()); }

fs::path textPath(std::string_view text) {
    if (text.find('\0') != std::string_view::npos) throw std::invalid_argument("路径包含无效字符。");
    return fs::path(wide(std::string(text)));
}

std::wstring environment(std::wstring_view key) {
    const std::wstring name(key);
    auto size = GetEnvironmentVariableW(name.c_str(), nullptr, 0);
    if (!size) return {};
    std::wstring value(size, L'\0');
    size = GetEnvironmentVariableW(name.c_str(), value.data(), size);
    if (size >= value.size()) throw std::runtime_error("环境变量在读取期间变化。");
    value.resize(size);
    return value;
}

fs::path dataHome() {
    if (const auto custom = environment(L"FACE_MATCHING_HOME"); !custom.empty()) return localPath(custom);
    const auto appData = environment(L"LOCALAPPDATA");
    if (appData.empty()) throw std::runtime_error("找不到本机应用数据目录。");
    return localPath(fs::path(appData) / L"FaceMatching");
}

std::string utcTimestamp() {
    SYSTEMTIME time{};
    GetSystemTime(&time);
    std::array<char, 32> result{};
    std::snprintf(result.data(), result.size(), "%04u-%02u-%02uT%02u:%02u:%02u+00:00",
        time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond);
    return result.data();
}

std::string newId() {
    std::array<unsigned char, 16> bytes{};
    if (BCryptGenRandom(nullptr, bytes.data(), static_cast<ULONG>(bytes.size()), BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) {
        throw std::runtime_error("生成记录标识失败。");
    }
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (const auto b : bytes) {
        result += hex[b >> 4];
        result += hex[b & 15];
    }
    return result;
}

Json readJson(const fs::path& path, uintmax_t limit) {
    const auto file = localPath(path);
    if (!fs::is_regular_file(file) || fs::file_size(file) > limit) throw std::runtime_error("JSON 文件丢失或超过大小上限。");
    std::ifstream input(file, std::ios::binary);
    return Json::parse(input);
}

void writeJson(const fs::path& path, const Json& value) {
    const auto file = localPath(path);
    fs::create_directories(file.parent_path());
    const auto stage = file.parent_path() / (L".visual-" + wide(newId()) + L".part");
    std::ofstream output(stage, std::ios::binary | std::ios::trunc);
    output << value.dump(2, ' ', false, Json::error_handler_t::replace) << '\n';
    output.close();
    if (!output || !MoveFileExW(stage.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::error_code ignored;
        fs::remove(stage, ignored);
        throw std::runtime_error("写入本机报告失败。");
    }
}

Json featureBlob(std::span<const float> feature) {
    std::vector<uint8_t> bytes(feature.size_bytes());
    if (!bytes.empty()) std::memcpy(bytes.data(), feature.data(), bytes.size());
    return Json::binary(std::move(bytes));
}

Feature readFeature(const Json& blob, size_t dimension) {
    if (!blob.is_binary() || dimension < 1 || dimension > 10000 ||
        blob.get_binary().size() != dimension * sizeof(float)) return {};
    Feature feature(dimension);
    std::memcpy(feature.data(), blob.get_binary().data(), dimension * sizeof(float));
    for (const auto value : feature) if (!std::isfinite(value)) return {};
    return feature;
}

Database::Database(const fs::path& path) {
    const auto file = localPath(path);
    fs::create_directories(file.parent_path());
    const auto opened = sqlite3_open_v2(pathText(file).c_str(), &handle_,
                                       SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr);
    try {
        if (opened != SQLITE_OK) throw std::runtime_error(handle_ ? sqlite3_errmsg(handle_) : "无法打开数据库。");
        sqlite3_busy_timeout(handle_, 30000);
        script("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL;");
    } catch (...) {
        sqlite3_close(handle_);
        throw;
    }
}

Database::~Database() { sqlite3_close(handle_); }

void Database::script(const char* sql) {
    const auto code = sqlite3_exec(handle_, sql, nullptr, nullptr, nullptr);
    if (code != SQLITE_OK) throw std::runtime_error(sqlite3_errmsg(handle_));
}

Json Database::query(const char* sql, std::initializer_list<Json> arguments) {
    return query(sql,std::span<const Json>(arguments.begin(),arguments.size()));
}

Json Database::query(const char* sql, std::span<const Json> arguments) {
    sqlite3_stmt* raw = nullptr;
    if (sqlite3_prepare_v2(handle_, sql, -1, &raw, nullptr) != SQLITE_OK) {
        throw std::runtime_error(sqlite3_errmsg(handle_));
    }
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> statement(raw, sqlite3_finalize);
    const auto expected=static_cast<size_t>(sqlite3_bind_parameter_count(raw));
    if (arguments.size() != expected) {
        throw std::invalid_argument("数据库参数数量不符 (expected "+std::to_string(expected)+
            ", received "+std::to_string(arguments.size())+"): "+sql);
    }
    for (size_t index = 0; index < arguments.size(); ++index) {
        const auto& value = arguments[index];
        const int key = static_cast<int>(index + 1);
        int result = SQLITE_MISMATCH;
        if (value.is_null()) result = sqlite3_bind_null(raw, key);
        else if (value.is_number_integer()) result = sqlite3_bind_int64(raw, key, value.get<int64_t>());
        else if (value.is_number()) result = sqlite3_bind_double(raw, key, value.get<double>());
        else if (value.is_string()) {
            const auto& text = value.get_ref<const std::string&>();
            result = sqlite3_bind_text(raw, key, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT);
        } else if (value.is_binary()) {
            const auto& bytes = value.get_binary();
            result = sqlite3_bind_blob(raw, key, bytes.data(), static_cast<int>(bytes.size()), SQLITE_TRANSIENT);
        }
        if (result != SQLITE_OK) throw std::runtime_error("绑定数据库参数失败。");
    }
    Json rows = Json::array();
    int code{};
    while ((code = sqlite3_step(raw)) == SQLITE_ROW) {
        Json row = Json::object();
        for (int index = 0; index < sqlite3_column_count(raw); ++index) {
            const auto* name = sqlite3_column_name(raw, index);
            switch (sqlite3_column_type(raw, index)) {
            case SQLITE_INTEGER: row[name] = sqlite3_column_int64(raw, index); break;
            case SQLITE_FLOAT: row[name] = sqlite3_column_double(raw, index); break;
            case SQLITE_TEXT: row[name] = reinterpret_cast<const char*>(sqlite3_column_text(raw, index)); break;
            case SQLITE_BLOB: {
                const auto* bytes = static_cast<const uint8_t*>(sqlite3_column_blob(raw, index));
                const auto length = sqlite3_column_bytes(raw, index);
                row[name] = Json::binary(length > 0 ? std::vector<uint8_t>(bytes, bytes + length)
                                                     : std::vector<uint8_t>{});
                break;
            }
            default: row[name] = nullptr; break;
            }
        }
        rows.push_back(std::move(row));
    }
    if (code != SQLITE_DONE) throw std::runtime_error(sqlite3_errmsg(handle_));
    return rows;
}

int Database::changes() const { return sqlite3_changes(handle_); }
int64_t Database::lastId() const { return sqlite3_last_insert_rowid(handle_); }

Transaction::Transaction(Database& database) : db_(database) { db_.script("BEGIN IMMEDIATE"); }
Transaction::~Transaction() {
    if (active_) {
        try { db_.script("ROLLBACK"); } catch (...) {}
    }
}
void Transaction::commit() {
    db_.script("COMMIT");
    active_ = false;
}

void initializePeople(Database& database) {
    Transaction transaction(database);
    database.script(R"sql(
        CREATE TABLE IF NOT EXISTS persons (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, id_card TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS face_samples (
            id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
            image_path TEXT NOT NULL, embedding BLOB NOT NULL, embedding_dim INTEGER NOT NULL,
            model_id TEXT NOT NULL, quality REAL NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_samples_person ON face_samples(person_id);
        CREATE INDEX IF NOT EXISTS idx_samples_model ON face_samples(model_id);
        CREATE TABLE IF NOT EXISTS recognition_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
            source TEXT NOT NULL, track_id INTEGER NOT NULL, score REAL NOT NULL, quality REAL NOT NULL,
            occurred_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recognition_annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, source TEXT NOT NULL,
            track_id INTEGER NOT NULL, frame_index INTEGER NOT NULL, bbox TEXT, snapshot_path TEXT, face_path TEXT,
            score REAL, quality REAL, model_id TEXT,
            action TEXT NOT NULL CHECK(action IN ('reject','associate','undo_rejection','remove_association')),
            person_id TEXT REFERENCES persons(id) ON DELETE SET NULL, occurred_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_annotations_session ON recognition_annotations(session_id, track_id);
        CREATE TABLE IF NOT EXISTS portrait_libraries (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE);
        INSERT OR IGNORE INTO portrait_libraries VALUES('default','默认库');
        CREATE TABLE IF NOT EXISTS person_libraries (
            person_id TEXT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
            library_id TEXT NOT NULL REFERENCES portrait_libraries(id) ON DELETE CASCADE,
            PRIMARY KEY(person_id,library_id)
        );
    )sql");
    if (database.query("PRAGMA user_version").front().at("user_version").get<int>() < 1) {
        database.script("INSERT OR IGNORE INTO person_libraries SELECT id,'default' FROM persons; PRAGMA user_version=1");
    }
    transaction.commit();
}

}
