#pragma once

#include <filesystem>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

struct sqlite3;

namespace visual {

using Json = nlohmann::json;
using Feature = std::vector<float>;

std::string utf8(std::wstring_view text);
std::string pathText(const std::filesystem::path& path);
std::filesystem::path textPath(std::string_view text);
std::filesystem::path dataHome();
std::wstring environment(std::wstring_view key);
std::string utcTimestamp();
std::string newId();
Json readJson(const std::filesystem::path& path, uintmax_t limit = 16 * 1024 * 1024);
void writeJson(const std::filesystem::path& path, const Json& value);
Json featureBlob(std::span<const float> feature);
Feature readFeature(const Json& blob, size_t dimension);

class Database {
public:
    explicit Database(const std::filesystem::path& path);
    ~Database();

    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;

    void script(const char* sql);
    Json query(const char* sql, std::initializer_list<Json> arguments = {});
    Json query(const char* sql, std::span<const Json> arguments);
    int changes() const;
    int64_t lastId() const;
private:
    sqlite3* handle_{};
};

class Transaction {
public:
    explicit Transaction(Database& database);
    ~Transaction();
    void commit();
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
private:
    Database& db_;
    bool active_ = true;
};

void initializePeople(Database& database);

}
