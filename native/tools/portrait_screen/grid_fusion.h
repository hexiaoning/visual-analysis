#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>

namespace visual::screen {

std::size_t runGridFusion(std::istream& manifest, const std::filesystem::path& output);

}
