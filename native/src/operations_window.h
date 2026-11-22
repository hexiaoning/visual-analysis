#pragma once

#include <filesystem>
#include <span>
#include <Windows.h>

namespace visual {

std::span<const wchar_t* const> workflowNames();
void openWorkflow(HWND parent,const std::filesystem::path& models,const std::filesystem::path& worker,size_t workflow);

}
