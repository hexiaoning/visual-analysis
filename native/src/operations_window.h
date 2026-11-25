#pragma once

#include <filesystem>
#include <Windows.h>

namespace visual {

enum class WorkflowPage : size_t {
    VehicleAnalysis, FaceSearch, ObjectDetection, VisualSearch, SemanticSearch,
    PlateRecognition, People, Annotations, Settings, FaceCompare, IncidentScreen, PortraitRestoration, Count
};

const wchar_t* workflowName(WorkflowPage workflow);
HWND createWorkflowPage(HWND parent,const std::filesystem::path& models,
                        const std::filesystem::path& worker,WorkflowPage workflow);
bool cancelWorkflow(HWND handle);

}
