#include "local_images.h"
#include "inference_backend.h"
#include "portrait_video.h"
#include "operations.h"
#include "operations_window.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#include <Windows.h>
#include <shellapi.h>

namespace {
namespace fs = std::filesystem;
using visual::WorkflowPage;

constexpr wchar_t className[] = L"VisualAnalysisNative";
constexpr UINT primaryControl=100, secondaryControl=200, settingsControl=300;

struct Tab {
    const wchar_t* name;
    std::array<WorkflowPage,3> pages;
    size_t count;
};
constexpr std::array tabs{
    Tab{L"人像检索",{WorkflowPage::FaceSearch},1},
    Tab{L"目标查找",{WorkflowPage::ObjectDetection,WorkflowPage::VisualSearch,WorkflowPage::SemanticSearch},3},
    Tab{L"车辆分析",{WorkflowPage::PlateRecognition,WorkflowPage::VehicleAnalysis},2},
    Tab{L"人像工具",{WorkflowPage::FaceCompare,WorkflowPage::PortraitRestoration},2},
    Tab{L"警情筛选",{WorkflowPage::IncidentScreen},1},
    Tab{L"资料库",{WorkflowPage::People,WorkflowPage::Annotations},2},
};

struct Window {
    fs::path models,worker;
    std::array<HWND,tabs.size()> primary{};
    std::array<HWND,3> secondary{};
    std::array<HWND,static_cast<size_t>(WorkflowPage::Count)> pages{};
    std::array<size_t,tabs.size()> selectedPage{};
    HWND settings{},placeholder{};
    size_t selectedTab{};
    bool settingsOpen{};
};

fs::path applicationDirectory() {
    std::wstring path(32768,L'\0');
    const auto length=GetModuleFileNameW(nullptr,path.data(),static_cast<DWORD>(path.size()));
    if(!length || length>=path.size()) throw std::runtime_error("无法定位程序目录。");
    path.resize(length);
    return fs::path(path).parent_path();
}

HWND button(HWND handle,const wchar_t* label,UINT id) {
    auto control=CreateWindowExW(0,L"BUTTON",label,WS_CHILD|WS_TABSTOP|BS_CHECKBOX|BS_PUSHLIKE,
                                 0,0,0,0,handle,reinterpret_cast<HMENU>(static_cast<UINT_PTR>(id)),nullptr,nullptr);
    SendMessageW(control,WM_SETFONT,reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)),TRUE);
    return control;
}

WorkflowPage selected(const Window& state) {
    return state.settingsOpen?WorkflowPage::Settings:tabs[state.selectedTab].pages[state.selectedPage[state.selectedTab]];
}

void layout(HWND handle,const Window& state) {
    RECT client{};
    GetClientRect(handle,&client);
    constexpr int inset=12, height=34;
    const int width=client.right;
    const int navWidth=std::max(0,width-2*inset-86);
    for(size_t i=0;i<tabs.size();++i) {
        const int left=inset+static_cast<int>(i)*navWidth/static_cast<int>(tabs.size());
        const int right=inset+static_cast<int>(i+1)*navWidth/static_cast<int>(tabs.size());
        MoveWindow(state.primary[i],left,inset,std::max(0,right-left-5),height,TRUE);
    }
    MoveWindow(state.settings,width-inset-76,inset,76,height,TRUE);
    const auto page=selected(state);
    const bool secondLine=!state.settingsOpen && tabs[state.selectedTab].count>1;
    for(size_t i=0;i<state.secondary.size();++i) {
        MoveWindow(state.secondary[i],inset+static_cast<int>(i)*140,58,134,30,TRUE);
    }
    const int top=secondLine?94:54;
    if(page==WorkflowPage::PortraitRestoration) {
        MoveWindow(state.placeholder,inset,top,width-2*inset,std::max(0,int(client.bottom)-top-inset),TRUE);
    } else if(auto child=state.pages[static_cast<size_t>(page)]) {
        MoveWindow(child,inset,top,width-2*inset,std::max(0,int(client.bottom)-top-inset),TRUE);
    }
}

void activate(HWND handle,Window& state) {
    const auto page=selected(state);
    if(page!=WorkflowPage::PortraitRestoration) {
        auto& child=state.pages[static_cast<size_t>(page)];
        if(!child) child=visual::createWorkflowPage(handle,state.models,state.worker,page);
    }
    const auto& tab=tabs[state.selectedTab];
    for(size_t i=0;i<state.primary.size();++i) {
        SendMessageW(state.primary[i],BM_SETCHECK,!state.settingsOpen && i==state.selectedTab?BST_CHECKED:BST_UNCHECKED,0);
        ShowWindow(state.primary[i],SW_SHOW);
    }
    SendMessageW(state.settings,BM_SETCHECK,state.settingsOpen?BST_CHECKED:BST_UNCHECKED,0);
    ShowWindow(state.settings,SW_SHOW);
    for(size_t i=0;i<state.secondary.size();++i) {
        const bool show=!state.settingsOpen && tab.count>1 && i<tab.count;
        if(show) {
            SetWindowTextW(state.secondary[i],visual::workflowName(tab.pages[i]));
            SendMessageW(state.secondary[i],BM_SETCHECK,i==state.selectedPage[state.selectedTab]?BST_CHECKED:BST_UNCHECKED,0);
        }
        ShowWindow(state.secondary[i],show?SW_SHOW:SW_HIDE);
    }
    for(size_t i=0;i<state.pages.size();++i) {
        if(state.pages[i]) ShowWindow(state.pages[i],i==static_cast<size_t>(page)?SW_SHOW:SW_HIDE);
    }
    ShowWindow(state.placeholder,page==WorkflowPage::PortraitRestoration?SW_SHOW:SW_HIDE);
    layout(handle,state);
}

LRESULT CALLBACK dispatch(HWND handle,UINT message,WPARAM wparam,LPARAM lparam) {
    auto* state=reinterpret_cast<Window*>(GetWindowLongPtrW(handle,GWLP_USERDATA));
    switch(message) {
    case WM_NCCREATE: {
        const auto* create=reinterpret_cast<const CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(handle,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(create->lpCreateParams));
        return TRUE;
    }
    case WM_CREATE:
        try {
            for(size_t i=0;i<tabs.size();++i) state->primary[i]=button(handle,tabs[i].name,primaryControl+static_cast<UINT>(i));
            for(size_t i=0;i<state->secondary.size();++i) state->secondary[i]=button(handle,L"",secondaryControl+static_cast<UINT>(i));
            state->settings=button(handle,L"设置",settingsControl);
            state->placeholder=CreateWindowExW(0,L"STATIC",L"开发中",WS_CHILD|SS_CENTER|SS_CENTERIMAGE,
                                                0,0,0,0,handle,nullptr,nullptr,nullptr);
            SendMessageW(state->placeholder,WM_SETFONT,reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)),TRUE);
            activate(handle,*state);
            return 0;
        } catch(const std::exception&) {return -1;}
    case WM_SIZE:
        if(state && state->placeholder) layout(handle,*state);
        return 0;
    case WM_GETMINMAXINFO:
        reinterpret_cast<MINMAXINFO*>(lparam)->ptMinTrackSize={860,690};
        return 0;
    case WM_COMMAND:
        if(state && HIWORD(wparam)==BN_CLICKED) {
            const auto command=LOWORD(wparam);
            try {
                if(command>=primaryControl && command<primaryControl+tabs.size()) {
                    state->selectedTab=command-primaryControl;
                    state->settingsOpen=false;
                } else if(command>=secondaryControl && command<secondaryControl+tabs[state->selectedTab].count) {
                    state->selectedPage[state->selectedTab]=command-secondaryControl;
                } else if(command==settingsControl) {
                    state->settingsOpen=true;
                } else break;
                activate(handle,*state);
            } catch(const std::exception& error) {
                MessageBoxW(handle,visual::wide(error.what()).c_str(),L"Visual Analysis",MB_OK|MB_ICONERROR);
            }
            return 0;
        }
        break;
    case WM_CLOSE:
        if(state) {
            bool busy=false;
            for(const auto page:state->pages) if(page && visual::cancelWorkflow(page)) busy=true;
            if(busy) return 0;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(handle,message,wparam,lparam);
}

int showWindow(HINSTANCE module,int show,const fs::path& models) {
    WNDCLASSW type{};
    type.hInstance=module;
    type.lpfnWndProc=dispatch;
    type.lpszClassName=className;
    type.hCursor=LoadCursorW(nullptr,IDC_ARROW);
    type.hbrBackground=GetSysColorBrush(COLOR_BTNFACE);
    if(!RegisterClassW(&type)) throw std::runtime_error("无法注册主窗口。");

    Window state;
    state.models=models;
    state.worker=applicationDirectory()/L"VehicleCatalogueWorker.exe";
    const auto caption=L"Visual Analysis v2.14.0 · "+visual::inferenceBackendLabel();
    const auto handle=CreateWindowExW(WS_EX_CONTROLPARENT,className,caption.c_str(),
        WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,CW_USEDEFAULT,1060,800,nullptr,nullptr,module,&state);
    if(!handle) throw std::runtime_error("无法创建主窗口。");
    ShowWindow(handle,show);
    MSG message{};
    BOOL status{};
    while((status=GetMessageW(&message,nullptr,0,0))>0) {
        if(!IsDialogMessageW(handle,&message)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    return status<0?2:static_cast<int>(message.wParam);
}

}


int WINAPI wWinMain(HINSTANCE module, HINSTANCE, PWSTR, int show) {
    // Suppress modal errors during scripted comparison runs.
    bool commandLine = false;
    try {
        fs::path models = applicationDirectory() / L"models";
        int count{};
        auto raw = CommandLineToArgvW(GetCommandLineW(), &count);
        if (!raw) {
            return 2;
        }
        std::vector<std::wstring> args(raw + 1, raw + count);
        LocalFree(raw);
        commandLine = !args.empty();
        fs::path inputs, outputs, report, video, videoOutput, operationRequest;
        std::string operation;
        bool recall = false;
        for (size_t index = 0; index < args.size(); ++index) {
            const auto flag = args[index];
            if (flag == L"--prefer-recall") {
                recall = true;
                continue;
            }
            if (++index == args.size()) {
                throw std::runtime_error("Missing command-line path.");
            }
            if (flag == L"--check-restoration-cases") inputs = args[index];
            else if (flag == L"--case-output") outputs = args[index];
            else if (flag == L"--report") report = args[index];
            else if (flag == L"--models") models = args[index];
            else if (flag == L"--restore-portrait-video") video = args[index];
            else if (flag == L"--video-output") videoOutput = args[index];
            else if (flag == L"--native-operation") operation = visual::utf8(args[index]);
            else if (flag == L"--request") operationRequest = args[index];
            else throw std::runtime_error("Unknown command-line argument.");
        }
        if (commandLine) {
            if (!operation.empty() && !operationRequest.empty() && !report.empty()) {
                try {
                    std::atomic<bool> cancel=false;
                    auto result=visual::runOperation(operation,visual::readJson(operationRequest),visual::localPath(models),
                        applicationDirectory()/L"VehicleCatalogueWorker.exe",cancel);
                    visual::writeJson(report,{{"ok",true},{"result",std::move(result)}});
                    return 0;
                } catch(const std::exception& error) {
                    visual::writeJson(report,{{"ok",false},{"error",error.what()}});
                    return 2;
                }
            }
            if (!video.empty() && !videoOutput.empty() && inputs.empty()) {
                visual::VideoProgress progress;
                visual::restorePortraitVideo(video, videoOutput, models, progress);
                return progress.frames > 0 ? 0 : 2;
            }
            if (inputs.empty() || outputs.empty() || report.empty()) {
                return 2;
            }
            return visual::runCases(inputs, outputs, report, visual::localPath(models), recall);
        }
        return showWindow(module, show, models);
    } catch (const std::exception& error) {
        if (!commandLine) {
            MessageBoxW(nullptr, visual::wide(error.what()).c_str(), L"Visual Analysis", MB_OK | MB_ICONERROR);
        }
        return 2;
    }
}
