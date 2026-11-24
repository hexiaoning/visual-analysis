#include "local_images.h"
#include "inference_backend.h"
#include "portrait_video.h"
#include "operations.h"
#include "operations_window.h"

#include <algorithm>
#include <array>
#include <future>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <Windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <shellapi.h>
#include <opencv2/imgproc.hpp>

namespace {
namespace fs = std::filesystem;

enum Control { chooseImage = 100, chooseMode, restoreImage, preferRecall, chooseCandidate, exportImage,
               firstWorkflow = 300 };
constexpr UINT_PTR workerTimer = 1;
constexpr UINT workerPollMs = 100;
constexpr wchar_t className[] = L"VisualAnalysisNative";

struct Window {
    fs::path models;
    fs::path worker;
    fs::path videoSource;
    cv::Mat original;
    visual::Candidates candidates;
    std::future<visual::Candidates> work;
    std::shared_ptr<visual::VideoProgress> videoWork;
    std::optional<cv::Rect> region;
    std::optional<cv::Point> dragStart;
    cv::Point dragEnd;
    HWND open{}, mode{}, restore{}, recall{}, list{}, save{}, status{};
};

std::wstring label(const visual::Candidate& item) {
    std::wostringstream text;
    text << visual::wide(item.model) << L" · " << std::fixed << std::setprecision(2)
         << item.score << L" 分 · 置信度 " << (item.moderateConfidence ? L"中" : L"低");
    return text.str();
}

fs::path applicationDirectory() {
    std::wstring path(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (!length || length >= path.size()) {
        throw std::runtime_error("Could not resolve the Visual Analysis executable.");
    }
    path.resize(length);
    return fs::path(path).parent_path();
}

std::optional<fs::path> selectFile(HWND owner, bool exporting, bool video = false) {
    std::array<wchar_t, 32768> file{};
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = owner;
    dialog.lpstrFile = file.data();
    dialog.nMaxFile = static_cast<DWORD>(file.size());
    if (video) {
        dialog.lpstrFilter = exporting ? L"Matroska 视频\0*.mkv\0\0"
            : L"本机视频\0*.mp4;*.mkv;*.mov;*.avi;*.webm;*.ts;*.wmv;*.flv;*.mpg\0\0";
    } else {
        dialog.lpstrFilter = exporting ? L"PNG 图片\0*.png\0\0"
            : L"本机图片\0*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff;*.gif;*.avif\0\0";
    }
    dialog.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    dialog.lpstrDefExt = exporting ? (video ? L"mkv" : L"png") : nullptr;
    const bool selected = exporting ? GetSaveFileNameW(&dialog)
        : (dialog.Flags |= OFN_FILEMUSTEXIST, GetOpenFileNameW(&dialog));
    if (selected) {
        return fs::path(file.data());
    }
    return std::nullopt;
}

void layout(HWND handle, const Window& state) {
    RECT client{};
    GetClientRect(handle, &client);
    const int width = client.right;
    MoveWindow(state.open, 16, 14, 120, 30, TRUE);
    MoveWindow(state.mode, 144, 14, 102, 140, TRUE);
    MoveWindow(state.restore, 254, 14, 106, 30, TRUE);
    MoveWindow(state.recall, 368, 14, 82, 30, TRUE);
    MoveWindow(state.list, 458, 14, std::max(width - 606, 120), 160, TRUE);
    MoveWindow(state.save, width - 138, 14, 122, 30, TRUE);
    MoveWindow(state.status, 16, 57, std::max(width - 32, 1), 42, TRUE);
    InvalidateRect(handle, nullptr, TRUE);
}

RECT fitImage(RECT area, const cv::Mat& image) {
    area.top += 28;
    if (image.empty() || area.right <= area.left || area.bottom <= area.top) {
        return {};
    }
    const double scale = std::min(double(area.right - area.left) / image.cols,
                                  double(area.bottom - area.top) / image.rows);
    const int width = std::max(1, cvRound(image.cols * scale));
    const int height = std::max(1, cvRound(image.rows * scale));
    const LONG left = (area.left + area.right - width) / 2;
    const LONG top = (area.top + area.bottom - height) / 2;
    return {left, top, left + width, top + height};
}

RECT sourceArea(HWND handle) {
    RECT client{};
    GetClientRect(handle, &client);
    return {16, 107, client.right / 2 - 8, client.bottom - 16};
}

void drawPreview(HDC dc, RECT area, const cv::Mat& image, const wchar_t* title) {
    SetBkMode(dc, TRANSPARENT);
    DrawTextW(dc, title, -1, &area, DT_TOP | DT_SINGLELINE | DT_CENTER);
    const auto fitted = fitImage(area, image);
    area.top += 28;
    FillRect(dc, &area, GetSysColorBrush(COLOR_WINDOW));
    if (IsRectEmpty(&fitted)) {
        return;
    }
    cv::Mat bgra;
    cv::cvtColor(image, bgra, cv::COLOR_BGR2BGRA);
    BITMAPINFO bitmap{};
    bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmap.bmiHeader.biWidth = image.cols;
    bitmap.bmiHeader.biHeight = -image.rows;
    bitmap.bmiHeader.biPlanes = 1;
    bitmap.bmiHeader.biBitCount = 32;
    bitmap.bmiHeader.biCompression = BI_RGB;
    SetStretchBltMode(dc, HALFTONE);
    StretchDIBits(dc, fitted.left, fitted.top, fitted.right - fitted.left, fitted.bottom - fitted.top,
                  0, 0, image.cols, image.rows,
                  bgra.data, &bitmap, DIB_RGB_COLORS, SRCCOPY);
}

bool plateMode(const Window& state) {
    return SendMessageW(state.mode, CB_GETCURSEL, 0, 0) == 1;
}

bool videoMode(const Window& state) {
    return SendMessageW(state.mode, CB_GETCURSEL, 0, 0) == 2;
}

void drawRegion(HDC dc, HWND handle, const Window& state) {
    std::optional<cv::Rect> region = state.region;
    if (state.dragStart) {
        region = cv::Rect(*state.dragStart, state.dragEnd);
    }
    if (!region || state.original.empty()) {
        return;
    }
    const auto fitted = fitImage(sourceArea(handle), state.original);
    const auto scaleX = double(fitted.right - fitted.left) / state.original.cols;
    const auto scaleY = double(fitted.bottom - fitted.top) / state.original.rows;
    const auto pen = CreatePen(PS_SOLID, 2, RGB(0, 184, 240));
    const auto oldPen = SelectObject(dc, pen);
    const auto oldBrush = SelectObject(dc, GetStockObject(NULL_BRUSH));
    Rectangle(dc, fitted.left + cvRound(region->x * scaleX), fitted.top + cvRound(region->y * scaleY),
              fitted.left + cvRound(region->br().x * scaleX), fitted.top + cvRound(region->br().y * scaleY));
    SelectObject(dc, oldBrush);
    SelectObject(dc, oldPen);
    DeleteObject(pen);
}

const visual::Candidate* selection(const Window& state) {
    const auto index = SendMessageW(state.list, CB_GETCURSEL, 0, 0);
    return index >= 0 && size_t(index) < state.candidates.size() ? &state.candidates[index] : nullptr;
}

void refresh(HWND handle, Window& state) {
    SendMessageW(state.list, CB_RESETCONTENT, 0, 0);
    for (const auto& item : state.candidates) {
        const auto text = label(item);
        SendMessageW(state.list, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(text.c_str()));
    }
    if (!state.candidates.empty()) {
        SendMessageW(state.list, CB_SETCURSEL, 0, 0);
    }
    EnableWindow(state.open, !state.work.valid());
    EnableWindow(state.mode, !state.work.valid());
    EnableWindow(state.recall, !state.work.valid() && !plateMode(state) && !videoMode(state));
    const bool ready = videoMode(state) ? !state.videoSource.empty()
        : !state.original.empty() && (!plateMode(state) || state.region.has_value());
    const bool canCancel = state.videoWork && !state.videoWork->cancel;
    EnableWindow(state.restore, canCancel || (!state.work.valid() && ready));
    SetWindowTextW(state.restore, state.videoWork ? L"取消处理" : L"离线还原");
    SetWindowTextW(state.open, videoMode(state) ? L"选择本机视频" : L"选择本机图片");
    EnableWindow(state.save, !state.candidates.empty());
    InvalidateRect(handle, nullptr, TRUE);
}

LRESULT CALLBACK dispatch(HWND handle, UINT message, WPARAM wparam, LPARAM lparam) {
    auto* state = reinterpret_cast<Window*>(GetWindowLongPtrW(handle, GWLP_USERDATA));
    switch (message) {
    case WM_NCCREATE: {
        const auto* create = reinterpret_cast<const CREATESTRUCTW*>(lparam);
        SetWindowLongPtrW(handle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create->lpCreateParams));
        return TRUE;
    }
    case WM_CREATE: {
        auto child = [&](const wchar_t* type, const wchar_t* name, DWORD style, UINT_PTR id) {
            auto result = CreateWindowExW(0, type, name, WS_CHILD | WS_VISIBLE | style, 0, 0, 0, 0,
                handle, reinterpret_cast<HMENU>(id), nullptr, nullptr);
            SendMessageW(result, WM_SETFONT, reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
            return result;
        };
        state->open = child(L"BUTTON", L"选择本机图片", WS_TABSTOP, chooseImage);
        state->mode = child(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, chooseMode);
        SendMessageW(state->mode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"人像"));
        SendMessageW(state->mode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"车牌"));
        SendMessageW(state->mode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"人像视频"));
        SendMessageW(state->mode, CB_SETCURSEL, 0, 0);
        state->restore = child(L"BUTTON", L"离线还原", WS_TABSTOP, restoreImage);
        state->recall = child(L"BUTTON", L"偏召回", BS_AUTOCHECKBOX | WS_TABSTOP, preferRecall);
        state->list = child(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP | WS_VSCROLL, chooseCandidate);
        state->save = child(L"BUTTON", L"保存候选 PNG", WS_TABSTOP, exportImage);
        state->status = child(L"STATIC", L"选择一张本机图片，使用 PCFlow 和 GFPGAN 生成人像还原候选。", SS_LEFT, 0);
        refresh(handle, *state);
        layout(handle, *state);
        return 0;
    }
    case WM_SIZE:
        if (state && state->status) layout(handle, *state);
        return 0;
    case WM_GETMINMAXINFO: {
        auto* info = reinterpret_cast<MINMAXINFO*>(lparam);
        info->ptMinTrackSize = {760, 400};
        return 0;
    }
    case visual::loadSearchRestoration: {
        if(state->work.valid() || !lparam) return FALSE;
        const auto& selected=*reinterpret_cast<const visual::SearchRestorationInput*>(lparam);
        if(selected.image.empty()) return FALSE;
        SendMessageW(state->mode,CB_SETCURSEL,0,0);
        state->original=selected.image;
        state->videoSource.clear();
        state->candidates.clear();
        state->region=selected.region&cv::Rect(0,0,selected.image.cols,selected.image.rows);
        if(state->region->empty()) state->region.reset();
        SetWindowTextW(state->status,L"已加载命中画面，可按选区还原。");
        refresh(handle,*state);
        return TRUE;
    }
    case WM_COMMAND:
        try {
            const auto command=LOWORD(wparam);
            if(command>=firstWorkflow && command<firstWorkflow+visual::workflowNames().size()) {
                visual::openWorkflow(handle,state->models,state->worker,command-firstWorkflow);
                return 0;
            }
            switch (LOWORD(wparam)) {
            case chooseImage:
                if (!state->work.valid()) {
                    const bool video = videoMode(*state);
                    if (const auto path = selectFile(handle, false, video)) {
                        if (video) {
                            state->videoSource = visual::localPath(*path);
                            state->original.release();
                        } else {
                            state->original = visual::readImage(*path);
                            state->videoSource.clear();
                        }
                        state->candidates.clear();
                        state->region.reset();
                        SetWindowTextW(state->status, plateMode(*state)
                            ? L"在原图上拖动鼠标框选车牌。" : path->filename().c_str());
                        refresh(handle, *state);
                    }
                }
                break;
            case restoreImage:
                if (state->videoWork) {
                    state->videoWork->cancel = true;
                    refresh(handle, *state);
                } else if (videoMode(*state) && !state->videoSource.empty()) {
                    if (const auto target = selectFile(handle, true, true)) {
                        auto progress = std::make_shared<visual::VideoProgress>();
                        state->work = std::async(std::launch::async, [input = state->videoSource, output = *target,
                                                                     models = state->models, progress] {
                            visual::restorePortraitVideo(input, output, models, *progress);
                            return visual::Candidates{};
                        });
                        state->videoWork = std::move(progress);
                        SetTimer(handle, workerTimer, workerPollMs, nullptr);
                        SetWindowTextW(state->status, L"正在离线处理视频，输出保留原始音轨与帧时间戳。");
                        refresh(handle, *state);
                    }
                } else if (!state->work.valid() && !state->original.empty()) {
                    SetTimer(handle, workerTimer, workerPollMs, nullptr);
                    const bool recall = SendMessageW(state->recall, BM_GETCHECK, 0, 0) == BST_CHECKED;
                    const bool plate=plateMode(*state);
                    state->work = std::async(std::launch::async, [image = state->original, models = state->models,
                                                                 region = state->region, recall, plate] {
                        if (plate) return visual::restorePlate(image, *region);
                        auto results = visual::restorePortraits({image}, visual::localPath(models), region, recall);
                        return std::move(results.front());
                    });
                    state->candidates.clear();
                    SetWindowTextW(state->status, L"正在本机执行图像还原…");
                    refresh(handle, *state);
                }
                break;
            case chooseMode:
                if (HIWORD(wparam) == CBN_SELCHANGE) {
                    state->candidates.clear();
                    state->region.reset();
                    SetWindowTextW(state->status,
                        videoMode(*state) ? L"选择本机视频并导出 MKV。"
                        : plateMode(*state) ? L"打开本机图片，再在原图上拖动鼠标框选车牌。"
                                            : L"选择本机图片，使用 PCFlow 和 GFPGAN 生成人像还原候选。");
                    refresh(handle, *state);
                }
                break;
            case chooseCandidate:
                InvalidateRect(handle, nullptr, TRUE);
                break;
            case preferRecall:
                SetWindowTextW(state->status, L"偏召回会多试去屏纹输入。候选仍需通过原图一致性与差异筛选。");
                break;
            case exportImage:
                if (const auto* candidate = selection(*state)) {
                    if (const auto path = selectFile(handle, true)) {
                        visual::writePng(*path, candidate->pixels);
                        SetWindowTextW(state->status, L"已保存还原候选。");
                    }
                }
                break;
            }
        } catch (const std::exception& error) {
            SetWindowTextW(state->status, visual::wide(error.what()).c_str());
        }
        return 0;
    case WM_LBUTTONDOWN:
    case WM_MOUSEMOVE:
    case WM_LBUTTONUP: {
        if (state->work.valid() || state->original.empty() || videoMode(*state)) break;
        const auto area = fitImage(sourceArea(handle), state->original);
        if (IsRectEmpty(&area)) break;
        const POINT cursor{GET_X_LPARAM(lparam), GET_Y_LPARAM(lparam)};
        const auto point = cv::Point(
            std::clamp(cvRound(double(cursor.x - area.left) * state->original.cols / (area.right - area.left)),
                       0, state->original.cols),
            std::clamp(cvRound(double(cursor.y - area.top) * state->original.rows / (area.bottom - area.top)),
                       0, state->original.rows));
        if (message == WM_LBUTTONDOWN && PtInRect(&area, cursor)) {
            state->dragStart = state->dragEnd = point;
            SetCapture(handle);
        } else if (state->dragStart) {
            state->dragEnd = point;
            if (message == WM_LBUTTONUP) {
                auto region = cv::Rect(*state->dragStart, point);
                state->dragStart.reset();
                ReleaseCapture();
                const bool validSize = !region.empty() &&
                    (!plateMode(*state) || (region.width>=16 && region.width<=1024 && region.height>=8 && region.height<=512));
                state->region = validSize ? std::optional(region) : std::nullopt;
                state->candidates.clear();
                SetWindowTextW(state->status, validSize ? L"已框选区域，点击离线还原。"
                    : L"车牌选区宽度须为 16～1024 像素，高度须为 8～512 像素。");
                refresh(handle, *state);
            }
        }
        if (state->dragStart) InvalidateRect(handle, nullptr, TRUE);
        return 0;
    }
    case WM_CAPTURECHANGED:
        if (state->dragStart) {
            state->dragStart.reset();
            InvalidateRect(handle, nullptr, TRUE);
        }
        return 0;
    case WM_TIMER:
        if (wparam == workerTimer && state->work.valid() &&
            state->work.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            KillTimer(handle, workerTimer);
            auto video = std::move(state->videoWork);
            try {
                state->candidates = state->work.get();
                SetWindowTextW(state->status, video ? (video->cancel ? L"视频处理已取消。" : L"已保存还原视频。")
                    : state->candidates.empty()
                    ? L"未找到可对齐的人脸。"
                    : plateMode(*state) ? L"分数用于排列去拖影候选。请与原始车牌区域并列核对。"
                    : L"分数用于排列同源候选。置信度反映原图和两模型支持程度。");
            } catch (const std::exception& error) {
                SetWindowTextW(state->status, visual::wide(error.what()).c_str());
            }
            refresh(handle, *state);
        } else if (state->videoWork) {
            const auto& progress = *state->videoWork;
            const auto status = progress.cancel.load() ? L"正在取消视频处理。"
                : L"已处理 " + std::to_wstring(progress.frames) + L" 帧，已还原 "
                + std::to_wstring(progress.faces) + L" 处人脸。";
            SetWindowTextW(state->status, status.c_str());
        }
        return 0;
    case WM_PAINT: {
        PAINTSTRUCT paint{};
        const auto dc = BeginPaint(handle, &paint);
        RECT client{};
        GetClientRect(handle, &client);
        const auto* candidate = selection(*state);
        drawPreview(dc, sourceArea(handle), state->original, L"原图");
        drawRegion(dc, handle, *state);
        drawPreview(dc, {client.right / 2 + 8, 107, client.right - 16, client.bottom - 16},
                    candidate ? candidate->pixels : cv::Mat{}, L"还原候选");
        EndPaint(handle, &paint);
        return 0;
    }
    case WM_CLOSE:
        if (state->work.valid()) {
            if (state->videoWork) state->videoWork->cancel = true;
            SetWindowTextW(state->status, L"当前模型正在本机运行，完成后可关闭。");
            return 0;
        }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(handle, message, wparam, lparam);
}

int showWindow(HINSTANCE module, int show, const fs::path& models) {
    WNDCLASSW windowClass{};
    windowClass.hInstance = module;
    windowClass.lpfnWndProc = dispatch;
    windowClass.lpszClassName = className;
    windowClass.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    windowClass.hbrBackground = GetSysColorBrush(COLOR_BTNFACE);
    if (!RegisterClassW(&windowClass)) {
        throw std::runtime_error("Could not register the visual analysis window.");
    }
    Window state;
    state.models = models;
    state.worker = applicationDirectory() / L"VehicleCatalogueWorker.exe";
    const auto menu=CreateMenu();
    for(size_t index=0;index<visual::workflowNames().size();++index) {
        AppendMenuW(menu,MF_STRING,firstWorkflow+index,visual::workflowNames()[index]);
    }
    const auto caption=L"Visual Analysis · "+visual::inferenceBackendLabel();
    const auto handle = CreateWindowExW(0, className, caption.c_str(),
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1060, 700, nullptr, menu, module, &state);
    if (!handle) {
        DestroyMenu(menu);
        throw std::runtime_error("Could not create the visual analysis window.");
    }
    ShowWindow(handle, show);
    MSG message{};
    BOOL status{};
    while ((status = GetMessageW(&message, nullptr, 0, 0)) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    return status < 0 ? 2 : static_cast<int>(message.wParam);
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
