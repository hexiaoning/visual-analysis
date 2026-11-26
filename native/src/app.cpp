#include "local_images.h"
#include "portrait_video.h"
#include "operations.h"
#include "operations_window.h"
#include "workspace_theme.h"
#include "inference_backend.h"

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
namespace theme = visual::theme;

enum Control { chooseImage = 100, chooseMode, restoreImage, preferRecall, chooseCandidate, exportImage,
               firstWorkflow = 300 };
constexpr UINT_PTR workerTimer = 1;
constexpr UINT workerPollMs = 100;
constexpr wchar_t className[] = L"VisualAnalysisNative";
constexpr size_t restorationTab=3;
constexpr int headingHeight=48;
constexpr int navigationHeight=38;
constexpr int contentTop=headingHeight+navigationHeight+12;
constexpr int inset=12;
struct Tab {const wchar_t* title;int page;};
constexpr std::array tabs{
    Tab{L"人脸检索",1},Tab{L"人车检测",2},Tab{L"车型识别",0},Tab{L"图像还原",-1},
    Tab{L"视频车牌",5},Tab{L"以图搜索",3},Tab{L"语义搜索",4},Tab{L"人员库",6},Tab{L"标注库",7},Tab{L"设置",8},
};
constexpr size_t workflowCount=tabs.size()-1;

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
    std::array<HWND,tabs.size()> navigation{};
    std::array<HWND,workflowCount> pages{};
    size_t activeTab=restorationTab;
    int page=-1;
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

int previewSplit(const RECT& client) {
    return client.right*45/100;
}

void layout(HWND handle, const Window& state) {
    RECT client{};
    GetClientRect(handle, &client);
    const int width = client.right;
    const auto space=width-2*inset;
    for(size_t index=0;index<tabs.size();++index) {
        const int left=inset+int(index)*space/int(tabs.size());
        const int right=inset+int(index+1)*space/int(tabs.size());
        MoveWindow(state.navigation[index],left,headingHeight,right-left,navigationHeight,TRUE);
    }
    const int toolY=contentTop+theme::panelHeaderHeight+8;
    const int right=previewSplit(client)+inset+8;
    MoveWindow(state.open,22,toolY,123,28,TRUE);
    MoveWindow(state.mode,151,toolY+2,90,145,TRUE);
    MoveWindow(state.recall,248,toolY,67,28,TRUE);
    MoveWindow(state.restore,320,toolY,80,28,TRUE);
    MoveWindow(state.list,right,toolY+2,std::max(width-150-right,1),166,TRUE);
    MoveWindow(state.save,width-142,toolY,120,28,TRUE);
    MoveWindow(state.status,22,client.bottom-33,std::max(width-44,1),25,TRUE);
    if(state.page>=0 && state.pages[state.page]) {
        MoveWindow(state.pages[state.page],inset,contentTop,space,std::max(int(client.bottom)-contentTop-inset,1),TRUE);
    }
    InvalidateRect(handle, nullptr, TRUE);
}

void activate(HWND owner,Window& state,size_t tab,int page) {
    // Create the destination before hiding the current workspace.
    if(page>=0 && !state.pages[page]) state.pages[page]=visual::createWorkflowPage(owner,state.models,state.worker,page);
    if(GetCapture()==owner) ReleaseCapture();
    state.activeTab=tab;
    state.page=page;
    layout(owner,state);
    for(size_t index=0;index<state.pages.size();++index) {
        if(state.pages[index]) ShowWindow(state.pages[index],page==int(index)?SW_SHOW:SW_HIDE);
    }
    for(const auto item:{state.open,state.mode,state.restore,state.recall,state.list,state.save,state.status}) {
        ShowWindow(item,page<0?SW_SHOW:SW_HIDE);
    }
    for(const auto item:state.navigation) InvalidateRect(item,nullptr,TRUE);
}

RECT fitImage(RECT area, const cv::Mat& image) {
    InflateRect(&area,-10,-10);
    area.top += theme::panelHeaderHeight+43;
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
    return {inset,contentTop,previewSplit(client),client.bottom-40};
}

void drawPreview(HDC dc, RECT area, const cv::Mat& image, const wchar_t* title) {
    theme::panel(dc,area,title);
    const auto toolbarBottom=area.top+theme::panelHeaderHeight+43;
    theme::fill(dc,{area.left+1,toolbarBottom,area.right-1,toolbarBottom+1},theme::border);
    const auto fitted = fitImage(area, image);
    if (IsRectEmpty(&fitted)) {
        theme::text(dc,wcscmp(title,L"原图")==0?L"选择本机图片开始":L"还原结果将显示在这里",
                    {area.left+18,toolbarBottom+1,area.right-18,area.bottom-12},
                    theme::Font::Body,theme::muted,DT_CENTER|DT_VCENTER|DT_SINGLELINE);
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
            theme::styleControl(result);
            return result;
        };
        for(size_t index=0;index<tabs.size();++index) {
            state->navigation[index]=child(L"BUTTON",tabs[index].title,WS_TABSTOP|BS_OWNERDRAW,firstWorkflow+index);
        }
        state->open = child(L"BUTTON", L"选择本机图片", WS_TABSTOP|BS_OWNERDRAW, chooseImage);
        state->mode = child(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP, chooseMode);
        SendMessageW(state->mode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"人像"));
        SendMessageW(state->mode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"车牌"));
        SendMessageW(state->mode, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"人像视频"));
        SendMessageW(state->mode, CB_SETCURSEL, 0, 0);
        state->restore = child(L"BUTTON", L"离线还原", WS_TABSTOP|BS_OWNERDRAW, restoreImage);
        state->recall = child(L"BUTTON", L"偏召回", BS_AUTOCHECKBOX | WS_TABSTOP, preferRecall);
        state->list = child(L"COMBOBOX", L"", CBS_DROPDOWNLIST | WS_TABSTOP | WS_VSCROLL, chooseCandidate);
        SendMessageW(state->list,CB_SETDROPPEDWIDTH,460,0);
        state->save = child(L"BUTTON", L"保存候选 PNG", WS_TABSTOP|BS_OWNERDRAW, exportImage);
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
        info->ptMinTrackSize = {950, 670};
        return 0;
    }
    case WM_DRAWITEM: {
        const auto* item=reinterpret_cast<const DRAWITEMSTRUCT*>(lparam);
        if(item->CtlID>=firstWorkflow && item->CtlID<firstWorkflow+tabs.size()) {
            theme::drawButton(*item,item->CtlID-firstWorkflow==state->activeTab
                ?theme::Button::SelectedNavigation:theme::Button::Navigation);
        } else {
            theme::drawButton(*item,item->CtlID==restoreImage?theme::Button::Primary:theme::Button::Quiet);
        }
        return TRUE;
    }
    case WM_CTLCOLORSTATIC:
    case WM_CTLCOLORBTN:
    case WM_CTLCOLOREDIT:
    case WM_CTLCOLORLISTBOX: {
        const auto dc=reinterpret_cast<HDC>(wparam);
        const auto control=reinterpret_cast<HWND>(lparam);
        SetTextColor(dc,control==state->status?theme::muted:theme::textColor);
        SetBkColor(dc,theme::white);
        return reinterpret_cast<LRESULT>(theme::surface());
    }
    case visual::loadSearchRestoration: {
        if(state->work.valid() || !lparam) return FALSE;
        const auto& selected=*reinterpret_cast<const visual::SearchRestorationInput*>(lparam);
        if(selected.image.empty()) return FALSE;
        activate(handle,*state,restorationTab,-1);
        SetFocus(state->open);
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
            if(command>=firstWorkflow && command<firstWorkflow+tabs.size()) {
                const auto tab=command-firstWorkflow;
                activate(handle,*state,tab,tabs[tab].page);
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
            const auto details=visual::wide(error.what());
            if(LOWORD(wparam)>=firstWorkflow) MessageBoxW(handle,details.c_str(),L"视频图像分析",MB_OK|MB_ICONERROR);
            else SetWindowTextW(state->status,details.c_str());
        }
        return 0;
    case WM_LBUTTONDOWN:
    case WM_MOUSEMOVE:
    case WM_LBUTTONUP: {
        if (state->page>=0 || state->work.valid() || state->original.empty() || videoMode(*state)) break;
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
        theme::fill(dc,client,theme::canvas);
        theme::fill(dc,{0,0,client.right,headingHeight},theme::blue);
        theme::fill(dc,{0,headingHeight,client.right,headingHeight+navigationHeight},theme::blueDark);
        theme::text(dc,L"视频图像分析",{22,0,client.right-150,headingHeight},theme::Font::Brand,
                    theme::white,DT_LEFT|DT_VCENTER|DT_SINGLELINE);
        theme::text(dc,visual::inferenceBackendLabel().c_str(),
                    {client.right*2/5,0,client.right-22,headingHeight},theme::Font::Body,
                    theme::white,DT_RIGHT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
        if(state->page<0) {
            theme::card(dc,{inset,client.bottom-36,client.right-inset,client.bottom-8});
            const auto* candidate = selection(*state);
            drawPreview(dc,sourceArea(handle),state->original,L"原图");
            drawRegion(dc,handle,*state);
            drawPreview(dc,{previewSplit(client)+inset,contentTop,client.right-inset,client.bottom-40},
                        candidate?candidate->pixels:cv::Mat{},L"还原候选");
        }
        EndPaint(handle, &paint);
        return 0;
    }
    case WM_CLOSE: {
        auto pendingTab=tabs.size();
        for(size_t index=0;index<tabs.size();++index) {
            const int page=tabs[index].page;
            if(page>=0 && state->pages[page] && !visual::requestWorkflowClose(state->pages[page])) {
                if(pendingTab==tabs.size() || index==state->activeTab) pendingTab=index;
            }
        }
        if(state->work.valid()) {
            if(state->videoWork) state->videoWork->cancel=true;
            SetWindowTextW(state->status,L"当前模型正在本机运行，完成后可关闭。");
            pendingTab=restorationTab;
        }
        if(pendingTab!=tabs.size()) {
            activate(handle,*state,pendingTab,tabs[pendingTab].page);
            SetFocus(state->navigation[pendingTab]);
            return 0;
        }
        break;
    }
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
    windowClass.hbrBackground = theme::background();
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    if (!RegisterClassW(&windowClass)) {
        throw std::runtime_error("Could not register the visual analysis window.");
    }
    Window state;
    state.models = models;
    state.worker = applicationDirectory() / L"VehicleCatalogueWorker.exe";
    const auto handle = CreateWindowExW(WS_EX_CONTROLPARENT, className, L"Visual Analysis",
        WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, 1320, 850, nullptr, nullptr, module, &state);
    if (!handle) {
        throw std::runtime_error("Could not create the visual analysis window.");
    }
    theme::titleBar(handle);
    ShowWindow(handle, show);
    MSG message{};
    BOOL status{};
    while ((status = GetMessageW(&message, nullptr, 0, 0)) > 0) {
        if(!IsDialogMessageW(handle,&message)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
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
