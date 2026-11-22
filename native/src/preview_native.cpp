#include "preview_native.h"
#include "portrait_video.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <windowsx.h>
#include <opencv2/imgproc.hpp>

namespace visual {
namespace {

constexpr wchar_t className[]=L"VisualAnalysisImagePreview";
struct Viewer {
    cv::Mat pixels;
    bool select{},closed{};
    std::optional<cv::Rect> answer,overlay;
    std::optional<cv::Point> drag;
    cv::Point end;
    std::mutex mutex;
    cv::Mat next;
    std::atomic<bool> cancel{};
    std::jthread playback;
};

RECT shown(HWND window,const cv::Mat& pixels) {
    RECT client{};
    GetClientRect(window,&client);
    const double scale=std::min(double(client.right-20)/pixels.cols,double(client.bottom-20)/pixels.rows);
    const int w=std::max(1,cvRound(pixels.cols*scale)),h=std::max(1,cvRound(pixels.rows*scale));
    const long left=(client.right-w)/2,top=(client.bottom-h)/2;
    return {left,top,left+w,top+h};
}

LRESULT CALLBACK route(HWND window,UINT message,WPARAM wParam,LPARAM lParam) {
    auto* view=reinterpret_cast<Viewer*>(GetWindowLongPtrW(window,GWLP_USERDATA));
    if(message==WM_NCCREATE) {
        view=static_cast<Viewer*>(reinterpret_cast<LPCREATESTRUCTW>(lParam)->lpCreateParams);
        SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(view));
        return TRUE;
    }
    if(!view) return DefWindowProcW(window,message,wParam,lParam);
    switch(message) {
    case WM_PAINT: {
        PAINTSTRUCT paint{};
        auto dc=BeginPaint(window,&paint);
        auto frame=shown(window,view->pixels);
        BITMAPINFO bitmap{};
        bitmap.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
        bitmap.bmiHeader.biPlanes=1; bitmap.bmiHeader.biBitCount=32;
        bitmap.bmiHeader.biWidth=view->pixels.cols; bitmap.bmiHeader.biHeight=-view->pixels.rows;
        SetStretchBltMode(dc,HALFTONE);
        StretchDIBits(dc,frame.left,frame.top,frame.right-frame.left,frame.bottom-frame.top,
            0,0,view->pixels.cols,view->pixels.rows,view->pixels.data,&bitmap,DIB_RGB_COLORS,SRCCOPY);
        auto area=view->drag ? std::optional(cv::Rect(*view->drag,view->end)) : view->overlay;
        if(area) {
            const double sx=double(frame.right-frame.left)/view->pixels.cols;
            const double sy=double(frame.bottom-frame.top)/view->pixels.rows;
            auto pen=CreatePen(PS_SOLID,2,RGB(0,190,245));
            const auto oldPen=SelectObject(dc,pen);
            const auto oldBrush=SelectObject(dc,GetStockObject(NULL_BRUSH));
            Rectangle(dc,frame.left+cvRound(area->x*sx),frame.top+cvRound(area->y*sy),
                frame.left+cvRound(area->br().x*sx),frame.top+cvRound(area->br().y*sy));
            SelectObject(dc,oldBrush);SelectObject(dc,oldPen);DeleteObject(pen);
        }
        EndPaint(window,&paint);
        return 0;
    }
    case WM_SIZE: InvalidateRect(window,nullptr,TRUE);return 0;
    case WM_TIMER: {
        std::lock_guard lock(view->mutex);
        if(!view->next.empty()) {
            view->pixels=std::move(view->next);
            view->overlay.reset();
            InvalidateRect(window,nullptr,TRUE);
        }
        return 0;
    }
    case WM_LBUTTONDOWN:
    case WM_MOUSEMOVE:
    case WM_LBUTTONUP: {
        if(!view->select) break;
        const auto bounds=shown(window,view->pixels);
        const POINT cursor{GET_X_LPARAM(lParam),GET_Y_LPARAM(lParam)};
        const cv::Point pos{
            std::clamp(cvRound(double(cursor.x-bounds.left)*view->pixels.cols/(bounds.right-bounds.left)),0,view->pixels.cols),
            std::clamp(cvRound(double(cursor.y-bounds.top)*view->pixels.rows/(bounds.bottom-bounds.top)),0,view->pixels.rows)
        };
        if(message==WM_LBUTTONDOWN && PtInRect(&bounds,cursor)) {
            view->drag=view->end=pos;
            SetCapture(window);
        } else if(view->drag) {
            view->end=pos;
            if(message==WM_LBUTTONUP) {
                const cv::Rect region(*view->drag,pos);
                view->drag.reset();
                ReleaseCapture();
                if(region.width>0 && region.height>0) {
                    view->answer=region;
                    DestroyWindow(window);
                    return 0;
                }
            }
        }
        InvalidateRect(window,nullptr,TRUE);
        return 0;
    }
    case WM_KEYDOWN:
        if(wParam==VK_ESCAPE) {DestroyWindow(window);return 0;}
        break;
    case WM_DESTROY:
        view->cancel=true;
        view->closed=true;
        return 0;
    }
    return DefWindowProcW(window,message,wParam,lParam);
}
void showViewer(HWND owner,Viewer& state,const wchar_t* caption) {
    WNDCLASSW type{};
    type.hInstance=GetModuleHandleW(nullptr);
    type.lpszClassName=className;
    type.lpfnWndProc=route;
    type.hCursor=LoadCursorW(nullptr,IDC_ARROW);
    type.hbrBackground=GetSysColorBrush(COLOR_WINDOW);
    if(!RegisterClassW(&type) && GetLastError()!=ERROR_CLASS_ALREADY_EXISTS) throw std::runtime_error("无法打开画面预览。");
    auto window=CreateWindowExW(0,className,caption,WS_OVERLAPPEDWINDOW,CW_USEDEFAULT,CW_USEDEFAULT,940,700,
        owner,nullptr,type.hInstance,&state);
    if(!window) throw std::runtime_error("无法创建画面预览。");
    EnableWindow(owner,FALSE);
    ShowWindow(window,SW_SHOWNORMAL);
    if(state.playback.joinable()) SetTimer(window,1,25,nullptr);
    MSG message{};
    int status=1;
    while(!state.closed && (status=GetMessageW(&message,nullptr,0,0))>0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    if(!state.closed && IsWindow(window)) DestroyWindow(window);
    state.cancel=true;
    if(status==0) PostQuitMessage(static_cast<int>(message.wParam));
    EnableWindow(owner,TRUE);
    SetActiveWindow(owner);
}
}

std::optional<cv::Rect> previewNative(HWND owner,const cv::Mat& image,const wchar_t* caption,
                                     bool select,std::optional<cv::Rect> overlay) {
    if(image.empty()) throw std::runtime_error("没有可预览的画面。");
    Viewer state;
    state.select=select;
    state.overlay=overlay;
    cv::cvtColor(image,state.pixels,cv::COLOR_BGR2BGRA);
    showViewer(owner,state,caption);
    return state.answer;
}

void playNativeVideo(HWND owner,const std::filesystem::path& source,int first,
                     const cv::Mat& initial,cv::Rect initialTarget) {
    Viewer state;
    state.overlay=initialTarget;
    cv::cvtColor(initial,state.pixels,cv::COLOR_BGR2BGRA);
    state.playback=std::jthread([&]{
        using namespace std::chrono;
        std::optional<double> startTime;
        auto start=steady_clock::now();
        try {
            visitVideo(source,[&](const cv::Mat& frame,int index,double seconds){
                if(index<first) return !state.cancel;
                if(!startTime) {
                    startTime=seconds;start=steady_clock::now();
                    return !state.cancel;
                }
                const auto due=start+duration_cast<steady_clock::duration>(duration<double>(seconds-*startTime));
                while(!state.cancel && steady_clock::now()<due) std::this_thread::sleep_for(milliseconds(10));
                if(state.cancel) return false;
                cv::Mat converted;
                cv::cvtColor(frame,converted,cv::COLOR_BGR2BGRA);
                std::lock_guard lock(state.mutex);
                state.next=std::move(converted);
                return true;
            },1,state.cancel);
        } catch(const std::exception&) {}
    });
    try {showViewer(owner,state,L"从命中帧播放本机源视频");}
    catch(...) {state.cancel=true;throw;}
}
}
