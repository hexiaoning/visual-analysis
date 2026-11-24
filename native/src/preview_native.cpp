#include "preview_native.h"
#include "portrait_video.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <windowsx.h>
#include <opencv2/imgproc.hpp>

namespace visual {
namespace {

constexpr wchar_t className[]=L"VisualAnalysisImagePreview";
constexpr int playbackControl=301,hitControl=302;

struct ReviewFrame {
    cv::Mat pixels;
    int index{};
    double seconds{};
};

struct Viewer {
    cv::Mat pixels;
    bool select{},closed{};
    bool selectLine{};
    std::optional<cv::Rect> answer,overlay,hitOverlay;
    std::optional<VideoLine> line;
    std::optional<cv::Point> drag;
    cv::Point end;
    std::mutex mutex;
    ReviewFrame next;
    std::optional<int> requestedFrame;
    std::condition_variable controls;
    HWND slider{},play{},returnToHit{},position{};
    int hitFrame{},currentFrame{};
    double currentSeconds{-1.};
    bool draggingSlider{};
    std::atomic<int> totalFrames{};
    std::atomic<bool> cancel{},interrupted{},paused{},finished{},failed{};
    std::jthread playback;

    void seek(int index,bool pause) {
        std::lock_guard lock(mutex);
        requestedFrame=index;
        paused=pause;
        finished=false;
        failed=false;
        next={};
        interrupted=true;
        controls.notify_one();
    }

    void close() {
        std::lock_guard lock(mutex);
        cancel=true;
        interrupted=true;
        controls.notify_one();
    }
};

RECT shown(HWND window,const Viewer& view) {
    RECT client{};
    GetClientRect(window,&client);
    if(view.slider) client.bottom-=72;
    const auto& pixels=view.pixels;
    const double scale=std::min(double(client.right-20)/pixels.cols,double(client.bottom-20)/pixels.rows);
    const int w=std::max(1,cvRound(pixels.cols*scale)),h=std::max(1,cvRound(pixels.rows*scale));
    const long left=(client.right-w)/2,top=(client.bottom-h)/2;
    return {left,top,left+w,top+h};
}

void layoutPlayback(HWND window,const Viewer& view) {
    RECT client{};
    GetClientRect(window,&client);
    MoveWindow(view.slider,12,client.bottom-70,std::max(1L,client.right-24),24,TRUE);
    MoveWindow(view.play,12,client.bottom-40,85,27,TRUE);
    MoveWindow(view.returnToHit,106,client.bottom-40,128,27,TRUE);
    MoveWindow(view.position,247,client.bottom-36,std::max(1L,client.right-259),23,TRUE);
}

void updatePlayback(const Viewer& view) {
    SetWindowTextW(view.play,view.paused?L"播放":L"暂停");
    auto label=L"第 "+std::to_wstring(view.currentFrame+1)+L" 帧";
    if(view.failed) label=L"无法继续读取源视频。";
    else if(view.currentSeconds>=0.) {
        const auto total=static_cast<long long>(view.currentSeconds);
        const auto seconds=std::to_wstring(total%60);
        label+=L" · "+std::to_wstring(total/60)+L":"+(seconds.size()==1?L"0":L"")+seconds;
    }
    SetWindowTextW(view.position,label.c_str());
    if(!view.draggingSlider) {
        const int last=std::max(view.currentFrame,std::max(0,view.totalFrames.load()-1));
        SCROLLINFO info{sizeof(info),SIF_RANGE|SIF_PAGE|SIF_POS,0,last,1,view.currentFrame,0};
        SetScrollInfo(view.slider,SB_CTL,&info,TRUE);
    }
}

void togglePlayback(Viewer& view) {
    if(view.finished) view.seek(view.hitFrame,false);
    else {
        std::lock_guard lock(view.mutex);
        view.paused=!view.paused.load();
        view.controls.notify_one();
    }
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
        auto frame=shown(window,*view);
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
            if(view->selectLine && view->drag) {
                MoveToEx(dc,frame.left+cvRound(view->drag->x*sx),frame.top+cvRound(view->drag->y*sy),nullptr);
                LineTo(dc,frame.left+cvRound(view->end.x*sx),frame.top+cvRound(view->end.y*sy));
            } else {
                Rectangle(dc,frame.left+cvRound(area->x*sx),frame.top+cvRound(area->y*sy),
                    frame.left+cvRound(area->br().x*sx),frame.top+cvRound(area->br().y*sy));
            }
            SelectObject(dc,oldBrush);SelectObject(dc,oldPen);DeleteObject(pen);
        }
        EndPaint(window,&paint);
        return 0;
    }
    case WM_SIZE:
        if(view->slider) layoutPlayback(window,*view);
        InvalidateRect(window,nullptr,TRUE);
        return 0;
    case WM_TIMER: {
        ReviewFrame frame;
        {
            std::lock_guard lock(view->mutex);
            frame=std::move(view->next);
        }
        if(!frame.pixels.empty()) {
            view->pixels=std::move(frame.pixels);
            view->currentFrame=frame.index;
            view->currentSeconds=frame.seconds;
            view->overlay=frame.index==view->hitFrame?view->hitOverlay:std::nullopt;
            InvalidateRect(window,nullptr,TRUE);
        }
        if(view->slider) updatePlayback(*view);
        return 0;
    }
    case WM_HSCROLL:
        if(view->slider && reinterpret_cast<HWND>(lParam)==view->slider) {
            SCROLLINFO info{sizeof(info),SIF_ALL};
            GetScrollInfo(view->slider,SB_CTL,&info);
            const auto action=LOWORD(wParam);
            const auto step=std::max(1,info.nMax/50);
            int64_t frame=info.nPos;
            switch(action) {
            case SB_THUMBTRACK:
                view->draggingSlider=true;
                {
                    std::lock_guard lock(view->mutex);
                    view->paused=true;
                    view->controls.notify_one();
                }
                SetScrollPos(view->slider,SB_CTL,info.nTrackPos,TRUE);
                return 0;
            case SB_THUMBPOSITION: frame=info.nTrackPos;break;
            case SB_PAGELEFT: frame-=step;break;
            case SB_PAGERIGHT: frame+=step;break;
            case SB_LINELEFT: --frame;break;
            case SB_LINERIGHT: ++frame;break;
            case SB_LEFT: frame=0;break;
            case SB_RIGHT: frame=info.nMax;break;
            default: view->draggingSlider=false;return 0;
            }
            view->draggingSlider=false;
            view->seek(static_cast<int>(std::clamp<int64_t>(frame,0,info.nMax)),true);
            return 0;
        }
        break;
    case WM_COMMAND:
        if(view->slider) {
            if(LOWORD(wParam)==playbackControl) togglePlayback(*view);
            else if(LOWORD(wParam)==hitControl) view->seek(view->hitFrame,true);
        }
        return 0;
    case WM_LBUTTONDOWN:
    case WM_MOUSEMOVE:
    case WM_LBUTTONUP: {
        if(!view->select) break;
        const auto bounds=shown(window,*view);
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
                const auto start=*view->drag;
                view->drag.reset();
                ReleaseCapture();
                if(view->selectLine && start!=pos) {
                    view->line=VideoLine{start,pos};
                    DestroyWindow(window);
                    return 0;
                }
                const cv::Rect region(start,pos);
                if(!view->selectLine && !region.empty()) {
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
        if(wParam==VK_SPACE && view->playback.joinable()) {
            togglePlayback(*view);
            return 0;
        }
        if(wParam==VK_ESCAPE) {DestroyWindow(window);return 0;}
        break;
    case WM_DESTROY:
        view->close();
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
    auto window=CreateWindowExW(0,className,caption,WS_OVERLAPPEDWINDOW|WS_CLIPCHILDREN,CW_USEDEFAULT,CW_USEDEFAULT,940,700,
        owner,nullptr,type.hInstance,&state);
    if(!window) throw std::runtime_error("无法创建画面预览。");
    if(state.playback.joinable()) {
        auto child=[&](const wchar_t* cls,const wchar_t* title,DWORD style,int id) {
            return CreateWindowExW(0,cls,title,WS_CHILD|WS_VISIBLE|style,0,0,0,0,
                window,reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),type.hInstance,nullptr);
        };
        state.slider=child(L"SCROLLBAR",L"",SBS_HORZ,0);
        state.play=child(L"BUTTON",L"暂停",BS_PUSHBUTTON,playbackControl);
        state.returnToHit=child(L"BUTTON",L"返回命中位置",BS_PUSHBUTTON,hitControl);
        state.position=child(L"STATIC",L"",SS_LEFT,0);
        layoutPlayback(window,state);
        updatePlayback(state);
    }
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
    state.close();
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

std::optional<VideoLine> selectNativeLine(HWND owner,const cv::Mat& image) {
    if(image.empty()) throw std::runtime_error("没有可预览的画面。");
    Viewer state;
    state.select=state.selectLine=true;
    cv::cvtColor(image,state.pixels,cv::COLOR_BGR2BGRA);
    showViewer(owner,state,L"在画面上拖动鼠标绘制触发线");
    return state.line;
}

void playNativeVideo(HWND owner,const std::filesystem::path& source,int first,
                     const cv::Mat& initial,cv::Rect initialTarget) {
    Viewer state;
    state.hitFrame=state.currentFrame=first;
    state.requestedFrame=first;
    state.overlay=state.hitOverlay=initialTarget;
    cv::cvtColor(initial,state.pixels,cv::COLOR_BGR2BGRA);
    state.playback=std::jthread([&]{
        using namespace std::chrono;
        while(!state.cancel) {
            int target;
            {
                std::unique_lock lock(state.mutex);
                state.controls.wait(lock,[&]{return state.cancel || state.requestedFrame.has_value();});
                if(state.cancel) return;
                target=*state.requestedFrame;
                state.requestedFrame.reset();
                state.interrupted=false;
            }
            std::optional<double> startTime;
            auto start=steady_clock::now();
            try {
                const auto frames=visitVideo(source,[&](const cv::Mat& frame,int index,double seconds){
                    if(!startTime) {
                        startTime=seconds;
                        start=steady_clock::now();
                    } else {
                        const auto offset=duration_cast<steady_clock::duration>(duration<double>(seconds-*startTime));
                        std::unique_lock lock(state.mutex);
                        while(!state.interrupted) {
                            if(state.paused) {
                                const auto pausedAt=steady_clock::now();
                                state.controls.wait(lock,[&]{return !state.paused || state.interrupted;});
                                start+=steady_clock::now()-pausedAt;
                            } else if(steady_clock::now()>=start+offset) break;
                            else state.controls.wait_until(lock,start+offset);
                        }
                    }
                    if(state.interrupted) return false;
                    cv::Mat converted;
                    cv::cvtColor(frame,converted,cv::COLOR_BGR2BGRA);
                    std::lock_guard lock(state.mutex);
                    if(state.interrupted) return false;
                    state.next={std::move(converted),index,seconds};
                    return true;
                },1,state.interrupted,target,{},&state.totalFrames);
                std::lock_guard lock(state.mutex);
                if(state.interrupted) continue;
                if(frames>0 && target>=frames) state.requestedFrame=frames-1;
                else {
                    state.finished=true;
                    state.paused=true;
                }
            } catch(const std::exception&) {
                std::lock_guard lock(state.mutex);
                if(!state.interrupted) {
                    state.failed=true;
                    state.paused=true;
                    state.finished=true;
                }
            }
        }
    });
    try {showViewer(owner,state,L"从命中帧播放本机源视频 · 空格暂停");}
    catch(...) {state.close();throw;}
}
}
