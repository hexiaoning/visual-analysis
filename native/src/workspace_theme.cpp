#include "workspace_theme.h"

#include <array>
#include <dwmapi.h>

namespace visual::theme {
namespace {

struct Resources {
    HFONT body = CreateFontW(-15,0,0,0,FW_NORMAL,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,
                             CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Microsoft YaHei UI");
    HFONT heading = CreateFontW(-15,0,0,0,FW_BOLD,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,
                                CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Microsoft YaHei UI");
    HFONT brand = CreateFontW(-24,0,0,0,FW_SEMIBOLD,FALSE,FALSE,FALSE,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,
                              CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Microsoft YaHei UI");
    HBRUSH surface = CreateSolidBrush(white);
    HBRUSH background = CreateSolidBrush(canvas);
    HBRUSH frame = CreateSolidBrush(border);

    ~Resources() {
        DeleteObject(body);
        DeleteObject(heading);
        DeleteObject(brand);
        DeleteObject(surface);
        DeleteObject(background);
        DeleteObject(frame);
    }
};

Resources& resources() {
    static Resources instance;
    return instance;
}

void box(HDC dc,RECT area,COLORREF fillColor,COLORREF borderColor,int radius) {
    const auto brush=CreateSolidBrush(fillColor);
    const auto pen=CreatePen(PS_SOLID,1,borderColor);
    const auto oldBrush=SelectObject(dc,brush);
    const auto oldPen=SelectObject(dc,pen);
    RoundRect(dc,area.left,area.top,area.right,area.bottom,radius,radius);
    SelectObject(dc,oldPen);
    SelectObject(dc,oldBrush);
    DeleteObject(pen);
    DeleteObject(brush);
}

}

HFONT font(Font role) {
    auto& owned=resources();
    auto result=role==Font::Brand?owned.brand:role==Font::Heading?owned.heading:owned.body;
    return result?result:static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
}

HBRUSH surface() {return resources().surface;}
HBRUSH background() {return resources().background;}

void styleControl(HWND control) {
    SendMessageW(control,WM_SETFONT,reinterpret_cast<WPARAM>(font(Font::Body)),TRUE);
}

void fill(HDC dc,RECT area,COLORREF color) {
    const auto brush=CreateSolidBrush(color);
    FillRect(dc,&area,brush);
    DeleteObject(brush);
}

void card(HDC dc,RECT area) {
    FillRect(dc,&area,surface());
    FrameRect(dc,&area,resources().frame);
}

void panel(HDC dc,RECT area,const wchar_t* title) {
    card(dc,area);
    const auto bottom=area.top+panelHeaderHeight;
    fill(dc,{area.left+1,area.top+1,area.right-1,bottom-1},panelTitle);
    fill(dc,{area.left+1,bottom-1,area.right-1,bottom},border);
    text(dc,title,{area.left+12,area.top,area.right-10,bottom},Font::Heading,blueDark,
         DT_LEFT|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
}

void text(HDC dc,const wchar_t* value,RECT area,Font role,COLORREF color,UINT flags) {
    const auto selected=SelectObject(dc,font(role));
    const auto old=SetTextColor(dc,color);
    const auto mode=SetBkMode(dc,TRANSPARENT);
    DrawTextW(dc,value,-1,&area,flags|DT_NOPREFIX);
    SetBkMode(dc,mode);
    SetTextColor(dc,old);
    SelectObject(dc,selected);
}

void drawButton(const DRAWITEMSTRUCT& item,Button kind) {
    const bool pressed=(item.itemState&ODS_SELECTED)!=0;
    const bool disabled=(item.itemState&(ODS_DISABLED|ODS_GRAYED))!=0;
    const bool focused=(item.itemState&ODS_FOCUS)!=0;
    auto area=item.rcItem;
    const bool selected=kind==Button::SelectedNavigation;
    const bool navigation=kind==Button::Navigation || selected;
    const bool primary=kind==Button::Primary;
    auto foreground=textColor;
    if(navigation) {
        fill(item.hDC,area,selected?RGB(52,106,203):pressed?blue:blueDark);
        foreground=white;
        if(selected) {
            fill(item.hDC,{area.left,area.bottom-3,area.right,area.bottom},RGB(162,205,255));
        }
    } else {
        const auto tone=disabled?RGB(232,238,246):primary?(pressed?blueDark:primaryBlue):pressed?panelTitle:white;
        const auto edge=primary&&!disabled?primaryBlue:(focused?primaryBlue:border);
        foreground=disabled?RGB(111,128,150):primary?white:textColor;
        fill(item.hDC,area,white);
        box(item.hDC,area,tone,edge,4);
    }
    std::array<wchar_t,256> caption{};
    GetWindowTextW(item.hwndItem,caption.data(),static_cast<int>(caption.size()));
    InflateRect(&area,-6,-3);
    text(item.hDC,caption.data(),area,primary||selected?Font::Heading:Font::Body,
         foreground,DT_CENTER|DT_VCENTER|DT_SINGLELINE|DT_END_ELLIPSIS);
    if(focused && !disabled && (item.itemState&ODS_NOFOCUSRECT)==0) {
        InflateRect(&area,1,0);
        DrawFocusRect(item.hDC,&area);
    }
}

void titleBar(HWND window) {
    const COLORREF caption=RGB(245,249,255);
    DwmSetWindowAttribute(window,DWMWA_CAPTION_COLOR,&caption,sizeof(caption));
    DwmSetWindowAttribute(window,DWMWA_TEXT_COLOR,&textColor,sizeof(textColor));
}

}
