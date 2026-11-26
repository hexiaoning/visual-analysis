#pragma once

#include <Windows.h>

namespace visual::theme {

inline constexpr COLORREF canvas = RGB(239, 242, 246);
inline constexpr COLORREF blue = RGB(24, 72, 164);
inline constexpr COLORREF blueDark = RGB(16, 53, 127);
inline constexpr COLORREF primaryBlue = RGB(31, 89, 191);
inline constexpr COLORREF white = RGB(255, 255, 255);
inline constexpr COLORREF textColor = RGB(37, 49, 64);
inline constexpr COLORREF muted = RGB(76, 92, 113);
inline constexpr COLORREF border = RGB(201, 211, 223);
inline constexpr COLORREF panelTitle = RGB(232, 239, 248);
inline constexpr int panelHeaderHeight = 34;

enum class Font { Body, Heading, Brand };
enum class Button { Quiet, Primary, Navigation, SelectedNavigation };

HFONT font(Font role);
HBRUSH surface();
HBRUSH background();
void styleControl(HWND control);
void fill(HDC dc,RECT area,COLORREF color);
void card(HDC dc,RECT area);
void panel(HDC dc,RECT area,const wchar_t* title);
void text(HDC dc,const wchar_t* value,RECT area,Font role,COLORREF color,UINT flags);
void drawButton(const DRAWITEMSTRUCT& item,Button kind);
void titleBar(HWND window);

}
