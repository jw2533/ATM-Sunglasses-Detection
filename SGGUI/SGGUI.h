
// SGGUI.h : PROJECT_NAME Ӧ�ó������ͷ�ļ�
//

#pragma once

#ifndef __AFXWIN_H__
	#error "�ڰ������ļ�֮ǰ������stdafx.h�������� PCH �ļ�"
#endif

#include "resource.h"		// ������


// CSGGUIApp: 
// �йش����ʵ�֣������ SGGUI.cpp
//

class CSGGUIApp : public CWinApp
{
public:
	CSGGUIApp();

// ��д
public:
	virtual BOOL InitInstance();

// ʵ��

	DECLARE_MESSAGE_MAP()
};

extern CSGGUIApp theApp;