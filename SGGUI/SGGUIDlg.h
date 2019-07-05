
// SGGUIDlg.h : 头文件
//

#pragma once


// CSGGUIDlg 对话框
class CSGGUIDlg : public CDialogEx
{
// 构造
public:
	CSGGUIDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_SGGUI_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	int m_editCMRNO;
	afx_msg void OnBnClickedCmrButton();
	CString m_editCMRCDT;
	afx_msg void OnEnChangeCmrcdtEdit();
	afx_msg void OnStnClickedFrame();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnBnClickedCmrstButton();
	afx_msg void OnStnClickedDettmStatic();
	double m_editDettm;
	CString m_editGL;
	afx_msg void OnBnClickedCancel();
	afx_msg void OnStnClickedCmrcdtStatic();
	afx_msg void OnBnClickedOpenadrButton();
	afx_msg void OnStnClickedhead();
};
