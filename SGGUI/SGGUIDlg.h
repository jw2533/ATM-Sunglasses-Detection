
// SGGUIDlg.h : ͷ�ļ�
//

#pragma once


// CSGGUIDlg �Ի���
class CSGGUIDlg : public CDialogEx
{
// ����
public:
	CSGGUIDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_SGGUI_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
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
