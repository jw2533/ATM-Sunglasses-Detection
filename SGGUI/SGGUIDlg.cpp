// SGGUIDlg.cpp : ʵ���ļ�
//

#include "stdafx.h"
#include "SGGUI.h"
#include "SGGUIDlg.h"
#include "afxdialogex.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include <stdio.h> 

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "CvvImage.h" 

#include "face_detection.h"
#include "Resource.h"

#include <caffe/caffe.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/concat_layer.hpp"

#include "HtpClient.h"


using namespace std;
using namespace caffe;  // NOLINT(build/namespaces)
typedef struct tm timeinfo;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// ����Ӧ�ó��򡰹��ڡ��˵���� CAboutDlg �Ի���
cv::Mat frame;
//cv::VideoCapture cap(0);
cv::VideoCapture cap;
CvvImage cimg1;
CvvImage cimg2;
cv::VideoCapture capture;
HtpClient c;


/***************************classification****************************************/
extern INSTANTIATE_CLASS(InputLayer);
extern INSTANTIATE_CLASS(FlattenLayer);
extern INSTANTIATE_CLASS(ConcatLayer);
extern INSTANTIATE_CLASS(InnerProductLayer);
extern INSTANTIATE_CLASS(DropoutLayer);
typedef std::pair<string, float> Prediction;

class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	std::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
	/* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	SetMean(mean_file);

	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}
static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	* filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}
std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}
void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
/**********************************************************************************/

seeta::FaceDetection detector("E:/VS Projects/FaceDetection/FaceDetection/model/seeta_fd_frontal_v1.0.bin"); // ģ��
string last_text;
string llast_text;

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// �Ի�������
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV ֧��

// ʵ��
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CSGGUIDlg �Ի���



CSGGUIDlg::CSGGUIDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_SGGUI_DIALOG, pParent)
	, m_editCMRNO(0)
	, m_editCMRCDT(_T(""))
	, m_editDettm(0)
	, m_editGL(_T(""))
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSGGUIDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_CMRNO_EDIT, m_editCMRNO);
	DDX_Text(pDX, IDC_CMRCDT_EDIT, m_editCMRCDT);
	DDX_Text(pDX, IDC_DETTM_EDIT, m_editDettm);
	DDX_Text(pDX, IDC_GL_EDIT, m_editGL);
}

BEGIN_MESSAGE_MAP(CSGGUIDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_CMR_BUTTON, &CSGGUIDlg::OnBnClickedCmrButton)
	ON_EN_CHANGE(IDC_CMRCDT_EDIT, &CSGGUIDlg::OnEnChangeCmrcdtEdit)
	ON_STN_CLICKED(IDC_FRAME, &CSGGUIDlg::OnStnClickedFrame)
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_CMRST_BUTTON, &CSGGUIDlg::OnBnClickedCmrstButton)
	ON_STN_CLICKED(IDC_DETTM_STATIC, &CSGGUIDlg::OnStnClickedDettmStatic)
	ON_BN_CLICKED(IDCANCEL, &CSGGUIDlg::OnBnClickedCancel)
	ON_STN_CLICKED(IDC_CMRCDT_STATIC, &CSGGUIDlg::OnStnClickedCmrcdtStatic)
	ON_BN_CLICKED(IDC_OPENADR_BUTTON, &CSGGUIDlg::OnBnClickedOpenadrButton)
	ON_STN_CLICKED(IDC_head, &CSGGUIDlg::OnStnClickedhead)
END_MESSAGE_MAP()


// CSGGUIDlg ��Ϣ�������

BOOL CSGGUIDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// ��������...���˵�����ӵ�ϵͳ�˵��С�

	// IDM_ABOUTBOX ������ϵͳ���Χ�ڡ�
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// ���ô˶Ի����ͼ�ꡣ  ��Ӧ�ó��������ڲ��ǶԻ���ʱ����ܽ��Զ�
	//  ִ�д˲���
	SetIcon(m_hIcon, TRUE);			// ���ô�ͼ��
	SetIcon(m_hIcon, FALSE);		// ����Сͼ��

	// TODO: �ڴ���Ӷ���ĳ�ʼ������

	return TRUE;  // ���ǽ��������õ��ؼ������򷵻� TRUE
}

void CSGGUIDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// �����Ի��������С����ť������Ҫ����Ĵ���
//  �����Ƹ�ͼ�ꡣ  ����ʹ���ĵ�/��ͼģ�͵� MFC Ӧ�ó���
//  �⽫�ɿ���Զ���ɡ�

void CSGGUIDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // ���ڻ��Ƶ��豸������

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// ʹͼ���ڹ����������о���
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// ����ͼ��
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//���û��϶���С������ʱϵͳ���ô˺���ȡ�ù��
//��ʾ��
HCURSOR CSGGUIDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CSGGUIDlg::OnBnClickedCmrButton()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������

	UpdateData(true);
	int camera = m_editCMRNO;

	CDC *pDC = GetDlgItem(IDC_FRAME)->GetDC();
	HDC hdc = pDC->GetSafeHdc();
	CRect rect;// ������ 
	GetDlgItem(IDC_FRAME)->GetClientRect(&rect);

	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f); //��ֵ
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// classification
	string model_file = "./deploy_48_color.prototxt"; // deploy �ļ�
	string trained_file = "real.caffemodel";//caffeģ��
	string mean_file = "real_mean.binaryproto";//��ֵ
	string label_file = "synset_words.txt";//��ǩ
	Classifier classifier(model_file, trained_file, mean_file, label_file);//�������󲢳�ʼ�����硢ģ�͡���ֵ����ǩ�������  


	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 2, 8);  //��ʼ������  


	if (!cap.open(camera))//������ͷ
	{
		m_editCMRCDT = "Capture from camera didn't work";
		INT_PTR nRes;
 
		nRes = MessageBox(_T("��ѡ�������ͷ�޷�ʹ�ã�"), _T("����"), MB_OK | MB_ICONWARNING);
		// �ж���Ϣ�Ի��򷵻�ֵ�����ΪIDCANCEL��return�������������ִ��   
		if (IDOK == nRes)
			return;
	}
	else
	{ 
			m_editCMRCDT = "Video capturing has been started ...";


			cap >> frame;


			cv::Mat img;
			cap >> img;

/**********************************************************************************/
			CvSize sz;
			sz.width = img.cols;
			sz.height = img.rows;
			IplImage* show = cvCreateImage(sz, IPL_DEPTH_8U, 3);
/**********************************************************************************/



/********************************�������*******************************************/
			cv::Mat img_gray;

			if (img.channels() != 1)  //ת��Ϊ�Ҷ�ͼ��
				cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
			else
				img_gray = img;

			seeta::ImageData img_data;
			img_data.data = img_gray.data;
			img_data.width = img_gray.cols;
			img_data.height = img_gray.rows;
			img_data.num_channels = 1;

			long t0 = cv::getTickCount();
			std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);  //�������
			long t1 = cv::getTickCount();
			double secs = (t1 - t0) / cv::getTickFrequency();  //��ʱ


/**********************************************************************************/

			cv::Rect face_rect;
			int32_t num_face = static_cast<int32_t>(faces.size());  //ǿ������ת��  ����ʶ�������������
			cout << "face number : " << num_face << endl;

			for (int32_t i = 0; i < num_face; i++) {                //Ϊʶ����������������ο�   
				face_rect.x = faces[i].bbox.x;
				face_rect.y = faces[i].bbox.y;
				face_rect.width = faces[i].bbox.width;
				face_rect.height = faces[i].bbox.height;

				cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
			}
			cvCopy(&(IplImage)img, show);


/**********************************************************************************/
			time_t rawtime;												 //ʱ������  
			timeinfo *timeinfos;										 //ʱ��ṹ�� ָ�����
			char cloc[100];
			char sendname[100];
			char filename[100];
			time(&rawtime);											//��ȡʱ�����������1970��1��1�տ�ʼ������rawtime  
			timeinfos = localtime(&rawtime);						//������תΪ����ʱ��  
			sprintf_s(cloc, "%d-%d-%d %dh%dm%ds",
				timeinfos->tm_year + 1900,
				timeinfos->tm_mon + 1,
				timeinfos->tm_mday,
				timeinfos->tm_hour,
				timeinfos->tm_min,
				timeinfos->tm_sec);
			sprintf_s(filename, "E:\\SG_WARNING\\Found SG_%s.jpg", cloc);
			sprintf_s(sendname, "Found SG_%s.jpg", cloc);

			time_t now = time(0);
			tm ltm;
			localtime_s(&ltm, &now);
			char temp[64];
			char prttm[64];
			strftime(prttm, sizeof(temp), "%Y-%m-%d %H:%M:%S", &ltm); //��ʾʱ���
			strftime(temp, sizeof(temp), "Datetime: %Y-%m-%d %H:%M:%S", &ltm);//����ʱ���
/**********************************************************************************/
			cvPutText(show, prttm, cvPoint(10, 30), &font, cvScalar(255, 255, 255, NULL));//��ʾʱ��


/**********************************************************************************/
			IplImage img1 = IplImage(frame);
			cimg1.CopyOf(&img1);
			cimg1.DrawToHDC(hdc, &rect);
			SetTimer(1, 1, NULL);//��һ��1Ϊ��ʱ�����ƣ��ڶ���Ϊʱ��������λ����  
 /**********************************************************************************/
			if (face_rect.x + face_rect.width <= img.cols  && face_rect.y + face_rect.height <= img.rows)  //��֤ROI����С��ԭͼ������
			{
				if (face_rect.x + face_rect.width != 0 && face_rect.y + face_rect.height != 0)
				{
					cv::Mat image_roi = img(face_rect);  //��ȡ����

					std::vector<Prediction> predictions = classifier.Classify(image_roi);
					Prediction label = predictions[0];
					string text = label.first;
					/* Print the top N predictions. */
					for (size_t i = 0; i < predictions.size(); ++i) {
						Prediction p = predictions[i];

					}




					cvPutText(show, text.c_str(), cvPoint(10, 55), &font, cvScalar(0, 0, 255, NULL));

					if (text == "sunglasses" && last_text != "sunglasses" && llast_text !="sunglasses")
					{

						cvSaveImage(filename, show);
						//c.level = "Level: 1";
						//c.description = "Description: 1";
						//c.alarmPicture = filename;
						//c.alarmPictureName = sendname;
						//c.device = "Device: 4396";
						//c.datetime = temp;
						//int res = c.Doclient();
						//cout << res << endl;

					}
					last_text = text;
				}

			}

/**********************************************************************************/
			cimg1.CopyOf(show);

			cimg1.DrawToHDC(hdc, &rect);
			SetTimer(1, 1, NULL);//��һ��1Ϊ��ʱ�����ƣ��ڶ���Ϊʱ��������λ����  
/**********************************************************************************/

		
	}


	UpdateData(false);


}


void CSGGUIDlg::OnEnChangeCmrcdtEdit()
{
	// TODO:  ����ÿؼ��� RICHEDIT �ؼ���������
	// ���ʹ�֪ͨ��������д CDialogEx::OnInitDialog()
	// ���������� CRichEditCtrl().SetEventMask()��
	// ͬʱ�� ENM_CHANGE ��־�������㵽�����С�

	// TODO:  �ڴ���ӿؼ�֪ͨ����������
}


void CSGGUIDlg::OnStnClickedFrame()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CSGGUIDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: �ڴ������Ϣ�����������/�����Ĭ��ֵ
	CDC *pDC = GetDlgItem(IDC_FRAME)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ�룬������ʾframe   
	HDC hdc = pDC->GetSafeHdc();                      // ��ȡ�豸�����ľ��      
	CRect rect;// ������     
	GetDlgItem(IDC_FRAME)->GetClientRect(&rect);

	CDC *pDC2 = GetDlgItem(IDC_head)->GetDC();//����ID��ô���ָ���ٻ�ȡ��ô��ڹ�����������ָ�룬������ʾhead
	HDC hdc2 = pDC2->GetSafeHdc();                      // ��ȡ�豸�����ľ��      
	CRect rect2;// ������     
	GetDlgItem(IDC_head)->GetClientRect(&rect2);

	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f); //��ֵ
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);

	// classification
	string model_file = "./deploy_48_color.prototxt"; // deploy �ļ�
	string trained_file = "real.caffemodel";//caffeģ��
	string mean_file = "real_mean.binaryproto";//��ֵ
	string label_file = "synset_words.txt";//��ǩ
	Classifier classifier(model_file, trained_file, mean_file, label_file);//�������󲢳�ʼ�����硢ģ�͡���ֵ����ǩ�������  


	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 2, 8);  //��ʼ������  

	if (cap.isOpened())
	{
		cv::Mat img;
		cap >> img;

/**********************************************************************************/
		CvSize sz;
		sz.width = img.cols;
		sz.height = img.rows;
		IplImage* show = cvCreateImage(sz, IPL_DEPTH_8U, 3);
/**********************************************************************************/



/********************************�������*******************************************/
		cv::Mat img_gray;

		if (img.channels() != 1)  //ת��Ϊ�Ҷ�ͼ��
			cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		else
			img_gray = img;

		seeta::ImageData img_data;
		img_data.data = img_gray.data;
		img_data.width = img_gray.cols;
		img_data.height = img_gray.rows;
		img_data.num_channels = 1;

		long t0 = cv::getTickCount();
		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);  //�������
		long t1 = cv::getTickCount();
		double secs = (t1 - t0) / cv::getTickFrequency();  //��ʱ

		m_editDettm = secs;

/**********************************************************************************/

		cv::Rect face_rect;
		int32_t num_face = static_cast<int32_t>(faces.size());  //ǿ������ת��  ����ʶ�������������
//		cout << "face number : " << num_face << endl;

		for (int32_t i = 0; i < num_face; i++) {                //Ϊʶ����������������ο�   
			face_rect.x = faces[i].bbox.x;
			face_rect.y = faces[i].bbox.y;
			face_rect.width = faces[i].bbox.width;
			face_rect.height = faces[i].bbox.height;

			cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
		}
		cvCopy(&(IplImage)img, show);


/**********************************************************************************/
		time_t rawtime;												 //ʱ������  
		timeinfo *timeinfos;										 //ʱ��ṹ�� ָ�����
		char cloc[100];
		char sendname[100];
		char filename[100];
		time(&rawtime);											//��ȡʱ�����������1970��1��1�տ�ʼ������rawtime  
		timeinfos = localtime(&rawtime);						//������תΪ����ʱ��  
		sprintf_s(cloc, "%d-%d-%d %dh%dm%ds",
			timeinfos->tm_year + 1900,
			timeinfos->tm_mon + 1,
			timeinfos->tm_mday,
			timeinfos->tm_hour,
			timeinfos->tm_min,
			timeinfos->tm_sec);
		sprintf_s(filename, "E:\\SG_WARNING\\Found SG_%s.jpg", cloc);
		sprintf_s(sendname, "Found SG_%s.jpg", cloc);

		time_t now = time(0);
		tm ltm;
		localtime_s(&ltm, &now);
		char temp[64];
		char prttm[64];
		strftime(prttm, sizeof(temp), "%Y-%m-%d %H:%M:%S", &ltm); //��ʾʱ���
		strftime(temp, sizeof(temp), "Datetime: %Y-%m-%d %H:%M:%S", &ltm);//����ʱ���
/**********************************************************************************/
		cvPutText(show, prttm, cvPoint(10, 30), &font, cvScalar(255, 255, 255, NULL));//��ʾʱ��


/**********************************************************************************/





/**********************************************************************************/
		if (face_rect.x + face_rect.width <= img.cols  && face_rect.y + face_rect.height <= img.rows)  //��֤ROI����С��ԭͼ������
		{
			if (face_rect.x + face_rect.width != 0 && face_rect.y + face_rect.height != 0)
			{
				if(50<=face_rect.x && 50 <= face_rect.y  && face_rect.x<= img.cols-50 && face_rect.y<= img.rows-50)
				{ 
				cv::Mat image_roi = img(face_rect);  //��ȡ����,������ʾ��IDC_head��
				IplImage head = IplImage(image_roi);
				cimg2.CopyOf(&head);
				cimg2.DrawToHDC(hdc2, &rect2);
				CString glstr1, glstr2;
				CString pre, last;
				pre = "---------- Prediction ----------";
				last = "-----------------------------------";

				std::vector<Prediction> predictions = classifier.Classify(image_roi);
				Prediction label = predictions[0];
				string text = label.first;
				/* Print the top N predictions. */
				for (size_t i = 0; i < predictions.size(); ++i)
				{
					Prediction p = predictions[i];
					if (i == 0)
					{
						glstr1.Format(_T("%.4f -- \"%s\" \r\n"), p.second, p.first);
					}
					else if (i == 1)
					{
						glstr2.Format(_T("%.4f -- \"%s\" \r\n"), p.second, p.first);// p.second�������ֵ��p.first��������ǩ
					}

					m_editGL = pre + "\r\n" + glstr1 + +glstr2 + last;

				}



				cvPutText(show, text.c_str(), cvPoint(10, 55), &font, cvScalar(0, 0, 255, NULL));

				if (text == "sunglasses" && last_text != "sunglasses")
				{
					cvSaveImage(filename, show);


					//c.level = "Level: 1";                    //���Ͳ���
					//c.description = "Description: 1";
					//c.alarmPicture = filename;
					//c.alarmPictureName = sendname;
					//c.device = "Device: 4396";
					//c.datetime = temp;
					//int res = c.Doclient();
					//cout << res << endl;
				}
				last_text = text;
				}
			}
		}
/**********************************************************************************/

		cimg1.CopyOf(show);
		cimg1.DrawToHDC(hdc, &rect);
		SetTimer(1, 1, NULL);//��һ��1Ϊ��ʱ�����ƣ��ڶ���Ϊʱ��������λ����  
/**********************************************************************************/


	}

	UpdateData(false);
	CDialogEx::OnTimer(nIDEvent);
}


void CSGGUIDlg::OnBnClickedCmrstButton()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	KillTimer(1);//��ʹ�ü�ʱ��  
}


void CSGGUIDlg::OnStnClickedDettmStatic()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CSGGUIDlg::OnBnClickedCancel()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	INT_PTR nRes;

	// ��ʾ��Ϣ�Ի���   
	nRes = MessageBox(_T("��ȷ��Ҫ�˳������"), _T("��ʾ"), MB_OKCANCEL | MB_ICONQUESTION);
	if (IDCANCEL == nRes)
		return;
	CDialogEx::OnCancel();
}


void CSGGUIDlg::OnStnClickedCmrcdtStatic()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}


void CSGGUIDlg::OnBnClickedOpenadrButton()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
	ShellExecute(NULL, "open", "E:\\SG_WARNING\\",NULL, NULL, SW_SHOW);
}


void CSGGUIDlg::OnStnClickedhead()
{
	// TODO: �ڴ���ӿؼ�֪ͨ����������
}
