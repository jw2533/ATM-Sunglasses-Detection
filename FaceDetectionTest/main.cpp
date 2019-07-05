//opencv3.4.0 + caffe + classification 
//first vision  : 2018-5-20
//		vision 2: 2018-5-22  窗口可以显示分类标签

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include <stdio.h> 

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "face_detection.h"


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

HtpClient c;

/***************************classification****************************************/

using namespace caffe;  // NOLINT(build/namespaces)

extern INSTANTIATE_CLASS(InputLayer);
extern INSTANTIATE_CLASS(FlattenLayer);
extern INSTANTIATE_CLASS(ConcatLayer);
extern INSTANTIATE_CLASS(InnerProductLayer);
extern INSTANTIATE_CLASS(DropoutLayer);

typedef struct tm timeinfo;	


/* Pair (label, confidence) representing a prediction. */
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
/* Return the indices of the top N values of vector v. */
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

/* Return the top N predictions. */
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
/* Load the mean file in binaryproto format. */
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
/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
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


seeta::FaceDetection detector("E:/VS Projects/FaceDetection/FaceDetection/model/seeta_fd_frontal_v1.0.bin"); // 模型

int main(int argc, char** argv) {


	detector.SetMinFaceSize(40);
	detector.SetScoreThresh(2.f); //阈值
	detector.SetImagePyramidScaleFactor(0.8f);
	detector.SetWindowStep(4, 4);


	// classification
	string model_file = "./deploy_48_color.prototxt"; // deploy 文件
	string trained_file = "real.caffemodel";//caffe模型
	string mean_file = "real_mean.binaryproto";//均值
	string label_file = "synset_words.txt";//标签
	Classifier classifier(model_file, trained_file, mean_file, label_file);//创建对象并初始化网络、模型、均值、标签各类对象  


	cv::VideoCapture capture;  
	int camera = 0;
	if (!capture.open(camera))//打开摄像头
	cout << "Capture from camera #" << camera << " didn't work" << endl;
	cout << "Video capturing has been started ..." << endl;

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX, 1.0, 1.0, 0, 2, 8);  //初始化字体  

	string last_text;

	for (;;)
	{
		cv::Mat img;
		capture >> img;
		if (img.empty())
			break;
/**********************************************************************************/
		CvSize sz;
		sz.width = img.cols;
		sz.height = img.rows;
		IplImage* show = cvCreateImage(sz, IPL_DEPTH_8U, 3);

/**********************************************************************************/



/********************************人脸检测*******************************************/
		cv::Mat img_gray;

		if (img.channels() != 1)  //转换为灰度图像
			cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
		else
			img_gray = img;

		seeta::ImageData img_data;
		img_data.data = img_gray.data;
		img_data.width = img_gray.cols;
		img_data.height = img_gray.rows;
		img_data.num_channels = 1;

		long t0 = cv::getTickCount();
		std::vector<seeta::FaceInfo> faces = detector.Detect(img_data);  //人脸检测
		long t1 = cv::getTickCount();
		double secs = (t1 - t0) / cv::getTickFrequency();  //计时

		cout << "Detections takes " << secs << " seconds " << endl;
/**********************************************************************************/





		cout << "Image size (wxh): " << img_data.width << "x"
			<< img_data.height << endl;

		cv::Rect face_rect;
		int32_t num_face = static_cast<int32_t>(faces.size());  //强制类型转换  返回识别出的人脸数量
		cout << "face number : "<< num_face << endl;

		for (int32_t i = 0; i < num_face; i++) {                //为识别出的人脸画出矩形框   
			face_rect.x = faces[i].bbox.x;
			face_rect.y = faces[i].bbox.y;
			face_rect.width = faces[i].bbox.width;
			face_rect.height = faces[i].bbox.height;

			cv::rectangle(img, face_rect, CV_RGB(0, 0, 255), 4, 8, 0);
		}

		cvCopy(&(IplImage)img, show);

		time_t rawtime;												 //时间类型  
		timeinfo *timeinfos;										 //时间结构体 指针变量
		char cloc[100];
		char sendname[100];
		char filename[100];
		time(&rawtime);											//获取时间的秒数，从1970年1月1日开始，存入rawtime  
		timeinfos = localtime(&rawtime);						//将秒数转为当地时间  
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
		strftime(prttm, sizeof(temp), "%Y-%m-%d %H:%M:%S", &ltm); //显示时间戳
		strftime(temp, sizeof(temp), "Datetime: %Y-%m-%d %H:%M:%S", &ltm);//发送时间戳



		cvPutText(show, prttm, cvPoint(10, 30), &font, cvScalar(255, 255, 255, NULL));//显示时间


		if (face_rect.x + face_rect.width <= img.cols  && face_rect.y + face_rect.height <= img.rows)  //保证ROI区域小于原图像区域
		{
			if (face_rect.x + face_rect.width != 0 && face_rect.y + face_rect.height != 0)
			{
				if (50 <= face_rect.x && 50 <= face_rect.y  && face_rect.x <= img.cols - 50 && face_rect.y <= img.rows - 50)
				{

				cv::Mat image_roi = img(face_rect);  //提取人脸


				std::cout << "---------- Prediction" << " ----------" << std::endl;
				std::vector<Prediction> predictions = classifier.Classify(image_roi);
				Prediction label = predictions[0];
				string text = label.first;
				/* Print the top N predictions. */
				for (size_t i = 0; i < predictions.size(); ++i) {
					Prediction p = predictions[i];
					std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
						<< p.first << "\"" << std::endl;// p.second代表概率值，p.first代表类别标签
				}
				//std::cout << "text:" <<text.c_str() << std::endl;
				std::cout << "-------------------------" << std::endl;


				cvPutText(show, text.c_str(), cvPoint(10, 55), &font, cvScalar(0, 0, 255, NULL));

				if (text == "sunglasses" && last_text != "sunglasses")
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


				cv::imshow("face", image_roi);
				}
			}

			cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
			cvShowImage("Test",show);
			cv::waitKey(30);


			char c = (char)cv::waitKey(1);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
												 
		}



	}
}
