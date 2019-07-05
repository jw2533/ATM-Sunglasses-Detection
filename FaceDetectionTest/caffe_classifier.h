#ifndef CAFFE_CLASSFIER_H_
#define CAFFE_CLASSFIER_H_

#include <mutex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ostream>
#include <opencv2/opencv.hpp>



/* Pair (label, confidence) representing a prediction. */
typedef std::pair<std::string, float> Prediction;
inline std::ostream& operator<< (std::ostream& out, const Prediction&pred)
{
    out << "[" << pred.first << ", " << pred.second * 100.0 << "%]";
    return out;
}
static bool PairCompare (const std::pair<float, int>& lhs,
                         const std::pair<float, int>& rhs);
/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax (const std::vector<float>& v, int N);

/* Return the indices of the top N values of vector v. */
std::vector<int> Argmax (const std::vector<float>& v, int N);

class Classifier
{
    public:
        Classifier (const std::string& model_file,
                    const std::string& trained_file,
                    const std::string& mean_file,
                    const std::string& label_file);
        //使用约定的方式来初始化classifier，规定文件夹内文件命名方式如下：
        //model_file   :deploy.prototxt
        //trained_file :model.caffemodel
        //mean_file    :mean.binaryproto
        //label_file   :label.txt
        explicit Classifier (const std::string& cnn_model_path);
        ~Classifier();
        Classifier (const Classifier&) = delete;
        Classifier (Classifier&&) = delete;
        Classifier&operator = (const Classifier&) = delete;
        
        std::vector<Prediction> Classify (const cv::Mat& img, int N = 2); //线程安全
        
    private:
        Classifier&operator = (Classifier&&);//禁止外界的移动
        void SetMean (const std::string& mean_file);
        
        std::vector<float> Predict (const cv::Mat& img);
        
        void WrapInputLayer (std::vector<cv::Mat>* input_channels);
        
        void Preprocess (const cv::Mat& img, std::vector<cv::Mat>* input_channels);
        
        
    private:
    
        std::shared_ptr<std::mutex> classify_mtx_ptr_{new std::mutex};//用于Classify的互斥
        //std::shared_ptr<caffe::Net<float> > net_;
        void *net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        std::vector<std::string> labels_;
};

//pool中的每一个分类器都有一个对应的mutex
class ClassifierPool
{
    public:
        //第二个参数表示分类器的个数
        ClassifierPool (const std::string &caffe_model_path, int num);
        Classifier& get_classifier();
    private:
        std::vector<Classifier> classifiers_;
        std::vector<std::mutex> mutex_vec_;
};
inline bool PairCompare (const std::pair<float, int>& lhs,
                         const std::pair<float, int>& rhs)
{
    return lhs.first > rhs.first;
}



#endif // !CAFFE_CLASSFIER.H


