#include"caffe_classifier.h"
#include <caffe/caffe.hpp>
#include <mutex>

#ifdef  AAA

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
#ifdef CAFFE_WITH_3RD_LAYERS
    #include "caffe/layers/shuffle_channel_layer.hpp"
    #include "caffe/layers/shuffle_channel_layer.hpp"
    #include "caffe/layers/scale_layer.hpp"
    #include "caffe/layers/concat_layer.hpp"
#endif
namespace caffe
{
    extern INSTANTIATE_CLASS (BatchNormLayer);
    extern INSTANTIATE_CLASS (BiasLayer);
    extern INSTANTIATE_CLASS (InputLayer);
    extern INSTANTIATE_CLASS (InnerProductLayer);
    extern INSTANTIATE_CLASS (DropoutLayer);
    extern INSTANTIATE_CLASS (ConvolutionLayer);
    //REGISTER_LAYER_CLASS (Convolution);
    extern INSTANTIATE_CLASS (ReLULayer);
    //REGISTER_LAYER_CLASS (ReLU);
    extern INSTANTIATE_CLASS (PoolingLayer);
    //REGISTER_LAYER_CLASS (Pooling);
    extern INSTANTIATE_CLASS (LRNLayer);
    //REGISTER_LAYER_CLASS (LRN);
    extern INSTANTIATE_CLASS (SoftmaxLayer);
    //REGISTER_LAYER_CLASS (Softmax);
    extern INSTANTIATE_CLASS (ConcatLayer);
    extern INSTANTIATE_CLASS (PReLULayer);
#ifdef CAFFE_WITH_3RD_LAYERS
    //REGISTER_LAYER_CLASS (ShuffleChannel);
    extern INSTANTIATE_CLASS (ShuffleChannelLayer);
    extern INSTANTIATE_CLASS (ScaleLayer);
    extern INSTANTIATE_CLASS (ConcatLayer);
#endif
}
#endif



using namespace caffe;  // NOLINT(build/namespaces)
Classifier::Classifier (const string& model_file,
                        const string& trained_file,
                        const string& mean_file,
                        const string& label_file)
{
#ifdef CPU_ONLY
    Caffe::set_mode (Caffe::CPU);
#else
    Caffe::set_mode (Caffe::GPU);
#endif
    /* Load the network. */
    net_ = new Net<float> (model_file, TEST);
    static_cast<Net<float>*> (net_)->CopyTrainedLayersFrom (trained_file);
    CHECK_EQ (static_cast<Net<float>*> (net_)->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ (static_cast<Net<float>*> (net_)->num_outputs(), 1) << "Network should have exactly one output.";
    Blob<float>* input_layer = static_cast<Net<float>*> (net_)->input_blobs() [0];
    num_channels_ = input_layer->channels();
    CHECK (num_channels_ == 3 || num_channels_ == 1)
            << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size (input_layer->width(), input_layer->height());
    /* Load the binaryproto mean file. */
    SetMean (mean_file);
    /* Load labels. */
    std::ifstream labels (label_file.c_str());
    CHECK (labels) << "Unable to open labels file " << label_file;
    string line;
    
    while (std::getline (labels, line))
    {
        labels_.push_back (string (line));
    }
    
    Blob<float>* output_layer = static_cast<Net<float>*> (net_)->output_blobs() [0];
    CHECK_EQ (labels_.size(), output_layer->channels())
            << "Number of labels is different from the output layer dimension.";
}

Classifier::~Classifier()
{
    //std::cout << "deconstructor Classifier object" << std::endl;
    delete static_cast<Net<float>*> (net_);
}

Classifier::Classifier (const string& cnn_model_path)
{
    std::string suffix;
    
    if (cnn_model_path.back() != '\\' || cnn_model_path.back() != '/')
    {
        suffix += '/';
    }
    
    string deploy_file = cnn_model_path + suffix + "deploy.prototxt";
    string model_file = cnn_model_path + suffix + "model.caffemodel";
    string mean_file = cnn_model_path + suffix + "mean.binaryproto";
    string label_file = cnn_model_path + suffix + "label.txt";
    Classifier (deploy_file, model_file, mean_file, label_file);
    //可以使用构造函数委托
    *this = Classifier (deploy_file, model_file, mean_file, label_file);//移动赋值
}

//赋值一般需要考虑三个问题：自赋值、异常安全、资源释放
//故swap和赋值是绝配
Classifier& Classifier::operator = (Classifier&&c)
{
    std::lock (*classify_mtx_ptr_, *c.classify_mtx_ptr_);
    std::lock_guard<std::mutex> lg_1 (*classify_mtx_ptr_, std::adopt_lock);
    std::lock_guard<std::mutex> lg_2 (*c.classify_mtx_ptr_, std::adopt_lock);
    std::swap (classify_mtx_ptr_, c.classify_mtx_ptr_);
    std::swap (net_, c.net_);
    std::swap (input_geometry_, c.input_geometry_);
    std::swap (num_channels_, c.num_channels_);
    std::swap (mean_, c.mean_);
    std::swap (labels_, c.labels_);
    // c.net_ = nullptr;
    return *this;
}

std::vector<Prediction> Classifier::Classify (const cv::Mat& img, int N)
{
    std::vector<float> output;
    {
        std::lock_guard<std::mutex> lg (*classify_mtx_ptr_);
        output = Predict (img);
        N = std::min<int> (labels_.size(), N);
    }
    std::vector<int> maxN = Argmax (output, N);
    std::vector<Prediction> predictions;
    
    for (int i = 0; i < N; ++i)
    {
        int idx = maxN[i];
        predictions.push_back (std::make_pair (labels_[idx], output[idx]));
    }
    
    return predictions;
}

void Classifier::SetMean (const string& mean_file)
{
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie (mean_file.c_str(), &blob_proto);
    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto (blob_proto);
    CHECK_EQ (mean_blob.channels(), num_channels_)
            << "Number of channels of mean file doesn't match input layer.";
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    
    for (int i = 0; i < num_channels_; ++i)
    {
        /* Extract an individual channel. */
        cv::Mat channel (mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        channels.push_back (channel);
        data += mean_blob.height() * mean_blob.width();
    }
    
    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge (channels, mean);
    /* Compute the global mean pixel value and create a mean image
    * filled with this value. */
    cv::Scalar channel_mean = cv::mean (mean);
    mean_ = cv::Mat (input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict (const cv::Mat& img)
{
    Blob<float>* input_layer = static_cast<Net<float>*> (net_)->input_blobs() [0];
    input_layer->Reshape (1, num_channels_,
                          input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    static_cast<Net<float>*> (net_)->Reshape();
    std::vector<cv::Mat> input_channels;
    WrapInputLayer (&input_channels);
    Preprocess (img, &input_channels);
    static_cast<Net<float>*> (net_)->Forward();
    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = static_cast<Net<float>*> (net_)->output_blobs() [0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float> (begin, end);
}

void Classifier::WrapInputLayer (std::vector<cv::Mat>* input_channels)
{
    Blob<float>* input_layer = static_cast<Net<float>*> (net_)->input_blobs() [0];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel (height, width, CV_32FC1, input_data);
        input_channels->push_back (channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess (const cv::Mat& img,
                             std::vector<cv::Mat>* input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    
    if (img.channels() == 3 && num_channels_ == 1)
    {
        cv::cvtColor (img, sample, cv::COLOR_BGR2GRAY);
    }
    
    else
        if (img.channels() == 4 && num_channels_ == 1)
        {
            cv::cvtColor (img, sample, cv::COLOR_BGRA2GRAY);
        }
        
        else
            if (img.channels() == 4 && num_channels_ == 3)
            {
                cv::cvtColor (img, sample, cv::COLOR_BGRA2BGR);
            }
            
            else
                if (img.channels() == 1 && num_channels_ == 3)
                {
                    cv::cvtColor (img, sample, cv::COLOR_GRAY2BGR);
                }
                
                else
                {
                    sample = img;
                }
                
    cv::Mat sample_resized;
    
    if (sample.size() != input_geometry_)
    {
        cv::resize (sample, sample_resized, input_geometry_);
    }
    
    else
    {
        sample_resized = sample;
    }
    
    cv::Mat sample_float;
    
    if (num_channels_ == 3)
    {
        sample_resized.convertTo (sample_float, CV_32FC3);
    }
    
    else
    {
        sample_resized.convertTo (sample_float, CV_32FC1);
    }
    
    cv::Mat sample_normalized;
    cv::subtract (sample_float, mean_, sample_normalized);
    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split (sample_normalized, *input_channels);
    CHECK (reinterpret_cast<float*> (input_channels->at (0).data)
           == static_cast<Net<float>*> (net_)->input_blobs() [0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
}

std::vector<int> Argmax (const std::vector<float>& v, int N)
{
    std::vector<std::pair<float, int> > pairs;
    
    for (size_t i = 0; i < v.size(); ++i)
    {
        pairs.push_back (std::make_pair (v[i], static_cast<int> (i)));
    }
    
    std::partial_sort (pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
    std::vector<int> result;
    
    for (int i = 0; i < N; ++i)
    {
        result.push_back (pairs[i].second);
    }
    
    return result;
}

