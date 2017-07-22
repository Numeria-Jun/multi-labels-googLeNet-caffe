#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             string& label_result);
           //  const vector<string>& label_files);

  std::vector<vector<Prediction> > Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<vector<float> > Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);
    
  //void Classifier::split(const string& str,const string& pattern);
  

 private:
  vector<string>label_file;
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<vector<string> > labels_; //multi
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       string& label_result) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  //CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  const string pattern="\t";
  std::string::size_type pos;
  label_result=label_result+pattern;
  int size=label_result.size();
  for(int i=0; i<size; i++)
  {
      pos=label_result.find(pattern,i);
      if(pos<size)
      {
          string s=label_result.substr(i,pos-i);
          label_file.push_back(s);
          i=pos+pattern.size()-1;
      }
  }
  
  //label_file=split(label_result,pattern);
  //2 labels should read
  string line;
  for (int i = 0; i < label_file.size(); i++)
  {
	  std::ifstream labels(label_file[i].c_str());
	  CHECK(labels) << "Unable to open labels file " << label_file[i];
	  vector<string> label_array;
	  while (std::getline(labels, line))
	  {
		  label_array.push_back(line);
	  }
	  Blob<float>* output_layer = net_->output_blobs()[i];
	  CHECK_EQ(label_array.size(), output_layer->channels())
		  << "Number of labels is different from the output layer dimension.";
	  labels_.push_back(label_array);
  }
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

//std::vector<std::string> void Classifier::split(const string& str,const string& pattern)
//{
/*      size_type pos;
      vector<string> result;
      str+=pattern;
      int size=str.size();
     
      for(int i=0; i<size; i++)
          {
                pos=str.find(pattern,i);
                if(pos<size)
                    {
                          string s=str.substr(i,pos-i);
                          result.push_back(s);
                          i=pos+pattern.size()-1;
                        }
              }
      return result;
}*/

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
//revise 1.
std::vector<vector<Prediction> > Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<vector<float> >output;
  output = Predict(img);
  int N1 = std::min<int>(labels_[0].size(), N);
  int N2 = std::min<int>(labels_[1].size(), N);
  int N3 = std::min<int>(labels_[2].size(), N);
  std::vector<int> maxN1 = Argmax(output[0], N1);
  std::vector<int> maxN2 = Argmax(output[1], N2);
  std::vector<int> maxN3 = Argmax(output[2], N3);
  std::vector<Prediction> predictions1;
  std::vector<Prediction> predictions2;
  std::vector<Prediction> predictions3;

  for (int i = 0; i < N1; ++i) {
    int idx = maxN1[i];
    predictions1.push_back(std::make_pair(labels_[0][idx], output[0][idx]));
  }
  for (int i = 0; i < N2; ++i) {
	  int idx = maxN2[i];
	  predictions2.push_back(std::make_pair(labels_[1][idx], output[1][idx]));
  }
  for (int i = 0; i < N3; ++i) {
        int idx = maxN3[i];
        predictions3.push_back(std::make_pair(labels_[2][idx], output[2][idx]));
  }
  
  vector<vector<Prediction> > predictions;
  predictions.push_back(predictions1);
  predictions.push_back(predictions2);
  predictions.push_back(predictions3);
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

std::vector<vector<float> > Classifier::Predict(const cv::Mat& img) {
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
//revise 2.
  Blob<float>* output_layer1 = net_->output_blobs()[0];
  Blob<float>* output_layer2 = net_->output_blobs()[1];
  Blob<float>* output_layer3 = net_->output_blobs()[2];
  const float* begin1 = output_layer1->cpu_data();
  const float* end1 = begin1+ output_layer1->channels();
  const float* begin2 = output_layer2->cpu_data();
  const float* end2 = begin2 + output_layer2->channels();
  const float* begin3 = output_layer3->cpu_data();
  const float* end3 = begin3 + output_layer3->channels();

  std::vector<float> prob1(begin1, end1);
  std::vector<float> prob2(begin2, end2);
  std::vector<float> prob3(begin3, end3);
  vector<vector<float> > prob_matrix;
  prob_matrix.push_back(prob1);
  prob_matrix.push_back(prob2);
  prob_matrix.push_back(prob3);
  return prob_matrix;
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
//revise 3.
int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto label1.txt label2.txt label3.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file1   = argv[4];
  string label_file2   = argv[5];
//revise 4.
  string label_file3   = argv[6];
//  vector<string> label_file;
//  label_file.push_back(label_file1);
//  label_file.push_back(label_file2);
//revise 5.
//  label_file.push_back(label_file3);
   string label_file;
  label_file=label_file1+'\t'+label_file2+'\t'+label_file3;
 // std::cout << "the labels' channel:"<<label_file.size() << std::endl;
  Classifier classifier(model_file, trained_file, mean_file, label_file);
//revise 6.
  string file = argv[7];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;
  
  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  vector<vector<Prediction> > predictions;
  predictions = classifier.Classify(img);
  std::cout << "have runed classifier.Classify" << std::endl;
  /* Print the top N predictions. */
//revise 7.
  std::cout << "---------- car year------------" << std::endl;
  for (size_t i = 0; i < predictions[0].size(); ++i) {
    Prediction p = predictions[0][i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
  std::cout << " ----------car type------------" << std::endl;
  for (size_t i = 0; i < predictions[1].size(); ++i) {
	  Prediction p = predictions[1][i];
	  std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
		        << p.first << "\"" << std::endl;
  }
    std::cout << " ----------car name------------" << std::endl;
  for (size_t i = 0; i < predictions[2].size(); ++i) {
      Prediction p = predictions[2][i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                << p.first << "\"" << std::endl;
  }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
