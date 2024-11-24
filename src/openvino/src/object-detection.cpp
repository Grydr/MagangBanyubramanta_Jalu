#include <filesystem>
#include <iomanip>
#include <memory>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "interfaces/msg/object.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#define N_CLASSES 3
#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640
#define CONF_THRESH 0.4
#define SCORE_THRESH 0.4
#define NMS_THRESH 0.4

using namespace cv;
using namespace std;
using namespace dnn;

const char* coconame[] = {"Baskom", "Celana", "Flare"};

const float color_list[80][3] = {
    {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556}, {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184}, {0.300, 0.300, 0.300}, {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000}, {1.000, 0.500, 0.000}, {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 1.000}, {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000}, {0.333, 0.667, 0.000}, {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000}, {0.667, 0.667, 0.000}, {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000}, {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500}, {0.000, 0.667, 0.500}, {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500}, {0.333, 0.333, 0.500}, {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500}, {0.667, 0.000, 0.500}, {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500}, {0.667, 1.000, 0.500}, {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500}, {1.000, 0.667, 0.500}, {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000}, {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000}, {0.333, 0.333, 1.000}, {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000}, {0.667, 0.000, 1.000}, {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000}, {0.667, 1.000, 1.000}, {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000}, {1.000, 0.667, 1.000}, {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000}, {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 0.167}, {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500}, {0.000, 0.000, 0.667}, {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000}, {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714}, {0.857, 0.857, 0.857}, {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741}, {0.50, 0.5, 0}};

struct Config {
   float confThreshold;
   float nmsThreshold;
   float scoreThreshold;
   int inpWidth;
   int inpHeight;
   std::string onnx_path;
};

struct Resize {
   cv::Mat resized_image;
   int dw;
   int dh;
};

struct Detection {
   int class_id;
   float confidence;
   cv::Rect box;
};

class YOLOV5 {
  public:
   YOLOV5(Config config);
   ~YOLOV5();
   void detect(cv::Mat& frame);

  private:
   float confThreshold;
   float nmsThreshold;
   float scoreThreshold;
   int inpWidth;
   int inpHeight;
   float rx;  // the width ratio of original image and resized image
   float ry;  // the height ratio of original image and resized image
   std::string onnx_path;
   Resize resize;
   ov::Tensor input_tensor;
   ov::InferRequest infer_request;
   ov::CompiledModel compiled_model;
   void initialmodel();
   void preprocess_img(cv::Mat& frame);
   void postprocess_img(cv::Mat& frame, float* detections,
                        ov::Shape& output_shape);
};

YOLOV5::YOLOV5(Config config) {
   this->confThreshold = config.confThreshold;
   this->nmsThreshold = config.nmsThreshold;
   this->scoreThreshold = config.scoreThreshold;
   this->inpWidth = config.inpWidth;
   this->inpHeight = config.inpHeight;
   this->onnx_path = config.onnx_path;
   this->initialmodel();
}
YOLOV5::~YOLOV5() {}
void YOLOV5::detect(Mat& frame) {
   preprocess_img(frame);
   infer_request.infer();
   const ov::Tensor& output_tensor = infer_request.get_output_tensor();
   ov::Shape output_shape = output_tensor.get_shape();
   float* detections = output_tensor.data<float>();
   this->postprocess_img(frame, detections, output_shape);
}

void YOLOV5::initialmodel() {
   ov::Core core;
   std::shared_ptr<ov::Model> model = core.read_model(this->onnx_path);
   ov::preprocess::PrePostProcessor ppp =
       ov::preprocess::PrePostProcessor(model);
   ppp.input()
       .tensor()
       .set_element_type(ov::element::u8)
       .set_layout("NHWC")
       .set_color_format(ov::preprocess::ColorFormat::RGB);
   ppp.input()
       .preprocess()
       .convert_element_type(ov::element::f32)
       .convert_color(ov::preprocess::ColorFormat::RGB)
       .scale({255, 255, 255});  // .scale({ 112, 112, 112 });
   ppp.input().model().set_layout("NCHW");
   ppp.output().tensor().set_element_type(ov::element::f32);
   model = ppp.build();
   this->compiled_model = core.compile_model(model, "CPU");
   this->infer_request = compiled_model.create_infer_request();
}

void YOLOV5::preprocess_img(Mat& frame) {
   float width = frame.cols;
   float height = frame.rows;
   cv::Size new_shape = cv::Size(inpWidth, inpHeight);
   float r = float(new_shape.width / max(width, height));
   int new_unpadW = int(round(width * r));
   int new_unpadH = int(round(height * r));

   cv::resize(frame, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0,
              0, cv::INTER_AREA);
   resize.dw = new_shape.width - new_unpadW;
   resize.dh = new_shape.height - new_unpadH;
   cv::Scalar color = cv::Scalar(100, 100, 100);
   cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh,
                      0, resize.dw, cv::BORDER_CONSTANT, color);

   this->rx =
       (float)frame.cols / (float)(resize.resized_image.cols - resize.dw);
   this->ry =
       (float)frame.rows / (float)(resize.resized_image.rows - resize.dh);
   float* input_data = (float*)resize.resized_image.data;
   input_tensor = ov::Tensor(compiled_model.input().get_element_type(),
                             compiled_model.input().get_shape(), input_data);
   infer_request.set_input_tensor(input_tensor);
}

void YOLOV5::postprocess_img(Mat& frame, float* detections,
                             ov::Shape& output_shape) {
   std::vector<cv::Rect> boxes;
   vector<int> class_ids;
   vector<float> confidences;
   for (int i = 0; i < output_shape[1]; i++) {
      float* detection = &detections[i * output_shape[2]];

      float confidence = detection[4];
      if (confidence >= this->confThreshold) {
         float* classes_scores = &detection[5];
         cv::Mat scores(1, output_shape[2] - 5, CV_32FC1, classes_scores);
         cv::Point class_id;
         double max_class_score;
         cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
         if (max_class_score > this->scoreThreshold) {
            confidences.push_back(confidence);
            class_ids.push_back(class_id.x);
            float x = detection[0];
            float y = detection[1];
            float w = detection[2];
            float h = detection[3];
            float xmin = x - (w / 2);
            float ymin = y - (h / 2);

            boxes.push_back(cv::Rect(xmin, ymin, w, h));
         }
      }
   }

   std::vector<int> nms_result;
   cv::dnn::NMSBoxes(boxes, confidences, this->scoreThreshold,
                     this->nmsThreshold, nms_result);

   std::vector<Detection> output;
   for (int i = 0; i < nms_result.size(); i++) {
      Detection result;
      int idx = nms_result[i];
      result.class_id = class_ids[idx];
      result.confidence = confidences[idx];
      result.box = boxes[idx];
      output.push_back(result);
   }
   cout << "output_size:" << output.size() << endl;

   for (int i = 0; i < output.size(); i++) {
      auto detection = output[i];
      auto box = detection.box;
      auto classId = detection.class_id;
      // if (classId != 0) continue;
      auto confidence = detection.confidence;

      box.x = this->rx * box.x;
      box.y = this->ry * box.y;
      box.width = this->rx * box.width;
      box.height = this->ry * box.height;
      float xmax = box.x + box.width;
      float ymax = box.y + box.height;
      cv::Scalar color =
          cv::Scalar(color_list[classId][0], color_list[classId][1],
                     color_list[classId][2]);
      float c_mean = cv::mean(color)[0];
      cv::Scalar txt_color;
      if (c_mean > 0.5) {
         txt_color = cv::Scalar(0, 0, 0);
      } else {
         txt_color = cv::Scalar(255, 255, 255);
      }
      cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(xmax, ymax),
                    color * 255, 2);
      int baseLine = 0;
      char text[256];
      sprintf(text, "%s %0.1f%%", coconame[classId], confidence * 100);
      cv::Size label_size =
          cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
      cv::Scalar txt_bk_color = color * 0.7 * 255;
      cv::rectangle(
          frame,
          cv::Rect(cv::Point(box.x, box.y),
                   cv::Size(label_size.width, label_size.height + baseLine)),
          txt_bk_color, -1);
      cv::putText(frame, text, cv::Point(box.x, box.y + label_size.height),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
   }
}

class ObjectDetection : public rclcpp::Node {
   void image_callback() {
      try {
         const std::string input_video_path = "../include/third.mp4";
         const std::string output_video_path = "../include/result.mp4";
         // confThreshold;nmsThreshold;scoreThreshold;inpWidth;inpHeight;onnx_path;
         Config config = {0.4, 0.4, 0.4, 640, 640, "../include/best.onnx"};
         YOLOV5 yolomodel(config);

         cv::VideoCapture cap(input_video_path);
         if (!cap.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Cant open input video file");
            return;
         }

         cv::Mat frame;
         cap.read(frame);
         if (frame.empty())
            RCLCPP_ERROR(this->get_logger(), "Cant process video frame");
         yolomodel.detect(frame);
         cap.release();

         RCLCPP_INFO(this->get_logger(), "Video processing complete");
         image_with_box_pub_->publish()
      } catch (const std::exception& ex) {
         RCLCPP_ERROR(this->get_logger(), "%s", ex.what());
         return;
      }
   };

  public:
   ObjectDetection() : Node("object_detection") {
      image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
          "camera", rclcpp::SensorDataQoS(),
          std::bind(&ObjectDetection::image_callback, this,
                    std::placeholders::_1));

      object_pub_ =
          this->create_publisher<interfaces::msg::Object>("objects", 10);

      image_with_box_pub_ =
          this->create_publisher<sensor_msgs::msg::Image>("objects_box", 10);
   }

   rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
   rclcpp::Publisher<interfaces::msg::Object>::SharedPtr object_pub_;
   rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_with_box_pub_;

   std::shared_ptr<ov::Core> core_;
   ov::CompiledModel compiled_model_;
   ov::InferRequest infer_request_;
};

int main(int argc, char* argv[]) {
   rclcpp::init(argc, argv);
   rclcpp::spin(std::make_shared<ObjectDetection>());
   rclcpp::shutdown();
   return 0;
}
