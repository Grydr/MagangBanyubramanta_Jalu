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

const char* coconame[] = {"Baskom", "Celana", "Flare"};

struct Detection {
   float conf;
   int class_;
   cv::Rect box;
};

class ObjectDetection : public rclcpp::Node {
  public:
   ObjectDetection() : Node("object_detection") {
      this->declare_parameter<std::string>("model_path",
                                           "src/openvino/include/best.onnx");

      image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
          "camera", rclcpp::SensorDataQoS(),
          std::bind(&ObjectDetection::image_callback, this,
                    std::placeholders::_1));

      object_pub_ =
          this->create_publisher<interfaces::msg::Object>("objects", 10);

      image_with_box_pub_ =
          this->create_publisher<sensor_msgs::msg::Image>("objects_box", 10);

      init_model();
   }

  private:
   void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
      cv::Mat frame;
      cv_bridge::CvImagePtr cv_ptr;
      try {
         cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
         frame = cv_ptr->image;
      } catch (cv_bridge::Exception& e) {
         RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      }

      if (frame.empty()) {
         RCLCPP_WARN(this->get_logger(), "Received empty frame!");
      }

      float scale_x = static_cast<float>(frame.cols) / INPUT_WIDTH;
      float scale_y = static_cast<float>(frame.rows) / INPUT_HEIGHT;

      std::vector<Detection> detections = process_frame(frame);

      for (auto& detection : detections) {
         detection.box.x = static_cast<int>(detection.box.x * scale_x);
         detection.box.y = static_cast<int>(detection.box.y * scale_y);
         detection.box.width = static_cast<int>(detection.box.width * scale_x);
         detection.box.height =
             static_cast<int>(detection.box.height * scale_y);
      }

      for (const auto& detection : detections) {
         cv::Rect valid_box =
             detection.box & cv::Rect(0, 0, frame.cols, frame.rows);
         cv::rectangle(frame, valid_box, cv::Scalar(0, 255, 0), 2);

         std::ostringstream label;
         label << coconame[detection.class_] << " (" << std::fixed
               << std::setprecision(2) << detection.conf * 100 << "%)"
               << " [x: " << detection.box.x << ", y: " << detection.box.y
               << ", w: " << detection.box.width
               << ", h: " << detection.box.height << "]";

         int baseline = 0;
         cv::Size text_size = cv::getTextSize(
             label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
         cv::Point text_origin(detection.box.x,
                               detection.box.y - text_size.height - 5);

         int label_x = std::max(0, text_origin.x);
         int label_y = std::max(0, text_origin.y);

         cv::rectangle(
             frame, cv::Point(label_x, label_y + baseline),
             cv::Point(label_x + text_size.width, label_y - text_size.height),
             cv::Scalar(0, 255, 0), cv::FILLED);

         cv::putText(frame, label.str(), cv::Point(label_x, label_y),
                     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
      }

      sensor_msgs::msg::Image::SharedPtr out_msg =
          cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
      image_with_box_pub_->publish(*out_msg);

      for (const auto& detection : detections) {
         interfaces::msg::Object obj_msg;
         obj_msg.name = coconame[detection.class_];
         obj_msg.x = detection.box.x;
         obj_msg.y = detection.box.y;
         obj_msg.width = detection.box.width;
         obj_msg.height = detection.box.height;
         obj_msg.confidence = detection.conf;

         object_pub_->publish(obj_msg);
      }
   }

   std::vector<Detection> process_frame(cv::Mat& frame) {
      std::vector<Detection> detections;

      cv::Mat resized;
      cv::resize(frame, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), 0, 0,
                 cv::INTER_LINEAR);

      ov::Tensor input_tensor =
          ov::Tensor(compiled_model_.input().get_element_type(),
                     {1, INPUT_HEIGHT, INPUT_WIDTH, 3}, resized.data);
      infer_request_.set_input_tensor(input_tensor);

      infer_request_.infer();

      const ov::Tensor& output_tensor = infer_request_.get_output_tensor();
      if (!output_tensor) {
         RCLCPP_ERROR(this->get_logger(), "Inference output tensor is null.");
         return detections;
      }

      ov::Shape output_shape = output_tensor.get_shape();
      float* output_data = output_tensor.data<float>();

      std::vector<cv::Rect> boxes;
      std::vector<int> class_ids;
      std::vector<float> confidences;

      size_t num_detections = output_shape[1];
      size_t num_attributes = output_shape[2];

      for (size_t i = 0; i < num_detections; i++) {
         float* detection = output_data + (i * num_attributes);
         float confidence = detection[4];

         if (confidence < CONF_THRESH) continue;

         float* class_scores = detection + 5;
         cv::Mat scores(1, N_CLASSES, CV_32FC1, class_scores);
         cv::Point class_id_point;
         double max_class_score;
         cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

         if (max_class_score < SCORE_THRESH) continue;

         int class_id = class_id_point.x;

         if (class_id < 0 || class_id >= N_CLASSES) continue;

         float x_center = detection[0];
         float y_center = detection[1];
         float w = detection[2];
         float h = detection[3];
         float x_min = x_center - (w / 2);
         float y_min = y_center - (h / 2);

         confidences.push_back(confidence);
         class_ids.push_back(class_id);
         boxes.push_back(cv::Rect(x_min, y_min, w, h));
      }

      std::vector<int> nms_result;
      cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESH, NMS_THRESH,
                        nms_result);

      for (int idx : nms_result) {
         Detection d;
         d.class_ = class_ids[idx];
         d.conf = confidences[idx];
         d.box = boxes[idx];
         detections.push_back(d);
      }

      return detections;
   }

   void init_model() {
      try {
         core_ = std::make_shared<ov::Core>();
         std::string model_path = this->get_parameter("model_path").as_string();

         if (model_path[0] != '/') {
            model_path = std::filesystem::absolute(model_path).string();
         }

         RCLCPP_INFO(this->get_logger(), "Using model path: %s",
                     model_path.c_str());

         // debug verif file model
         if (!std::filesystem::exists(model_path)) {
            RCLCPP_ERROR(this->get_logger(),
                         "Model file does not exist at path: %s",
                         model_path.c_str());
            throw std::runtime_error("Model file not found");
         }

         std::shared_ptr<ov::Model> model = core_->read_model(model_path);

         ov::preprocess::PrePostProcessor ppp =
             ov::preprocess::PrePostProcessor(model);
         ppp.input()
             .tensor()
             .set_element_type(ov::element::u8)
             .set_layout("NHWC")
             .set_color_format(ov::preprocess::ColorFormat::BGR);
         ppp.input()
             .preprocess()
             .convert_element_type(ov::element::f32)
             .convert_color(ov::preprocess::ColorFormat::RGB);
         ppp.input().model().set_layout("NCHW");
         ppp.output().tensor().set_element_type(ov::element::f32);
         model = ppp.build();

         compiled_model_ = core_->compile_model(model, "CPU");
         infer_request_ = compiled_model_.create_infer_request();

         RCLCPP_INFO(this->get_logger(),
                     "OpenVINO model initialized successfully.");
      } catch (const std::exception& e) {
         RCLCPP_ERROR(this->get_logger(),
                      "Error initializing OpenVINO model: %s", e.what());
      }
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
