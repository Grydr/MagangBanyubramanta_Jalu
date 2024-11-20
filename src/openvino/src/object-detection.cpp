#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include <cv_bridge/cv_bridge.h> 
#include <image_transport/image_transport.hpp> 
#include <opencv2/opencv.hpp> 

using namespace std::chrono_literals;

#define N_CLASSES 80
#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640
#define CONF_THRESH 0.4
#define SCORE_THRESH 0.4
#define NMS_THRESH 0.4

const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

class ObjectDetection : public rclcpp::Node {
   public:
      ObjectDetection() : Node("object_detection") {
         sub_raw_ = this->create_subscription<sensor_msgs::msg::Image>("camera", 10, std::bind(&ObjectDetection::detectionCallback, this, std::placeholders::_1));
         pub_obj_ = this->create_publisher<sensor_msgs::msg::Image>("object", 10);

         initModel();
      }

   private:
      void initModel() {
         ov::Core core;
         std::shared_ptr<ov::Model> model = core.read_model("src/openvino/include/best.onnx");

         ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
         ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
         ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB);
         ppp.input().model().set_layout("NCHW");
         ppp.output(0).tensor().set_element_type(ov::element::f32);
         model = ppp.build();

         ov::CompiledModel compiled_model = core.compile_model(model, "CPU"); // CPU / CPU_ARM / GPU / dGPU / NPU / AUTO / BATCH / HETERO
         ov::InferRequest infer_request = compiled_model.create_infer_request();
      }

      void detectionCallback(const sensor_msgs::msg::Image::SharedPtr &msg) {
         cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
         cv::Mat frame = cv_ptr->image;

         interferenceProcessing(frame);
      }

      void interferenceProcessing(const cv::Mat &frame) {
         // Preprocessing

         float width = frame.cols;
         float height = frame.rows;

         cv::Size new_shape = cv::Size(INPUT_WIDTH, INPUT_HEIGHT);

         float ratio = float(new_shape.width / std::max(width, height));

         int new_width = int(std::round(width * ratio));
         int new_height = int(std::round(height * ratio));

         int padding_x = new_shape.width - new_width;
         int padding_y = new_shape.height - new_height;
         cv::Scalar padding_color = cv::Scalar(100, 100, 100);
         
         cv::Mat input_frame;
         cv::resize(frame, input_frame, cv::Size(new_width, new_height), 0, 0, cv::INTER_AREA);
         cv::copyMakeBorder(input_frame, input_frame, 0, padding_y, 0, padding_x, cv::BORDER_CONSTANT, padding_color);

         float ratio_x = (float)frame.cols / (float)(input_frame.cols - padding_x);
         float ratio_y = (float)frame.rows / (float)(input_frame.rows - padding_y);

         // cv::imshow("input", input_frame);
         
         float *input_data = (float *)input_frame.data;
         ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
         infer_request.set_input_tensor(input_tensor);

         infer_request.infer();

         const ov::Tensor& output_tensor = infer_request.get_output_tensor();
         ov::Shape output_shape = output_tensor.get_shape();
         float *detections = output_tensor.data<float>();

         // Post processing

         std::vector<cv::Rect> boxes;
         std::vector<int> class_ids;
         std::vector<float> confidences;

         for (int i = 0; i < output_shape[1]; i++) {
            float *detection = detections + (i * output_shape[2]);
            float confidence = detection[4];

            if (confidence < CONF_THRESH)
               continue;
            
            float *class_scores = detection + 5;
            cv::Mat scores(1, N_CLASSES, CV_32FC1, class_scores);
            cv::Point class_id;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score < SCORE_THRESH)
               continue;
            
            float x = detection[0];
            float y = detection[1];
            float w = detection[2];
            float h = detection[3];
            float x_min = x - (w / 2);
            float y_min = y - (h / 2);

            confidences.push_back(confidence);
            class_ids.push_back(class_id.x);
            boxes.push_back(cv::Rect(x_min, y_min, w, h));
         }

         std::vector<int> nms_result;
         cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESH, NMS_THRESH, nms_result);

         // Prepare output message
         sensor_msgs::msg::Image output_image;
         output_image.header.stamp = this->now();
         output_image.height = frame.rows
         output_image.height = frame.rows;
         output_image.width = frame.cols;
         output_image.encoding = "bgr8"; // Assuming the image is in BGR format
         output_image.is_bigendian = 0;
         output_image.step = frame.step[0];
         output_image.data.resize(frame.total() * frame.elemSize());
         std::memcpy(output_image.data.data(), frame.data, output_image.data.size());

         std::vector<Detection> output;
         
         for (int &i : nms_result) {	
            Detection d;
            d.class_ = class_ids[i];
            d.conf = confidences[i];
            d.box = boxes[i];
            output.push_back(d);
         }
         
         // Display Results
         for (Detection &d : output) {
            cv::Rect &box = d.box;

            box.x /= ratio_x;
            box.y /= ratio_y;
            box.width /= ratio_x;
            box.height /= ratio_y;

            float x_max = box.x + box.width;
            float y_max = box.y + box.height;

            cv::Scalar color = cv::Scalar(color_list[d.class_][0], color_list[d.class_][1], color_list[d.class_][2]);
            float color_mean = cv::mean(color)[0];
            
            cv::Scalar text_color = color_mean > 0.5 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
            
            cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(x_max, y_max), color * 255, 2);
            
            int baseline = 0;
            // char text[256];
            // sprintf(text, "%s %0.1f%%", coconame[d.class_], d.conf * 100);
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
            cv::Scalar text_background_color = color * 0.7 * 255;
            
            cv::rectangle(frame, cv::Rect(cv::Point(box.x, box.y), cv::Size(label_size.width, label_size.height + baseline)), text_background_color, -1);
            auto label = std::string(coconame[d.class_]) + ": " + std::to_string(d.conf * 100) + "%";
            cv::putText(frame, text, cv::Point(box.x, box.y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);

         }

         // publish 
         pub_obj_->publish(output_image);
      }

      rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_raw_;
      rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_obj_;
};

int main(int argc, char **argv) {
   rclcpp::Init(argc, argv);
   rclcpp::Spin(std::make_shared<ObjectDetection>());
   rclcpp::Shutdown();
   return 0;
}