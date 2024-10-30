#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include <cv_bridge/cv_bridge.h> 
#include <image_transport/image_transport.hpp> 
#include <opencv2/opencv.hpp> 

using namespace std::chrono_literals;

class OpenCVNode : public rclcpp::Node {
   public:
      OpenCVNode() : Node("opencv_node") {
         // create publisher for both the raw & masked image
         pub_raw_ = this->create_publisher<sensor_msgs::msg::Image>("camera", 10);
         pub_mask_ = this->create_publisher<sensor_msgs::msg::Image>("mask", 10);

         // open video file
         cap_.open("src/opencv/include/fourth.mp4");
         if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open");
            rclcpp::shutdown();
         }

         // create timer callback
         timer_ = this->create_wall_timer(33ms, std::bind(&OpenCVNode::timer_callback, this));
      }

   private:
      void timer_callback() {
         cv::Mat frame;

         if (!cap_.read(frame)) {
            RCLCPP_INFO(this->get_logger(), "Video finished");
            rclcpp::shutdown();
            return;
         }

         int iLowH = 0;
         int iHighH = 35;

         int iLowS = 95; 
         int iHighS = 235;

         int iLowV = 0;
         int iHighV = 255;

         cv::Scalar minHSV = cv::Scalar(iLowH, iLowS, iLowV);
         cv::Scalar maxHSV = cv::Scalar(iHighH, iHighS, iHighV);

         cv::Mat imgHSV;
         cv::cvtColor(frame, imgHSV, cv::COLOR_BGR2HSV);

         cv::Mat imgThresh;
         cv::inRange(imgHSV, minHSV, maxHSV, imgThresh);
            
         //morphological opening (remove small objects from the foreground)
         cv::erode(imgThresh, imgThresh, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
         cv::dilate( imgThresh, imgThresh, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) ); 
         //morphological closing (fill small holes in the foreground)
         cv::dilate( imgThresh, imgThresh, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) ); 
         cv::erode(imgThresh, imgThresh, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

         // cv::imshow("Thresh Image", imgThresh); //show the thresholded image
         // cv::imshow("Original", frame); //show the original image

         publish_raw(frame, pub_raw_);
         publish_masked(imgThresh, pub_mask_);
      }

      void publish_raw(const cv::Mat &img, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &pub) {
         auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
         pub->publish(*msg);         
      }

      void publish_masked(const cv::Mat &img, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr &pub) {
         auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", img).toImageMsg();
         pub->publish(*msg);         
      }

      cv::VideoCapture cap_;
      rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_raw_;
      rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mask_;
      rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv) {
   rclcpp::init(argc, argv);
   rclcpp::spin(std::make_shared<OpenCVNode>());
   rclcpp::shutdown();
   return 0;
}