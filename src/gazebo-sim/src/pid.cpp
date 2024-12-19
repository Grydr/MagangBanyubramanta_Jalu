#include "pid.hpp"

PID::PID(double kp, double ki, double kd) : kp_(kp), ki_(ki), kd_(kd), integral_(0.0), previous_error_(0.0) {}

double PID::compute(double setpoint, double actual, double dt) {
   double error = setpoint - actual;
   integral_ += error * dt;
   double derivative = (error - previous_error_) / dt;
   double out = kp_ * error + ki_ * integral_ + kd_ * derivative;
   previous_error_ = error;
   return out;
}