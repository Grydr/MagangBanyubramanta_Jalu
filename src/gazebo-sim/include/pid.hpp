#ifndef PID_HPP
#define PID_HPP

class PID {
 public:
   PID(double kp, double ki, double kd);
   double compute(double setpoint, double actual, double dt);

 private:
   double kp_;
   double ki_;
   double kd_;
   double integral_;
   double previous_error_;
};

#endif // PID_HPP