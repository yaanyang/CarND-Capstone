import sys
import rospy

from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
    wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        # Yaw controller
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        # PID controller
        # mn: minimum throttle
        # mx: maximun throttle
        self.pid = PID(kp=0.3, ki=0.1, kd=0.0, mn=0.0, mx=0.2)

        # Low pass filter
        # tau: cut-off frequency
        # ts: sample time 
        self.lowpassfilter = LowPassFilter(tau=0.5, ts=0.02)

        # Some vehicle properties
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_vel = None
        self.last_time = rospy.get_time()

    def control(self, linear_velocity, angular_velocity, curr_linear_velocity, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        # Reset if dbw not enabled
        if not dbw_enabled:
            self.pid.reset()
            return 0.0, 0.0, 0.0

        # Throttle control
        vel_err = linear_velocity - curr_linear_velocity
        self.last_vel = curr_linear_velocity

        curr_time = rospy.get_time()
        sample_time = curr_time - self.last_time
        self.last_time = curr_time

        throttle = self.pid.step(vel_err, sample_time)

        # Brake control
        brake = 0.0

        # To hold the car from moving
        if abs(linear_velocity) < sys.float_info.epsilon and curr_linear_velocity < 0.1:
            throttle = 0.0
            brake = 700.0 # Nm of torque to hold the car
        
        # Deceleration
        elif throttle < 0.1 and vel_err < 0.0:
            throttle = 0.0
            decel = max(vel_err, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque (Nm)

        # Steering control
        steering = self.yaw_controller.get_steering(linear_velocity, angular_velocity, curr_linear_velocity)
        
        return throttle, brake, steering
