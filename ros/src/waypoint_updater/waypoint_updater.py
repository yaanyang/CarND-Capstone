#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 20 # Number of waypoints we will publish. You can change this number
DECEL_LIMIT = 0.5
STOP_PRIOR_WPS = 5 # Number of waypoints to stop prior to stop line.


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stop_line_wp_idx = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints: 
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_cood = self.waypoints_2d[closest_idx]
        prev_cood = self.waypoints_2d[closest_idx-1]

        # Hyperplane vectors
        cl_vect = np.array(closest_cood)
        prev_vect = np.array(prev_cood)
        pos_vect = np.array([x, y])

        # Check dot products to determine if ahead of behind
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
    
    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        lane_waypoints = self.base_waypoints.waypoints[closest_idx: farthest_idx]            

        if self.stop_line_wp_idx == -1 or (self.stop_line_wp_idx >= farthest_idx):
            lane.waypoints = lane_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(lane_waypoints, closest_idx)            
        
        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):        
        temp_wps = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose            
            # Stop fwe waypoints earlier from the stop line
            # or stop right away if no such waypoint
            stop_wp_idx = max(self.stop_line_wp_idx - closest_idx - STOP_PRIOR_WPS, 0)
            d = self.distance(waypoints, i, stop_wp_idx)
            vel = math.sqrt(2 * DECEL_LIMIT * d)
            if vel < 1.0 : 
                vel = 0.0
            
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp_wps.append(p)

        return temp_wps
    
    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints =  waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stop_line_wp_idx = msg.data        

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
