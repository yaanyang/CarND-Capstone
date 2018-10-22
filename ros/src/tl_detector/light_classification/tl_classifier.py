import rospy
import tensorflow as tf
import numpy as np

from styx_msgs.msg import TrafficLight

MIN_SCORE_THRESHOLD = 0.5
CLASS_DICT = {1: 'Green', 2: 'Red', 3: 'Yellow'}

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        if is_site:
            PATH_TO_MODEL = r'light_classification/models/site/frozen_inference_graph.pb'
        else:
            PATH_TO_MODEL = r'light_classification/models/sim/frozen_inference_graph.pb'
        self.state = TrafficLight.UNKNOWN
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
                        
            if scores[0] > MIN_SCORE_THRESHOLD:
                rospy.loginfo('Detecting %s!! Score: %4f', CLASS_DICT[classes[0]], scores[0])
                if classes[0] == 1:                    
                    self.state = TrafficLight.GREEN
                elif classes[0] == 2:                    
                    self.state = TrafficLight.RED
                elif classes[0] == 3:
                    self.state = TrafficLight.YELLOW
            else:
                rospy.loginfo('No Traffic Light Detected!!')
                    
        return self.state
