import time
import cv2
import os
from openvino.inference_engine import IENetwork, IECore

UPDATE_BACKGROUND_AFTER = 1000 # frame
MOVEMENT_THRESHOLD = 20 # a scale from 10 to 90 to set the ratio of (moving pixels / frame size)

# Abstract class
class Detector(object):
    def infer(self, frame):
        raise NotImplementedError

class MovementDetector:
    """
    init_frame : initial background frame
    reset_counter: to update background after x frame
    """
    def __init__(self, init_frame):
        self.is_movement = False
        self.init_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        self.reset_cnt = UPDATE_BACKGROUND_AFTER

    def infer(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        frame_diff = cv2.absdiff(gray, self.init_gray)
        res_frame = cv2.threshold(frame_diff, 50, 255, cv2.THRESH_BINARY)[1]
        # calculate the ratio of white pixels to the whole image
        white_px = cv2.countNonZero(res_frame)
        white_ratio = white_px / (gray.shape[0] * gray.shape[1])
        # print(white_ratio)
        self.is_movement = white_ratio > ((1.*MOVEMENT_THRESHOLD)/100)
        #
        self.reset_cnt -= 1

        if self.reset_cnt <= 0:
            self.init_gray = gray
            self.reset_cnt = UPDATE_BACKGROUND_AFTER
            self.is_movement = False


    def isMovementDetected(self):   
        return self.is_movement 



class SSD_Detector(object):
    def __init__(self, model_xml, accelerator='CPU', cpu_extension=None, prob_thresh=0.5):
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        ie = IECore()
        if (accelerator == 'CPU') and cpu_extension:
            ie.add_extension(cpu_extension, accelerator)
        # loading network
        net = IENetwork(model=model_xml, weights=model_bin)
        assert len(net.outputs) == 1, "This detector supports only single output topologies"

        self.feed_dict = {}
        self.input_blob = None
        self.network_inshape = []

        for blob_name in net.inputs:
            if len(net.inputs[blob_name].shape) == 4:
                self.input_blob = blob_name
                self.network_inshape = net.inputs[self.input_blob].shape # n,c,h,w

            elif len(net.inputs[blob_name].shape) == 2:
                self.feed_dict[blob_name] = [
                    self.network_inshape[2],
                    self.network_inshape[3],
                    1
                ]
            else:
                raise RuntimeError("Unsupported input layer")


        self.exec_net = ie.load_network(network=net, num_requests=2, device_name=accelerator)

        self.out_blob = next(iter(net.outputs))

        self.prob_threshold = prob_thresh
        self.person_detected = False
        
    def infer(self, frame):
        inf_start = time.time()
        in_frame = cv2.resize(frame, (self.network_inshape[3], self.network_inshape[2]))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        in_frame = in_frame.reshape(self.network_inshape)
        self.feed_dict[self.input_blob] = in_frame
        self.exec_net.start_async(request_id=0, inputs=self.feed_dict) # request_id=0 for sync request/ 1 for async request
        if self.exec_net.requests[0].wait(-1) == 0: # request_id = 0
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = self.exec_net.requests[0].outputs[self.out_blob]
            # Filter detections
            detections = [obj for obj in res[0][0] if obj[2] > self.prob_threshold]
            self.person_detected = len(detections) > 0

    def isPerson(self):
        return self.person_detected