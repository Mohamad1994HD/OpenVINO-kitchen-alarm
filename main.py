from argparse import ArgumentParser, SUPPRESS
import logging as log
import sys
import cv2

import gi
gi.require_version('Notify', '0.7')
from gi.repository import Notify

from detectors import SSD_Detector, MovementDetector

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input",
                      help="Required. Path to video file or camera", 
                      required=True, 
                      type=str)
    args.add_argument("-o", "--output",
                      help="Optional. For output video file to save",
                      type=str)

    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("--debug", help="Debugging mode", action='store_true')

    return parser

if __name__ == '__main__':
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    # Initialize the detector
    log.info("Creating Inference Engine...")
    personDetector = SSD_Detector(args.model, prob_thresh=0.8)

    # Load video input
    log.info("Opening stream source")
    input_stream = args.input if args.input != "0" else 0
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + input_stream
    
    _, init_frame = cap.read()

    # initialize motion detector
    motionDetector =  MovementDetector(init_frame)  
   
    # state variable
    is_handled = False
    is_notified = False

    def onNotificationHandled(*args):
        global is_handled
        is_handled = True

     # initialize notifier 
    Notify.init("Kitchen watcher")
    notification = Notify.Notification.new("ALARM", "KITCHEN UNDER ATTACK", "dialog-error")
    notification.set_urgency(2)
    notification.add_action("action_click", "Show me", onNotificationHandled)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        motionDetector.infer(frame)
        if motionDetector.isMovementDetected():
            #log.info("Motion Detected !!!")
            # infer the motion reason
            personDetector.infer(frame)

            if personDetector.isPerson() and not is_notified:
                is_notified = True
                # pop up imshow
                notification.show()

        # Check if notification is handled to show alarm
        if is_handled:
            # Show alarm
            cv2.putText(frame, "Kitchen Under ATTACK!!", (15, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", frame)    
            


        if args.debug:
            cv2.imshow("debug", frame)

      
      
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    cap.release()
    exit(0)