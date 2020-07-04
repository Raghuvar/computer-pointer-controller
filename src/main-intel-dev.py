import os
import sys
import json
import time
import cv2

from threading import Thread
from collections import namedtuple
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import logging as log
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmark_detection import FaceLandmarksDetection
from gaze_estimation import GazeEstimation

POSE_CHECKED = False

def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fm", "--facemodel", required=True,
                        help="Path to an .xml file with a pre-trained"
                        "face detection model")
    parser.add_argument("-pm", "--posemodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                        "head pose model")
    parser.add_argument("-lm", "--landmarksmodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                        "landmarks model")
    parser.add_argument("-gm", "--gazemodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                        "gaze estimation model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image."
                        "'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                        "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                        "for a suitable plugin for device specified"
                        "(CPU by default)")
    parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        help="Probability threshold for detections filtering")
    parser.add_argument("-o", "--output_dir", help = "Path to output directory", type = str, default = None)
    parser.add_argument("-m", "--mode", help = "async or sync mode", type = str, default = 'async')
    parser.add_argument("-wi", "--write_intermediate", default=None, type=str,
                        help="Select between yes | no ")

    return parser


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    global POSE_CHECKED
    
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = args_parser().parse_args()
    logger = log.getLogger()

    if args.input == 'cam':
       input_stream = 0
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)
    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(os.path.join(args.output_dir, "output.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
    
    if args.write_intermediate == 'yes':
        out_fm = cv2.VideoWriter(os.path.join(args.output_dir, "output_fm.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_lm = cv2.VideoWriter(os.path.join(args.output_dir, "output_lm.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_pm = cv2.VideoWriter(os.path.join(args.output_dir, "output_pm.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
        out_gm = cv2.VideoWriter(os.path.join(args.output_dir, "output_gm.mp4"), 
            cv2.VideoWriter_fourcc(*"MP4V"), fps, (initial_w, initial_h), True)
    
    frame_count = 0

    job_id = 1

    infer_time_start = time.time()

    if input_stream:
        cap.open(args.input)
        # Adjust DELAY to match the number of FPS of the video file

    if not cap.isOpened():
        logger.error("ERROR! Unable to open video source")
        return

    if args.mode == 'sync':
        async_mode = False
    else:
        async_mode = True

    # Initialise the class
    if args.cpu_extension:
        face_det = FaceDetection(args.facemodel, args.confidence,extensions=args.cpu_extension, async_mode = async_mode)
        pose_det = HeadPoseEstimation(args.posemodel, args.confidence,extensions=args.cpu_extension, async_mode = async_mode)
        land_det = FaceLandmarksDetection(args.landmarksmodel, args.confidence,extensions=args.cpu_extension, async_mode = async_mode)
        gaze_est = GazeEstimation(args.gazemodel, args.confidence,extensions=args.cpu_extension, async_mode = async_mode)
    else:
        face_det = FaceDetection(args.facemodel, args.confidence, async_mode = async_mode)
        pose_det = HeadPoseEstimation(args.posemodel, args.confidence, async_mode = async_mode)
        land_det = FaceLandmarksDetection(args.landmarksmodel, args.confidence, async_mode = async_mode)
        gaze_est = GazeEstimation(args.gazemodel, args.confidence, async_mode = async_mode)

    # infer_network_pose = Network()
    # Load the network to IE plugin to get shape of input layer
    face_det.load_model()
    pose_det.load_model()
    land_det.load_model()
    gaze_est.load_model()

    model_load_time = time.time() - infer_time_start

    print("All models are loaded successfully")

    try:
        pass
    except Exception as e:
        print("Could not run Inference: ", e)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print ("checkpoint *BREAKING")
            break

        frame_count += 1
        looking = 0
        POSE_CHECKED = False

        if frame is None:
            log.error("checkpoint ERROR! blank FRAME grabbed")
            break

        initial_w = int(cap.get(3))
        initial_h = int(cap.get(4))

        # Start asynchronous inference for specified request
        inf_start_fd = time.time()
        
        # Results of the output layer of the network
        coords, frame = face_det.predict(frame)
        
        if args.write_intermediate == 'yes':
            out_fm.write(frame)

        det_time_fd = time.time() - inf_start_fd
        
        if len(coords) > 0:
            [xmin,ymin,xmax,ymax] = coords[0] # use only the first detected face
            head_pose = frame[ymin:ymax, xmin:xmax]
            inf_start_hp = time.time()
            is_looking, pose_angles = pose_det.predict(head_pose)
            if args.write_intermediate == 'yes':
                p = "Pose Angles {}, is Looking? {}".format(pose_angles,is_looking)
                cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255,0, 0), 1)
                out_pm.write(frame)

            if is_looking:
                det_time_hp = time.time() - inf_start_hp
                POSE_CHECKED = True
                inf_start_lm = time.time()
                coords,f = land_det.predict(head_pose)
                
                frame[ymin:ymax, xmin:xmax] = f
                
                if args.write_intermediate == "yes":
                    out_lm.write(frame)

                det_time_lm = time.time() - inf_start_lm
                [[xlmin,ylmin,xlmax,ylmax],[xrmin,yrmin,xrmax,yrmax]] = coords
                left_eye_image = f[ylmin:ylmax, xlmin:xlmax]
                right_eye_image = f[yrmin:yrmax, xrmin:xrmax]

                output,gaze_vector = gaze_est.predict(left_eye_image,right_eye_image,pose_angles)
                
                if args.write_intermediate == 'yes':
                    p = "Gaze Vector {}".format(gaze_vector)
                    cv2.putText(frame, p, (50, 15), cv2.FONT_HERSHEY_COMPLEX,
                            0.5, (255, 0, 0), 1)
                    fl = draw_gaze(left_eye_image, gaze_vector)
                    fr = draw_gaze(right_eye_image, gaze_vector)
                    f[ylmin:ylmax, xlmin:xlmax] = fl
                    f[yrmin:yrmax, xrmin:xrmax] = fr
                    # cv2.arrowedLine(f, (xlmin, ylmin), (xrmin, yrmin), (0,0,255), 5)
                    out_gm.write(frame)

        # Draw performance stats
        inf_time_message = "Face Inference time: {:.3f} ms.".format(det_time_fd * 1000)
        #
        if POSE_CHECKED:
            cv2.putText(frame, "Head pose Inference time: {:.3f} ms.".format(det_time_hp * 1000), (0, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, inf_time_message, (0, 15), cv2.FONT_HERSHEY_COMPLEX,
                        0.5, (255, 0, 0), 1)
        out.write(frame)
        if frame_count%10 == 0:
            print("Inference time = ", int(time.time()-infer_time_start))
            print('Frame count {} and vidoe len {}'.format( frame_count, video_len))
        if args.output_dir:
            total_time = time.time() - infer_time_start
            with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
                f.write(str(round(total_time, 1))+'\n')
                f.write(str(frame_count)+'\n')

    if args.output_dir:
        with open(os.path.join(args.output_dir, 'stats.txt'), 'a') as f:
            f.write(str(round(model_load_time))+'\n')
            
    # Clean all models
    face_det.clean()
    pose_det.clean()
    land_det.clean()
    gaze_est.clean()
    # release cv2 cap
    cap.release()
    cv2.destroyAllWindows()
    # release all out writer
    out.release()
    if args.write_intermediate == 'yes':
        out_fm.release()
        out_pm.release()
        out_lm.release()
        out_gm.release()


def draw_gaze(screen_img, gaze_pts, gaze_colors=None, scale=4, return_img=False, cross_size=16, thickness=10):

    """ Draws an "x"-shaped cross on a screen for given gaze points, ignoring missing ones
    """
    width = int(cross_size * scale)
   
    draw_cross(screen_img, gaze_pts[0] * scale, gaze_pts[1] * scale, 
        (0, 0, 255), width, thickness)
    return  screen_img

def draw_cross(bgr_img,x, y,color=(255, 255, 255), width=2, thickness=0.5):

    """ Draws an "x"-shaped cross at (x,y)
    """
    x, y, w = int(x), int(y), int(width / 2)  # ensure points are ints for cv2 methods

    cv2.line(bgr_img, (x - w , y - w), (x + w , y + w), color, thickness)
    cv2.line(bgr_img, (x - w , y + w), (x + w, y - w), color, thickness)

if __name__ == '__main__':
    main()
    sys.exit()