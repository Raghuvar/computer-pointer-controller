## Pre-requisites:

1. Download all required models
    * `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001 --output_dir models`
    * `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --output_dir models`
    * `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --output_dir models`
    * `python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --output_dir models`


## Run application on your local machine:
1. cd to the project directory
2. Download all models using above described commands
3. Checkout out the arguments using help argument
    `python3 main.py -h`
4. Run application using below command:
   1. Run demo using existing demo vidoe:
   `python3 src/main.py -fm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -pm  models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -lm models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -gm models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4  -o . -d "CPU" -c 0.5 -m 'async' -wi 'yes'`
   2. Run demo with web cam:
    `python3 src/main.py -fm models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -pm  models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -lm models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -gm models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i 'cam'  -o . -d "CPU" -c 0.5 -m 'async' -wi 'yes'`

### Main application arguments:
  ```
  -h, --help            show this help message and exit
  -fm FACEMODEL, --facemodel FACEMODEL
                        Path to an .xml file with a pre-trainedface detection
                        model
  -pm POSEMODEL, --posemodel POSEMODEL
                        Path to an .xml file with a pre-trained modelhead pose
                        model
  -lm LANDMARKSMODEL, --landmarksmodel LANDMARKSMODEL
                        Path to an .xml file with a pre-trained modellandmarks
                        model
  -gm GAZEMODEL, --gazemodel GAZEMODEL
                        Path to an .xml file with a pre-trained modelgaze
                        estimation model
  -i INPUT, --input INPUT
                        Path to video file or image.'cam' for capturing video
                        stream from camera
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA
                        or MYRIAD is acceptable. Looksfor a suitable plugin
                        for device specified(CPU by default)
  -c CONFIDENCE, --confidence CONFIDENCE
                        Probability threshold for detections filtering
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to output directory
  -m MODE, --mode MODE  async or sync mode
  -wi WRITE_INTERMEDIATE, --write_intermediate WRITE_INTERMEDIATE
                        Select between yes | no

```

## Run this application on intel dev cloud
- Log in to the Intel® DevCloud:
- Sign into the Intel DevCloud account with your credentials from [here](https://software.intel.com/en-us/devcloud/edge)
- If you are new user, Register into the Intel DevCloud account from [here](https://inteliotgnew.secure.force.com/devcloudsignup)
- In home page, Under "Advanced Tab", Click on "Connect and Create"
- Click on My Files, then you will be navigated to your Home Directory.
- Open a new linux terminal and clone this repo
- Navigate to the downloaded code and open computer-controller.ipynb from src directory


### NOTE: the whole experiment has been performed in MacBookPro with OS X. You might see some addition errors while installing the requirement and running the application. Below I have listed few error that occured. 

## Error and their resolution:
1. Error log: 
   ```
    Traceback (most recent call last):
    File "main.py", line 5, in <module>
    import cv2
    ImportError: dlopen(/opt/intel/openvino_2020.1.023/python/python2.7/cv2.so, 2): Symbol not found: _PyCObject_Type
    Referenced from: /opt/intel/openvino_2020.1.023/python/python2.7/cv2.so
    Expected in: flat namespace
    in /opt/intel/openvino_2020.1.023/python/python2.7/cv2.so
    ```

Under `/opt/intel/openvino_2020.1.023/python` -- it has cv2.so for all python version.  
**Resolution** .. just copy pasted python3/cv2.so to python2.7

2. Error log: ModuleNotFoundError: No module named 'openvino'  
    **Resolution**: Add this line to your .bashrc or zhrc file `source /opt/intel/openvino/bin/setupvars.sh`

3. Mouse Pointer Not Moving:
    If you are using Mac OSX then you might need to allow  your Terminal/iTerm to controll your computer. You can allow this behaviour by doing as :  
    `Open System Prefrerence --> Security & Privacy --> Select Terminal/iTerm to controll you computer and then save.`


