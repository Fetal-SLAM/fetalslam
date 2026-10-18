import numpy as np
import yaml
import logging
import csv
import cv2
import os
from .runtime_config import runtime_settings, configure_logging

import logging as _logging

_logger = _logging.getLogger(__name__)


class Parameter:
    
    class CameraIntrinsic:
        def __init__(self):
            self.fx = 0.0
            self.fy = 0.0
            self.cx = 0.0
            self.cy = 0.0
            self.k1 = 0.0
            self.k2 = 0.0
            self.p1 = 0.0
            self.p2 = 0.0
            self.k3 = 0.0
            self.width = 0
            self.height = 0
            self.cam_port = 0

    class Extrinsic:
        def __init__(self):
            self.W_R_C = np.eye(3)
            self.W_p_C = np.zeros((3,1))


    class SystemConfig:
        def __init__(self):
            self.__dict__.update(vars(runtime_settings("system_config")))
    
    class TrackerConfig:
        def __init__(self):
            self.max_kpts = runtime_settings("tracker_config").max_kpts


    def __init__(self):
        self.save_result = True
        self.cam_intrinsic = self.CameraIntrinsic()
        self.system_config = self.SystemConfig()
        self.extrinsic = self.Extrinsic()
        self.tracker = self.TrackerConfig()

    def load_sys_param(self, sys_param, cam_param):
        with open(sys_param, 'r') as file:
            yaml_data = yaml.safe_load(file)
            if not isinstance(yaml_data, dict):
                raise ValueError(
                    f"System parameter YAML '{sys_param}' is empty or does not contain a mapping."
                )

            runtime = yaml_data.get('runtime_config', {})
            if not isinstance(runtime, dict):
                raise ValueError('runtime_config must be a mapping')
            sections = ('logging', 'tracking', 'features', 'local_mapping', 'optimizer', 'viewer', 'loop_closing', 'reconstruction')
            if set(runtime) - set(sections):
                raise ValueError('Unknown runtime_config section')
            self.runtime_config = {name: runtime_settings(name, runtime.get(name)) for name in sections}
            configure_logging(self.runtime_config['logging'])
            sys_data = yaml_data['system_config']
            from .dense_config import dense_settings
            self.dense_config = dense_settings(yaml_data.get('dense_map_config'))
            sys_data = dict(sys_data)
            if 'debug_flag' in sys_data:
                sys_data['debug'] = sys_data.pop('debug_flag')
            self.system_config.__dict__.update(vars(runtime_settings('system_config', sys_data)))
            self.tracker.__dict__.update(vars(runtime_settings('tracker_config', yaml_data.get('tracker_config'))))


        with open(cam_param, 'r') as file:
            cam_intr = yaml.safe_load(file)
            if not isinstance(cam_intr, dict):
                raise ValueError(
                    f"Camera parameter YAML '{cam_param}' is empty or does not contain a mapping."
                )
            self.cam_intrinsic.fx = cam_intr['cam_fx']
            self.cam_intrinsic.fy = cam_intr['cam_fy']
            self.cam_intrinsic.cx = cam_intr['cam_cx']
            self.cam_intrinsic.cy = cam_intr['cam_cy']
            self.cam_intrinsic.k1 = cam_intr['cam_k1']
            self.cam_intrinsic.k2 = cam_intr['cam_k2']
            self.cam_intrinsic.p1 = cam_intr['cam_p1']
            self.cam_intrinsic.p2 = cam_intr['cam_p2']
            self.cam_intrinsic.k3 = cam_intr['cam_k3']
            self.cam_intrinsic.width = cam_intr['cam_width']
            self.cam_intrinsic.height = cam_intr['cam_height']
            self.cam_intrinsic.cam_port = cam_intr['cam_port']


    def load_sensor_data(self, encoder_file_path, time_data_map):
        try:
            with open(encoder_file_path, 'r') as encoder_file:
                for line_str in encoder_file:
                    parts = line_str.strip().split(',', 1)
                    if len(parts) < 2:
                        logging.error("[LoadSensorData]: Find a bad line in the encoder file.: %s", line_str)
                        return False
                    time_str, data_str = parts
                    time_data_map[time_str] = data_str
        except IOError:
            logging.error("[LoadSensorData]: Failed to open encoder file.")
            return False
    
        return True
    
    def get_cam_K_distco(self):
        fx = self.cam_intrinsic.fx
        fy = self.cam_intrinsic.fy
        cx = self.cam_intrinsic.cx
        cy = self.cam_intrinsic.cy
        k1 = self.cam_intrinsic.k1
        k2 = self.cam_intrinsic.k2
        k3 = self.cam_intrinsic.k3
        p1 = self.cam_intrinsic.p1
        p2 = self.cam_intrinsic.p2

        K_matrix = np.array([[fx,  0.0, cx],
                                [0.0, fy,  cy],
                                [0.0, 0.0, 1.0]])
        distco = np.array([k1,k2,p1,p2,k3])

        return K_matrix, distco

    def load_multi_sensor_data(self, data_folder):
        '''
        load multi sensor data from the data folder
        input: data folder path
        output: 1. the flag if data loading is successful 2. images_list 3. timestamps_list
        
        '''


        # Load all encoder data to buffer.
        time_encoder_map = {}
        data_list = []

        if not self.load_sensor_data(f"{data_folder}/sensor_data/ee_pos.csv", time_encoder_map):
            logging.error("[main]: Failed to load encoder data.")
            return False, None, None
        
        with open(f"{data_folder}/sensor_data/data_stamp.csv", 'r') as file_data_stamp:
            reader = csv.reader(file_data_stamp)
            kToSecond = 1e-9

            for row in reader:
                data_list.append(row)


        _logger.debug('%s', "finish data loading!")
        return True, data_list, time_encoder_map
    

    def load_multi_sensor_data_from_npy(self, data_folder):
        image = data_folder + "/xfeat_outputsave_ref_frames.npy"

        
        image_list = np.load(image)
        timestamp_list = list(range(0, image_list.shape[0] + 1))
        return True, image_list, timestamp_list

    def load_image_sequence_data(self, data_folder, image_subdir="undist_image"):
        image_dir = os.path.join(data_folder, image_subdir)
        image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        if not os.path.isdir(image_dir):
            logging.error("[LoadImageSequence]: Failed to find image folder: %s", image_dir)
            return False, None, None

        image_paths = sorted(
            os.path.join(image_dir, filename)
            for filename in os.listdir(image_dir)
            if filename.lower().endswith(image_extensions)
        )
        if not image_paths:
            logging.error("[LoadImageSequence]: No images found in: %s", image_dir)
            return False, None, None

        image_list = []
        timestamp_list = []
        for image_idx, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            if image is None:
                logging.error("[LoadImageSequence]: Failed to read image: %s", image_path)
                return False, None, None

            image_list.append(image)
            timestamp_list.append(float(image_idx))

        _logger.debug('%s', f"finish image sequence loading with {len(image_list)} images")
        return True, image_list, timestamp_list
