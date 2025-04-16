import os
import cv2
import time
import logging
import threading
from ultralytics import YOLO
# import io
# import sys
# import json
# import time
# import torch
# import pickle
# import socket
# import struct
# import requests
# import datetime
# import threading
# import lz4.frame
# import numpy as np
# from torchvision import transforms
# from collections import defaultdict
# from ultralytics.engine.results import Results

from flask import Flask, request, render_template, Response, jsonify, redirect, url_for

app = Flask(__name__)

VALID_STATUSES = {
    'idle' : 0,

    'offload_raw' : 1,
    'load_raw' : 2,

    'offload_jpg' : 3,
    'load_jpg' : 4,

    'offload_backbone' : 5,
    'load_backbone' : 6,

    'inference' : 7,
}

# Status variables with the defaults or the values from the docker-compose environment
STATS_PRINT = os.getenv('STATS_PRINT', 'true').lower() in ('true', '1', 't', 'yes', 'y')
RUNNING_PORT = int(os.getenv('RUNNING_PORT', 5000))     # Set RUNNING_PORT to True if the environment variable 'STATS_PRINT' is {true, 1, t, yes, y}, else False
STATUS = VALID_STATUSES.get(os.getenv('STARTUP_STATUS', 'idle').lower(), VALID_STATUSES['idle'])        # Get status code from environment, defaulting to 'idle' if unset or invalid
MODEL_DIR = '/models'
MODEL_LIST = {  os.path.relpath(filepath, MODEL_DIR): os.path.abspath(filepath)
                for root, _, files in os.walk(MODEL_DIR)
                for f in files
                if not f.startswith('.')  # and f.endswith('.pt')                      # Optional extension filter check
                for filepath in [os.path.join(root, f)]
            }
MODEL = os.getenv('STARTUP_MODEL', '') if os.getenv('STARTUP_MODEL', '') in MODEL_LIST else next(iter(MODEL_LIST), None)
INPUT_SRC = os.getenv('INPUT_SRC', '/dev/video0')   # Default is the camera, but can also be an online stream URL

# For adhering to Flask's best practices, then a global variable is later on modified
app.config['STATUS'] = STATUS
app.config['MODEL'] = MODEL
app.config['INPUT_SRC'] = INPUT_SRC

# Permit to have the printing of all the app messages from INFO to ERROR
if STATS_PRINT:
    app.logger.setLevel(logging.INFO)

app.logger.info("###### Configuration Recap:")
app.logger.info(f"Valid Statuses: {list(VALID_STATUSES.keys())}")
app.logger.info(f"Detected Models: {list(MODEL_LIST.values())}")
app.logger.info(f"Running Port: {RUNNING_PORT}")
app.logger.info(f"Current Status: {app.config['STATUS']}")
app.logger.info(f"Current Model: {app.config['MODEL']}")
app.logger.info(f"Current Input Video: {app.config['INPUT_SRC']}")
app.logger.info("######")

# Reference for the control loop thread, for tracking during execution
control_background_thread = None

# MODEL = YOLO(f'{MODEL_FOLDER}Yolo/bestSanRossore.pt')          # [print(f"Layer {i}: {layer}") for i, layer in enumerate(MODEL.model.model)]
# MODEL = YOLO(f'{MODEL_FOLDER}Yolo/yolov8m.pt')          # [print(f"Layer {i}: {layer}") for i, layer in enumerate(MODEL.model.model)]
# MODEL = MODEL.to('cuda')
# MODEL_E = False
# BACKBONE = MODEL.model.model[0]                         # Backbone part of the model
# NECK = MODEL.model.model[1]                             # Neck part of the model
# HEAD = MODEL.model.model[2]                             # Head part of the model


# tensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# def get_the_size(obj, seen=None):
#     if seen is None:
#         seen = set()

#     # Handle circular references
#     obj_id = id(obj)
#     if obj_id in seen:
#         return 0
#     seen.add(obj_id)

#     size = sys.getsizeof(obj)  # Base size of the object

#     # Handle specific types
#     if isinstance(obj, torch.Tensor):
#         size += obj.element_size() * obj.numel()  # Tensor's data size
#     elif isinstance(obj, (list, tuple, set, frozenset)):
#         size += sum(get_the_size(item, seen) for item in obj)  # Sum sizes of elements
#     elif isinstance(obj, dict):
#         size += sum(get_the_size(k, seen) + get_the_size(v, seen) for k, v in obj.items())  # Keys and values
#     elif hasattr(obj, '__dict__'):
#         size += get_the_size(vars(obj), seen)  # Object's attributes
#     elif hasattr(obj, '__slots__'):
#         size += sum(get_the_size(getattr(obj, slot), seen) for slot in obj.__slots__ if hasattr(obj, slot))

#     return (size * 8)

# def compress_tensor(tensor):
#     # Serialize the tensor to bytes using BytesIO
#     buffer = io.BytesIO()
#     torch.save(tensor, buffer)
#     buffer.seek(0)
#     tensor_bytes = buffer.read()

#     # Compress the serialized tensor data using lz4
#     compressed_tensor = lz4.frame.compress(tensor_bytes)

#     return compressed_tensor

# def decompress_tensor(compressed_tensor):
#     # Decompress the tensor data using lz4
#     decompressed_tensor_bytes = lz4.frame.decompress(compressed_tensor)

#     # Convert back to tensor
#     buffer = io.BytesIO(decompressed_tensor_bytes)
#     buffer.seek(0)
#     decompressed_tensor = torch.load(buffer)

#     return decompressed_tensor

# ###
# #   OPERATIVE PARTS (as capable device)
# ###

# ### yield is a keyword in Python that allows a function to return a value and pause its execution, so that it can later resume where it left off

# def generate_raw():
#     print(f"Raw stream prepering on {DEVICE_IP}:5111")
#     try:
#         tensor_socket.bind((DEVICE_IP, 5111))
#         tensor_socket.listen(1)
#     except Exception as e:
#         print(f"Error in socket: {e}. Check with 'netstat -tuln| grep 5111'")
#         tensor_socket.close()
#         return
#     print("Raw stream is ready")
#     conn, addr = tensor_socket.accept()
#     print("Connection...")
#     cap = cv2.VideoCapture(INPUT_SRC)

#     if not cap.isOpened():
#         print(f"Error: Unable to access the video feed from '{SOURCE_DATA}'. Is the stream active elsewhere?")
#         return

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_fps_time = time.perf_counter()
#     start_bps_time = time.perf_counter()

#     while STATUS in ["send_raw"]:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from the video feed.")
#             break   # Using 'continue' allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

#         if(STATS_PRINT):
#             frame_count += 1                                        # Update the frame count
#             if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                 elapsed_time1 = time.perf_counter() - start_fps_time
#                 if elapsed_time1 > 0:
#                     fps = frame_count / elapsed_time1
#                     frame_count = 0                                     # Restart the counter
#                     start_fps_time = time.perf_counter()                    # Restart the times

#         # Overlay FPS, image size, and camera FPS on the frame
#         if(RUNNING_VIDEO):
#             # cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)    # The triplete (x, x, x) is the color of the text
#             # cv2.putText(frame, f"Frame Rate: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             if(STATS_PRINT):
#                 cv2.putText(frame, f"Estimated FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
#                 (frame.shape[1] - cv2.getTextSize(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0][0] - 10, 
#                 frame.shape[0] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

#         slice = frame

#         serialized_frame = pickle.dumps(slice)
#         # Split the data into chunks for big datas
#         chunk_size = 1024  # Size of each chunk (in bytes)
#         data_length = len(serialized_frame)
        
        
#         print(f"-------------------> Size of FRAME sent in bits: {get_the_size(serialized_frame)} Bits")
#         print(f"-------------------> Streaming at {fps} FRAME PER SECOND")
        
        
#         if(STATS_PRINT):
#             bytes_sent = len(serialized_frame)
#             bytes_count += bytes_sent                             # Update the bytes count
#             elapsed_time2 = time.perf_counter() - start_bps_time
#             if elapsed_time2 > 2:
#                 bps = bytes_count / elapsed_time2
#                 bytes_count = 0                                             # Restart the counter
#                 start_bps_time = time.perf_counter()                        # Restart the times

#             print(f"Bytes of the frame: {bytes_sent}")
#             print(f"Frame per Seconds:  {fps}")
#             print(f"Bytes per Seconds:  {bps}")

#         # Send the tensor
#         conn.sendall(struct.pack('!I', data_length))
#         for i in range(0, data_length, chunk_size):
#             chunk = serialized_frame[i:i+chunk_size]
#             conn.sendall(chunk)

#     cap.release()
#     conn.close()
#     tensor_socket.close()
#     print("Raw stream is terminated")

# def generate_jpg():
#     cap = cv2.VideoCapture(INPUT_SRC)
#     if not cap.isOpened():
#         print(f"Error: Unable to access the video feed from '{INPUT_SRC}'. Is the stream active elsewhere?")
#         return
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     # Verify the settings
#     current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     current_fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Camera Resolution: {current_width}x{current_height}")
#     print(f"Camera Frame Rate: {current_fps}")

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_fps_time = time.perf_counter()
#     start_bps_time = time.perf_counter()
#     while STATUS == "send_jpg":
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from the video feed.")
#             break   # Using continue allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

#         if(STATS_PRINT):
#             frame_count += 1                                        # Update the frame count
#             if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                 elapsed_time1 = time.perf_counter() - start_fps_time
#                 if elapsed_time1 > 0:
#                     fps = frame_count / elapsed_time1
#                     frame_count = 0                                     # Restart the counter
#                     start_fps_time = time.perf_counter()                    # Restart the times

#         # Overlay FPS, image size, and camera FPS on the frame
#         if(RUNNING_VIDEO):
#             cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)    # The triplete (x, x, x) is the color of the text
#             cv2.putText(frame, f"Frame Rate: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             if(STATS_PRINT):
#                 cv2.putText(frame, f"Estimated FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
#                 (frame.shape[1] - cv2.getTextSize(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0][0] - 10, 
#                 frame.shape[0] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

#         # Encode the frame as JPEG
#         print(len(frame.tobytes()))
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         print(f"-------------------> Size of FRAME sent in bits: {get_the_size(frame_bytes)} Bits")
#         print(f"-------------------> Streaming at {fps} FRAME PER SECOND")

#         print(len(frame_bytes))
#         if(STATS_PRINT):
#             bytes_sent = len(frame_bytes)
#             bytes_count += bytes_sent                             # Update the bytes count
#             elapsed_time2 = time.perf_counter() - start_bps_time
#             if elapsed_time2 > 2:
#                 bps = bytes_count / elapsed_time2
#                 bytes_count = 0                                             # Restart the counter
#                 start_bps_time = time.perf_counter()                        # Restart the times

#             print(f"Bytes of the frame: {bytes_sent}")
#             print(f"Frame per Seconds:  {fps}")
#             print(f"Bytes per Seconds:  {bps}")

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()
#     print("Jpg stream is terminated")

# def generate_back():
#     print(f"Back stream prepering on {DEVICE_IP}:5222")
#     try:
#         tensor_socket.bind((DEVICE_IP, 5222))
#         tensor_socket.listen(1)
#     except Exception as e:
#         print(f"Error in socket: {e}. Check with 'netstat -tuln| grep 5222'")
#         tensor_socket.close()
#         return
#     print("Back stream is ready")
#     conn, addr = tensor_socket.accept()
#     print("Connection...")
#     cap = cv2.VideoCapture(INPUT_SRC)

#     if not cap.isOpened():
#         print(f"Error: Unable to access the video feed from '{SOURCE_DATA}'. Is the stream active elsewhere?")
#         return

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_fps_time = time.perf_counter()
#     start_bps_time = time.perf_counter()

#     while STATUS in ["send_back"]:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from the video feed.")
#             break   # Using 'continue' allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

#         if(STATS_PRINT):
#             frame_count += 1                                        # Update the frame count
#             if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                 elapsed_time1 = time.perf_counter() - start_fps_time
#                 if elapsed_time1 > 0:
#                     fps = frame_count / elapsed_time1
#                     frame_count = 0                                     # Restart the counter
#                     start_fps_time = time.perf_counter()                    # Restart the times

#         # Pre-process the frame to match YOLO input size
#         frame_resized = cv2.resize(frame, (640, 640))               # Resize frame to the desired input size
#         frame_rgb = frame_resized[..., ::-1]                        # Convert BGR to RGB (YOLO typically expects RGB)
#         frame_normalized = frame_rgb / 255.0                        # Normalize the image (YOLO uses values in range [0, 1])
#         transform = transforms.ToTensor()                           # Convert to Tensor and add batch dimension
#         frame_tensor = transform(frame_normalized).unsqueeze(0)     # Add batch dimension
#         frame_tensor = frame_tensor.to(torch.float32)               # Convert to float32 as the first layer require

#         # Can also use this to pre-process
#         # input_tensor = cv2.resize(frame, (640, 640))
#         # input_tensor = input_tensor[..., ::-1]                            # Convert BGR to RGB
#         # input_tensor = np.copy(input_tensor)                              # Create a copy of the array to avoid negative strides
#         # input_tensor = np.transpose(input_tensor, (2, 0, 1))              # Change to (C, H, W)
#         # input_tensor = np.expand_dims(input_tensor, axis=0)               # Add batch dimension
#         # input_tensor = torch.from_numpy(input_tensor).float() / 255.0     # Normalize to [0, 1]

#         # MODEL = MODEL.to('cuda')                    # Move the model to GPU
#         frame_tensor = frame_tensor.to('cuda')      # Ensure the input tensor is on the same device as the model

#         # Stage 1: Backbone (feature extraction)
#         b0 = MODEL.model.model[0](frame_tensor)
#         b1 = MODEL.model.model[1](b0)
#         b2 = MODEL.model.model[2](b1)
#         b3 = MODEL.model.model[3](b2)
#         b4 = MODEL.model.model[4](b3)
#         b5 = MODEL.model.model[5](b4)
#         b6 = MODEL.model.model[6](b5)
#         b7 = MODEL.model.model[7](b6)
#         b8 = MODEL.model.model[8](b7)

#         slice = [b6,b4,b8]

#         serialized_tensor = pickle.dumps(slice)
#         # Split the data into chunks for big datas
#         chunk_size = 1024  # Size of each chunk (in bytes)
#         data_length = len(serialized_tensor)
#         print(f"-------------------> Size of FRAME sent in bits: {get_the_size(serialized_tensor)} Bits")
#         print(f"-------------------> Streaming at {fps} FRAME PER SECOND")

#         if(STATS_PRINT):
#             bytes_sent = data_length
#             bytes_count += bytes_sent                             # Update the bytes count
#             elapsed_time2 = time.perf_counter() - start_bps_time
#             if elapsed_time2 > 2:
#                 bps = bytes_count / elapsed_time2
#                 bytes_count = 0                                             # Restart the counter
#                 start_bps_time = time.perf_counter()                        # Restart the times

#             print(f"Bytes of the frame: {bytes_sent}")
#             print(f"Frame per Seconds:  {fps}")
#             print(f"Bytes per Seconds:  {bps}")

#         # Send the tensor
#         conn.sendall(struct.pack('!I', data_length))
#         for i in range(0, data_length, chunk_size):
#             chunk = serialized_tensor[i:i+chunk_size]
#             conn.sendall(chunk)

#     cap.release()
#     conn.close()
#     tensor_socket.close()
#     print("Back stream is terminated")
    
# def generate_neck():
#     print(f"Neck stream prepering on {DEVICE_IP}:5333")
#     try:
#         tensor_socket.bind((DEVICE_IP, 5333))
#         tensor_socket.listen(1)
#     except Exception as e:
#         print(f"Error in socket: {e}. Check with 'netstat -tuln| grep 5333'")
#         tensor_socket.close()
#         return
#     print("Neck stream is ready")
#     conn, addr = tensor_socket.accept()
#     print("Connection...")
#     cap = cv2.VideoCapture(INPUT_SRC)

#     if not cap.isOpened():
#         print(f"Error: Unable to access the video feed from '{SOURCE_DATA}'. Is the stream active elsewhere?")
#         return

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_fps_time = time.perf_counter()
#     start_bps_time = time.perf_counter()

#     while STATUS in ["send_neck"]:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from the video feed.")
#             break   # Using 'continue' allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

#         if(STATS_PRINT):
#             frame_count += 1                                        # Update the frame count
#             if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                 elapsed_time1 = time.perf_counter() - start_fps_time
#                 if elapsed_time1 > 0:
#                     fps = frame_count / elapsed_time1
#                     frame_count = 0                                     # Restart the counter
#                     start_fps_time = time.perf_counter()                    # Restart the times

#         # Pre-process the frame to match YOLO input size
#         frame_resized = cv2.resize(frame, (640, 640))               # Resize frame to the desired input size
#         frame_rgb = frame_resized[..., ::-1]                        # Convert BGR to RGB (YOLO typically expects RGB)
#         frame_normalized = frame_rgb / 255.0                        # Normalize the image (YOLO uses values in range [0, 1])
#         transform = transforms.ToTensor()                           # Convert to Tensor and add batch dimension
#         frame_tensor = transform(frame_normalized).unsqueeze(0)     # Add batch dimension
#         frame_tensor = frame_tensor.to(torch.float32)               # Convert to float32 as the first layer require

#         # Can also use this to pre-process
#         # input_tensor = cv2.resize(frame, (640, 640))
#         # input_tensor = input_tensor[..., ::-1]                            # Convert BGR to RGB
#         # input_tensor = np.copy(input_tensor)                              # Create a copy of the array to avoid negative strides
#         # input_tensor = np.transpose(input_tensor, (2, 0, 1))              # Change to (C, H, W)
#         # input_tensor = np.expand_dims(input_tensor, axis=0)               # Add batch dimension
#         # input_tensor = torch.from_numpy(input_tensor).float() / 255.0     # Normalize to [0, 1]

#         # MODEL = MODEL.to('cuda')                    # Move the model to GPU
#         frame_tensor = frame_tensor.to('cuda')      # Ensure the input tensor is on the same device as the model

#         # Stage 1: Backbone (feature extraction)
#         b0 = MODEL.model.model[0](frame_tensor)
#         b1 = MODEL.model.model[1](b0)
#         b2 = MODEL.model.model[2](b1)
#         b3 = MODEL.model.model[3](b2)
#         b4 = MODEL.model.model[4](b3)
#         b5 = MODEL.model.model[5](b4)
#         b6 = MODEL.model.model[6](b5)
#         b7 = MODEL.model.model[7](b6)
#         b8 = MODEL.model.model[8](b7)

#         # Stage 2: Neck (Feature Refinement)
#         b9 = MODEL.model.model[9](b8)
#         b10 = MODEL.model.model[10](b9)
#         b11 = MODEL.model.model[11]([b10,b6])
#         b12 = MODEL.model.model[12](b11)
#         b13 = MODEL.model.model[13](b12)
#         b14 = MODEL.model.model[14]([b13,b4])
#         b15 = MODEL.model.model[15](b14)
#         b16 = MODEL.model.model[16](b15)
#         b17 = MODEL.model.model[17]([b16,b12])
#         b18 = MODEL.model.model[18](b17)
#         b19 = MODEL.model.model[19](b18)
#         b20 = MODEL.model.model[20]([b19,b9])
#         b21 = MODEL.model.model[21](b20)

#         slice = [b15,b18,b21]

#         serialized_tensor = pickle.dumps(slice)
#         # Split the data into chunks for big datas
#         chunk_size = 1024  # Size of each chunk (in bytes)
#         data_length = len(serialized_tensor)
#         print(f"-------------------> Size of FRAME sent in bits: {get_the_size(serialized_tensor)} Bits")
#         print(f"-------------------> Streaming at {fps} FRAME PER SECOND")

#         if(STATS_PRINT):
#             bytes_sent = data_length
#             bytes_count += bytes_sent                             # Update the bytes count
#             elapsed_time2 = time.perf_counter() - start_bps_time
#             if elapsed_time2 > 2:
#                 bps = bytes_count / elapsed_time2
#                 bytes_count = 0                                             # Restart the counter
#                 start_bps_time = time.perf_counter()                        # Restart the times

#             print(f"Bytes of the frame: {bytes_sent}")
#             print(f"Frame per Seconds:  {fps}")
#             print(f"Bytes per Seconds:  {bps}")

#         # Send the tensor
#         conn.sendall(struct.pack('!I', data_length))
#         for i in range(0, data_length, chunk_size):
#             chunk = serialized_tensor[i:i+chunk_size]
#             conn.sendall(chunk)

#     cap.release()
#     conn.close()
#     tensor_socket.close()
#     print("Neck stream is terminated")
    
# def generate_head():
#     print(f"Head stream prepering on {DEVICE_IP}:5444")
#     try:
#         tensor_socket.bind((DEVICE_IP, 5444))
#         tensor_socket.listen(1)
#     except Exception as e:
#         print(f"Error in socket: {e}. Check with 'netstat -tuln| grep 5444'")
#         tensor_socket.close()
#         return
#     print("Head stream is ready")
#     conn, addr = tensor_socket.accept()
#     print("Connection...")
#     cap = cv2.VideoCapture(INPUT_SRC)

#     if not cap.isOpened():
#         print(f"Error: Unable to access the video feed from '{SOURCE_DATA}'. Is the stream active elsewhere?")
#         return

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_fps_time = time.perf_counter()
#     start_bps_time = time.perf_counter()

#     while STATUS in ["send_head"]:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from the video feed.")
#             break   # Using 'continue' allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

#         if(STATS_PRINT):
#             frame_count += 1                                        # Update the frame count
#             if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                 elapsed_time1 = time.perf_counter() - start_fps_time
#                 if elapsed_time1 > 0:
#                     fps = frame_count / elapsed_time1
#                     frame_count = 0                                     # Restart the counter
#                     start_fps_time = time.perf_counter()                    # Restart the times

#         # Pre-process the frame to match YOLO input size
#         frame_resized = cv2.resize(frame, (640, 640))               # Resize frame to the desired input size
#         frame_rgb = frame_resized[..., ::-1]                        # Convert BGR to RGB (YOLO typically expects RGB)
#         frame_normalized = frame_rgb / 255.0                        # Normalize the image (YOLO uses values in range [0, 1])
#         transform = transforms.ToTensor()                           # Convert to Tensor and add batch dimension
#         frame_tensor = transform(frame_normalized).unsqueeze(0)     # Add batch dimension
#         frame_tensor = frame_tensor.to(torch.float32)               # Convert to float32 as the first layer require

#         # Can also use this to pre-process
#         # input_tensor = cv2.resize(frame, (640, 640))
#         # input_tensor = input_tensor[..., ::-1]                            # Convert BGR to RGB
#         # input_tensor = np.copy(input_tensor)                              # Create a copy of the array to avoid negative strides
#         # input_tensor = np.transpose(input_tensor, (2, 0, 1))              # Change to (C, H, W)
#         # input_tensor = np.expand_dims(input_tensor, axis=0)               # Add batch dimension
#         # input_tensor = torch.from_numpy(input_tensor).float() / 255.0     # Normalize to [0, 1]

#         # MODEL = MODEL.to('cuda')                    # Move the model to GPU
#         frame_tensor = frame_tensor.to('cuda')      # Ensure the input tensor is on the same device as the model

#         # Stage 1: Backbone (feature extraction)    # Input a torch.Size([1, 3, 640, 640]) !FOR yolov8 SanRossore.pt!
#         b0 = MODEL.model.model[0](frame_tensor)     # Output a torch.Size([1, 48, 320, 320])
#         b1 = MODEL.model.model[1](b0)               # Output a torch.Size([1, 96, 160, 160])
#         b2 = MODEL.model.model[2](b1)               # Output a torch.Size([1, 96, 160, 160])
#         b3 = MODEL.model.model[3](b2)               # Output a torch.Size([1, 192, 80, 80])
#         b4 = MODEL.model.model[4](b3)               # Output a torch.Size([1, 192, 80, 80])
#         b5 = MODEL.model.model[5](b4)               # Output a torch.Size([1, 384, 40, 40])
#         b6 = MODEL.model.model[6](b5)               # Output a torch.Size([1, 384, 40, 40])
#         b7 = MODEL.model.model[7](b6)               # Output a torch.Size([1, 576, 20, 20])
#         b8 = MODEL.model.model[8](b7)               # Output a torch.Size([1, 576, 20, 20])

#         # Stage 2: Neck (Feature Refinement)
#         b9 = MODEL.model.model[9](b8)               # Output a torch.Size([1, 576, 20, 20])
#         b10 = MODEL.model.model[10](b9)             # Output a torch.Size([1, 576, 40, 40])
#         b11 = MODEL.model.model[11]([b10,b6])       # Output a torch.Size([1, 960, 40, 40])
#         b12 = MODEL.model.model[12](b11)            # Output a torch.Size([1, 384, 40, 40])
#         b13 = MODEL.model.model[13](b12)            # Output a torch.Size([1, 384, 80, 80])
#         b14 = MODEL.model.model[14]([b13,b4])       # Output a torch.Size([1, 576, 80, 80])
#         b15 = MODEL.model.model[15](b14)            # Output a torch.Size([1, 192, 80, 80])
#         b16 = MODEL.model.model[16](b15)            # Output a torch.Size([1, 192, 40, 40])
#         b17 = MODEL.model.model[17]([b16,b12])      # Output a torch.Size([1, 576, 40, 40])
#         b18 = MODEL.model.model[18](b17)            # Output a torch.Size([1, 384, 40, 40])
#         b19 = MODEL.model.model[19](b18)            # Output a torch.Size([1, 384, 20, 20])
#         b20 = MODEL.model.model[20]([b19,b9])       # Output a torch.Size([1, 960, 20, 20])
#         b21 = MODEL.model.model[21](b20)            # Output a torch.Size([1, 576, 20, 20])

#         # Stage 3: Head (Final Predictions)
#         slice = MODEL.model.model[22]([b15,b18,b21])

#         serialized_tensor = pickle.dumps(slice)
#         # Split the data into chunks for big datas
#         chunk_size = 1024  # Size of each chunk (in bytes)
#         data_length = len(serialized_tensor)
#         print(f"-------------------> Size of FRAME sent in bits: {get_the_size(serialized_tensor)} Bits")
#         print(f"-------------------> Streaming at {fps} FRAME PER SECOND")

#         if(STATS_PRINT):
#             bytes_sent = data_length
#             bytes_count += bytes_sent                             # Update the bytes count
#             elapsed_time2 = time.perf_counter() - start_bps_time
#             if elapsed_time2 > 2:
#                 bps = bytes_count / elapsed_time2
#                 bytes_count = 0                                             # Restart the counter
#                 start_bps_time = time.perf_counter()                        # Restart the times

#             print(f"Bytes of the frame: {bytes_sent}")
#             print(f"Frame per Seconds:  {fps}")
#             print(f"Bytes per Seconds:  {bps}")

#         # Send the tensor
#         conn.sendall(struct.pack('!I', data_length))
#         for i in range(0, data_length, chunk_size):
#             chunk = serialized_tensor[i:i+chunk_size]
#             conn.sendall(chunk)

#     cap.release()
#     conn.close()
#     tensor_socket.close()
#     print("Head stream is terminated")

#TODO qui inferenza completa
# def generate_inf():
#     cap = cv2.VideoCapture(INPUT_SRC)

#     if not cap.isOpened():
#         print(f"Error: Unable to access the video feed from '{INPUT_SRC}'. Is the stream active elsewhere?")
#         return
    
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
#     cap.set(cv2.CAP_PROP_FPS, 30)
#     # Verify the settings
#     current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     current_fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Resolution: {current_width}x{current_height}")
#     print(f"Frame Rate (FPS): {current_fps}")

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_time = time.perf_counter()
#     start_bps_time = time.perf_counter()
#     if(MODEL_E):
#         engine_path = model_path.replace('.pt', '.engine')
#         if os.path.exists(engine_path):
#             current_model = YOLO(engine_path)
#         else:
#             print(f"Exporting model {model_path} to TensorRT format...")
#             model = YOLO(model_path)
#             model.export(format='engine', device=0)  # 指定设备
#             while not os.path.exists(engine_path):
#                 time.sleep(1)
#             print(f"TensorRT engine exported and saved to {engine_path}")
#             current_model = YOLO(engine_path)

#     while STATUS == "send_inf":
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from the video feed.")
#             break   # Using continue allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume
#         if(MODEL_E):
#             results = current_model(frame)
#         else:
#             results = MODEL.predict(source=frame,device=0)
            

#         frame_count += 1                                        # Update the frame count
#         if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#             elapsed_time = time.perf_counter() - start_time
#             if elapsed_time > 0:
#                 fps = frame_count / elapsed_time
#             frame_count = 0                                     # Restart the counter
#             start_time = time.perf_counter()                    # Restart the times

#         extracted_res = {
#             'inference_time': results[0].speed['inference'],        # Espressed in ms
#             'boxes': results[0].boxes.data.tolist() if results[0].boxes else [],        # Contain: [x1, y1, x2, y2, confidence, class_id]
#             # 'keypoints': results[0].keypoints.data.tolist() if results[0].keypoints else [],
#             # 'masks': results[0].masks.data.tolist() if results[0].masks else [],
#             'names': results[0].names,
#             # 'path': results[0].path,
#         }
#         yoloFps = 1000/results[0].speed['inference']
#         box_groups = defaultdict(list)                  # Dictionary of boxes grouped by class ID
#         for box in extracted_res['boxes']:
#             class_id = box[5]                           # The 6th value represents the class ID
#             box_groups[class_id].append(box[:5])        # Append the box without the class ID
#         rewritten_res = {
#             'inference_time': extracted_res['inference_time'],
#             'box_count': {extracted_res['names'][class_id]: len(boxes) for class_id, boxes in box_groups.items()},
#             'boxes': box_groups,
#             'yolo_fps': yoloFps,
#             'fps': fps,
#             'names': extracted_res['names']
#         }

#         # Overlay FPS, image size, and camera FPS on the frame
#         if(RUNNING_VIDEO):
#             cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)    # The triplete (x, x, x) is the color of the text
#             cv2.putText(frame, f"Frame Rate: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             if(STATS_PRINT):
#                 cv2.putText(frame, f"Estimated FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#                 cv2.putText(frame, f"Yolo FPS: {yoloFps:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
#                 (frame.shape[1] - cv2.getTextSize(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0][0] - 10, 
#                 frame.shape[0] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

#             # Encode the frame as JPEG
#             _, buffer = cv2.imencode('.jpg', results[0].plot())
#             frame_bytes = buffer.tobytes()
#             print(f"-------------------> Size of FRAME sent in bits: {get_the_size(frame_bytes)} Bits")
#             print(f"-------------------> Streaming at {fps} FRAME PER SECOND")
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         else:
#             # Convert the result data dictionary to JSON string
#             result_json = json.dumps(rewritten_res)
#             print(f"-------------------> Size of FRAME sent in bits: {get_the_size(result_json)} Bits")
#             print(f"-------------------> Streaming at {fps} FRAME PER SECOND")

#             if(STATS_PRINT):
#                 bytes_sent = len(result_json)
#                 bytes_count += bytes_sent                             # Update the bytes count
#                 elapsed_time2 = time.perf_counter() - start_bps_time
#                 if elapsed_time2 > 2:
#                     bps = bytes_count / elapsed_time2
#                     bytes_count = 0                                             # Restart the counter
#                     start_bps_time = time.perf_counter()                        # Restart the times

#                 print(f"Bytes of the frame: {bytes_sent}")
#                 print(f"Frame per Seconds:  {fps}")
#                 print(f"Bytes per Seconds:  {bps}")

#             # Yield the JSON-encoded results
#             yield (b'--frame\r\n'
#                 b'Content-Type: application/json\r\n\r\n' + result_json.encode('utf-8') + b'\r\n')
        
#     cap.release()
#     print("Inference stream is terminated")

# ###
# #   OPERATIVE PARTS (as helping device)
# ###

# ### with torch.no_grad():
# ### is a context manager in PyTorch that disables gradient computation, which is useful during inference or evaluation.
# ### When used can save memory and computational resources, during model evaluation or inference you don’t need gradients since you’re not updating any parameters.

# ### print(MODEL.model)  or look at https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml for layer structure
# ### as HEAD output, referred as 'detectHead', we can find:
# ### detectHead[0]: Primary predictions (e.g., processed for object detection).
# ### detectHead[1]: Feature maps used for further calculations or analysis (e.g., visualization, debugging, additional model layers).
# ### Can examine better the content with:
# ###  if isinstance(b23, (list, tuple)):
# ###      for i, tensor in enumerate(b23):
# ###          print(f"Type of element {i} in b23: {type(tensor)}")
# ###          if isinstance(tensor, torch.Tensor):
# ###              print(f"Shape of tensor {i} in b23: {tensor.shape}")
# ###          elif isinstance(tensor, list):
# ###              print(f"Element {i} in b23 is a list with length: {len(tensor)}")
# ###          else:
# ###              print(f"Unexpected type for element {i}: {type(tensor)}")
# ###  else:
# ###      print("Unexpected output type for b23:", type(b23))
# ###  if isinstance(b23, (list, tuple)):
# ###      for i, item in enumerate(b23):
# ###          if isinstance(item, list):
# ###              for j, sub_item in enumerate(item):
# ###                  print(f"Type of sub-item {j} in list at b23[{i}]: {type(sub_item)}")
# ###                  if isinstance(sub_item, torch.Tensor):
# ###                      print(f"Shape of sub-item {j}: {sub_item.shape}")

# def consume_raw():
#     try:
#         tensor_socket.connect((SOURCE_DATA, 5111))
#     except Exception as e:
#         print(f"Error in socket.connect: {e}. Get stream from {SOURCE_DATA}:5111")
#         tensor_socket.close()
#         return

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_time = time.perf_counter()
#     start_bps_time = time.perf_counter()
#     while STATUS in ["get_raw"]:
#         try:
#             # Receive the total length of the data first (4 bytes)
#             data_length = struct.unpack('!I', tensor_socket.recv(4))[0]

#             # Receive data in chunks and append to reconstruct the full data
#             buffer = b''  # To hold the entire serialized data
#             while len(buffer) < data_length:
#                 chunk = tensor_socket.recv(min(1024, data_length - len(buffer)))  # Receive a chunk
#                 buffer += chunk
#             if buffer:
#                 # Deserialize the data
#                 frame = pickle.loads(buffer)
#                 results = MODEL.predict(source=frame)

#                 extracted_res = {
#                     'inference_time': results[0].speed['inference'],        # Espressed in ms
#                     'boxes': results[0].boxes.data.tolist() if results[0].boxes else [],        # Contain: [x1, y1, x2, y2, confidence, class_id]
#                     # 'keypoints': results[0].keypoints.data.tolist() if results[0].keypoints else [],
#                     # 'masks': results[0].masks.data.tolist() if results[0].masks else [],
#                     'names': results[0].names,
#                     # 'path': results[0].path,
#                 }
#                 yoloFps = 1000/results[0].speed['inference']
#                 box_groups = defaultdict(list)                  # Dictionary of boxes grouped by class ID
#                 for box in extracted_res['boxes']:
#                     class_id = box[5]                           # The 6th value represents the class ID
#                     box_groups[class_id].append(box[:5])        # Append the box without the class ID
#                 rewritten_res = {
#                     'inference_time': extracted_res['inference_time'],
#                     'box_count': {extracted_res['names'][class_id]: len(boxes) for class_id, boxes in box_groups.items()},
#                     'boxes': box_groups,
#                     'yolo_fps': yoloFps,
#                     'fps': fps,
#                     'names': extracted_res['names']
#                 }

#                 frame_count += 1                                        # Update the frame count
#                 if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                     elapsed_time = time.perf_counter() - start_time
#                     if elapsed_time > 0:
#                         fps = frame_count / elapsed_time
#                     frame_count = 0                                     # Restart the counter
#                     start_time = time.perf_counter()                    # Restart the times

#                 # Overlay FPS, image size, and camera FPS on the frame
#                 if(RUNNING_VIDEO):
#                     # cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)    # The triplete (x, x, x) is the color of the text
#                     # cv2.putText(frame, f"Frame Rate: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#                     if(STATS_PRINT):
#                         cv2.putText(frame, f"Estimated FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#                         # cv2.putText(frame, f"Yolo FPS: {yoloFps:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#                     cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
#                         (frame.shape[1] - cv2.getTextSize(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0][0] - 10, 
#                         frame.shape[0] - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

#                     # Encode the frame as JPEG
#                     _, buffer = cv2.imencode('.jpg', results[0].plot())
#                     frame_bytes = buffer.tobytes()
#                     print(f"-------------------> Size of FRAME sent in bits: {get_the_size(frame_bytes)} Bits")
#                     print(f"-------------------> Streaming at {fps} FRAME PER SECOND")
#                     yield (b'--frame\r\n'
#                         b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#                 else:
#                     # Convert the result data dictionary to JSON string
#                     result_json = json.dumps(rewritten_res)
#                     print(f"-------------------> Size of FRAME sent in bits: {get_the_size(result_json)} Bits")
#                     print(f"-------------------> Streaming at {fps} FRAME PER SECOND")

#                     if(STATS_PRINT):
#                         bytes_sent = len(result_json)
#                         bytes_count += bytes_sent                             # Update the bytes count
#                         elapsed_time2 = time.perf_counter() - start_bps_time
#                         if elapsed_time2 > 2:
#                             bps = bytes_count / elapsed_time2
#                             bytes_count = 0                                             # Restart the counter
#                             start_bps_time = time.perf_counter()                        # Restart the times

#                         print(f"Bytes of the frame: {bytes_sent}")
#                         print(f"Frame per Seconds:  {fps}")
#                         print(f"Bytes per Seconds:  {bps}")

#                     # Yield the JSON-encoded results
#                     yield (b'--frame\r\n'
#                         b'Content-Type: application/json\r\n\r\n' + result_json.encode('utf-8') + b'\r\n')

#             else:
#                 print("No data recieved")
#                 break
#         except Exception as e:
#             print(f"Error receiving data: {e}")
#             break
#     tensor_socket.close()
#     print("Consume raw is terminated")


# def consume_jpg():
#     cap = cv2.VideoCapture(SOURCE_DATA)

#     if not cap.isOpened():
#         print(f"Error: Unable to access the video feed from '{INPUT_SRC}'. Is the stream active elsewhere?")
#         return
    
#     # Stream settings
#     current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     current_fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Resolution: {current_width}x{current_height}")
#     print(f"Frame Rate (FPS): {current_fps}")

#     # Initialize FPS calculation
#     fps = 0
#     bps = 0
#     frame_count = 0
#     bytes_count = 0
#     start_time = time.perf_counter()
#     start_bps_time = time.perf_counter()
#     while STATUS == "get_jpg":
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from the video feed.")
#             break   # Using continue allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume

#         results = MODEL.predict(source=frame)

#         frame_count += 1                                        # Update the frame count
#         if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#             elapsed_time = time.perf_counter() - start_time
#             if elapsed_time > 0:
#                 fps = frame_count / elapsed_time
#             frame_count = 0                                     # Restart the counter
#             start_time = time.perf_counter()                    # Restart the times

#         extracted_res = {
#             'inference_time': results[0].speed['inference'],        # Espressed in ms
#             'boxes': results[0].boxes.data.tolist() if results[0].boxes else [],        # Contain: [x1, y1, x2, y2, confidence, class_id]
#             # 'keypoints': results[0].keypoints.data.tolist() if results[0].keypoints else [],
#             # 'masks': results[0].masks.data.tolist() if results[0].masks else [],
#             'names': results[0].names,
#             # 'path': results[0].path,
#         }
#         yoloFps = 1000/results[0].speed['inference']
#         box_groups = defaultdict(list)                  # Dictionary of boxes grouped by class ID
#         for box in extracted_res['boxes']:
#             class_id = box[5]                           # The 6th value represents the class ID
#             box_groups[class_id].append(box[:5])        # Append the box without the class ID
#         rewritten_res = {
#             'inference_time': extracted_res['inference_time'],
#             'box_count': {extracted_res['names'][class_id]: len(boxes) for class_id, boxes in box_groups.items()},
#             'boxes': box_groups,
#             'yolo_fps': yoloFps,
#             'fps': fps,
#             'names': extracted_res['names']
#         }

#         # Overlay FPS, image size, and camera FPS on the frame
#         if(RUNNING_VIDEO):
#             cv2.putText(frame, f"Image Size: {current_width}x{current_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)    # The triplete (x, x, x) is the color of the text
#             # cv2.putText(frame, f"Frame Rate: {current_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             if(STATS_PRINT):
#                 cv2.putText(frame, f"Estimated FPS: {fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#                 cv2.putText(frame, f"Yolo FPS: {yoloFps:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#             cv2.putText(frame, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
#                 (frame.shape[1] - cv2.getTextSize(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)[0][0] - 10, 
#                 frame.shape[0] - 10), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

#             # Encode the frame as JPEG
#             _, buffer = cv2.imencode('.jpg', results[0].plot())
#             frame_bytes = buffer.tobytes()
#             print(f"-------------------> Size of FRAME sent in bits: {get_the_size(frame_bytes)} Bits")
#             print(f"-------------------> Streaming at {fps} FRAME PER SECOND")
#             yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         else:
#             # Convert the result data dictionary to JSON string
#             result_json = json.dumps(rewritten_res)
#             print(f"-------------------> Size of FRAME sent in bits: {get_the_size(result_json)} Bits")
#             print(f"-------------------> Streaming at {fps} FRAME PER SECOND")

#             if(STATS_PRINT):
#                 bytes_sent = len(result_json)
#                 bytes_count += bytes_sent                             # Update the bytes count
#                 elapsed_time2 = time.perf_counter() - start_bps_time
#                 if elapsed_time2 > 2:
#                     bps = bytes_count / elapsed_time2
#                     bytes_count = 0                                             # Restart the counter
#                     start_bps_time = time.perf_counter()                        # Restart the times

#                 print(f"Bytes of the frame: {bytes_sent}")
#                 print(f"Frame per Seconds:  {fps}")
#                 print(f"Bytes per Seconds:  {bps}")

#             # Yield the JSON-encoded results
#             yield (b'--frame\r\n'
#                 b'Content-Type: application/json\r\n\r\n' + result_json.encode('utf-8') + b'\r\n')
        
#     cap.release()
#     print("Consume jpg is terminated")

# def consume_back():
#     try:
#         tensor_socket.connect((SOURCE_DATA, 5222))
#     except Exception as e:
#         print(f"Error in socket.connect: {e}. Get stream from {SOURCE_DATA}:5222")
#         tensor_socket.close()
#         return

#     # Initialize FPS calculation
#     fps = 0
#     # bps = 0
#     frame_count = 0
#     # bytes_count = 0
#     start_time = time.perf_counter()
#     # start_bps_time = time.perf_counter()
#     while STATUS in ["get_back"]:
#         try:
#             # Receive the total length of the data first (4 bytes)
#             data_length = struct.unpack('!I', tensor_socket.recv(4))[0]

#             # Receive data in chunks and append to reconstruct the full data
#             buffer = b''  # To hold the entire serialized data
#             while len(buffer) < data_length:
#                 chunk = tensor_socket.recv(min(1024, data_length - len(buffer)))  # Receive a chunk
#                 buffer += chunk
#             if buffer:
#                 # Deserialize the data
#                 slice = pickle.loads(buffer)

#                 # Stage 2: Neck (Feature Refinement)
#                 b6 = slice[0].to('cuda')      # Ensure the input tensor is on the same device as the model
#                 b4 = slice[1].to('cuda')      # Ensure the input tensor is on the same device as the model
#                 b8 = slice[2].to('cuda')      # Ensure the input tensor is on the same device as the model
#                 b9 = MODEL.model.model[9](b8)
#                 b10 = MODEL.model.model[10](b9)
#                 b11 = MODEL.model.model[11]([b10,b6])
#                 b12 = MODEL.model.model[12](b11)
#                 b13 = MODEL.model.model[13](b12)
#                 b14 = MODEL.model.model[14]([b13,b4])
#                 b15 = MODEL.model.model[15](b14)
#                 b16 = MODEL.model.model[16](b15)
#                 b17 = MODEL.model.model[17]([b16,b12])
#                 b18 = MODEL.model.model[18](b17)
#                 b19 = MODEL.model.model[19](b18)
#                 b20 = MODEL.model.model[20]([b19,b9])
#                 b21 = MODEL.model.model[21](b20)

#                 # Stage 3: Head (Final Predictions)
#                 b23 = MODEL.model.model[22]([b15,b18,b21])

#                 # Post-process the predictions, converting predictions into bounding boxes, confidences, and class IDs
                
#                 # Stream the results

#                 frame_count += 1                                        # Update the frame count
#                 if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                     elapsed_time = time.perf_counter() - start_time
#                     if elapsed_time > 0:
#                         fps = frame_count / elapsed_time
#                     frame_count = 0                                     # Restart the counter
#                     start_time = time.perf_counter()                    # Restart the times

#                 print(f"-------------------> Size of FRAME sent in bits: {get_the_size(b23)} Bits")
#                 print(f"-------------------> Streaming at {fps} FRAME PER SECOND")
#                 if(STATS_PRINT):
#                     # bytes_sent = len(result_json)
#                     # bytes_count += bytes_sent                             # Update the bytes count
#                     # elapsed_time2 = time.perf_counter() - start_bps_time
#                     # if elapsed_time2 > 2:
#                         # bps = bytes_count / elapsed_time2
#                         # bytes_count = 0                                             # Restart the counter
#                         # start_bps_time = time.perf_counter()                        # Restart the times

#                     # print(f"Bytes of the frame: {bytes_sent}")
#                     print(f"Frame per Seconds:  {fps}")
#                     # print(f"Bytes per Seconds:  {bps}")

#             else:
#                 print("No data recieved")
#                 break
#         except Exception as e:
#             print(f"Error receiving data: {e}")
#             break
    
# def consume_neck():
#     try:
#         tensor_socket.connect((SOURCE_DATA, 5333))
#     except Exception as e:
#         print(f"Error in socket.connect: {e}. Get stream from {SOURCE_DATA}:5333")
#         tensor_socket.close()
#         return

#     # Initialize FPS calculation
#     fps = 0
#     # bps = 0
#     frame_count = 0
#     # bytes_count = 0
#     start_time = time.perf_counter()
#     # start_bps_time = time.perf_counter()
#     while STATUS in ["get_neck"]:
#         try:
#             # Receive the total length of the data first (4 bytes)
#             data_length = struct.unpack('!I', tensor_socket.recv(4))[0]

#             # Receive data in chunks and append to reconstruct the full data
#             buffer = b''  # To hold the entire serialized data
#             while len(buffer) < data_length:
#                 chunk = tensor_socket.recv(min(1024, data_length - len(buffer)))  # Receive a chunk
#                 buffer += chunk
#             if buffer:
#                 # Deserialize the data
#                 slice = pickle.loads(buffer)

#                 # Stage 3: Head (Final Predictions)
#                 b15 = slice[0].to('cuda')      # Ensure the input tensor is on the same device as the model
#                 b18 = slice[1].to('cuda')      # Ensure the input tensor is on the same device as the model
#                 b21 = slice[2].to('cuda')      # Ensure the input tensor is on the same device as the model
#                 b23 = MODEL.model.model[22]([b15,b18,b21])

#                 # Post-process the predictions, converting predictions into bounding boxes, confidences, and class IDs
                
#                 # Stream the results

#                 frame_count += 1                                        # Update the frame count
#                 if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                     elapsed_time = time.perf_counter() - start_time
#                     if elapsed_time > 0:
#                         fps = frame_count / elapsed_time
#                     frame_count = 0                                     # Restart the counter
#                     start_time = time.perf_counter()                    # Restart the times
#                 print(f"-------------------> Size of FRAME sent in bits: {get_the_size(b23)} Bits")
#                 print(f"-------------------> Streaming at {fps} FRAME PER SECOND")
#                 if(STATS_PRINT):
#                     # bytes_sent = len(result_json)
#                     # bytes_count += bytes_sent                             # Update the bytes count
#                     # elapsed_time2 = time.perf_counter() - start_bps_time
#                     # if elapsed_time2 > 2:
#                         # bps = bytes_count / elapsed_time2
#                         # bytes_count = 0                                             # Restart the counter
#                         # start_bps_time = time.perf_counter()                        # Restart the times

#                     # print(f"Bytes of the frame: {bytes_sent}")
#                     print(f"Frame per Seconds:  {fps}")
#                     # print(f"Bytes per Seconds:  {bps}")
                
#             else:
#                 print("No data recieved")
#                 break
#         except Exception as e:
#             print(f"Error receiving data: {e}")
#             break
    
# def consume_head():
#     try:
#         tensor_socket.connect((SOURCE_DATA, 5444))
#     except Exception as e:
#         print(f"Error in socket.connect: {e}. Get stream from {SOURCE_DATA}:5444")
#         tensor_socket.close()
#         return

#     # Initialize FPS calculation
#     fps = 0
#     # bps = 0
#     frame_count = 0
#     # bytes_count = 0
#     start_time = time.perf_counter()
#     # start_bps_time = time.perf_counter()
#     while STATUS in ["get_head"]:
#         try:
#             # Receive the total length of the data first (4 bytes)
#             data_length = struct.unpack('!I', tensor_socket.recv(4))[0]

#             # Receive data in chunks and append to reconstruct the full data
#             buffer = b''  # To hold the entire serialized data
#             while len(buffer) < data_length:
#                 chunk = tensor_socket.recv(min(1024, data_length - len(buffer)))  # Receive a chunk
#                 buffer += chunk
#             if buffer:
#                 # Deserialize the data
#                 slice = pickle.loads(buffer)

#                 # Post-process the predictions, converting predictions into bounding boxes, confidences, and class IDs
                
#                 # Stream the results

#                 frame_count += 1                                        # Update the frame count
#                 if frame_count % 30 == 0:                               # Calculate FPS every 30 frames
#                     elapsed_time = time.perf_counter() - start_time
#                     if elapsed_time > 0:
#                         fps = frame_count / elapsed_time
#                     frame_count = 0                                     # Restart the counter
#                     start_time = time.perf_counter()                    # Restart the times
#                 print(f"-------------------> Size of FRAME sent in bits: {get_the_size(slice)} Bits")
#                 print(f"-------------------> Streaming at {fps} FRAME PER SECOND")
#                 if(STATS_PRINT):
#                     # bytes_sent = len(result_json)
#                     # bytes_count += bytes_sent                             # Update the bytes count
#                     # elapsed_time2 = time.perf_counter() - start_bps_time
#                     # if elapsed_time2 > 2:
#                         # bps = bytes_count / elapsed_time2
#                         # bytes_count = 0                                             # Restart the counter
#                         # start_bps_time = time.perf_counter()                        # Restart the times

#                     # print(f"Bytes of the frame: {bytes_sent}")
#                     print(f"Frame per Seconds:  {fps}")
#                     # print(f"Bytes per Seconds:  {bps}")

#             else:
#                 print("No data recieved")
#                 break
#         except Exception as e:
#             print(f"Error receiving data: {e}")
#             break
#     tensor_socket.close()
#     print("Consume head is terminated")

# ###
# #   ENDPOINTS
# ###

# ### mimetype specifies the media type of the HTTP response:
# ### - multipart/x-mixed-replace is used for server push content, where the server continuously sends new parts
# ### - boundary=frame defines the boundary string that separates individual parts of the data stream

# @app.route("/stream_raw")
# def stream_raw():
#     global STATUS
#     if STATUS != "send_raw":
#         return "Raw stream not active", 400
#     return Response(generate_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/stream_jpg")
# def stream_jpg():
#     global STATUS
#     if STATUS != "send_jpg":
#         return "Jpg stream not active", 400
#     return Response(generate_jpg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/stream_back")
# def stream_back():
#     global STATUS
#     if STATUS != "send_back":
#         return "Backbone stream not active", 400
#     return Response(generate_back(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/stream_neck")
# def stream_neck():
#     global STATUS
#     if STATUS != "send_neck":
#         return "Neck stream not active", 400
#     return Response(generate_neck(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/stream_head")
# def stream_head():
#     global STATUS
#     if STATUS not in ["send_head"]:
#         return "Head stream not active", 400
#     return Response(generate_head(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/stream_inf")
# def stream_inf():
#     global STATUS
#     if STATUS != "send_inf":
#         return "Inference stream not active", 400
#     return Response(generate_inf(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/help_raw")
# def help_raw():
#     global STATUS
#     if STATUS != "get_raw":
#         return "HELP Raw stream not active", 400
#     return Response(consume_raw(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/help_jpg")
# def help_jpg():
#     global STATUS
#     if STATUS != "get_jpg":
#         return "HELP Jpg stream not active", 400
#     return Response(consume_jpg(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/help_back")
# def help_back():
#     global STATUS
#     if STATUS != "get_back":
#         return "HELP Backbone stream not active", 400
#     return Response(consume_back(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/help_neck")
# def help_neck():
#     global STATUS
#     if STATUS != "get_neck":
#         return "HELP Neck stream not active", 400
#     return Response(consume_neck(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/help_head")
# def help_head():
#     global STATUS
#     if STATUS != "get_head":
#         return "HELP Head stream not active", 400
#     return Response(consume_head(), mimetype="multipart/x-mixed-replace; boundary=frame")

# @app.route("/change_status", methods=["POST"])
# def change_status():
#     global STATUS
#     ols_stat = STATUS
#     status = request.form.get("status")
#     if status not in ["idle", "send_raw", "send_back", "send_neck", "send_head", "send_inf", "get_raw", "get_jpg", "send_jpg", "get_back", "get_neck", "get_head"]:
#         print("Invalid status")
#         return redirect("/")

#     STATUS = "idle"             # To stop previous streaming threads tasks
#     threading.Event().wait(1)   # introduces a 1-second delay

#     if status == "send_raw":
#         STATUS = "send_raw"
#         generate_raw();
#     elif status == "send_jpg":
#         STATUS = "send_jpg"
#     elif status == "send_back":
#         STATUS = "send_back"
#         generate_back();
#     elif status == "send_neck":
#         STATUS = "send_neck"
#         generate_neck();
#     elif status == "send_head":
#         STATUS = "send_head"
#         generate_head();
#     elif status == "send_inf":
#         STATUS = "send_inf"
#     elif status == "get_raw":
#         STATUS = "get_raw"
#     elif status == "get_jpg":
#         STATUS = "get_jpg"
#     elif status == "get_back":
#         STATUS = "get_back"
#     elif status == "get_neck":
#         STATUS = "get_neck"
#     elif status == "get_head":
#         STATUS = "get_head"
#     else:
#         STATUS = "idle"

#     return f"Status changed from '{ols_stat}' to '{STATUS}'", 200


# @app.route("/")
# def home():
#     return """
#     <h1>Flask Wild Detector Control Panel</h1>
#     <form action="/set_env" method="POST">
#         <label>Set external data source, actually '{source_data}':</label>
#         <input type="text" name="value">
#         <button type="submit">Set</button>
#     </form>
#     <form action="/set_env" method="POST">
#         <form action="/change_status" method="POST">
#         <label>Set inference model, actually 'yolov8m.pt':</label>
#         <input type="text" name="value2">
#         <button type="submit">Set</button>
#     </form>
#     <form action="/change_status" method="POST">
#         <label>Change Status:</label>
#         <select name="status">
#             <option value="idle">Idle</option>
#             <option value="send_raw">Share Raw</option>
#             <option value="send_jpg">Share Jpg</option>
#             <option value="send_back">Share Backbone</option>
#             <option value="send_neck">Share Neck</option>
#             <option value="send_head">Share Head</option>
#             <option value="send_inf">Share Inference</option>
#             <option value="none">----------------</option>
#             <option value="get_raw">Help in Raw</option>
#             <option value="get_jpg">Help in Jpg</option>
#             <option value="get_back">Help in Backbone</option>
#             <option value="get_neck">Help in Neck</option>
#             <option value="get_head">Help in Head</option>
#         </select>
#         <button type="submit">Change</button>
#     </form>
#     <p>Current Status: {status}</p>
#     """.format(source_data=SOURCE_DATA, status=STATUS)

def init_source_settings(cap, target_width, target_height, target_fps):
    """
    Initializes the source settings, like a camera, with the specified resolution and FPS.

    :param cap: The cv2.VideoCapture object
    :param target_width: The desired width for the source feed
    :param target_height: The desired height for the source feed
    :param target_fps: The desired frames per second for the source feed
    :return: None, updates the settings on the provided cap object
    """
    app.logger.info("Initializing video source")

    # Impose the settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Verify the settings
    current_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    current_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    current_fps = cap.get(cv2.CAP_PROP_FPS)

    app.logger.info(f"Requested resolution: {target_width}x{target_height}")
    app.logger.info(f"Requested Frame Rate (FPS): {target_fps}")
    app.logger.info(f"Source resolution: {current_width}x{current_height}")
    app.logger.info(f"Source Frame Rate (FPS): {current_fps}")

    # Check if the camera actually supports the desired FPS/resolution
    if abs(current_fps - target_fps) > 1:
        app.logger.warning(f"Warning: Desired FPS of {target_fps} is not supported. Using {current_fps} instead.")
    if current_width != target_width or current_height != target_height:
        app.logger.warning(f"Warning: Desired resolution of {target_width}x{target_height} is not supported. Using {current_width}x{current_height} instead.")

def control_loop():
    """
    Is the core of the program, the control loop permit to handle state machine and react as a consequence.
    """
    if app.config['STATUS'] == 0: #idle
        return

    model_path = MODEL_LIST.get(app.config.get('MODEL', None))
    if model_path is None:
        app.logger.error("Error: model configuration is missing.")
        cap.release()
        return
    cnn = YOLO(model_path)     # for i, layer in enumerate(cnn.model.model): print(f"Layer {i}: {layer}", flush=True)

    cap = cv2.VideoCapture(app.config['INPUT_SRC'])
    if not cap.isOpened():
        app.logger.error("Error: Could not open video source.")
        app.logger.info("Trying opening stream from gRPC...")
        if not 0:
            app.logger.error("Error: Could not open gRPC source.")
        else:
            app.logger.info("Initializing gRPC source")
            # TODO aggiungi per avere un effettivo uso di gRPC
            # init_gRPC()
    else:
        init_source_settings(cap, 1080, 720, 30)

    while True:     # TODO testa la velocita di inferenza normale e aprendo ogni layer
        if app.config['STATUS'] == 0: #idle
            break   #TODO aggiungi il caso in cui lo stream finisce, pensare se break oppure aspettare e poi break
###################################
#
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the video feed.")
            break   # Using continue allows the loop to skip the current iteration and attempt to read the next frame. This approach assumes that the issue is transient and the video feed will resume
        else:
            results = cnn.predict(source=frame)

        extracted_res = {
            'inference_time': results[0].speed['inference'],        # Espressed in ms
            'boxes': results[0].boxes.data.tolist() if results[0].boxes else [],        # Contain: [x1, y1, x2, y2, confidence, class_id]
            # 'keypoints': results[0].keypoints.data.tolist() if results[0].keypoints else [],
            # 'masks': results[0].masks.data.tolist() if results[0].masks else [],
            'names': results[0].names,
            # 'path': results[0].path,
        }
        print(f"------ {extracted_res}", flush=True)
#
###################################
        app.logger.info("loop")
        time.sleep(1) #TODO Can be used for tuning the FPS output

    cap.release()

def update_config_variable(variable_name, current_value, new_value, dict_entry, valid_values=None):
    """
    Update a Flask config variable.
    
    :param variable_name: The name of the config variable to update.
    :param current_value: The current value of the variable.
    :param new_value: The new value from the user input (e.g., form submission).
    :param dict_entry: Can be 0 or 1 depending on if putting the value as for determinated dictionary key(1) or straight the value(0)
    :param valid_values: Optional dictionary or list of valid values for the variable.
                          If None, no validation is performed.
    :return: The updated value or current value if invalid.
    """
    if valid_values is None or new_value in valid_values:
        if dict_entry:
            app.config[variable_name] = valid_values.get(new_value, current_value)
        else:
            app.config[variable_name] = new_value
        return new_value
    else:
        return current_value

###
#   ENDPOINTS
###
@app.route('/change_status', methods=['POST'])
def change_status():
    new_status = request.form.get('status')     # Get the selected status from the form
    update_config_variable(
        'STATUS', 
        app.config['STATUS'],  # Current status
        new_status,
        1,
        valid_values=VALID_STATUSES  # Use the valid statuses dictionary
    )
    return redirect(url_for('index'))   # Redirect back to the main page

@app.route('/change_model', methods=['POST'])
def change_model():
    new_model = request.form.get('model')     # Get the selected model from the form
    update_config_variable(
        'MODEL',
        app.config['MODEL'],    # Current model
        new_model,
        0,
        valid_values=MODEL_LIST  # Use the valid modeles dictionary
    )
    return redirect(url_for('index'))   # Redirect back to the main page

@app.route('/change_input_video', methods=['POST'])
def change_input_video():
    new_input = request.form.get('input_video')     # Get the selected model from the form
    update_config_variable(
        'INPUT_SRC',
        app.config['INPUT_SRC'],    # Current video source
        new_input,
        0,
    )
    return redirect(url_for('index'))   # Redirect back to the main page

@app.route('/start_loop', methods=['POST'])
def start_loop():
    global control_background_thread

    # The control thread is set and if is running is no started another one. When the thread function returns the thread is killed
    if control_background_thread is None or not control_background_thread.is_alive():
        app.logger.info("Starting control loop thread")
        control_background_thread = threading.Thread(target=control_loop, daemon=True)
        control_background_thread.start()
    else:
        app.logger.info("Control loop thread already running. Doing nothing.")
    
    return redirect('/')

@app.route('/')
def index():
    return render_template('index.html',
                            status=next((key for key, value in VALID_STATUSES.items() if value == app.config['STATUS']), 'Error'),
                            valid_statuses=sorted(VALID_STATUSES.items()),
                            model=app.config['MODEL'],
                            model_list=sorted(MODEL_LIST.items()),
                            input=app.config['INPUT_SRC'],
                            )

###
#   MAIN
###
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=RUNNING_PORT)
