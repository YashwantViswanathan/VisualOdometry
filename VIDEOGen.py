#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:20:45 2025

@author: yashwant
"""

import cv2
import os

# Directory with annotated frames
frame_dir = 'distance_angle_conf07'
output_video = 'joint_vid_Midas_4.mp4'

# Get sorted list of frame files
frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
if not frames:
    raise ValueError("No frames found in the directory.")

# Read first frame to get dimensions
first_frame = cv2.imread(os.path.join(frame_dir, frames[0]))
height, width, layers = first_frame.shape

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, 20.0, (width, height))

# Write each frame
for frame_name in frames:
    frame_path = os.path.join(frame_dir, frame_name)
    frame = cv2.imread(frame_path)
    out.write(frame)

out.release()
print(f"Saved video to {output_video}")
