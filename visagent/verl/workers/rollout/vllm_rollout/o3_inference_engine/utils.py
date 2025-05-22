import os
import json
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile

# Function to encode the video
def encode_video(video_path, for_get_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

    # Ensure the last frame is included
    if total_frame_num - 1 not in uniform_sampled_frames:
        uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        base64_frames.append(base64_str)

    return base64_frames

def encode_video_and_save(video_path, for_get_frames_num):
    # Create a temporary directory to store frames
    temp_dir = tempfile.mkdtemp()
    frame_paths = []
    
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

    # Ensure the last frame is included
    if total_frame_num - 1 not in uniform_sampled_frames:
        uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        img.save(frame_path, format="PNG")
        frame_paths.append(frame_path)

    return frame_paths