import os
import cv2
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio



# Base directory
base_dir = "/home/k8suser/hdd1/MEAD_EXTRACTED"

def audio_encode(fa, video_path, output_path):

    # Create output directory
    base_directory = os.path.dirname(output_path)
    os.makedirs(base_directory, exist_ok=True)
    video_name = os.path.splitext(output_path)[0]
    output_file = os.path.join(base_directory, f"{video_name}.npz")
    if os.path.exists(output_file):
        print('skip ', output_file)
        return
    
    waveform, sample_rate = torchaudio.load(video_path)
    # resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    # waveform = resampler(waveform)
    waveform = waveform.mean(dim=-2, keepdim=False)
    
    input_values = processor(waveform , sampling_rate=16000, return_tensors="pt").input_values

    # Extract features
    with torch.no_grad():
        vec = detector(input_values)


        # Convert list to NumPy array and save to .npz
        if vec:
            vec = np.array(vec.last_hidden_state  )  # Shape: (num_frames, num_landmarks, 2)
            np.savez(output_file, vec=vec)
            print(f"Saved vector for frames to {output_file}")
        else:
            print("No vector detected in the video.")
        return output_file
    # print("Landmark extraction complete.")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
detector = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
detector.eval()

# Iterate through the base directory
for root, dirs, files in os.walk(base_dir):
    # Initialize Face Alignment library
    # fa.get_landmarks_from_batch(torch.zeros((10, 3, 1920, 1080)))
    for dir_name in dirs:
        if dir_name == "audio":  # Check for 'front' directory
            front_dir_path = os.path.join(root, dir_name)
            for emotion_folder in os.listdir(front_dir_path):
                emotion_path = os.path.join(front_dir_path, emotion_folder)
                if os.path.isdir(emotion_path):  # Check if it's a directory
                    print(f"Processing emotion folder: {emotion_folder}")
                    # print(emotion_path)
                    # Process videos inside the emotion folder
                    for emotion_level in os.listdir(emotion_path):
                        emotion_level_path = os.path.join(emotion_path, emotion_level)
                        for video_file in os.listdir(emotion_level_path):
                            video_path = os.path.join(emotion_level_path, video_file)
                            landmark_path = video_path.replace('/audio/', '/audio_vec/')
                            if video_file.endswith((".m4a", )):  # Filter video files
                                print(f"Found video: {video_path}")
                                try:
                                    audio_encode(detector, video_path, landmark_path)
                                except Exception as e:
                                    print(f"Error on video: {video_path}")
                                # print(f'Save to: {landmark_path}')
                                # exit()