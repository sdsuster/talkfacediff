import os
import cv2
import tensorflow as tf
from mtcnn import MTCNN
import numpy as np

# Base directory
base_dir = "/home/k8suser/hdd1/MEAD_EXTRACTED"


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
def bbox_extract(fa, video_path, output_path):

    # Create output directory
    base_directory = os.path.dirname(output_path)
    os.makedirs(base_directory, exist_ok=True)
    video_name = os.path.splitext(output_path)[0]
    output_file = os.path.join(base_directory, f"{video_name}.npz")
    if os.path.exists(output_file):
        print('skip ', output_file)
        return
    # Open the video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    all_landmarks = []
    # batch_frames = []
    
    # batch_frames = []

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     # Convert the frame to RGB (Face Alignment requires RGB images)
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    #     # Add the frame to the batch
    #     batch_frames.append(rgb_frame)

    #     # Process the batch once it reaches the batch_size
    #     if len(batch_frames) >= batch_size:
    #         # Convert list to a numpy array for batch processing
    #         batch_frames_np = np.array(batch_frames)

    #         # Detect landmarks for the batch
    #         batch_landmarks = fa.get_landmarks_from_batch(torch.Tensor(batch_frames_np.transpose((0, 3, 1, 2))))

    #         # Collect the landmarks for each frame in the batch
    #         for i, landmarks in enumerate(batch_landmarks):
    #             if landmarks is not None:
    #                 all_landmarks.append(np.array(landmarks[0]))  # Only use the first face detected
    #         print(f"Processed batch {frame_idx // batch_size + 1}")

    #         # Reset the batch
    #         batch_frames = []

    #     frame_idx += 1

    #  # Process any remaining frames that didn't complete the batch
    # if len(batch_frames) > 0:
    #     batch_frames_np = np.array(batch_frames)

    #     # Detect landmarks for the batch
    #     batch_landmarks = fa.get_landmarks_from_batch(torch.Tensor(batch_frames_np.transpose((0, 3, 1, 2))))
    #     for i, landmarks in enumerate(batch_landmarks):
    #         if landmarks is not None:
    #             all_landmarks.append(np.array(landmarks[0]))  # Only use the first face detected
    #     print(f"Processed last batch with {len(batch_frames)} frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB (Face Alignment requires RGB images)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        faces = fa.detect_faces(rgb_frame)

        if faces:
            # Get the bounding box coordinates of the first detected face
            x, y, w, h = faces[0]['box']
            
            all_landmarks.append(np.array([x, y, w, h]))
        else:
            print("No faces detected.")
            all_landmarks.append(all_landmarks[-1])
        # print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()

    # Convert list to NumPy array and save to .npz
    if all_landmarks:
        all_landmarks = np.array(all_landmarks)  # Shape: (num_frames, num_landmarks, 2)
        np.savez(output_file, landmarks=all_landmarks)
        print(f"Saved landmarks for {frame_idx} frames to {output_file}")
    else:
        print("No landmarks detected in the video.")
    return output_file
    # print("Landmark extraction complete.")

detector = MTCNN()

# Iterate through the base directory
for root, dirs, files in os.walk(base_dir):
    # Initialize Face Alignment library
    # fa.get_landmarks_from_batch(torch.zeros((10, 3, 1920, 1080)))
    for dir_name in dirs:
        if dir_name == "front":  # Check for 'front' directory
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
                            landmark_path = video_path.replace('/video/', '/bbox/')
                            if video_file.endswith((".mp4", ".avi", ".mov")):  # Filter video files
                                print(f"Found video: {video_path}")
                                bbox_extract(detector, video_path, landmark_path)
                                # print(f'Save to: {landmark_path}')
                                # exit()