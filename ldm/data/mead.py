import os
import json
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import cv2
import random
from PIL import Image

def kfold_subjects(path: str):

    from sklearn.model_selection import KFold
    subjects = os.listdir(path)
    subjects_np =  np.array(subjects)

    # Initialize KFold with 5 splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    folds_train = []
    folds_test = []
    # Open the file to write
    fold_number = 0
    for train_index, test_index in kf.split(subjects):
        train_subjects = subjects_np[train_index]
        test_subjects = subjects_np[test_index]
        folds_train.append(train_subjects)
        folds_test.append(test_subjects)
        fold_number += 1

    return folds_train, folds_test, fold_number

def index_dataset(path: str, output_file='mead_split.json'):
    emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    orientations = ['front']
    subjects = os.listdir(path)

    subject_map = {}
    
    x = []

    train_split, test_split, n_fold = kfold_subjects(path)
    for subject in subjects:
        subject_obj = []
        full_subject_path = os.path.join(path, subject)

        for emotion in emotions:
            for orientation in orientations:
                level_path = os.path.join(full_subject_path, 'landmark', orientation, emotion)
                try:
                    levels = os.listdir(level_path)
                except:
                    continue
                for level in levels:
                        
                    landmark_path = os.path.join(full_subject_path, 'landmark', orientation, emotion, level)
                    video_path = os.path.join(full_subject_path, 'video', orientation, emotion, level)
                    audio_path = os.path.join(full_subject_path, 'audio', emotion, level)

                    landmarks = os.listdir(landmark_path)
                    videos = os.listdir(video_path)
                    audios = os.listdir(audio_path)

                    for landmark in landmarks:
                        base_name = os.path.splitext(landmark)[0]
                        obj = {
                            # 'landmark': os.path.join(landmark_path, landmark),
                            # 'video': os.path.join(f'{base_name}.mp4'),
                            # 'audio': os.path.join(f'{base_name}.m4a'),
                            'landmark': landmark,
                            'video': f'{base_name}.mp4',
                            'audio': f'{base_name}.m4a',
                            'subject': subject,
                            'level': level,
                            'emotion': emotion,
                            'orientation': orientation
                        }
                        subject_obj.append(obj)

                    if not ( len(landmarks) == len(videos) and len(landmarks) == len(audios) and len(videos) == len(audios)):
                        print(f'WARNING: Checking failed, please make sure you have all files in subjects {subject}/{orientation}/{emotion}/{level} {len(audios)}, {len(videos)}, {len(landmarks)}')

        subject_map[subject] = subject_obj

    
    for i in range(n_fold):
        trains, tests = train_split[i], test_split[i]
        trains_out = []
        tests_out = []
        for t in trains:
            for obj in subject_map[t]:

                trains_out.append(obj)
        
        for t in tests:
            for obj in subject_map[t]:
                tests_out.append(obj)
        
        x.append({'train': trains_out, 'test': tests_out})
    
    
    with open(output_file, "w") as file:
        json.dump(x, file, indent=4)



class MeadDataset(Dataset):
    
    def __init__(self,json_path, base_path, n_fold, training = False):
        super().__init__()

        with open(json_path, 'r') as file:
            data = json.load(file)

        self.data = data[n_fold]['train' if training else 'test']
        self.base_path = base_path
        self.training = training
        self.transform = self.get_image_transform()

    def __len__(self):
        return len(self.data)
    
    def get_image_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            transforms.RandomRotation(30),           # Random rotation by a maximum of 30 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
            transforms.Resize((512, 512)),
            transforms.ToTensor(),                   # Convert image to Tensor
        ])
    def landmarks_to_bbox(self, landmarks, padding=0):
        """
        Convert landmarks to a bounding box.
        
        Parameters:
            landmarks (numpy.ndarray): Array of shape (N, 2) where N is the number of landmarks.
            padding (int): Additional padding around the bounding box (default: 0).
        
        Returns:
            tuple: (x_min, y_min, x_max, y_max) defining the bounding box.
        """
        # Get minimum and maximum coordinates
        x_min = np.min(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        x_max = np.max(landmarks[:, 0])
        y_max = np.max(landmarks[:, 1])
        
        # Add padding
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max += padding
        y_max += padding
        
        return int(x_min), int(y_min), int(x_max), int(y_max)
        
    def video_sample(self, video_path, landmark_path):
        cap = cv2.VideoCapture(video_path)

        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Randomly select a frame index
        random_frame_index = random.randint(0, total_frames - 1)
        # Set the video to the selected frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)

        # Read the frame
        ret, frame = cap.read()
        if ret:
            # Show the frame (optional)
            x_min = (width - height)//2
            frame = frame[0:height, x_min:x_min+height]
            # cv2.rectangle(frame, (x_min, 0), (height + x_min, height), (255, 0, 0), 2)  # Blue rectangle
        else: 
            frame = np.zeros(height, height)
        # Release the video capture
        cap.release()
        return Image.fromarray(frame.astype('uint8'))

    def __getitem__(self, i):
        x = self.data[i]
        video_path = os.path.join(self.base_path, x['subject'], 'video', x['orientation'], x['emotion'], x['level'], x['video'])
        landmark_path = os.path.join(self.base_path, x['subject'], 'landmark', x['orientation'], x['emotion'], x['level'], x['landmark'])
        frame = self.video_sample(video_path, landmark_path)

        return {"image": self.transform(frame)}

# class BratsValDataset(BratsDataset):
    
#     def __init__(self,data_path, pad_size = [160, 160, 126], crop_size = [160, 160, 126], resize = None):
#         self.data = get_brats_dataset(data_path, pad_size, crop_size, resize, is_val=True)


# if __name__ == "__main__":
    # index_dataset('/home/k8suser/hdd1/MEAD_EXTRACTED')
    # ds = MeadDataset('./data/utils/mead_split.json', '/home/k8suser/hdd1/MEAD_EXTRACTED', 1, )
    # print(ds.__getitem__(1))