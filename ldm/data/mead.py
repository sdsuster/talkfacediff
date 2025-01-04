import os

# def preflight(path, subjects):
    
        

def index_dataset(path: str):
    emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    orientations = ['front']
    levels = ['level_1', 'level_2', 'level_3']
    subjects = os.listdir(path)

    for subject in subjects:
        full_subject_path = os.path.join(path, subject)

        for emotion in emotions:
            for orientation in orientations:
                for level in levels:
                        
                    landmark_path = os.path.join(full_subject_path, 'landmark', 'front', emotion, level)
                    video_path = os.path.join(full_subject_path, 'video', 'front', emotion, level)
                    audio_path = os.path.join(full_subject_path, 'audio', emotion, level)

                    landmarks = os.listdir(landmark_path)
                    videos = os.listdir(video_path)
                    audios = os.listdir(audio_path)

                    assert len(landmarks) == len(videos) and len(landmarks) == len(audios) and len(videos) == len(audios), f'Checking failed, please make sure you have all files in subjects {subject}'
 

print()