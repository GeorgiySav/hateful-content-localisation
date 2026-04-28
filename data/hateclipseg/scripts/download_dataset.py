import pandas as pd
import yt_dlp
import os

# Settings
base_dir = "./data/hateclipseg"
video_path = os.path.join(base_dir, "dataset", "videos")
csv_path = os.path.join(base_dir, "dataset", "segment_level_annotation.csv")
ffmpeg_path = os.path.join(base_dir, "../ffmpeg.exe") 

# Ensure the output directory exists
os.makedirs(video_path, exist_ok=True)

# Open annotations
try:
    df = pd.read_csv(csv_path)
    print("CSV loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find CSV file at {csv_path}")
    exit()

def download_with_ytdlp(url: str, video_id: str, output_folder: str, output_name: str):
    """
    Downloads video as MP4, then extracts audio as WAV.
    """
    
    # Check if BOTH files exist
    mp4_exists = os.path.exists(os.path.join(output_folder, f"{video_id}.mp4"))
    wav_exists = os.path.exists(os.path.join(output_folder, f"{video_id}.wav"))
    
    if mp4_exists and wav_exists:
        print("  - Files (mp4 & wav) already exist. Skipping.")
        return 

    ydl_opts = {
        # Looks for <= 1080p mp4 video + m4a audio, or falls back to best mp4
        'format': 'bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]',

        'ffmpeg_location': ffmpeg_path,  # Specify the path to ffmpeg

        # Ensures the final merged container is an MP4
        'merge_output_format': 'mp4', 
        
        # Saves the file using the video's title as the filename
        'outtmpl': f'{output_folder}/{output_name}.%(ext)s',
        
        'quiet': True 
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"  - Error processing {video_id}: {e}")

def download_video_from_youtube(youtube_id: str):
    url = f'https://www.youtube.com/watch?v={youtube_id}'
    download_with_ytdlp(url, youtube_id, video_path, "yt_" + youtube_id)

def download_video_from_bitchute(bitchute_id: str):
    url = f'https://www.bitchute.com/video/{bitchute_id}/'
    download_with_ytdlp(url, bitchute_id, video_path, "bit_" + bitchute_id)

if 'Video Id' in df.columns:
    unique_ids = df['Video Id'].unique()
    total_videos = len(unique_ids)
    
    print(f"Found {total_videos} unique videos to process.")

    for index, video_id in enumerate(unique_ids):
        print(f"[{index + 1}/{total_videos}] Processing {video_id}...", end=" ")
        
        try:
            if video_id.startswith('yt_'):
                download_video_from_youtube(video_id[3:])
                print("Done")

            elif video_id.startswith('bit_'):
                download_video_from_bitchute(video_id[4:])
                print("Done")

            else:
                print(f"Skipped (Unknown platform)")
                
        except Exception as e:
            print(f"\nFailed to download {video_id}: {e}")
else:
    print("Error: Column 'Video Id' not found in the CSV.")