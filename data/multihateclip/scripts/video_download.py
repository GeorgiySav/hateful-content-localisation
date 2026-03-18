import requests
import yt_dlp
import os
import csv
# pip install yt-dlp requests

'''
Settings
'''
base_dir = "data/multihateclip"
video_path = os.path.join(base_dir, "dataset", "videos")
# Adjust this path to point exactly to where your ffmpeg.exe is located
ffmpeg_path = os.path.join(base_dir, "../ffmpeg.exe")

bilibili_headers = {
    'User-Agent': "your_device's_user_agent",
    'Referer': 'https://www.bilibili.com/'
}
bilibili_cookies = {'your_cookie_name': 'your_cookie_value'}

# Bilibili video download func
def request_cid(bid):
  http = "https://api.bilibili.com/x/player/pagelist?"
  response = requests.get(http + "bvid=" + bid, headers=bilibili_headers, cookies=bilibili_cookies)
  data = response.json()
  if(data["code"] == 0 ):
    return data["data"][0]["cid"]
  return "error"

def request_url(bid):
  cid = request_cid(bid)
  if(cid != "error"):
    http ="https://api.bilibili.com/x/player/playurl?"
    response = requests.get(http + "bvid=" +bid+ "&cid="+str(cid)+"&qn=32", headers=bilibili_headers, cookies=bilibili_cookies)
    data = response.json()
    if(data["code"] == 0 ):
      return data["data"]["durl"][0]["url"]
  return "error"

def download_bilibili_video(url, file_path):
    response = requests.get(url, stream=True, headers=bilibili_headers, cookies=bilibili_cookies)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download video. Status code: {response.status_code}")

def request_video(bid, path):
  url = request_url(bid)
  if(url != "error"):
    download_bilibili_video(url, path)


def download_with_ytdlp(url: str, video_id: str, output_folder: str):
    """
    Downloads a YouTube video as MP4.
    """
    mp4_path = os.path.join(output_folder, f"{video_id}.mp4")

    if os.path.exists(mp4_path):
        print("  - File already exists. Skipping.")
        return

    ydl_opts = {
        'format': 'bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]',
        #'ffmpeg_location': ffmpeg_path,
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(output_folder, f'{video_id}.%(ext)s'),
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"  - Error processing {video_id}: {e}")


def download_video_func(video_ids, platform, data_folder):
  os.makedirs(data_folder, exist_ok=True)
  total = len(video_ids)
  for index, video_id in enumerate(video_ids):
      print(f"[{index + 1}/{total}] Processing {video_id}...", end=" ")

      if platform == "Bilibili":
        path = os.path.join(data_folder, f"{video_id}.mp4")
        if os.path.exists(path):
          print("already exists, skipping.")
          continue
        request_video(video_id, path)
        print("Done")

      elif platform == "YouTube":
        url = f"https://www.youtube.com/watch?v={video_id}"
        download_with_ytdlp(url, video_id, data_folder)
        print("Done")


if __name__ == "__main__":
  # load video ids from tsv (skip header row, deduplicate)
  seen = set()
  video_ids = []
  tsv_files = [
    "data/multihateclip/dataset/annotation/train.tsv",
    "data/multihateclip/dataset/annotation/valid.tsv",
    "data/multihateclip/dataset/annotation/test.tsv",
  ]
  for tsv_file in tsv_files:
    with open(tsv_file, 'r') as f:
      reader = csv.reader(f, delimiter='\t')
      next(reader)  # skip header
      for row in reader:
        vid = row[0]
        if vid not in seen:
          seen.add(vid)
          video_ids.append(vid)

  print(f"Total unique videos to download: {len(video_ids)}")

  platform = "YouTube"  # or "Bilibili"
  data_folder = "data/multihateclip/dataset/videos"
  download_video_func(video_ids, platform, data_folder)
