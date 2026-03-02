# -*- coding: utf-8 -*-
import subprocess
import os
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(script_dir, "sounds")
    os.makedirs(sounds_dir, exist_ok=True)
    
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    # 1. Download video
    video_file = os.path.join(sounds_dir, "video.webm")
    if not os.path.exists(video_file):
        print("[1] 비디오 다운로드 중...")
        subprocess.run([
            sys.executable, "-m", "yt_dlp",
            "-f", "bestvideo",
            "-o", video_file,
            "https://youtu.be/G07w0mG2Xe8"
        ])
    
    # 2. Extract specific frame (00:00:05.400 - 여우 얼굴이 선명할 때)
    frame_file = os.path.join(sounds_dir, "fox_frame.png")
    print("[2] 비디오 프레임 추출 중...")
    subprocess.run([
        ffmpeg_path, "-y", "-ss", "00:00:05.400", "-i", video_file,
        "-vframes", "1", "-q:v", "2", frame_file
    ])
    
    # 3. Remove background using rembg
    fox_head_file = os.path.join(sounds_dir, "fox_head.png")
    print("[3] 배경 제거 중 (rembg)...")
    try:
        from rembg import remove
        from PIL import Image
        
        if os.path.exists(frame_file):
            input_image = Image.open(frame_file)
            output_image = remove(input_image)
            output_image.save(fox_head_file)
            print("[3] 여우 머리 이미지 생성 완료:", fox_head_file)
        else:
            print("[ERROR] 프레임 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"[ERROR] rembg 실행 실패: {e}")
    
    # 4. Extract Audio with EXACT timings
    wav_file = os.path.join(sounds_dir, "full_audio.wav")
    kon_file = os.path.join(sounds_dir, "kon.wav")
    fox_file = os.path.join(sounds_dir, "fox_appear.wav")
    
    print("[4] 정확한 오디오 구간 재추출 중...")
    # 컹 (0:03.5 ~ 0:04.1) : 약 0.6초
    subprocess.run([
        ffmpeg_path, "-y", "-i", wav_file,
        "-ss", "00:00:03.500", "-to", "00:00:04.100",
        "-ar", "44100", "-ac", "1", kon_file
    ])
    
    # 으르렁 (0:04.4 ~ 0:08.5) : 약 4초
    subprocess.run([
        ffmpeg_path, "-y", "-i", wav_file,
        "-ss", "00:00:04.400", "-to", "00:00:08.500",
        "-ar", "44100", "-ac", "1", fox_file
    ])
    
    print("[DONE] 에셋 추출 완료!")

if __name__ == "__main__":
    main()
