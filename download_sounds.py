# -*- coding: utf-8 -*-
"""
YouTube에서 여우의 악마 사운드 이펙트를 추출하는 스크립트
"""
import subprocess
import os
import sys

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sounds_dir = os.path.join(script_dir, "sounds")
    os.makedirs(sounds_dir, exist_ok=True)
    
    # ffmpeg 경로 가져오기
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[INFO] ffmpeg 경로: {ffmpeg_path}")
    except ImportError:
        print("[ERROR] imageio-ffmpeg가 설치되어 있지 않습니다.")
        print("  pip install imageio-ffmpeg")
        sys.exit(1)
    
    # ffprobe 경로 (같은 폴더에 있음)
    ffmpeg_dir = os.path.dirname(ffmpeg_path)
    
    webm_file = os.path.join(sounds_dir, "full_audio.webm")
    wav_file = os.path.join(sounds_dir, "full_audio.wav")
    
    # Step 1: YouTube에서 다운로드 (이미 완료되었으면 스킵)
    if not os.path.exists(webm_file):
        print("[STEP 1] YouTube에서 오디오 다운로드 중...")
        result = subprocess.run([
            sys.executable, "-m", "yt_dlp",
            "-x", "--audio-format", "webm",
            "-o", webm_file,
            "https://youtu.be/G07w0mG2Xe8"
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
    else:
        print("[STEP 1] 오디오 파일이 이미 존재합니다. 스킵.")
    
    # Step 2: webm → wav 변환
    if not os.path.exists(wav_file):
        print("[STEP 2] webm → wav 변환 중...")
        result = subprocess.run([
            ffmpeg_path, "-i", webm_file,
            "-ar", "44100", "-ac", "1",
            wav_file, "-y"
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] 변환 실패: {result.stderr}")
            sys.exit(1)
        print("[STEP 2] 변환 완료!")
    else:
        print("[STEP 2] WAV 파일이 이미 존재합니다. 스킵.")
    
    # Step 3: 구간 추출
    # - "컹!" 소리: 대략 3~5초 부근 (짧은 외침)
    # - 여우 등장: 대략 4~8초 부근 (으르렁 + 등장 효과음)
    # 정확한 타이밍은 영상마다 다르므로, 넉넉하게 잘라서 사용
    
    kon_file = os.path.join(sounds_dir, "kon.wav")
    fox_file = os.path.join(sounds_dir, "fox_appear.wav")
    
    # "컹" 소리 추출 (2초~4초 구간, 짧은 외침)
    print("[STEP 3-1] '컹!' 사운드 추출 중 (2s ~ 4s)...")
    result = subprocess.run([
        ffmpeg_path, "-i", wav_file,
        "-ss", "2", "-to", "4",
        "-ar", "44100", "-ac", "1",
        kon_file, "-y"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[STEP 3-1] 완료: {kon_file}")
    else:
        print(f"[ERROR] {result.stderr}")
    
    # 여우 등장 효과음 추출 (3초~8초 구간)
    print("[STEP 3-2] 여우 등장 사운드 추출 중 (3s ~ 8s)...")
    result = subprocess.run([
        ffmpeg_path, "-i", wav_file,
        "-ss", "3", "-to", "8",
        "-ar", "44100", "-ac", "1",
        fox_file, "-y"
    ], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[STEP 3-2] 완료: {fox_file}")
    else:
        print(f"[ERROR] {result.stderr}")
    
    print()
    print("=" * 50)
    print("사운드 추출 완료!")
    print(f"  - 컹: {kon_file}")
    print(f"  - 여우 등장: {fox_file}")
    print("=" * 50)

if __name__ == "__main__":
    main()
