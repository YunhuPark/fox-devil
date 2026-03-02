# -*- coding: utf-8 -*-
"""
=============================================================================
여우의 악마(컹) 핸드 제스처 인식 시스템 v2.0
- 체인소맨(Chainsaw Man)의 '여우의 악마' 손모양을 실시간으로 인식
- MediaPipe Tasks API (HandLandmarker + FaceDetector) + OpenCV 기반
- 애니메이션 수준의 시각 이펙트 + 사운드 재생
=============================================================================

[사용법]
1. pip install opencv-python mediapipe numpy Pillow pygame imageio-ffmpeg
2. hand_landmarker.task, blaze_face_short_range.tflite 모델 파일 배치
3. sounds/ 폴더에 kon.wav, fox_appear.wav 배치 (download_sounds.py 실행)
4. python fox_devil_gesture.py
5. 'q' 키로 종료
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import os
import sys
import time
import random
import threading

# =============================================================================
# ★ 임계값 설정 (Threshold Configuration)
# =============================================================================

THUMB_TOUCH_THRESHOLD = 0.08
FINGER_EXTENDED_THRESHOLD = 0.02
FINGER_BENT_THRESHOLD = 0.01
GESTURE_STABILITY_FRAMES = 3
EFFECT_DURATION = 3.0        # 전체 이펙트 지속 시간 (초)
EFFECT_FONT_SCALE = 1.2
EFFECT_FONT_THICKNESS = 3

# 이펙트 타이밍 (초)
FOX_APPEAR_DELAY = 1.0        # 사운드 재생 후 여우 등장 지연 시간
FLASH_DURATION = 0.2          # 스크린 플래시 지속 시간
SHAKE_DURATION = 0.6          # 화면 흔들림 지속 시간 (좀 더 강하게)
FOX_SCALE_DURATION = 0.25     # 여우 머리가 화면을 덮치는 시간 (매우 짧고 빠르게!)
FOX_HOLD_DURATION = 1.2       # 화면에 꽉 찬 상태 (피바다/암전 연출)
FOX_FADE_DURATION = 1.0       # 여우 머리 페이드아웃 시간
PARTICLE_COUNT = 150          # 파티클 개수 증가 (와일드한 느낌)

# =============================================================================
# 랜드마크 인덱스
# =============================================================================
WRIST = 0
THUMB_TIP = 4
INDEX_PIP = 6; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_PIP = 14; RING_TIP = 16
PINKY_PIP = 18; PINKY_TIP = 20


# =============================================================================
# 제스처 판별 함수
# =============================================================================

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def is_finger_extended(landmarks, tip_idx, pip_idx):
    return (landmarks[pip_idx].y - landmarks[tip_idx].y) > FINGER_EXTENDED_THRESHOLD

def is_finger_bent(landmarks, tip_idx, pip_idx):
    return (landmarks[tip_idx].y - landmarks[pip_idx].y) > FINGER_BENT_THRESHOLD

def is_thumb_touching_fingers(landmarks):
    thumb_tip = landmarks[THUMB_TIP]
    return (distance(thumb_tip, landmarks[MIDDLE_TIP]) < THUMB_TOUCH_THRESHOLD or
            distance(thumb_tip, landmarks[RING_TIP]) < THUMB_TOUCH_THRESHOLD)

def is_fox_gesture(landmarks):
    return (is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP) and
            is_finger_extended(landmarks, PINKY_TIP, PINKY_PIP) and
            is_finger_bent(landmarks, MIDDLE_TIP, MIDDLE_PIP) and
            is_finger_bent(landmarks, RING_TIP, RING_PIP) and
            is_thumb_touching_fingers(landmarks))


# =============================================================================
# 이미지 오버레이 (여우 머리)
# =============================================================================

def overlay_transparent_image(bg_img, img_to_overlay_t, x, y, overlay_size=None, alpha_mult=1.0):
    """
    배경 이미지 위에 투명도를 가진 PNG 이미지를 오버레이합니다.
    (x, y)는 이미지의 중심점 좌표입니다.
    """
    try:
        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t, overlay_size)
        
        # 오버레이 이미지 추출 (BGR, Alpha)
        bgr = img_to_overlay_t[:,:,:3]
        alpha_mask = (img_to_overlay_t[:,:,3] / 255.0) * alpha_mult
        
        h, w = img_to_overlay_t.shape[:2]
        
        # 화면 경계를 벗어나는 영역 처리
        y1, y2 = max(0, y - h//2), min(bg_img.shape[0], y - h//2 + h)
        x1, x2 = max(0, x - w//2), min(bg_img.shape[1], x - w//2 + w)
        
        # 오버레이 영역 내에서의 인덱스
        y1o, y2o = max(0, -(y - h//2)), min(h, h - ((y - h//2 + h) - bg_img.shape[0]))
        x1o, x2o = max(0, -(x - w//2)), min(w, w - ((x - w//2 + w) - bg_img.shape[1]))
        
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return bg_img
        
        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        
        for c in range(3):
            bg_img[y1:y2, x1:x2, c] = (alpha * bgr[y1o:y2o, x1o:x2o, c] +
                                      (1 - alpha) * bg_img[y1:y2, x1:x2, c])
    except Exception as e:
        pass
    
    return bg_img


# =============================================================================
# 사운드 시스템
# =============================================================================

class SoundManager:
    """pygame 기반 사운드 재생 관리 (비동기)"""
    
    def __init__(self, sounds_dir):
        self.initialized = False
        self.user_sound = None
        
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.initialized = True
            
            # 사용자가 직접 넣을 사운드 파일 (이름을 user_sound.wav 등 원하는 대로 지정하세요)
            sound_path = os.path.join(sounds_dir, "user_sound.wav")
            
            if os.path.exists(sound_path):
                self.user_sound = pygame.mixer.Sound(sound_path)
                print(f"[SOUND] 사용자 커스텀 사운드 로드 완료: {sound_path}")
            else:
                print(f"[SOUND] {sound_path} 파일을 찾을 수 없습니다.")
                print("[SOUND] 원하는 효과음 파일을 'sounds/user_sound.wav' 이름으로 넣어주세요.")
                
        except Exception as e:
            print(f"[SOUND] pygame 초기화 실패: {e}")
    
    def play_sound(self):
        if self.initialized and self.user_sound:
            self.user_sound.play()
    
    def cleanup(self):
        if self.initialized:
            try:
                import pygame
                pygame.mixer.quit()
            except:
                pass


# =============================================================================
# 파티클 시스템
# =============================================================================

class Particle:
    """혈흔/파편 파티클"""
    def __init__(self, x, y):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 15)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - random.uniform(0, 5)
        self.size = random.randint(2, 8)
        self.life = random.uniform(0.5, 1.5)
        self.max_life = self.life
        # 혈흔 색상 변위 (짙은 빨강 ~ 검붉은색)
        self.color = (
            random.randint(0, 30),       # B
            random.randint(0, 30),       # G
            random.randint(120, 220)     # R
        )
        self.gravity = 0.3
    
    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.life -= dt
        return self.life > 0
    
    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = max(0, self.life / self.max_life)
        size = max(1, int(self.size * alpha))
        x, y = int(self.x), int(self.y)
        h, w = frame.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), size, self.color, -1, cv2.LINE_AA)


class ParticleSystem:
    """파티클 관리 시스템"""
    def __init__(self):
        self.particles = []
    
    def emit(self, x, y, count=PARTICLE_COUNT):
        for _ in range(count):
            self.particles.append(Particle(x, y))
    
    def update(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)]
    
    def draw(self, frame):
        for p in self.particles:
            p.draw(frame)


# 삭제됨: procedural draw_fox_head


# =============================================================================
# 스크린 이펙트
# =============================================================================

def screen_flash(frame, progress, color=(255, 255, 255)):
    """스크린 플래시 효과"""
    if progress <= 0:
        return frame
    overlay = frame.copy()
    overlay[:] = color
    alpha = progress * 0.7  # 최대 70% 불투명
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def screen_shake(frame, intensity=10):
    """화면 흔들림 효과"""
    h, w = frame.shape[:2]
    dx = random.randint(-intensity, intensity)
    dy = random.randint(-intensity, intensity)
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return frame


def draw_dark_vignette(frame, intensity=0.5):
    """어둡게 하는 비네팅 효과 (화면 가장자리 어둡게)"""
    h, w = frame.shape[:2]
    
    # 가우시안 기반 비네팅
    X = cv2.getGaussianKernel(w, w * 0.5)
    Y = cv2.getGaussianKernel(h, h * 0.5)
    M = Y * X.T
    M = M / M.max()
    
    mask = np.stack([M] * 3, axis=-1)
    # 비네팅 적용 (intensity가 클수록 어두움)
    darkened = (frame * (mask * (1 - intensity) + intensity)).astype(np.uint8)
    return darkened


# =============================================================================
# 디버그 UI
# =============================================================================

def draw_debug_info(frame, is_fox, finger_states):
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    status_color = (0, 255, 0) if is_fox else (0, 0, 255)
    status_text = "FOX GESTURE: DETECTED!" if is_fox else "FOX GESTURE: None"
    cv2.putText(frame, status_text, (10, y_offset), font, 0.5, status_color, 1, cv2.LINE_AA)
    
    y_offset += 25
    for name, state in finger_states.items():
        color = (0, 255, 0) if state else (100, 100, 100)
        cv2.putText(frame, f"  {name}: [{'O' if state else 'X'}]", (10, y_offset),
                   font, 0.5, color, 1, cv2.LINE_AA)
        y_offset += 20
    return frame


def draw_hand_landmarks(frame, landmarks, w, h):
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
    ]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in connections:
        cv2.line(frame, points[s], points[e], (0, 255, 0), 2, cv2.LINE_AA)
    for i, pt in enumerate(points):
        color = (0, 0, 255) if i in [4, 8, 12, 16, 20] else (255, 0, 0)
        cv2.circle(frame, pt, 5 if i in [4,8,12,16,20] else 3, color, -1, cv2.LINE_AA)
    return frame


# 삭제됨: draw_text_effect


# =============================================================================
# 이펙트 시퀀스 컨트롤러
# =============================================================================

class EffectController:
    """
    제스처 인식 → 사운드 + 시각 이펙트 시퀀스를 관리합니다.
    """
    
    def __init__(self, sound_manager, fox_img=None, fox_frames=None):
        self.sound = sound_manager
        self.fox_img = fox_img
        self.fox_frames = fox_frames
        self.particles = ParticleSystem()
        self.active = False
        self.start_time = 0
        self.face_cx = 0
        self.face_cy = 0
        self.sound_played = False
        self.particles_emitted = False
        
        # 여우 이미지/영상 원본 비율 계산
        self.fox_aspect = 1.0
        if self.fox_frames and len(self.fox_frames) > 0:
            self.fox_aspect = self.fox_frames[0].shape[1] / self.fox_frames[0].shape[0]
        elif self.fox_img is not None:
            self.fox_aspect = self.fox_img.shape[1] / self.fox_img.shape[0]
            
    # [이하 코드는 유지되거나 다음 청크에서 수정됨]
    
    def trigger(self, face_cx, face_cy):
        """이펙트 발동"""
        if self.active:
            return
        self.active = True
        self.start_time = time.time()
        self.face_cx = face_cx
        self.face_cy = face_cy
        self.sound_played = False
        self.particles_emitted = False
        
        print("[EFFECT] ★ 여우의 악마 소환! ★")
    
    def update_position(self, face_cx, face_cy):
        """얼굴 위치 업데이트 (이펙트 중에도 추적)"""
        if self.active:
            # 부드러운 추적 (보간)
            self.face_cx = int(self.face_cx * 0.7 + face_cx * 0.3)
            self.face_cy = int(self.face_cy * 0.7 + face_cy * 0.3)
    
    def render(self, frame, dt):
        """현재 프레임에 이펙트를 렌더링"""
        if not self.active:
            self.particles.update(dt)
            self.particles.draw(frame)
            return frame
        
        elapsed = time.time() - self.start_time
        total = FOX_APPEAR_DELAY + FOX_SCALE_DURATION + FOX_HOLD_DURATION + FOX_FADE_DURATION
        
        if elapsed > total:
            self.active = False
            return frame
        
        cx, cy = self.face_cx, self.face_cy
        
        # --- 사용자 커스텀 사운드 (발동 즉시 재생) ---
        if elapsed > 0.0 and not self.sound_played:
            self.sound.play_sound()
            self.sound_played = True
        
        # --- 파티클 폭발 지연 발동 ---
        if elapsed > FOX_APPEAR_DELAY and not getattr(self, 'particles_emitted', False):
            self.particles.emit(self.face_cx, self.face_cy, PARTICLE_COUNT)
            self.particles_emitted = True

        # --- 스크린 플래시 (파티클 터질 때 붉게 번쩍!) ---
        if FOX_APPEAR_DELAY < elapsed < FOX_APPEAR_DELAY + FLASH_DURATION:
            flash_progress = 1.0 - ((elapsed - FOX_APPEAR_DELAY) / FLASH_DURATION)
            # 피바다 연출을 위해 붉은색 플래시 (B, G, R)
            flash_color = (0, 0, 255)
            frame = screen_flash(frame, flash_progress, flash_color)
        
        # --- 화면 흔들림 (물어뜯는 반동!) ---
        if FOX_APPEAR_DELAY + 0.05 < elapsed < FOX_APPEAR_DELAY + SHAKE_DURATION:
            shake_progress = 1.0 - ((elapsed - (FOX_APPEAR_DELAY + 0.05)) / (SHAKE_DURATION - 0.05))
            intensity = int(30 * shake_progress) # 흔들림 강도 2배로
            if intensity > 0:
                frame = screen_shake(frame, intensity)
        
        # --- 비네팅 (어두운 분위기) ---
        vignette_start = FOX_APPEAR_DELAY + 0.1
        if elapsed > vignette_start:
            v_progress = min(1.0, (elapsed - vignette_start) / 0.3)
            # 페이드아웃 시 비네팅도 줄어듦
            fade_start = FOX_APPEAR_DELAY + FOX_SCALE_DURATION + FOX_HOLD_DURATION
            if elapsed > fade_start:
                v_progress *= max(0, 1.0 - (elapsed - fade_start) / FOX_FADE_DURATION)
            frame = draw_dark_vignette(frame, 0.3 * v_progress)
        
        # --- 여우 머리 (집어삼키는 3D 비디오/돌진 애니메이션) ---
        if self.fox_img is not None or (self.fox_frames and len(self.fox_frames) > 0):
            fox_start = FOX_APPEAR_DELAY
            if elapsed > fox_start:
                fox_elapsed = elapsed - fox_start
                h, w = frame.shape[:2]
                
                # 비디오 프레임이면 비디오 진행상태에 맞춤
                current_fox_img = None
                if self.fox_frames and len(self.fox_frames) > 0:
                    vid_duration = FOX_SCALE_DURATION + FOX_HOLD_DURATION + FOX_FADE_DURATION
                    frame_idx = int((fox_elapsed / vid_duration) * len(self.fox_frames))
                    if frame_idx >= len(self.fox_frames):
                        frame_idx = len(self.fox_frames) - 1
                    current_fox_img = self.fox_frames[frame_idx]
                else:
                    current_fox_img = self.fox_img

                target_w = w  # 화면 너비에 맞춤 (너무 확대되지 않도록)
                
                # 스케일 계산
                if fox_elapsed < FOX_SCALE_DURATION:
                    # 가속
                    t = fox_elapsed / FOX_SCALE_DURATION
                    scale = t ** 2  # 점프 속도 살짝 완화 (여우가 잘 보이게)
                    fox_w = max(20, int(target_w * scale))
                    fox_alpha = min(1.0, scale * 3 + 0.1)
                    
                elif fox_elapsed < FOX_SCALE_DURATION + FOX_HOLD_DURATION:
                    # 화면을 꽉 채운 상태 - 암전 이펙트
                    hold_t = fox_elapsed - FOX_SCALE_DURATION
                    hold_progress = hold_t / FOX_HOLD_DURATION
                    
                    fox_w = target_w
                    fox_alpha = 1.0
                    
                    # 피바다/암전 이펙트 레이어 추가 (완전히 먹힌 느낌)
                    overlay = frame.copy()
                    if hold_progress < 0.2:
                        overlay[:] = (0, 0, 150) # 다크 레드
                        alpha = hold_progress / 0.2
                    else:
                        overlay[:] = (0, 0, 0) # 완전 암흑
                        alpha = min(1.0, (hold_progress - 0.2) / 0.2)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                    
                else:
                    # 페이드아웃 및 원래 크기로 되돌아감
                    fade_t = fox_elapsed - FOX_SCALE_DURATION - FOX_HOLD_DURATION
                    fade_progress = min(1.0, fade_t / FOX_FADE_DURATION)
                    fox_w = int(target_w * (1.0 - fade_progress * 0.5))
                    fox_alpha = 1.0 - fade_progress
                
                if fox_w > 10 and fox_alpha > 0.01 and current_fox_img is not None:
                    fox_h = int(fox_w / self.fox_aspect)
                    
                    # 카메라 렌즈 중앙을 향해 돌진
                    offset_x = 0  
                    offset_y = 0  
                    
                    frame = overlay_transparent_image(frame, current_fox_img, cx + offset_x, cy + offset_y, (fox_w, fox_h), fox_alpha)
        
        # --- 파티클 업데이트 & 렌더링 ---
        self.particles.update(dt)
        self.particles.draw(frame)
        
        return frame


# =============================================================================
# 메인 실행 로직
# =============================================================================

def main():
    print("=" * 60)
    print("  Fox Devil v2.0 - Enhanced Edition")
    print("  Chainsaw Man - Fox Devil Gesture Recognition")
    print("=" * 60)
    print()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ----- 여우 영상/이미지 로드 -----
    sounds_dir = os.path.join(script_dir, "sounds")
    fox_img_path = os.path.join(sounds_dir, "fox_head.png")
    fox_vid_path = os.path.join(sounds_dir, "202603022227 (1).mp4")
    
    fox_img = None
    fox_frames = []
    
    # 1순위: 비디오가 있으면 비디오 로드
    if os.path.exists(fox_vid_path):
        print(f"[INIT] 3D 영상 에셋 로드 중... ({fox_vid_path})")
        cap_vid = cv2.VideoCapture(fox_vid_path)
        while cap_vid.isOpened():
            ret_v, frame_v = cap_vid.read()
            if not ret_v:
                break
            # RGBA 변환 (반투명 처리용)
            frame_rgba = cv2.cvtColor(frame_v, cv2.COLOR_BGR2BGRA)
            
            # 크로마키(초록색 배경) 또는 블랙 스크린 기초 제거
            # (만약 영상이 이미 투명한 배경을 의도했다면 알파 채널 편집)
            g_mask = (frame_rgba[:,:,1] > 180) & (frame_rgba[:,:,0] < 100) & (frame_rgba[:,:,2] < 100)
            frame_rgba[g_mask, 3] = 0  # 초록색 투명화
            
            fox_frames.append(frame_rgba)
        cap_vid.release()
        print(f"[INIT] 3D 영상 로드 완료: 총 {len(fox_frames)} 프레임")
    
    # 2순위: 비디오가 없으면 기본 이미지 로드
    elif os.path.exists(fox_img_path):
        img_array = np.fromfile(fox_img_path, np.uint8)
        fox_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        print("[INIT] 여우 머리 에셋(fox_head.png) 로드 완료")
    else:
        print("[WARNING] 영상 또는 이미지가 없습니다.")
    
    # ----- 사운드 초기화 -----
    sound_mgr = SoundManager(sounds_dir)
    
    # ----- HandLandmarker 초기화 -----
    print("[INIT] MediaPipe HandLandmarker 초기화 중...")
    hand_model = os.path.join(script_dir, "hand_landmarker.task")
    if not os.path.exists(hand_model):
        print(f"[ERROR] hand_landmarker.task 없음!")
        sys.exit(1)
    
    with open(hand_model, "rb") as f:
        hand_model_data = f.read()
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=hand_model_data),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)
    print("[INIT] HandLandmarker 완료!")
    
    # ----- FaceDetector 초기화 -----
    print("[INIT] MediaPipe FaceDetector 초기화 중...")
    face_model = os.path.join(script_dir, "blaze_face_short_range.tflite")
    face_detector = None
    
    if os.path.exists(face_model):
        try:
            with open(face_model, "rb") as f:
                face_model_data = f.read()
            
            FaceDetector = mp.tasks.vision.FaceDetector
            FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
            
            face_options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_buffer=face_model_data),
                running_mode=VisionRunningMode.VIDEO,
                min_detection_confidence=0.5
            )
            face_detector = FaceDetector.create_from_options(face_options)
            print("[INIT] FaceDetector 완료!")
        except Exception as e:
            print(f"[INIT] FaceDetector 초기화 실패: {e}")
            print("[INIT] 얼굴 추적 없이 손 위치 기반으로 동작합니다.")
    else:
        print("[INIT] blaze_face_short_range.tflite 없음 — 손 위치 기반 모드")
    
    # ----- 웹캠 초기화 -----
    print("[INIT] 웹캠 초기화 중...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다!")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[INIT] 웹캠 해상도: {int(cap.get(3))}x{int(cap.get(4))}")
    print()
    print("[READY] ★ 시스템 준비 완료! 여우 손모양을 보여주세요 ★")
    print("[INFO] 종료: 'q' 키")
    print()
    
    # ----- 상태 변수 -----
    gesture_counter = 0
    effect = EffectController(sound_mgr, fox_img, fox_frames)
    fps_time = time.time()
    frame_count = 0
    current_fps = 0
    frame_ts = 0
    last_time = time.time()
    
    face_cx, face_cy = 640, 360  # 기본 얼굴 위치
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            now = time.time()
            dt = now - last_time
            last_time = now
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33
            
            # --- 얼굴 감지 ---
            if face_detector:
                try:
                    face_result = face_detector.detect_for_video(mp_image, frame_ts)
                    if face_result.detections:
                        det = face_result.detections[0]
                        bbox = det.bounding_box
                        face_cx = int(bbox.origin_x + bbox.width / 2)
                        face_cy = int(bbox.origin_y + bbox.height / 2)
                        
                        # 얼굴 바운딩 박스 표시 (디버그, 이펙트 비활성 시만)
                        if not effect.active:
                            cv2.rectangle(frame,
                                (bbox.origin_x, bbox.origin_y),
                                (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                (255, 200, 0), 1)
                            cv2.putText(frame, "Face", (bbox.origin_x, bbox.origin_y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                except:
                    pass
            
            # --- 손 감지 ---
            fox_detected = False
            try:
                hand_result = hand_landmarker.detect_for_video(mp_image, frame_ts)
                
                if hand_result.hand_landmarks:
                    for hand_lms in hand_result.hand_landmarks:
                        frame = draw_hand_landmarks(frame, hand_lms, w, h)
                        
                        index_ext = is_finger_extended(hand_lms, INDEX_TIP, INDEX_PIP)
                        pinky_ext = is_finger_extended(hand_lms, PINKY_TIP, PINKY_PIP)
                        middle_bent = is_finger_bent(hand_lms, MIDDLE_TIP, MIDDLE_PIP)
                        ring_bent = is_finger_bent(hand_lms, RING_TIP, RING_PIP)
                        thumb_touch = is_thumb_touching_fingers(hand_lms)
                        
                        finger_states = {
                            "Index Extended": index_ext,
                            "Pinky Extended": pinky_ext,
                            "Middle Bent": middle_bent,
                            "Ring Bent": ring_bent,
                            "Thumb Touch": thumb_touch
                        }
                        
                        is_fox = is_fox_gesture(hand_lms)
                        frame = draw_debug_info(frame, is_fox, finger_states)
                        
                        if is_fox:
                            fox_detected = True
                            # 얼굴이 없으면 손 위치 사용
                            if not face_detector:
                                face_cx = int(hand_lms[MIDDLE_MCP].x * w)
                                face_cy = int(hand_lms[MIDDLE_MCP].y * h)
            except:
                pass
            
            # --- 제스처 안정화 & 이펙트 발동 ---
            if fox_detected:
                gesture_counter += 1
                if gesture_counter >= GESTURE_STABILITY_FRAMES and not effect.active:
                    effect.trigger(face_cx, face_cy)
            else:
                gesture_counter = 0
            
            # --- 이펙트 렌더링 ---
            if effect.active:
                effect.update_position(face_cx, face_cy)
            frame = effect.render(frame, dt)
            
            # --- FPS ---
            frame_count += 1
            if now - fps_time >= 1.0:
                current_fps = frame_count
                frame_count = 0
                fps_time = now
            
            cv2.putText(frame, f"FPS: {current_fps}", (w-120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press 'q' to quit", (w-180, h-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)
            
            cv2.imshow("Fox Devil v2.0", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[EXIT] 종료합니다.")
                break
    
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C")
    
    finally:
        print("[CLEANUP] 리소스 정리...")
        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()
        if face_detector:
            face_detector.close()
        sound_mgr.cleanup()
        print("[CLEANUP] 완료!")


if __name__ == "__main__":
    main()
