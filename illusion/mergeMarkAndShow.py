import cv2
import numpy as np

def composite_hidden_object(
    background_path: str,
    object_path: str,
    bg_blur_ksize: int = 5,        
    obj_dark_offset: int = -20,    
    alpha_blur_ksize: int = 31,    
    color_blend_ksize: int = 31,   
    alpha_value: float = 0.15
):
    """
    1) 배경, 오브젝트 불러오기
    2) 배경에 가우시안 블러(소량) -> 눈 피로도 감소
    3) 오브젝트 색 = 배경 평균색 (알파 채널 제외)
    4) 오브젝트 색 조금 더 어둡게
    5) 알파·색 블러(페더링) -> 테두리로 갈수록 alpha=0, RGB는 배경색에 수렴
    6) 최종 투명도 alpha_value
    7) 합성
    ---
    추가 지표:
    (A) 배경 vs. 오브젝트 대비값(contrast_value)
    (B) '75% 확대' 가정한 서브윈도우들의 색 분포(표준편차) -> 최대/평균/검사 횟수
    """

    # ---------------------------
    # 1) 배경, 오브젝트 로드
    # ---------------------------
    bg = cv2.imread(background_path)               # BGR
    obj = cv2.imread(object_path + ".png", cv2.IMREAD_UNCHANGED)  # BGRA 가능

    if bg is None:
        print("배경 이미지를 불러올 수 없습니다.")
        return
    if obj is None:
        print("오브젝트 이미지를 불러올 수 없습니다.")
        return

    h_bg, w_bg = bg.shape[:2]

    # 오브젝트 리사이즈 (배경 크기에 맞춤)
    obj = cv2.resize(obj, (w_bg, h_bg), interpolation=cv2.INTER_AREA)

    # ---------------------------
    # 2) 배경 소량 블러
    # ---------------------------
    if bg_blur_ksize % 2 == 0:
        bg_blur_ksize += 1
    bg_blurred = cv2.GaussianBlur(bg, (bg_blur_ksize, bg_blur_ksize), 0)

    # ---------------------------
    # 3) 오브젝트 색 = 배경 평균색
    # ---------------------------
    bg_mean_color = cv2.mean(bg_blurred)[:3]  # (B, G, R)
    if obj.shape[2] == 4:
        obj_bgr = obj[:, :, :3]
        obj_alpha = obj[:, :, 3]
    else:
        obj_bgr = obj
        obj_alpha = np.ones((h_bg, w_bg), dtype=np.uint8) * 255

    # 오브젝트 RGB를 배경 평균색으로
    obj_bgr[:] = np.array(bg_mean_color, dtype=np.uint8)

    # ---------------------------
    # 4) 오브젝트 색 더 어둡게
    # ---------------------------
    if obj_dark_offset != 0:
        tmp = obj_bgr.astype(np.int16) + obj_dark_offset
        tmp = np.clip(tmp, 0, 255)
        obj_bgr = tmp.astype(np.uint8)

    # ---------------------------
    # 5) 테두리 페더링
    # ---------------------------
    # (5-1) 알파 채널 블러 -> 테두리 alpha=0
    if alpha_blur_ksize % 2 == 0:
        alpha_blur_ksize += 1
    blurred_alpha = cv2.GaussianBlur(obj_alpha, (alpha_blur_ksize, alpha_blur_ksize), 0)

    # (5-2) RGB 채널 페더링 -> 테두리 배경색에 수렴
    if color_blend_ksize % 2 == 0:
        color_blend_ksize += 1
    center_mask = obj_alpha.copy()  # 0~255
    blurred_mask = cv2.GaussianBlur(center_mask, (color_blend_ksize, color_blend_ksize), 0)
    blurred_mask_f = blurred_mask.astype(np.float32) / 255.0

    dark_f = obj_bgr.astype(np.float32)
    base_f = np.full_like(dark_f, bg_mean_color, dtype=np.float32)  # 배경 평균색
    mask_3ch = np.dstack([blurred_mask_f, blurred_mask_f, blurred_mask_f])

    blended_rgb_f = dark_f * mask_3ch + base_f * (1.0 - mask_3ch)
    blended_rgb = np.clip(blended_rgb_f, 0, 255).astype(np.uint8)

    obj_bgr = blended_rgb
    obj_alpha = blurred_alpha

    # ---------------------------
    # 6) 전체 투명도 적용
    # ---------------------------
    alpha_float = obj_alpha.astype(np.float32) / 255.0
    alpha_float *= alpha_value
    alpha_float = np.clip(alpha_float, 0.0, 1.0)

    # ---------------------------
    # 7) 합성
    # ---------------------------
    bg_f = bg_blurred.astype(np.float32)
    obj_f = obj_bgr.astype(np.float32)
    alpha_3ch = np.dstack([alpha_float, alpha_float, alpha_float])

    composite_f = bg_f * (1.0 - alpha_3ch) + obj_f * alpha_3ch
    composite = np.clip(composite_f, 0, 255).astype(np.uint8)

    # ===========================
    # (A) 배경 vs. 오브젝트 대비 측정
    # ===========================
    # 방법: 최종 composite를 그레이스케일로 변환
    #       obj_mask = alpha_float > 임계값(예: 0.05)
    #       배경은 alpha_float < 0.05
    #       각 영역 평균 밝기 차이
    gray_comp = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    obj_mask = (alpha_float > 0.05)  # True/False
    bg_mask = ~obj_mask             # 반대

    if np.any(obj_mask) and np.any(bg_mask):
        mean_obj = np.mean(gray_comp[obj_mask])
        mean_bg = np.mean(gray_comp[bg_mask])
        contrast_value = abs(mean_obj - mean_bg)
    else:
        # 오브젝트가 전부 투명 or 전부 불투명일 경우 등
        contrast_value = 0.0

    # ===========================
    # (B) '75% 확대' 시 색 분포 측정
    # ===========================
    #  - 실제로는 '서브 윈도우(가로/세로 25% 크기)'를
    #    중심부(20%~80%) 영역 내에서 반씩 이동하며 스캔
    #  - 각 윈도우 내 그레이스케일 표준편차를 구함
    #  - 최대 표준편차, 평균 표준편차, 검사 횟수
    h, w = composite.shape[:2]
    # 중심부 범위
    x_start, x_end = int(0.2 * w), int(0.8 * w)
    y_start, y_end = int(0.2 * h), int(0.8 * h)

    # 서브 윈도우 크기(25%)
    sub_w = int(w * 0.25)
    sub_h = int(h * 0.25)

    # 그레이스케일로 검사
    gray_img = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)

    # 스캔할 좌표들
    # 가로 방향: x_start -> x_end-sub_w, step = sub_w//2
    # 세로 방향: y_start -> y_end-sub_h, step = sub_h//2
    max_std = 0.0
    sum_std = 0.0
    count = 0

    # 범위 넘어가지 않도록 최대한 맞춰줌
    for y_top in range(y_start, y_end - sub_h + 1, sub_h // 2 if sub_h>1 else 1):
        for x_left in range(x_start, x_end - sub_w + 1, sub_w // 2 if sub_w>1 else 1):
            roi = gray_img[y_top:y_top+sub_h, x_left:x_left+sub_w]
            # 표준편차(색 분포)
            std_val = float(np.std(roi))
            max_std = max(max_std, std_val)
            sum_std += std_val
            count += 1

    if count > 0:
        avg_std = sum_std / count
    else:
        avg_std = 0.0

    # ---------------------------
    # 결과 저장
    # ---------------------------
    output_path = f"{object_path}_{contrast_value:.3f}_{count}_{max_std:.3f}_{avg_std:.3f}.png"
    
    # 문제 문자를 치환
    for ch in [' ', ':', '\\', '/', '?', '*', '<', '>', '"']:
        output_path = output_path.replace(ch, '_')

    # 저장 시도
    save_success = cv2.imwrite(output_path, composite)
    print(f"cv2.imwrite 성공 여부: {save_success}")
    if save_success:
        print(f"[완료] 결과 저장: {output_path}")
    else:
        print("!!! 저장 실패: 경로/파일명 문제 또는 권한 문제일 수 있음 !!!")

    print("==== 요약 정보 ====")
    print(f"(A) 배경 vs. 오브젝트 대비값: {contrast_value:.3f}")
    print(f"(B) 서브윈도우 검사 횟수: {count}")
    print(f"    최대 표준편차 (max_std): {max_std:.3f}")
    print(f"    평균 표준편차 (avg_std): {avg_std:.3f}")


if __name__ == "__main__":
    composite_hidden_object(
        background_path="background.png",
        object_path="Airplane",
        bg_blur_ksize=5,
        obj_dark_offset=-25,
        alpha_blur_ksize=31,
        color_blend_ksize=31,
        alpha_value=0.2
    )
