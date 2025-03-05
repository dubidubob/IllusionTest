import cv2
import numpy as np

def composite_hidden_object(
    background_path: str,
    object_path: str,
    output_path: str = "result.png",
    bg_blur_ksize: int = 5,        # (2) 배경 가우시안 블러 정도
    obj_dark_offset: int = -20,    # (4) 오브젝트를 더 어둡게
    alpha_blur_ksize: int = 31,    # (5) 알파 채널 블러 (테두리로 갈수록 alpha→0)
    color_blend_ksize: int = 31,   # (5) 색 페더링용 마스크 블러
    alpha_value: float = 0.15      # (6) 전체 투명도
):
    """
    1. 배경화면, 상 불러오기
    2. 배경화면에 가우시안 블러를 조금 넣어 눈 피로도 줄인다.
    3. 상의 색깔을 배경화면의 색 평균값으로 전부 바꾸기(알파값 채널은 적용하지 않음)
    4. 상의 색깔만 조금 더 어둡게 만들기
    5. 상에 가우시안 블러를 이용해 테두리를 희미하게 만든다.
       - alpha 채널: 테두리로 갈수록 alpha=0
       - rgb 채널: 테두리로 갈수록 배경 평균색과 동일해지도록 페더링
    6. 상의 투명도를 alpha_value(0.15)로 지정
    7. 배경 위에 합성 (배경 노이즈가 살짝 비치도록)
    """

    # 1) 배경, 오브젝트 불러오기
    bg = cv2.imread(background_path)               # BGR
    obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)  # BGRA 가능

    if bg is None:
        print("배경 이미지를 불러올 수 없습니다.")
        return
    if obj is None:
        print("오브젝트 이미지를 불러올 수 없습니다.")
        return

    h_bg, w_bg = bg.shape[:2]

    # 오브젝트 리사이즈 (배경 크기에 맞춤, 필요 시)
    obj = cv2.resize(obj, (w_bg, h_bg), interpolation=cv2.INTER_AREA)

    # 2) 배경에 소량 가우시안 블러
    if bg_blur_ksize % 2 == 0:
        bg_blur_ksize += 1
    bg_blurred = cv2.GaussianBlur(bg, (bg_blur_ksize, bg_blur_ksize), 0)

    # 3) 상의 색깔을 배경 평균값으로 변경
    bg_mean_color = cv2.mean(bg_blurred)[:3]  # (B, G, R)
    if obj.shape[2] == 4:
        obj_bgr = obj[:, :, :3]
        obj_alpha = obj[:, :, 3]
    else:
        obj_bgr = obj
        obj_alpha = np.ones((h_bg, w_bg), dtype=np.uint8) * 255

    # 오브젝트의 RGB를 모두 배경 평균색으로
    obj_bgr[:] = np.array(bg_mean_color, dtype=np.uint8)

    # 4) 상의 색깔을 조금 더 어둡게 (obj_dark_offset)
    if obj_dark_offset != 0:
        tmp = obj_bgr.astype(np.int16) + obj_dark_offset
        tmp = np.clip(tmp, 0, 255)
        obj_bgr = tmp.astype(np.uint8)

    # (5-1) 알파 채널 블러 -> 테두리 alpha=0
    if alpha_blur_ksize % 2 == 0:
        alpha_blur_ksize += 1
    blurred_alpha = cv2.GaussianBlur(obj_alpha, (alpha_blur_ksize, alpha_blur_ksize), 0)

    # (5-2) RGB 채널도 테두리를 배경 평균색과 동일하게 페더링
    # 방법: center_mask = 원본 알파 (255=오브젝트), 0=배경
    # 큰 블러 -> 테두리로 갈수록 mask=0, center=1
    if color_blend_ksize % 2 == 0:
        color_blend_ksize += 1
    center_mask = obj_alpha.copy()  # 0~255
    # 가우시안 블러
    blurred_mask = cv2.GaussianBlur(center_mask, (color_blend_ksize, color_blend_ksize), 0)
    # 0~1 범위로
    blurred_mask_f = blurred_mask.astype(np.float32) / 255.0

    # "중심부"는 어둡게, "가장자리"는 배경 평균색 그대로 -> 
    # 실제로 우리는 이미 obj_bgr에 "어두운 배경 평균색"을 넣었음.
    # 하지만 테두리를 원래 배경 평균색(어둡게 하기 전)으로 부드럽게 보정하려면:
    # dark_color = obj_bgr
    # base_color = np.array(bg_mean_color, dtype=np.uint8)
    # -> blend = dark_color * mask + base_color * (1 - mask)
    dark_f = obj_bgr.astype(np.float32)
    base_f = np.full_like(dark_f, bg_mean_color, dtype=np.float32)  # 배경 평균색
    mask_3ch = np.dstack([blurred_mask_f, blurred_mask_f, blurred_mask_f])

    # 최종 RGB
    blended_rgb_f = dark_f * mask_3ch + base_f * (1.0 - mask_3ch)
    blended_rgb = np.clip(blended_rgb_f, 0, 255).astype(np.uint8)

    # 5) 결과로 obj_bgr, obj_alpha 업데이트
    obj_bgr = blended_rgb
    obj_alpha = blurred_alpha

    # 6) 상 전체 투명도(alpha_value) 적용
    # 최종 alpha = (obj_alpha / 255) * alpha_value
    alpha_float = obj_alpha.astype(np.float32) / 255.0
    alpha_float *= alpha_value
    alpha_float = np.clip(alpha_float, 0.0, 1.0)

    # 7) 배경 위 합성
    bg_f = bg_blurred.astype(np.float32)
    obj_f = obj_bgr.astype(np.float32)
    alpha_3ch = np.dstack([alpha_float, alpha_float, alpha_float])

    composite_f = bg_f * (1.0 - alpha_3ch) + obj_f * alpha_3ch
    composite = np.clip(composite_f, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, composite)
    print(f"[완료] 결과 저장: {output_path}")


if __name__ == "__main__":
    composite_hidden_object(
        background_path="background.png",
        object_path="croi2.png",
        output_path="result42.png",
        bg_blur_ksize=5,
        obj_dark_offset=-20,
        alpha_blur_ksize=31,
        color_blend_ksize=31,
        alpha_value=0.3
    )
