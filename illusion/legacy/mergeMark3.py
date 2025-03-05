import cv2
import numpy as np

def composite_low_alpha(
    background_path: str,
    object_path: str,
    output_path: str = "result.png",
    bg_blur_ksize: int = 5,
    obj_dark_offset: int = -20,
    obj_alpha_blur_ksize: int = 31,
    alpha_value: float = 0.15
):
    """
    1) 배경화면, 상 불러오기
    2) 배경화면에 가우시안 블러를 조금 넣어 눈 피로도 줄이기
    3) 상의 색깔을 배경화면의 색 평균값으로 전부 바꾸기(알파값 채널은 적용 X)
    4) 상의 색깔을 조금 더 어둡게 만들기
    5) 상에 가우시안 블러를 이용해 테두리 알파값을 0에 수렴시키기(페더링)
    6) 상의 투명도를 alpha_value(기본 0.15)로 지정(연하게)
    7) 이후 배경 위에 합성
    """

    # 1) 배경, 오브젝트 불러오기
    bg = cv2.imread(background_path)  # BGR
    obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)  # BGRA 가능

    if bg is None:
        print("배경 이미지를 불러올 수 없습니다.")
        return
    if obj is None:
        print("오브젝트 이미지를 불러올 수 없습니다.")
        return

    h_bg, w_bg = bg.shape[:2]

    # 배경과 오브젝트 크기가 다르면 리사이즈 (선택)
    obj = cv2.resize(obj, (w_bg, h_bg), interpolation=cv2.INTER_AREA)

    # 2) 배경에 소량 가우시안 블러 적용
    #    bg_blur_ksize가 홀수가 되도록 조정
    if bg_blur_ksize % 2 == 0:
        bg_blur_ksize += 1
    bg_blurred = cv2.GaussianBlur(bg, (bg_blur_ksize, bg_blur_ksize), 0)

    # 3) 상의 색깔을 배경화면의 색 평균값으로 변경(알파 채널 제외)
    bg_mean_color = cv2.mean(bg_blurred)[:3]  # (B, G, R) 평균
    # 오브젝트 분리
    if obj.shape[2] == 4:
        obj_bgr = obj[:, :, :3]
        obj_alpha = obj[:, :, 3]
    else:
        obj_bgr = obj
        obj_alpha = np.ones((h_bg, w_bg), dtype=np.uint8) * 255

    # 오브젝트 RGB를 배경 평균색으로 덮어씌움
    obj_bgr[:] = np.array(bg_mean_color, dtype=np.uint8)

    # 4) 상의 색깔을 조금 더 어둡게 (obj_dark_offset)
    #    예: -20이면 더 어둡게
    if obj_dark_offset != 0:
        obj_bgr_f = obj_bgr.astype(np.int16) + obj_dark_offset
        obj_bgr_f = np.clip(obj_bgr_f, 0, 255)
        obj_bgr = obj_bgr_f.astype(np.uint8)

    # 5) 상의 알파 채널을 큰 가우시안 블러로 페더링
    #    테두리로 갈수록 알파값이 0으로 떨어지게 만들려면
    #    원본 알파(255=불투명, 0=투명) -> 블러 -> 테두리 주변이 서서히 0
    if obj_alpha is not None:
        if obj_alpha_blur_ksize % 2 == 0:
            obj_alpha_blur_ksize += 1
        alpha_blurred = cv2.GaussianBlur(obj_alpha, (obj_alpha_blur_ksize, obj_alpha_blur_ksize), 0)
    else:
        # 알파 채널이 없는 경우, 임시로 전부 255
        alpha_blurred = np.ones((h_bg, w_bg), dtype=np.uint8) * 255

    # 6) 상의 투명도 alpha_value로 지정
    #    최종 알파 = alpha_blurred/255 * alpha_value
    alpha_float = alpha_blurred.astype(np.float32) / 255.0
    alpha_float *= alpha_value
    alpha_float = np.clip(alpha_float, 0.0, 1.0)

    # 7) 배경 위에 합성 (배경 노이즈가 상 뒤로 비치도록)
    #    result_pixel = bg_blurred * (1 - alpha) + obj_bgr * alpha
    bg_f = bg_blurred.astype(np.float32)
    obj_f = obj_bgr.astype(np.float32)
    alpha_3ch = np.dstack([alpha_float, alpha_float, alpha_float])

    composite_f = bg_f * (1.0 - alpha_3ch) + obj_f * alpha_3ch
    composite = np.clip(composite_f, 0, 255).astype(np.uint8)

    # 결과 저장
    cv2.imwrite(output_path, composite)
    print(f"결과가 저장되었습니다: {output_path}")


if __name__ == "__main__":
    composite_low_alpha(
        background_path="background.png",
        object_path="croi.png",
        output_path="result3.png",
        bg_blur_ksize=5,        # 2) 배경 블러 정도
        obj_dark_offset=-20,    # 4) 오브젝트 색을 더 어둡게
        obj_alpha_blur_ksize=31,# 5) 오브젝트 알파 페더링
        alpha_value=0.15        # 6) 오브젝트 전체 투명도
    )
