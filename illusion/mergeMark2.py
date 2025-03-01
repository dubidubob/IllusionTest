import cv2
import numpy as np

def overlay_low_alpha_object(
    background_path: str,
    object_path: str,
    output_path: str = "result.png",
    alpha_value: float = 0.15,
    blur_ksize: int = 15
):
    """
    1. 배경화면, 상 불러오기
    2. 상의 색깔을 배경화면의 색 평균값으로 전부 바꾸기(알파값은 그대로 유지)
    3. 가우시안 블러로 상의 테두리를 희미하게 만든다(페더링).
    4. 상의 투명도를 alpha_value(기본 0.15)로 지정한다.
    5. 배경 위에 합성한다.
    """

    # 1) 배경, 오브젝트 이미지 불러오기
    bg = cv2.imread(background_path)  # 배경 (BGR 3채널)
    obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)  # 상 (BGRA 4채널 가능)

    if bg is None:
        print("배경 이미지를 불러올 수 없습니다.")
        return
    if obj is None:
        print("오브젝트 이미지를 불러올 수 없습니다.")
        return

    h_bg, w_bg = bg.shape[:2]

    # 오브젝트 리사이즈(필요 시)
    # 배경과 크기가 다르면, 배경 크기에 맞춰 조정
    obj = cv2.resize(obj, (w_bg, h_bg), interpolation=cv2.INTER_AREA)

    # 2) 배경 평균 색을 구해, 오브젝트의 RGB 채널에 적용
    # 배경 평균 색 (B, G, R)
    bg_mean_color = cv2.mean(bg)[:3]  # (B, G, R) 세 채널 평균

    # 오브젝트 분리: obj_bgr, obj_alpha
    if obj.shape[2] == 4:
        obj_bgr = obj[:, :, :3]
        obj_alpha = obj[:, :, 3]
    else:
        # 만약 3채널이라면 알파 채널이 없는 상태
        obj_bgr = obj
        # 알파 채널이 없으니, 임시로 전부 불투명(255)로 생성
        obj_alpha = np.ones((h_bg, w_bg), dtype=np.uint8) * 255

    # 오브젝트 RGB 채널을 배경 평균 색으로 채움
    # np.full_like: shape 동일, 값만 지정
    obj_bgr[:] = np.array(bg_mean_color, dtype=np.uint8)

    # 3) 오브젝트 테두리를 가우시안 블러(페더링)
    #    여기서는 알파 채널(테두리)을 블러해 경계를 부드럽게 만든다.
    #    커널 크기는 blur_ksize x blur_ksize, 반드시 홀수
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred_alpha = cv2.GaussianBlur(obj_alpha, (blur_ksize, blur_ksize), 0)

    # 4) 상의 투명도(alpha_value) 적용
    #    최종 alpha = blurred_alpha * alpha_value
    #    [0,255] 범위 내에서 float 연산
    alpha_float = blurred_alpha.astype(np.float32) / 255.0
    alpha_float *= alpha_value
    # 클램핑
    alpha_float = np.clip(alpha_float, 0.0, 1.0)

    # 5) 배경 위에 합성
    # 최종 픽셀 = bg * (1 - alpha) + obj_color * alpha
    bg_f = bg.astype(np.float32)
    obj_f = obj_bgr.astype(np.float32)
    alpha_3ch = np.dstack([alpha_float, alpha_float, alpha_float])

    result_f = bg_f * (1.0 - alpha_3ch) + obj_f * alpha_3ch
    result = np.clip(result_f, 0, 255).astype(np.uint8)

    # 결과 저장
    cv2.imwrite(output_path, result)
    print(f"결과가 저장되었습니다: {output_path}")


if __name__ == "__main__":
    overlay_low_alpha_object(
        background_path="background.png",
        object_path="croi.png",
        output_path="result2.png",
        alpha_value=1,
        blur_ksize=15
    )
