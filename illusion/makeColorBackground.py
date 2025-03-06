import cv2
import numpy as np

def generate_subtle_high_freq_background(
    width=3840, 
    height=2160, 
    output_file="background.png",
    base_color=(128, 128, 128),  # (B, G, R) 튜플로 지정 가능
    noise_scale=15,
    saturation_scale=0.5,  # 색을 좀 더 선명하게 하고 싶으면 1.0에 가깝게
    contrast_factor=0.5
):
    """
    1. (height, width, 3) 크기의 무작위 컬러 노이즈 이미지를 만든다.
       - base_color를 (B, G, R) 튜플로 지정하면 해당 색 주변으로 노이즈가 발생.
    2. HSV 변환 -> 채도(saturation_scale)로 조절.
    3. 대비(contrast_factor) 낮추기.
    4. 최종 이미지 저장.
    
    예: base_color=(0, 255, 255)는 '노랑' 계열 (OpenCV는 BGR이므로 (B=0, G=255, R=255))
        base_color=(0, 165, 255)는 '주황' 계열
        base_color=(255, 0, 0)는 '파랑' 계열 등
    """

    # 1) 무작위 노이즈 이미지 생성
    # base_color가 (B, G, R) 튜플이라고 가정
    random_noise = np.zeros((height, width, 3), dtype=np.float32)

    # 채널별로 loc=base_color[c], scale=noise_scale
    for c in range(3):
        random_noise[..., c] = np.random.normal(
            loc=base_color[c],
            scale=noise_scale,
            size=(height, width)
        )

    # 범위 [0, 255]로 클램핑
    random_noise = np.clip(random_noise, 0, 255)

    # uint8로 변환
    img_bgr = random_noise.astype(np.uint8)

    # 2) HSV 변환 후 채도(S) 조절
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 채도 채널: img_hsv[..., 1]
    # scale 적용
    img_hsv[..., 1] = (img_hsv[..., 1].astype(np.float32) * saturation_scale)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 255)

    # 다시 BGR로 변환
    img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3) 대비 낮추기
    if contrast_factor < 1.0:
        img_f = img_bgr.astype(np.float32)
        # (픽셀 - 128)*contrast_factor + 128
        img_f = 128.0 + contrast_factor * (img_f - 128.0)
        img_bgr = np.clip(img_f, 0, 255).astype(np.uint8)

    # 4) 최종 결과 저장
    cv2.imwrite(output_file, img_bgr)

if __name__ == "__main__":
    # 예: 노란색 계열 배경 (B=0, G=255, R=255) 근처\
    generate_subtle_high_freq_background(
        width=3840,
        height=2160,
        output_file="colorbackground.png",
        base_color=(100, 0, 100),   # 노랑(사이안+빨강)
        noise_scale=25,
        saturation_scale=0.8,       # 채도를 조금 높여서 색감 강조
        contrast_factor=0.8
    )
    print("노란 계열 배경 이미지가 생성되었습니다!")
