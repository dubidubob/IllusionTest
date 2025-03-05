import cv2
import numpy as np

def generate_subtle_high_freq_background(
    width=3840, 
    height=2160, 
    output_file="background.png",
    base_color=128,
    noise_scale=15,
    saturation_scale=0.2,
    contrast_factor=0.5
):
    """
    1. (height, width, 3) 크기의 무작위 컬러 노이즈 이미지를 만든다. (고주파)
    2. 전체 톤이 중간 밝기(base_color) 근처가 되도록 설정.
    3. HSV로 변환해 채도를 낮춘다 (saturation_scale).
    4. 대비를 낮춘다 (contrast_factor).
    5. 최종 이미지를 저장한다.
    
    기본적으로 채도(saturation_scale)가 0.2이므로
    완전 무채색(grayscale)은 아니지만, 거의 회색에 가까운 미묘한 컬러 노이즈가 생성된다.
    """
    
    # 1) 무작위 노이즈 이미지 생성 (고주파)
    #    정규분포(평균 base_color, 표준편차 noise_scale)
    random_noise = np.random.normal(
        loc=base_color, 
        scale=noise_scale, 
        size=(height, width, 3)
    ).astype(np.float32)
    
    # 범위 [0, 255]로 클램핑
    random_noise = np.clip(random_noise, 0, 255)
    
    # 2) 채널 순서: OpenCV는 BGR
    img_bgr = random_noise.astype(np.uint8)
    
    # 3) HSV 변환 후 채도(S)를 낮춘다
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_hsv[..., 1] = (img_hsv[..., 1].astype(np.float32) * saturation_scale)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1], 0, 255)
    
    # 다시 BGR로 변환
    img_bgr = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 4) 대비 낮추기
    if contrast_factor < 1.0:
        img_f = img_bgr.astype(np.float32)
        img_f = 128.0 + contrast_factor * (img_f - 128.0)
        img_bgr = np.clip(img_f, 0, 255).astype(np.uint8)
    
    # 5) 결과 저장
    cv2.imwrite(output_file, img_bgr)

if __name__ == "__main__":
    generate_subtle_high_freq_background(
        width=3840, 
        height=2160,
        output_file="background.png",
        base_color=128,        # 중간 밝기
        noise_scale=15,        # 노이즈 진폭
        saturation_scale=0.2,  # 채도를 20%로 (거의 무채색에 가까움)
        contrast_factor=0.5    # 대비를 절반으로 낮춤
    )
    print("배경 이미지가 생성되었습니다: background.png (3840x2160)")
