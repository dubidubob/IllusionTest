import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise_using_image_mean(imageName, noise_scale=20):
    """
    1. 이미지를 불러오고, 알파 채널이 있는 경우 RGB와 알파를 분리합니다.
    2. RGB 채널의 각 채널별 평균을 계산합니다.
    3. 각 채널의 평균을 노이즈의 loc 파라미터로 사용하여 노이즈를 생성합니다.
    4. 생성된 노이즈를 RGB 이미지에 추가하고, 결과를 클리핑합니다.
    5. 알파 채널이 있는 경우, 흰색 배경과 합성하여 저장합니다.
    """
    input_path = f"{imageName}.png"
    output_path = f"{imageName}_{noise_scale}_add_composited.png"
    
    # 알파 채널 포함하여 이미지 불러오기
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {input_path}")
        return
    
    # 이미지 shape 확인
    height, width = image.shape[:2]
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    
    # 알파 채널 분리: 3채널 (RGB) 혹은 4채널 (RGBA)
    if num_channels == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3] / 255.0  # 정규화 (0~1)
    else:
        rgb = image
        alpha = None

    # 각 채널별 평균 계산 (float32로 계산)
    channel_means = np.mean(rgb.astype(np.float32), axis=(0, 1))
    print(f"계산된 RGB 채널 평균: {channel_means}")

    # 노이즈 생성: 각 픽셀의 R, G, B 채널에 대해 계산된 평균을 사용하여 생성
    noise = np.random.normal(loc=channel_means, scale=noise_scale, size=rgb.shape)

    # 노이즈를 RGB에 추가한 후, 0~255 범위로 클리핑
    result_rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 알파 채널이 있을 경우, 흰색 배경과 합성
    if alpha is not None:
        white_background = np.full_like(result_rgb, 255, dtype=np.uint8)  # 흰색 배경 생성
        alpha = alpha[:, :, np.newaxis]  # (H, W) -> (H, W, 1)로 변환
        
        # (1 - alpha) * 흰 배경 + alpha * 결과 이미지
        composited_image = (white_background * (1 - alpha) + result_rgb * alpha).astype(np.uint8)
    else:
        composited_image = result_rgb  # 알파 채널이 없으면 그냥 노이즈 이미지 사용

    # 결과 저장
    cv2.imwrite(output_path, composited_image)
    
    # matplotlib으로 출력 (cv2는 BGR이므로 RGB로 변환)
    result_disp = cv2.cvtColor(composited_image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(result_disp)
    plt.axis('off')
    plt.title("Image with Noise on White Background")
    plt.show()

if __name__ == "__main__":
    image_name = "stage01_darkMarkLarger"  # 이미지 파일 이름 (확장자 제외)
    noise_scale = 5  # 노이즈 강도 조절
    add_noise_using_image_mean(image_name, noise_scale)
