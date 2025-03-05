import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_noise_using_image_mean(imageName, noise_scale=20):
    """
    1. 이미지를 불러오고, 알파 채널이 있는 경우 RGB와 알파를 분리합니다.
    2. RGB 채널의 각 채널별 평균을 계산합니다.
    3. 각 채널의 평균을 노이즈의 loc 파라미터로 사용하여 노이즈를 생성합니다.
    4. 생성된 노이즈를 RGB 이미지에 추가하고, 결과를 클리핑합니다.
    5. 알파 채널이 있는 경우, RGB 노이즈 이미지와 알파 채널을 합칩니다.
    """
    input_path = f"{imageName}.png"
    output_path = f"{imageName}_{noise_scale}_add.png"
    
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
        alpha = image[:, :, 3]
    else:
        rgb = image
        alpha = None

    # 각 채널별 평균 계산 (float32로 계산)
    channel_means = np.mean(rgb.astype(np.float32), axis=(0, 1))
    print(f"계산된 RGB 채널 평균: {channel_means}")

    # 노이즈 생성: 각 픽셀의 R, G, B 채널에 대해 계산된 평균을 사용하여 생성
    # np.random.normal의 loc 파라미터에 (3,) shape의 배열을 넣으면, size에 맞춰 브로드캐스팅됩니다.
    noise = np.random.normal(loc=channel_means, scale=noise_scale, size=rgb.shape)

    # 노이즈를 RGB에 추가한 후, 0~255 범위로 클리핑
    #result_rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    result_rgb = np.clip(noise, 0, 255).astype(np.uint8)
    
    # 알파 채널이 있으면 합치기
    if alpha is not None:
        # alpha 채널은 원본 그대로 사용
        result = np.dstack((result_rgb, alpha))
    else:
        result = result_rgb

    # 결과 저장 및 출력
    cv2.imwrite(output_path, result)
    
    # matplotlib으로 출력 (알파 채널 있을 경우, cv2.cvtColor를 이용하여 BGR->RGB 변환)
    if result.shape[2] == 4:
        # OpenCV는 BGRA이므로, RGB 순서로 변환
        result_disp = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
    else:
        result_disp = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
    plt.imshow(result_disp)
    plt.axis('off')
    plt.title("Image with Noise")
    plt.show()

if __name__ == "__main__":
    image_name = "stage01_darkMarkLarger"  # 사용하실 이미지 파일 이름 (확장자 제외)
    noise_scale = 10         # 노이즈의 표준편차 값, 필요에 따라 조절
    add_noise_using_image_mean(image_name, noise_scale)
