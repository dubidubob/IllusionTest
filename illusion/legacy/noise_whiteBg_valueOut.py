import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def add_noise_and_measure(imageName, noise_scale=20, region_size=(200, 200), zoom_factor=1.6):
    """
    1. 이미지를 읽고, 알파 채널이 있으면 분리한다.
    2. RGB 채널 평균값으로 노이즈를 생성하여 합성한다.
    3. 알파 채널이 있으면 흰색 배경과 합성한다.
    4. 최종 결과 이미지를 그레이스케일로 변환하여 전체 대비(표준편차)를 계산한다.
    5. 최종 결과 이미지에서 랜덤 위치의 작은 영역을 추출하고,
       zoom_factor(1.6)만큼 확대하여 그 부분의 표준편차(색 분포)를 계산한다.
    6. 두 값을 출력한다.
    """

    # 1) 이미지 읽기
    input_path = f"{imageName}.png"
    output_path = f"{imageName}_{noise_scale}_composited.png"

    # 알파 채널 포함해서 읽기
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {input_path}")
        return

    height, width = image.shape[:2]
    if image.ndim == 3:
        num_channels = image.shape[2]
    else:
        num_channels = 1

    # 알파 채널 분리
    if num_channels == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3].astype(np.float32) / 255.0  # 0~1 범위로 정규화
    else:
        rgb = image
        alpha = None

    # 2) 노이즈 생성
    channel_means = np.mean(rgb.astype(np.float32), axis=(0, 1))
    noise = np.random.normal(loc=channel_means, scale=noise_scale, size=rgb.shape)
    result_rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3) 알파 채널 있으면 흰 배경과 합성
    if alpha is not None:
        white_bg = np.full_like(result_rgb, 255, dtype=np.uint8)
        alpha_3d = alpha[:, :, np.newaxis]
        composited_image = (white_bg * (1 - alpha_3d) + result_rgb * alpha_3d).astype(np.uint8)
    else:
        composited_image = result_rgb

    # 결과 저장
    cv2.imwrite(output_path, composited_image)

    # 4) 전체 이미지 대비(표준편차)
    gray_full = cv2.cvtColor(composited_image, cv2.COLOR_BGR2GRAY)
    contrast_full = np.std(gray_full)

    # 5) 랜덤 위치의 영역을 잘라서 확대 후 표준편차 측정
    region_w, region_h = region_size
    # 영역이 이미지보다 클 경우 대비해 최소값 설정
    region_w = min(region_w, width)
    region_h = min(region_h, height)

    # 랜덤 시작점
    start_x = random.randint(0, width - region_w)
    start_y = random.randint(0, height - region_h)

    # 영역 추출
    region = composited_image[start_y:start_y+region_h, start_x:start_x+region_w]

    # 확대 (zoom_factor = 1.6 => 160% 크기)
    zoom_w = int(region_w * zoom_factor)
    zoom_h = int(region_h * zoom_factor)
    zoomed_region = cv2.resize(region, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)

    # 확대된 영역의 표준편차(색 분포)
    gray_zoomed = cv2.cvtColor(zoomed_region, cv2.COLOR_BGR2GRAY)
    zoomed_std = np.std(gray_zoomed)

    # 결과 출력
    print(f"[{imageName}]")
    print(f" > 전체 이미지 대비(표준편차): {contrast_full:.3f}")
    print(f" > 랜덤 영역 {region_w}x{region_h} (확대 후 {zoom_w}x{zoom_h}) 색 분포 표준편차: {zoomed_std:.3f}")

    # 6) 시각화 (원본, 랜덤영역, 확대영역)
    plt.figure(figsize=(12, 4))

    # (a) 전체 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(composited_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Full Image\nSTD={contrast_full:.2f}")
    plt.axis('off')

    # (b) 잘라낸 영역
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    plt.title(f"Random Region\n{region_w}x{region_h}")
    plt.axis('off')

    # (c) 확대된 영역
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(zoomed_region, cv2.COLOR_BGR2RGB))
    plt.title(f"Zoomed Region\nSTD={zoomed_std:.2f}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 사용 예시
if __name__ == "__main__":
    image_name = "stage01_darkMarkLarger"  # 확장자 제외
    noise_scale = 10   # 노이즈 강도
    add_noise_and_measure(image_name, noise_scale, region_size=(200, 200), zoom_factor=1.6)
