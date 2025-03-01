import cv2
import numpy as np

def embed_lowfreq_object_in_background(
    background_path="background.png",
    object_path="object.png",
    output_path="result.png",
    blur_ksize=101,
    alpha_factor=0.3,
    brightness_offset=-20
):
    """
    1) 배경(고주파, 전체화면)과 오브젝트(검은 실루엣)를 불러온다.
    2) 오브젝트를 그레이스케일로 변환 후, 크게 가우시안 블러(저주파화).
    3) 블러 결과에 밝기 오프셋(brightness_offset)을 적용해 배경과 비슷한 톤으로 맞춤.
    4) alpha_factor로 투명도 조절하여 배경과 합성.
    5) 최종 결과 저장.
    
    :param background_path: 고주파 배경 이미지 경로
    :param object_path: 검은 실루엣 오브젝트 이미지 경로(투명 PNG면 좋음)
    :param output_path: 최종 저장 경로
    :param blur_ksize: 가우시안 블러 커널 크기 (커질수록 더 흐릿)
    :param alpha_factor: 오브젝트(저주파) 레이어를 얼마나 투명하게 할지 (0~1)
    :param brightness_offset: 블러된 오브젝트의 전체 밝기 보정
    """
    
    # 1) 이미지 불러오기
    bg = cv2.imread(background_path)  # 배경(BGR)
    obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)  # 오브젝트(BGRA 가능)
    
    if bg is None:
        print("배경 이미지를 불러올 수 없습니다.")
        return
    if obj is None:
        print("오브젝트 이미지를 불러올 수 없습니다.")
        return
    
    # 배경과 오브젝트 크기가 다를 수 있으니, 배경 크기로 리사이즈 (선택 사항)
    #h_bg, w_bg = bg.shape[:2]
    
    # 2) 오브젝트 -> 그레이스케일 (또는 알파 채널만 사용)
    #    오브젝트가 검은색 실루엣이라고 가정 -> 그레이 변환 후 흰색 영역은 0, 검정 영역은 255가 될 수도 있으므로
    #    실제 값 확인 후 필요하면 반전 처리
    if obj.shape[2] == 4:
        # RGBA -> Grayscale, 알파 채널 고려
        # 알파 채널이 0이면 투명, 255이면 불투명
        b, g, r, a = cv2.split(obj)
        # 알파가 거의 0인 영역은 무시, 255인 영역만 검정 실루엣일 가능성
        # 여기서는 간단히 BGR을 그레이 변환
        obj_bgr = cv2.merge((b, g, r))
        gray_obj = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2GRAY)
        # shape 맞추기
        alpha_mask = a
    else:
        # 3채널 (BGR)이라면 직접 그레이 변환
        gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        alpha_mask = np.ones_like(gray_obj, dtype=np.uint8) * 255
    
    # 리사이즈 (필요하면)
    gray_obj = cv2.resize(gray_obj, (w_bg, h_bg), interpolation=cv2.INTER_AREA)
    alpha_mask = cv2.resize(alpha_mask, (w_bg, h_bg), interpolation=cv2.INTER_AREA)
    
    # 검정 실루엣이 0(검정), 255(흰) 인지 혹은 반대인지 확인
    # 가정: 검정 영역 -> 작은 픽셀값, 배경(흰) -> 큰 픽셀값
    # 만약 반대로 되어 있다면 아래처럼 반전
    # if np.mean(gray_obj) > 128:
    #     gray_obj = 255 - gray_obj
    
    # 3) 큰 가우시안 블러(저주파화)
    # blur_ksize는 반드시 홀수
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred_obj = cv2.GaussianBlur(gray_obj, (blur_ksize, blur_ksize), 0)
    
    # 4) 밝기 보정
    # brightness_offset이 -20이면 전체적으로 더 어둡게
    # (0~255 범위 유지)
    blurred_obj = blurred_obj.astype(np.int16) + brightness_offset
    blurred_obj = np.clip(blurred_obj, 0, 255).astype(np.uint8)
    
    # 5) 오브젝트의 투명도(alpha_factor) 조절
    # 최종 mask = alpha_mask (오브젝트 영역) * alpha_factor
    # 하지만 blurred_obj 자체가 "저주파 이미지" 역할을 함.
    # 즉, blurred_obj를 color(=그레이)로, alpha_factor를 곱해서 합성
    alpha_float = alpha_mask.astype(np.float32) / 255.0
    # 여기서는 알파 채널을 별도로 고려하기보단, blurred_obj를 일정 투명도로 BG에 합성
    
    # blurred_obj -> 3채널로 만들어야 BGR에 합성 가능
    blurred_bgr = cv2.merge([blurred_obj, blurred_obj, blurred_obj])
    
    # 6) 합성
    # 최종 픽셀 = bg * (1 - total_alpha) + blurred_obj * total_alpha
    # total_alpha = alpha_float * alpha_factor
    total_alpha = alpha_float * alpha_factor
    
    # 배경 float 변환
    bg_f = bg.astype(np.float32)
    obj_f = blurred_bgr.astype(np.float32)
    
    # 합성
    # pixel_out = bg_f * (1 - total_alpha) + obj_f * total_alpha
    # alpha가 위치마다 다르므로, 위치별 연산
    # 편의상 for 루프 없이 브로드캐스팅
    alpha_3ch = np.dstack([total_alpha, total_alpha, total_alpha])
    result_f = bg_f * (1 - alpha_3ch) + obj_f * alpha_3ch
    
    result = np.clip(result_f, 0, 255).astype(np.uint8)
    
    # 7) 결과 저장
    cv2.imwrite(output_path, result)
    print(f"결과가 저장되었습니다: {output_path}")


if __name__ == "__main__":
    embed_lowfreq_object_in_background(
        background_path="background.png",  # 고주파 배경
        object_path="croi.png",         # 검정 실루엣
        output_path="result.png",
        blur_ksize=1001,       # 블러 강도 (커질수록 더 흐릿)
        alpha_factor=0.15,     # 최종 투명도 (0.3 -> 30%)
        brightness_offset=-20 # 블러된 실루엣을 조금 어둡게
    )
