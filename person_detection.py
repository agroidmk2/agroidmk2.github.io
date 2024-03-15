import openvino as ov
import cv2
import numpy as np
import ipywidgets as widgets
import os
import datetime


class ObjectDetection:
    def __init__(self, model_xml, model_bin, device="AUTO"):
        # 모델 파일의 경로와 사용할 디바이스를 입력으로 받아 초기화합니다.
        self.model_xml = model_xml
        self.model_bin = model_bin

        self.core = ov.Core()
        self.device = widgets.Dropdown(
            options=self.core.available_devices + ["AUTO"],
            value='AUTO',
            description='Device:',
            disabled=False,
        )
        # 모델을 로드하고, 실행할 디바이스에 맞게 최적화합니다.
        self.model = self.core.read_model(model=self.model_xml)
        self.compiled_model = self.core.compile_model(
            model=self.model, device_name=self.device.value)
        # 모델의 입력과 출력 키를 가져옵니다.
        self.input_keys = self.compiled_model.input(0)
        self.boxes_output_keys = self.compiled_model.output(0)
        self.labels_output_keys = self.compiled_model.output(1)
        # 이미지의 높이와 너비를 설정합니다.
        self.height = 640
        self.width = 640

    def preprocess_image(self, image):
        # 이미지를 모델 입력에 맞게 전처리합니다.
        resized_image = cv2.resize(image, (self.width, self.height))
        input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
        return input_image

    def detect_object(self, processed_image):
        # 전처리된 이미지에서 손을 감지합니다.
        results = self.compiled_model([processed_image])
        boxes = results[self.boxes_output_keys]
        labels = results[self.labels_output_keys]

        # 신뢰도 점수를 기준으로 가장 높은 값을 가진 객체를 찾습니다.
        max_confidence_index = np.argmax(boxes[0, :, 4])
        max_confidence_box = boxes[0, max_confidence_index]
        max_confidence_label = labels[0, max_confidence_index]
        return max_confidence_box, max_confidence_label

    def run(self, image):
        # 이미지를 입력받아 감지된 물체를 리턴합니다.
        processed_image = self.preprocess_image(image)
        box, label = self.detect_object(processed_image)
        # 레이블에 해당하는 동작 이름을 가져옵니다.
        object_naem = ['person', 'book']
        object = object_naem[label]
        confidence = box[4]*100
        return object, confidence, box


def person_detection():
    # 사용예제
    object_detector = ObjectDetection(model_xml='./models/model.xml',
                                      model_bin='./models/model.bin')

    cap = cv2.VideoCapture(4)

    # 프로그램 시작 시간
    start_time = datetime.datetime.now()

    while True:
        for i in range(0, 50):
            ret, frame = cap.read()
            cv2.imshow("frame", frame)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%s")

        if not ret:
            print("입력이 없습니다.")
            break

        label, confidence, box = object_detector.run(frame)

        # 라벨이 "person"인 경우
        if label == "person":
            save_path = './checkout'  # 이미지 저장 경로 설정
            # 디렉토리가 없으면 생성
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # 바운딩 박스 좌표
            x_min, y_min, x_max, y_max = map(int, box[:4])

            # 바운딩 박스에 해당하는 부분만 자르기
            cropped_image = frame[y_min:y_max, x_min:x_max]

            # 이미지가 비어 있지 않은 경우에만 저장
            if not cropped_image.size == 0:
                file_name = f'{save_path}/checkout_time_{current_time}.jpg'
                # 이미지를 저장
                cv2.imwrite(file_name, cropped_image)
                print(label)
                break

            elif label == "book":
                continue

        # 30초동안 인식되는 내용이 없는 경우 프로그램 종료
        if (datetime.datetime.now() - start_time).total_seconds() >= 30:
            break

        # 1ms 마다 키 입력 대기
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 비디오 캡처 객체 해제
    cap.release()
    # 모든 창 닫기
    cv2.destroyAllWindows()


person_detection()
