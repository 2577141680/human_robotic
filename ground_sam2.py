import cv2
import torch
import supervision as sv
import pycocotools.mask as mask_util
import numpy as np
from pathlib import Path

from PIL.ImageMath import imagemath_convert
from supervision.draw.color import ColorPalette
from zmq import NormMode

from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import yaml
from pathplanning import pathplan
from PIL import ImageDraw
from supervision.annotators.utils import ColorLookup
from scipy.interpolate import splprep, splev

class GroundingDINO_SAM:
    def __init__(self, config_file="config.yaml"):
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.GROUNDING_MODEL = config["Dino"]["grounding_model"]
        self.SAM2_CHECKPOINT = config["SAM2"]["checkpoint"]
        self.SAM2_MODEL_CONFIG = config["SAM2"]["model_config"]
        self.DINO_BOX_THRESHOLD = config["Dino"]["box_threshold"]
        self.DINO_TEXT_THRESHOLD = config["Dino"]["text_threshold"]
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        torch.autocast(device_type=self.DEVICE, dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.SAM2_model = build_sam2(self.SAM2_MODEL_CONFIG, self.SAM2_CHECKPOINT, device=self.DEVICE)
        self.SAM2_predictor = SAM2ImagePredictor(self.SAM2_model)

        self.processor = AutoProcessor.from_pretrained(self.GROUNDING_MODEL)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.GROUNDING_MODEL).to(self.DEVICE)

    
    def get_grounding_dino_boxes(self, image_path, text):
        # image = Image.open(image_path)
        ###li
        # image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image_path)

        ###
        inputs = self.processor(images=image_path, text=text, return_tensors="pt").to(self.DEVICE)
        
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.DINO_BOX_THRESHOLD,
            text_threshold=self.DINO_TEXT_THRESHOLD,
            target_sizes=[image_path.size[::-1]],
        )

        return results
    
    def get_sam2_masks(self, image_path, boxes):
        # image = Image.open(image_path)
        ###li

        ###

        self.SAM2_predictor.set_image(np.array(image_path.convert("RGB")))

        masks, scores, logits = self.SAM2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits

    def get_finesegment_feature_field(self, image_path, text, visualize=True, save_path="", target_classes = ['floor','step','ladder'],path=None):
        try:
            prediction = self.get_grounding_dino_boxes(image_path, text)
            boxes = prediction[0]["boxes"].cpu().numpy()
            masks, _, _ = self.get_sam2_masks(image_path, boxes)

            if target_classes is not None:
                class_ids = prediction[0]["labels"]  # 获取类别 ID
                target_mask = np.zeros(masks.shape[1:], dtype=np.uint8)  # 创建一个空掩码

                for idx, class_id in enumerate(class_ids):
                    if class_id in target_classes:  # 检查是否是目标类别
                        target_mask |= masks[idx].astype(np.uint8)  # 合并目标类别的掩码

                # 将合并后的目标掩码用于后续处理
                path = pathplan(target_mask)  # 处理合成的掩码

                # def draw_path_on_image(image, path, color=(255, 255, 255), radius=3):
                #     draw = ImageDraw.Draw(image)
                #     for point in path:
                #         # 绘制每个点为一个圆
                #         draw.ellipse([point[0] - radius, point[1] - radius,
                #                       point[0] + radius, point[1] + radius],
                #                      fill=color)
                #     return image
                def smooth_curve(points):
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    tck, u = splprep([x, y], s=0)
                    x_new, y_new = splev(np.linspace(0, 1, 100), tck)
                    return list(zip(x_new, y_new))

                def draw_path_on_image(image, path, color=(255, 255, 255), radius=1):
                    # 生成平滑曲线点
                    smooth_path = smooth_curve(path)

                    draw = ImageDraw.Draw(image)
                    for point in smooth_path:
                        # 绘制每个点为一个圆
                        draw.ellipse([point[0] - radius, point[1] - radius,
                                      point[0] + radius, point[1] + radius],
                                     fill=color)
                    return image

            return draw_path_on_image(self.save_visualization(image_path, masks, prediction, visualize, save_path),path), target_mask

            # self.save_visualization(image_path, masks, prediction, visualize, save_path)
        except Exception as e:
            print(f"Error: {e}")
            return None


        return draw_path_on_image(self.save_visualization(image_path, masks, prediction, visualize, save_path), path), target_mask

    def save_visualization(self, image_path, masks, prediction, visualize, save_path):
        # img = cv2.imread(image_path)
        ###li、
        img = image_path
        ###
        input_boxes = prediction[0]["boxes"].cpu().numpy()
        confidences = prediction[0]["scores"].cpu().numpy().tolist()
        class_names = prediction[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP),color_lookup=ColorLookup.INDEX)
        masked_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # if save_path is not None:
        #     cv2.imwrite("groundingdino_annotated_image.jpg", annotated_frame)
        #     cv2.imwrite("grounded_sam2_annotated_image_with_mask.jpg", masked_frame)

        # if visualize:
        #     cv2.imshow("GroundingDINO_SAM", masked_frame)
        #     cv2.imshow("GroundingDINO", annotated_frame)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        ###li
        return masked_frame


if __name__ == "__main__":
    image_path = "gpt_input_image.jpg"
    text = "floor. people. obstacle."
    Splitter = GroundingDINO_SAM()
    Splitter.get_finesegment_feature_field(image_path, text, visualize=True)
