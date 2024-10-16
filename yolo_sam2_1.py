import cv2
import torch
import supervision as sv
import numpy as np
# from pathlib import Path
from supervision.draw.color import ColorPalette
# from LM.pathplanning import img_patch
from utils.supervision_utils import CUSTOM_COLOR_MAP
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import yaml
from pathplanning import pathplan
from supervision.annotators.utils import ColorLookup
from pathplanning import  draw_path_on_image

# Import the Ultralytics YOLO library
from ultralytics import YOLO

class GroundingDINO_SAM:
    def __init__(self, config_file="config.yaml"):
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Removed GroundingDINO configurations
        self.SAM2_CHECKPOINT = config["SAM2"]["checkpoint"]
        self.SAM2_MODEL_CONFIG = config["SAM2"]["model_config"]
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        torch.autocast(device_type=self.DEVICE, dtype=torch.float16).__enter__()

        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            # Turn on TF32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.SAM2_model = build_sam2(self.SAM2_MODEL_CONFIG, self.SAM2_CHECKPOINT, device=self.DEVICE)
        self.SAM2_predictor = SAM2ImagePredictor(self.SAM2_model)

        # Load the YOLO model
        self.YOLO_MODEL = config["YOLO"]["model"]
        self.yolo_model = YOLO(self.YOLO_MODEL)

        # 创建一个颜色调色板，用于固定每个类别的颜色
        # num_classes = len(self.yolo_model.names)
        # self.palette = sv.ColorPalette()  # 不带参数



    def get_yolo_boxes(self, image_path):
        # 将图像转换为 NumPy 数组
        image_np = image_path

        # 使用 YOLO 模型进行检测
        results = self.yolo_model(image_np, conf=0.5)

        # 获取检测结果
        detections = results[0]

        # 提取边界框、置信度、类别 ID
        boxes = detections.boxes.xyxy.cpu().numpy()  # (n, 4)
        scores = detections.boxes.conf.cpu().numpy()  # (n,)
        class_ids = detections.boxes.cls.cpu().numpy().astype(int)  # (n,)

        # 将类别 ID 映射到类别名称
        class_names = [self.yolo_model.names[class_id] for class_id in class_ids]

        # 创建一个结果字典，包含所有检测到的对象
        result = [{
            "boxes": torch.tensor(boxes),
            "labels": class_names,
            "scores": torch.tensor(scores),
            "class_ids": class_ids  # 添加 class_ids
        }]

        return result

    def get_sam2_masks(self, image_path, boxes):
        self.SAM2_predictor.set_image(image_path)

        masks, scores, logits = self.SAM2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, scores, logits

    def get_finesegment_feature_field(self, image_path, text, visualize=True, save_path="", target_classes=['floor', 'steps', 'slope'], path=None):
        try:
            # Use the YOLO detection method
            prediction = self.get_yolo_boxes(image_path)
            boxes = prediction[0]["boxes"].cpu().numpy()
            masks, _, _ = self.get_sam2_masks(image_path, boxes)

            # 确保检测结果不受 target_classes 影响
            # 仅在生成 target_mask 时根据 target_classes 过滤

            if target_classes is not None:
                class_ids = prediction[0]["class_ids"]  # 获取类别 ID
                class_names = prediction[0]["labels"]  # 获取类别名称
                target_mask = np.zeros(masks.shape[1:], dtype=np.uint8)  # 创建一个空掩码

                for idx, (class_id, class_name) in enumerate(zip(class_ids, class_names)):
                    if class_name in target_classes:  # 检查是否是目标类别
                        target_mask |= masks[idx].astype(np.uint8)  # 合并目标类别的掩码

                # --- 新增代码开始 ---

                # 将掩码转换为 0 和 255
                target_mask_uint8 = (target_mask * 255).astype(np.uint8)

                # 定义形态学操作的核
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

                # 进行闭运算，填充小的孔洞
                closed_mask = cv2.morphologyEx(target_mask_uint8, cv2.MORPH_CLOSE, kernel)

                # 将处理后的掩码转换回 0 和 1
                cleaned_mask = (closed_mask > 0).astype(np.uint8)
                # --- 新增代码结束 ---

                # 将清理后的掩码用于后续处理
                path = pathplan(cleaned_mask)

            else:
                target_mask = None

            return draw_path_on_image(self.save_visualization(image_path, masks, prediction, visualize, save_path), path), target_mask

        except Exception as e:
            print(f"Error: {e}")
            return image_path, None

    def save_visualization(self, image_path, masks, prediction, visualize, save_path):
        img = np.array(image_path)
        input_boxes = prediction[0]["boxes"].cpu().numpy()
        confidences = prediction[0]["scores"].cpu().numpy().tolist()
        class_names = prediction[0]["labels"]
        class_ids = prediction[0]["class_ids"]  # 使用实际的类别 ID

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids  # 使用实际的类别 ID
        )

        # 使用固定的颜色调色板
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))

        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        masked_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # Optionally save the visualization
        if save_path:
            cv2.imwrite(save_path, masked_frame)

        return masked_frame
