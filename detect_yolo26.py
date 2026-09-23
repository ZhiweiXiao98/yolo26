#!/usr/bin/env python
"""YOLO26 目标检测脚本."""

import os

from ultralytics import YOLO

# ======================== 配置参数 ========================
# 模型配置
MODEL_CONFIG = "ultralytics/cfg/models/26/yolo26.yaml"  # YOLO26检测模型配置
MODEL_WEIGHTS = "yolo26.pt"  # 预训练权重文件（如果有的话）

# 数据源配置 - 请将您的图片/视频放在这里
DATA_SOURCE = "data/images"  # 图片文件夹或单个图片/视频路径
# 示例：
# DATA_SOURCE = "image.jpg"           # 单个图片
# DATA_SOURCE = "video.mp4"           # 视频文件
# DATA_SOURCE = "data/images"         # 图片文件夹
# DATA_SOURCE = 0                      # 网络摄像头
# DATA_SOURCE = "https://..."         # 在线图片/视频URL

# 输出配置
OUTPUT_DIR = "runs/detect"  # 结果保存目录
CONF_THRESHOLD = 0.25  # 置信度阈值
IOU_THRESHOLD = 0.45  # IOU阈值


# ======================== 主函数 ========================
def main():
    print("=" * 50)
    print("YOLO26 目标检测")
    print("=" * 50)

    # 1. 加载模型
    print(f"\n📦 加载模型: {MODEL_WEIGHTS or MODEL_CONFIG}")
    try:
        if os.path.exists(MODEL_WEIGHTS):
            model = YOLO(MODEL_WEIGHTS)
        else:
            # 如果没有权重，从配置文件创建模型
            model = YOLO(MODEL_CONFIG)
            print("⚠️  使用配置文件创建新模型（未训练）")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 检查数据源
    print(f"\n📁 数据源: {DATA_SOURCE}")
    if not os.path.exists(DATA_SOURCE) and not str(DATA_SOURCE).startswith("http"):
        print("❌ 数据源不存在！")
        print("\n💡 请将您的图片或视频放在以下位置：")
        print(f"   {os.path.abspath(DATA_SOURCE)}")
        print("\n   示例：")
        print(f"   - 图片: {os.path.abspath(os.path.join(DATA_SOURCE, 'image.jpg'))}")
        print(f"   - 视频: {os.path.abspath(os.path.join(DATA_SOURCE, 'video.mp4'))}")
        return

    # 3. 运行推理
    print("\n🚀 开始推理...\n")
    try:
        results = model.predict(
            source=DATA_SOURCE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            save=True,
            project=OUTPUT_DIR,
            name="results",
            device=0,  # GPU:0，如果没有GPU改为'cpu'
            verbose=True,
        )

        # 4. 输出结果统计
        print("\n✅ 推理完成！")
        print("📊 结果统计:")
        for result in results:
            if result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"   - 检测到 {num_detections} 个目标")
                for box in result.boxes:
                    conf = box.conf.item()
                    cls_id = int(box.cls.item())
                    cls_name = result.names[cls_id]
                    print(f"     • {cls_name}: {conf:.2%}")

        print(f"\n💾 结果已保存到: {os.path.abspath(OUTPUT_DIR)}")

    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return


if __name__ == "__main__":
    main()
