from pathlib import Path
from ultralytics import YOLO
import cv2

# 1. 搞定基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

# 输入视频路径
video_url = BASE_DIR / "vedio" / "77b2903f4d0f839cf20375fa3ccfaa80.mp4"
# YOLO 模型路径
model_path = BASE_DIR / "runs" / "safety_helmet" / "yolo11n_baseline_v1" / "weights" / "best.pt"

model = YOLO(model_path)
cap = cv2.VideoCapture(str(video_url))

# ----------------- 视频保存准备工作 -----------------
# 2. 获取原视频的参数（帧率、宽度、高度）
# 这样能保证我们保存出来的视频和原视频播放速度、画面大小一模一样
fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率（比如一秒30帧）
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 画面宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 画面高度
size = (width, height)  # 画面尺寸元组

# 3. 指定输出视频的保存路径和格式
save_video_path = str(BASE_DIR / "vedio" / "output_annotated.mp4")

# 4. 设置视频编码器（FourCC）
# 'mp4v' 是 mp4 格式常用的编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 5. 创建视频写入器对象
# 参数依次是：保存路径、编码器、帧率、画面大小
out = cv2.VideoWriter(save_video_path, fourcc, fps, size)
# ----------------------------------------------------

print("正在处理并保存视频，请稍候...")
print("提示：在弹出的画面中，按下键盘 's' 键可以截图保存当前帧，按下 'q' 键退出。")

frame_id = 0  # 用来给截图命名

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("视频处理完毕！")
        break

    # YOLO 进行识别
    results = model(frame, conf=0.2, verbose=False)
    
    # 获取画好框的图片帧
    annotated_frame = results[0].plot()

    # 6. 把这一帧“画好框的图”写进视频文件里
    # 就像往相册里一张张插照片一样，最后会自动连成视频
    out.write(annotated_frame)

    # 在屏幕上实时显示出来看效果
    cv2.imshow("saving video demo", annotated_frame)

    # 监听键盘按键
    key = cv2.waitKey(1) & 0xFF
    
    # 7. 截图功能：如果用户按下 's' 键，就把当前画好框的帧保存为一张普通的 .jpg 图片
    if key == ord('s'):
        frame_id += 1
        screenshot_path = str(BASE_DIR / "vedio" / f"screenshot_{frame_id}.jpg")
        # cv2.imwrite 是 OpenCV 保存图片的函数，参数为：(保存路径, 要保存的图像矩阵)
        cv2.imwrite(screenshot_path, annotated_frame)
        print(f"成功保存截图到: {screenshot_path}")

    # 如果按下 'q' 键，退出播放
    elif key == ord('q'):
        print("用户主动退出。")
        break

# 8. 必须释放所有资源！
cap.release()  # 关掉输入视频
out.release()  # 关掉输出视频（这一步不执行，保存的视频会打不开或者损坏）
cv2.destroyAllWindows()  # 关闭弹窗窗口
