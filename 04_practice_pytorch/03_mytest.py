import  numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.rcParams["font.sans-selrif"]=["Microsoft YaHei"]
boxes = np.array([
    # 目标A: 猫, 位置(50,50)~(150,150), 模型给了 3 个高度重叠的框
    #         置信度各不相同 → NMS 应该只保留 0.95 那个
    [50, 50,   150, 150,  0.95, 0],
    [60, 55,   145, 148,  0.82, 0],   # 和上面那个 IoU 很高 → 会被删
    [45, 60,   155, 140,  0.71, 0],   # 同上
    # 目标B: 另一只猫, 位置偏右(130,40)~(220,140), 离目标A很近但不完全重叠
    [130, 40,  220, 140,  0.88, 0],
    [125, 45,  215, 135,  0.65, 0],   # 和目标B的第1个框重叠 → 会被删
    # 目标C: 狗, 位置(250,60)~(350,160), 跟猫离很远 → 不会受影响
    [250, 60,  350, 160,  0.91, 1],
    [255, 55,  345, 165,  0.78, 1],   # 和目标C的第1个框重叠 → 会被删
    # 背景误检: 置信度 0.42, 位置比较偏 → 正常的 conf 过滤就可能删掉它
    [300, 200, 340, 240,  0.42, 0],
])
# 类别名和颜色映射（方便画图时区分猫和狗）
names = {0: "猫", 1: "狗"}
colors = {0: "blue", 1: "orange"}
def compute_iou(b1,b2):
    x1=max(b1[0],b2[0])
    y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2])
    y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b2[3]-b2[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    area=a1+a2-inter
    return inter/(area+1e-8)
def nms(boxes,iou_thresh=0.5):
    order=np.argsort[:,4][::-1]
    keep=[]
    while len(order)>0:
        cur=order[0]
        keep.append(cur)
        if len(order)==1:
            break
        ious=[compute_iou(boxes[cur],boxes[i]) for i in order[1:]]
        order=order[1:][np.array(ious)<=iou_thresh]
    return keep
def nms(boxes,iou_thresh=0.5,sigma=0.5):
    boxes=boxes.copy().astype(float)
    order=list(np.argsor[:,4][::-1])
    keep=[]
    while len(order)>0:
        cur=order.pop[0]
        cur=order[0]
        keep.append(cur)
        for idx in list(order):
            iou_val=compute_iou(boxes[cur],order[idx])
            if iou_val>=iou_thresh:
                boxes[idx,4]*=mp.exp(-iou_val*iou_val/sigma)
fig,axes=plt.subplots(1,3,figsize=(16,5))
titles=["Nms","nms" ,"iou"]
for ax,title,keep in zip(axes,titles,[list(range(8),nms(boxes,0.5),(soft_nms(boxes,0.5)))]):
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 300)
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    for i,b in enumerate(boxes):
        x1,y1,x2,y2=b[:4]
            if i in keep:
                rect=patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,
                edgecolor=colors[int(b[5])]),facecolor="none"
                ax.text(x1,y1-5,f"{names[b[5]]}{b[4]:.2f}",fontsize=8,color=colors[b[5]])
      else:
            # ═════════════════════════════════════════════════════════
            # 情况 B: 框被干掉了 → 画灰色虚线
            # ═════════════════════════════════════════════════════════
            # edgecolor="gray"    → 灰色边框 = 被淘汰的标记
            # linewidth=1         → 细线，不抢眼
            # linestyle="--"      → 虚线，一眼看出是"已经不存在的框"
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=1,
                                      edgecolor="gray",
                                      facecolor="none",
                                      linestyle="--")
            ax.add_patch(rect)

    # ── 每张子图左下角标注统计 ──
    # 位置 (10, 275): x=10(靠左), y=275(靠下)
    # bbox=dict(...): 给文字画一个圆角背景框 → 更醒目
    ax.text(10, 275, f"保留 {len(keep)}/{len(boxes)} 框",
            bbox=dict(boxstyle="round", facecolor="wheat"))
       

    
