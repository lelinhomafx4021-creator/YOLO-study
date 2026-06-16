import torch 
import torch.nn as nn
import numpy as np
import onnxruntime as ort 
import os
class TinyClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,8,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*8*8,10)

        )
    def forward(self,x):
        return self.net(x)
model=TinyClass()
model.eval()
dummy=torch.randn(1,3,32,32)
onnx_file="04_practice_pytoch/tinnycnn.onnx"
torch.onnx.export(model,dummy,onnx_file,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input":{0:"batch"},"output":{0:"batch}"}},opset_version=17)
print(f"ONNX导出{onnx_file},({os.path.getsize(onnx_file)/1024:.0f}KB)")
with torch.no_grad():
    pt_output=model(dummy)
    pt_output=pt_output.numpy()
print(pt_output)
session=ort.InferenceSession(onnx_file)
ort_output=session.run(None,{"input":dummy.numpy()})[0]
print(ort_output)
diff=np.abs(pt_output-ort_output).max()


