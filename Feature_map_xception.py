import torch
from Xception.xception_py import Xception
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms,models

#图像预处理，要与生成alexnet.pth文件的train预处理一致
data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()])

# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
model = Xception(num_classes=5)
print(model)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./measure_xception/xception.pth"  # "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
img = Image.open("./001_01.bmp")
# print(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
# print(img)
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
# print(out_put)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]print(model)
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    # plt.figure()
    for i in range(64):
        # ax = plt.subplot(8, 8, i+1)#行，列，索引
        # [H, W, C]
        plt.axis('off')
        plt.savefig('./feature_map_xception_raw/' + str(i) + '.png')
        plt.imshow(im[:, :, i])#cmap默认为蓝绿图
        # print(im[:, :, i])
    # plt.show()