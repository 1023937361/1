import torch
from Inception_V3.inception_v3 import Inception3
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
model = Inception3(num_classes=5,aux_logits=False)
print(model)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./measure_inception_v3/inception-v3.pth"  # "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
img = Image.open("./001_01_standard.bmp")
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
    for i in range(32):
        # ax = plt.subplot(4, 8, i+1)#行，列，索引
        # [H, W, C]
        plt.axis('off')

        plt.imshow(im[:, :, i])#cmap默认为蓝绿图
        plt.savefig('./feature_map_inception_standard2/' + str(i) + '.png', pad_inches=0.0)
        # print(im[:, :, i])
    # plt.show()