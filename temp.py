import cv2
import numpy as np
import os
from PIL import Image
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torchvision.utils import save_image

from tqdm import tqdm
from models.srcnn import *
from models.vdsr import *

from models.sr_model import *
from models.hqsr import *
from models.vdsr import *
from models.srresnet_ import *
from models.sr_model import *
from models.vdsr import *
from models.utils import *
from models.srcnn import *
from models.edsr import *
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
# edsr_srcnn = EDSR_srcnnfy()
# edsr  = EDSR_orig().to(device)
edsr = EDSR().to(device)
edrn_canny = EDRN(use_canny=True).to(device)
edrn_sobel = EDRN(use_sobel=True).to(device)
srresnet = SRResNet().to(device)
vdsr = VDSR().to(device)
srcnn = SRCNN().to(device)

# edsr.load_state_dict(torch.load('weight/best_edsrx4_model.pth', map_location=device))
edsr.load_state_dict(torch.load(f'weight_pcb/best_edsr.pth', map_location=device))
edrn_sobel.load_state_dict(torch.load('weight_pcb/best_sobel_srx4_model.pth', map_location=device))
edrn_canny.load_state_dict(torch.load('weight_pcb/best_canny_srx4_model.pth', map_location=device))
srresnet.load_state_dict(torch.load('weight_pcb/best_srresnet.pth', map_location=device))
vdsr.load_state_dict(torch.load('weight_pcb/best_vdsr.pth', map_location=device))
srcnn.load_state_dict(torch.load('weight_pcb/best_srcnn.pth', map_location=device))
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def calculate_metrics(img1, img2, max_pixel_value=1.0):
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_value = psnr(img1, img2)
    ssim_value = ssim(img1, img2)
    return psnr_value, ssim_value

import random
import time
from PIL import ImageDraw
transform = transforms.ToTensor()

sub = 'pcb'
box = 80
valid_lr_dir = f'benchmark/{sub}/LR'
valid_hr_dir = f'benchmark/{sub}/HR'
output_image_dir = f'benchmark/output/{sub}'
os.makedirs(output_image_dir, exist_ok=True)
lr_image_dir = valid_lr_dir
hr_image_dir = valid_hr_dir
# index = random.randint(0, len(os.listdir(lr_image_dir))-1)
# file = '0875.png'
# lr_image_file = file
# hr_image_file = file
with torch.no_grad():
    for (lr_image_file, hr_image_file) in zip(sorted(os.listdir(lr_image_dir)), sorted(os.listdir(hr_image_dir))):
        # Đường dẫn đến ảnh

        lr_image_path = os.path.join(lr_image_dir, lr_image_file)
        hr_image_path = os.path.join(hr_image_dir, hr_image_file)
        ####################################
        img = cv2.imread(hr_image_path)  # Thay bằng đường dẫn ảnh của bạn
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        drawing = False  # True khi đang nhấn chuột
        top_left_corner = [0,0]
        xy = [0, 0, 0, 0]
        def draw_rectangle(event, x, y, flags, param):
            global drawing, top_left_corner, img, xy
            
            # Khi nhấn chuột trái, bắt đầu vẽ
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                top_left_corner = [x, y]   # Lưu tọa độ của điểm bắt đầu
            
            # Khi thả chuột trái, kết thúc vẽ
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                
                xy[0] = top_left_corner[0]
                xy[1] = top_left_corner[1]
                xy[2] = top_left_corner[0] + box # type: ignore
                xy[3] = top_left_corner[1] + box
                
                cv2.destroyAllWindows()
                # Đóng cửa sổ ảnh sau khi vẽ xong
        # Đặt callback cho sự kiện chuột
        cv2.setMouseCallback("Image", draw_rectangle)

        # Hiển thị ảnh và đợi cho đến khi bounding box được vẽ
        cv2.waitKey(0)
        ####################################
        edsr_path = os.path.join(output_image_dir, lr_image_file[:-4] + '_edsr.jpg')
        edrn_sobel_path = os.path.join(output_image_dir, lr_image_file[:-4] + '_edrn_sobel.jpg')
        edrn_canny_path = os.path.join(output_image_dir, lr_image_file[:-4] + '_edrn_canny.jpg')
        bicubic_path = os.path.join(output_image_dir, lr_image_file[:-4] + '_bicubic.jpg')
        srresnet_path = os.path.join(output_image_dir, lr_image_file[:-4] + '_srresnet.jpg')
        srcnn_path = os.path.join(output_image_dir, lr_image_file[:-4] + '_srcnn.jpg')
        vdsr_path = os.path.join(output_image_dir, lr_image_file[:-4] + '_vdsr.jpg')

        hr_path = os.path.join(output_image_dir, hr_image_file)
        # Tải và chuyển đổi ảnh
        lr_image = Image.open(lr_image_path)
        hr_image = Image.open(hr_image_path)
        w, h = hr_image.size
        bicubic = lr_image.resize((w, h))
        lr_image_copy = lr_image.copy().convert('RGB')
        hr_image_copy = hr_image.copy()
        # lr_image_1dms = torch.Tensor(lr_image_1dms)
        lr_image = transform(lr_image).unsqueeze(0).to(device)  # Thêm batch dimension và chuyển sang CPU
        hr_image = transform(hr_image).unsqueeze(0).to(device)  # Thêm batch dimension và chuyển sang CPU
        bicubic_ = transform(bicubic).unsqueeze(0).to(device)
        # print(type(lr_image_1dms))
        
        
        
        # bicubic = lr_image_copy.resize((600, 600), resample=Image.BICUBIC) # type: ignore
        # bicubic_ = transform(bicubic).unsqueeze(0).to(device)
        # lr_image_1dms, _ = preprocess(lr_image_copy, device)
        # lr_image_copy, _ = preprocess(lr_image_copy, device)
        # hr_image_copy, _ = preprocess(HR, device)
        # _, ycbcr = preprocess(bicubic, device)
        time_pro = []
        def measure_inference_time(model, input_image, model_name):
            start_time = time.time()
            output = model(input_image)
            end_time = time.time()
            
            inference_time = end_time - start_time
            time_pro.append(inference_time)
            return output
        
        # Dự đoán
        output_edsr = edsr(lr_image)
        output_edrn_sobel = edrn_sobel(lr_image)
        output_edrn_canny = edrn_canny(lr_image)
        output_srresnet = srresnet(lr_image)
        output_srcnn = srcnn(bicubic_)
        output_vdsr = vdsr(bicubic_)
        models = ['EDSR', 'EDRN Sobel', 'EDRN Canny', 'SRResNet', 'SRCNN', 'VDSR']
        psnr_value = []  # Lưu giá trị PSNR cho mỗi mô hình
        ssim_value = []
        # Tính PSNR cho từng mô hình (giả sử output của các mô hình đã có)
        for model_output in [output_edsr, output_edrn_sobel, output_edrn_canny, output_srresnet, output_srcnn, output_vdsr]:
            # Tính PSNR và thêm vào danh sách
            psnr_value.append(calculate_metrics(model_output, hr_image)[0])
            ssim_value.append(calculate_metrics(model_output, hr_image)[1])
        # Ghi PSNR cho từng mô hình vào file
        with open(f"{output_image_dir}/results.txt", "a") as psnr_file:
            psnr_file.write(f"HR Image: {hr_image_file}\n")
            for i, model_name in enumerate(models):
                psnr_file.write(f"{model_name} PSNR/SSIM: {psnr_value[i]:.2f} dB/ {ssim_value[i]:.4f}\n")
            psnr_file.write(f"Bicubic PSNR/SSIM: {calculate_metrics(bicubic_, hr_image)[0]:.2f} dB/ {calculate_metrics(bicubic_, hr_image)[1]:.2f}\n")
            psnr_file.write("\n")  # Dòng trống giữa các lần lặp
            
            
        output_image_edsr = output_edsr.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
        output_image_edsr = transforms.ToPILImage()(output_image_edsr)  # Chuyển tensor thành ảnh PIL
        output_image_edsr = output_image_edsr.crop(xy)
        output_image_edsr.resize((100, 100))
        output_image_edsr.save(edsr_path)  # Lưu ảnh
        
        output_image_edrn_sobel = output_edrn_sobel.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
        output_image_edrn_sobel = transforms.ToPILImage()(output_image_edrn_sobel)  # Chuyển tensor thành ảnh PIL
        output_image_edrn_sobel = output_image_edrn_sobel.crop(xy)
        output_image_edrn_sobel.resize((100, 100))
        output_image_edrn_sobel.save(edrn_sobel_path)  # Lưu ảnh
        
        
        output_image_edrn_canny = output_edrn_canny.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
        output_image_edrn_canny = transforms.ToPILImage()(output_image_edrn_canny)  # Chuyển tensor thành ảnh PIL
        output_image_edrn_canny = output_image_edrn_canny.crop(xy)
        output_image_edrn_canny.resize((100, 100))
        output_image_edrn_canny.save(edrn_canny_path)  # Lưu ảnh
        
        output_image_srcnn = output_srcnn.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
        output_image_srcnn = transforms.ToPILImage()(output_image_srcnn)  # Chuyển tensor thành ảnh PIL
        output_image_srcnn = output_image_srcnn.crop(xy)
        output_image_srcnn.resize((100, 100))
        output_image_srcnn.save(srcnn_path)  # Lưu ảnh
        
        # output_vdsr = output_vdsr.mul(255.0).to(device).numpy().squeeze(0).squeeze(0)
        # output_image_vdsr = np.array([output_vdsr, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        # output_image_vdsr = np.clip(convert_ycbcr_to_rgb(output_image_vdsr ), 0.0, 255.0).astype(np.uint8)
        # output_image_vdsr = Image.fromarray(output_image_vdsr )
        # output_image_vdsr = output_image_vdsr.crop(xy)
        # output_image_vdsr .save(vdsr_path) 
        
        output_image_vdsr = output_vdsr.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
        output_image_vdsr = transforms.ToPILImage()(output_image_vdsr)  # Chuyển tensor thành ảnh PIL
        output_image_vdsr = output_image_vdsr.crop(xy)
        output_image_vdsr.resize((100, 100))
        output_image_vdsr.save(vdsr_path)  # Lưu ản
        
        output_image_srresnet = output_srresnet.squeeze(0).to(device)  # Loại bỏ batch dimension và chuyển tensor sang CPU
        output_image_srresnet = transforms.ToPILImage()(output_image_srresnet)  # Chuyển tensor thành ảnh PIL
        output_image_srresnet = output_image_srresnet.crop(xy)
        output_image_srresnet.resize((100, 100))
        output_image_srresnet.save(srresnet_path)  # Lưu ảnh
        
        # bicubic = lr_image.resize((600, 600))
        bicubic = bicubic.crop(xy)
        bicubic.resize((100, 100))
        bicubic.save(bicubic_path)
        
        hr_image_crop = hr_image_copy.crop(xy)
        hr_image_crop.resize((100, 100))
        hr_image_crop.save(hr_path)
        
        draw = ImageDraw.Draw(hr_image_copy)
        draw.rectangle(xy, outline="red", width=3)
        hr_image_copy = hr_image_copy.resize((600, 600))
        hr_image_copy.save(f"{output_image_dir}/{hr_image_file[:-4]}_with_bounding_box.jpg")
        print('Done')