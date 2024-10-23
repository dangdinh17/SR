import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
# Đường dẫn đến thư mục chứa ảnh gốc DVI2K
sub = 'BSDS100'
input_dir = f'archive/DIV2K_train_HR/DIV2K_train_HR'
# hr_output_dir = f'benchmark/{sub}/HR'
# lr_output_dir = f'benchmark/{sub}/LR'
hr_output_dir = 'dataset/Train/HR'


# # Đảm bảo các thư mục đầu ra tồn tại
os.makedirs(hr_output_dir, exist_ok=True)
# os.makedirs(lr_output_dir, exist_ok=True)

# Kích thước cắt và giảm kích thước
crop_size = 96

num_patches_per_image = 250

# Lấy tất cả các tệp ảnh trong thư mục input

def process_image(image_path, hr_output_dir, lr_output_dir):
    # Mở ảnh
    img = Image.open(image_path)
    width, height = img.size

    # Kiểm tra xem việc chia xuống 4 lần có nguyên không
    lr_width = width // 4
    lr_height = height // 4

    # Tính lại kích thước HR bằng cách nhân kích thước LR lên 4 lần
    new_hr_width = lr_width * 4
    new_hr_height = lr_height * 4

    # Resize ảnh xuống kích thước LR
    img_lr = img.resize((lr_width, lr_height), Image.BICUBIC)

    # Resize ảnh LR lên kích thước HR mới (lấy kích thước nguyên)
    img_hr = img.resize((new_hr_width, new_hr_height), Image.BICUBIC)

    # Lưu ảnh LR và HR vào thư mục tương ứng
    img_name = os.path.basename(image_path)
    lr_save_path = os.path.join(lr_output_dir, img_name)
    hr_save_path = os.path.join(hr_output_dir, img_name)

    img_lr.save(lr_save_path)
    img_hr.save(hr_save_path)

    # print(f"Ảnh {img_name} đã được lưu vào:\n  - LR: {lr_save_path}\n  - HR: {hr_save_path}")

# Đường dẫn tới ảnh và thư mục lưu kết quả


def process_image_folder(input_folder, hr_output_dir, lr_output_dir):
    # Tạo thư mục nếu chưa có
    os.makedirs(hr_output_dir, exist_ok=True)
    os.makedirs(lr_output_dir, exist_ok=True)

    # Duyệt qua tất cả các tệp trong thư mục đầu vào
    for filename in tqdm(os.listdir(input_folder), unit='img'):
        image_path = os.path.join(input_folder, filename)
        # Kiểm tra nếu tệp là ảnh
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            process_image(image_path, hr_output_dir, lr_output_dir)



# Xử lý cả thư mục ảnh
# process_image_folder(input_dir, hr_output_dir, lr_output_dir)
def crop_random_patches(image, num_patches, crop_size):
    width, height = image.size
    patches = []
    for _ in range(num_patches):
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        patch = image.crop((left, top, right, bottom))
        patches.append(patch)
    return patches

# Duyệt qua từng ảnh 2K
image_files = os.listdir(input_dir)
for index, image_file in enumerate(image_files):

    image_path = os.path.join(input_dir, image_file)
    image = Image.open(image_path)

    # Cắt 500 ảnh nhỏ 100x100
    hr_patches = crop_random_patches(image, num_patches_per_image, crop_size)

    for i, hr_patch in enumerate(tqdm(hr_patches, desc=f'Epoch {index+1}/{len(image_files)}', unit='batch')):
        # Lưu ảnh 100x100 vào thư mục Train/HR
        hr_patch_name = f'{os.path.splitext(image_file)[0]}_patch_{i}.png'
        hr_patch_path = os.path.join(hr_output_dir, hr_patch_name)
        hr_patch.save(hr_patch_path)

        # Rescale ảnh 100x100 thành 25x25 và lưu vào thư mục Train/LR
        for scale in [2, 3, 4]:
            lr_size = crop_size // scale
            lr_output_dir = f'dataset/Train/LR_{scale}'
            lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)
            lr_patch_name = f'{os.path.splitext(image_file)[0]}_patch_{i}.png'
            lr_patch_path = os.path.join(lr_output_dir, lr_patch_name)
            lr_patch.save(lr_patch_path)

print("Cắt và lưu ảnh thành công!")
