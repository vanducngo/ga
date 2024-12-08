import numpy as np
import torch
import math


def pytorch_switch(tensor_image):
    return tensor_image.permute(1, 2, 0)


def to_pytorch(tensor_image):
    return torch.from_numpy(tensor_image).permute(2, 0, 1)

'''
Tấn công đối kháng phi mục tiêu (untargeted adversarial attack)
Tấn công này cố gắng thay đổi dự đoán của mô hình để nhãn đầu ra khác với 
nhãn đúng (true), mà không yêu cầu nhãn đầu ra phải là một nhãn cụ thể.
'''
class UnTargeted:
    def __init__(self, model, true, unormalize=False, to_pytorch=False):
        # Mô hình học máy (machine learning model) được tấn công
        # Phương thức predict của mô hình sẽ được gọi để tính toán dự đoán cho một hình ảnh đầu vào.
        self.model = model
        
        # Nhãn đúng của hình ảnh đầu vào (dưới dạng số nguyên).
        # Được sử dụng để kiểm tra xem dự đoán của mô hình có khác với nhãn đúng không (phát hiện tấn công đối kháng).
        self.true = true

        # Boolean xác định xem hình ảnh có cần được chuyển đổi từ dạng chuẩn hóa về dạng ban đầu (0-255) không.
        self.unormalize = unormalize
        # Boolean xác định xem hình ảnh có cần được chuyển đổi thành tensor phù hợp với mô hình PyTorch không.
        self.to_pytorch = to_pytorch

    # Lấy nhãn dự đoán (y) của mô hình cho một hình ảnh (img).
    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    # Kiểm tra xem hình ảnh có phải là mẫu đối kháng (adversarial example) không.
    # Tính toán một giá trị định lượng sự khác biệt giữa nhãn đúng và nhãn có xác suất cao nhất khác (f_true - f_other).
    def __call__(self, img):

        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y != self.true else False

        # Tính toán giá trị log của xác suất liên quan đến nhãn đúng
        # Bảo vệ chống lỗi số học bằng cách thêm 1e-30.
        f_true = math.log(math.exp(preds[self.true]) + 1e-30)
        
        # Loại bỏ xác suất liên quan đến nhãn đúng
        preds[self.true] = -math.inf
        # Tìm xác suất cao nhất trong các nhãn còn lại
        f_other = math.log(math.exp(max(preds)) + 1e-30)

        # f_true - f_other: Chênh lệch giữa giá trị xác suất log của nhãn đúng và nhãn có xác suất cao nhất khác.
        return [is_adversarial, float(f_true - f_other)]

'''
Tấn công đối kháng có mục tiêu (targeted adversarial attack)
Mục tiêu là làm thay đổi dự đoán của mô hình thành một nhãn cụ thể (target), thay vì chỉ khác với nhãn đúng (true).
'''
class Targeted:
    def __init__(self, model, true, target, unormalize=False, to_pytorch=False):
        self.model = model
        self.true = true
        # target: Nhãn mục tiêu mong muốn mà tấn công đối kháng cố gắng đạt được.
        self.target = target
        self.unormalize = unormalize
        self.to_pytorch = to_pytorch

    def get_label(self, img):
        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        return y

    def __call__(self, img):

        if self.unormalize:
            img_ = img * 255.

        else:
            img_ = img

        if self.to_pytorch:
            img_ = to_pytorch(img_)
            img_ = img_[None, :]
            preds = self.model.predict(img_).flatten()
            y = int(torch.argmax(preds))
            preds = preds.tolist()
        else:
            preds = self.model.predict(np.expand_dims(img_, axis=0)).flatten()
            y = int(np.argmax(preds))

        is_adversarial = True if y == self.target else False
        #print("current label %d target label %d" % (y, self.target))
        f_target = preds[self.target]
        #preds[self.true] = -math.inf

        f_other = math.log(sum(math.exp(pi) for pi in preds))
        return [is_adversarial, f_other - f_target]
