import numpy as np
import matplotlib.pyplot as plt

# for i in range (0, 100):
#     # data = np.load(f"ResultUntargeted/result_untargeted_{i}.npy", allow_pickle=True).item()
#     data = np.load(f"Result2/result_targeted_{i}.npy", allow_pickle=True).item()
#     # Truy xuất thông tin từ file
#     front0_imgs = data["front0_imgs"]  # Hình ảnh đối kháng (numpy array)
#     true_label = data["true_label"]    # Nhãn đúng của hình ảnh gốc
#     adversarial_labels = data["adversarial_labels"]  # Nhãn đối kháng
    
#     isSuccess = False
#     for i, img in enumerate(front0_imgs):
#         label = data['adversarial_labels'][i]
#         if (int(label) != int(true_label)):
#             isSuccess = True

#     print(f'------------------Image {i}--------------------')
#     print("True Label:", true_label)
#     print("Adversarial Labels:", adversarial_labels)
#     print("Success:", isSuccess)
#     print('------------------------------------------------')



data = np.load(f"ResultUntargeted/result_untargeted_20.npy", allow_pickle=True).item()
plt.figure(figsize=(10, 6))
# for i in range(fitness_process.shape[1]):  # Nếu có nhiều mục tiêu (multi-objective)
#     plt.plot(fitness_process[:, i], label=f"Fitness Objective {i + 1}")

# # Tùy chỉnh biểu đồ
# plt.xlabel("Iterations")
# plt.ylabel("Fitness Value")
# plt.title("Fitness Process Over Iterations")
# plt.legend()
# plt.grid(True)
# plt.show()

# data = np.load(f"ResultUntargeted/result_untargeted_{i}.npy", allow_pickle=True).item()
data = np.load(f"Result3/result_targeted_70.npy", allow_pickle=True).item()
# Truy xuất thông tin từ file
front0_imgs = data["front0_imgs"]  # Hình ảnh đối kháng (numpy array)
x = data['x']
true_label = data["true_label"]    # Nhãn đúng của hình ảnh gốc
adversarial_labels = data["adversarial_labels"]  # Nhãn đối kháng

plt.figure(figsize=(10, 6))
for i, img in enumerate(front0_imgs):
    plt.imshow(img)
    plt.title(f"Adversarial Image {i+1}, Label: {data['adversarial_labels'][i]}")
    plt.axis("off")
    plt.show()

plt.imshow(x)
plt.title(f"Original Image H")
plt.axis("off")
plt.show()
# for i, img in enumerate(front0_imgs):
#     plt.imshow(img)
#     plt.title(f"Adversarial Image {i+1}, Label: {data['adversarial_labels'][i]}")
#     plt.axis("off")
#     plt.show()

