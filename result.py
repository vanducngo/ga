import numpy as np
for i in range (0, 100):
    data = np.load(f"ResultUntargeted/result_untargeted_{i}.npy", allow_pickle=True).item()
    # Truy xuất thông tin từ file
    front0_imgs = data["front0_imgs"]  # Hình ảnh đối kháng (numpy array)
    true_label = data["true_label"]    # Nhãn đúng của hình ảnh gốc
    adversarial_labels = data["adversarial_labels"]  # Nhãn đối kháng
    
    isSuccess = False
    for i, img in enumerate(front0_imgs):
        label = data['adversarial_labels'][i]
        if (int(label) != int(true_label)):
            isSuccess = True

    print(f'------------------Image {i}--------------------')
    print("True Label:", true_label)
    print("Adversarial Labels:", adversarial_labels)
    print("Success:", isSuccess)
    print('------------------------------------------------')


# originalImage = x_test[20]
# data = np.load(f"ResultUntargeted/result_untargeted_20.npy", allow_pickle=True).item()
# for i, img in enumerate(front0_imgs):
#     plt.imshow(img)
#     plt.title(f"Adversarial Image {i+1}, Label: {data['adversarial_labels'][i]}")
#     plt.axis("off")
#     plt.show()