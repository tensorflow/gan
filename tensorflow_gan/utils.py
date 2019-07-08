import matplotlib.pyplot as plt


def display_gray_images(imgs, rows=4, cols=4):
    # Rescale image pixel values to [0, 1]
    imgs = 0.5 * imgs + 0.5

    # Set image grid
    _, axs = plt.subplots(rows, cols, figsize=(4, 4), sharey=True, sharex=True)

    cnt = 0
    for i in range(rows):
        for j in range(cols):
            # Output a grid of images
            axs[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

