import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import os

def sample_image(model, language_model, output_image_dir, n_row, batches_done, dataloader):
    """Saves a grid of generated imagenet pictures with captions.
    """
    target_dir = os.path.join(output_image_dir, "samples/")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)

    captions = []
    # get sample captions
    for (_, captions_batch) in dataloader:
        captions += captions_batch
        if len(captions) > n_row ** 2:
            captions = captions[:n_row ** 2]
            break

    captions_embd = language_model(captions)
    gen_imgs = model.sample(captions_embd).cpu().numpy()
    gen_imgs = np.clip(gen_imgs, 0, 1)

    fig = plt.figure(figsize=((8, 8)))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_row, n_row), axes_pad=0.2)

    for i in range(n_row ** 2):
        grid[i].imshow(gen_imgs[i].transpose([1, 2, 0]))
        grid[i].set_title(captions[i])
        grid[i].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=True)

    save_file = os.path.join(target_dir, "{:013d}.png".format(batches_done))
    plt.savefig(save_file)
    print("saved  {}".format(save_file))
