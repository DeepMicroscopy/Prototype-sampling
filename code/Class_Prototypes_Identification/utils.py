import matplotlib.pyplot as plt
import numpy as np
import os.path
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class sourceDataset(Dataset):
    def __init__(self, filenames, patch_size, do_norm=False):
        super(sourceDataset, self).__init__()
        self.filenames = filenames

        self.base_transform = [
            transforms.CenterCrop(patch_size),
            transforms.ToTensor()
        ]

        if do_norm:
            self.base_transform.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            )

        self.base_transform = transforms.Compose(self.base_transform)

    def __getitem__(self, index):
        image_path = self.filenames[index]
        I = Image.open(image_path).convert("RGB")
        I = self.base_transform(I)
        return I, index

    def get_original_image(self, index):
        image_path = self.filenames[index]
        I = Image.open(image_path).convert("RGB")
        I = np.asarray(I) / 255.
        return I, index

    def __len__(self):
        return len(self.filenames)


def get_features(model, ds, dl, descriptors_dimension, device):
    model = model.eval().to(device)
    descriptors = torch.empty((len(ds), descriptors_dimension), device=device)

    with torch.inference_mode():
        for images, indices in dl:
            f = model(images.to(device))
            f = f.to(device)
            f /= torch.linalg.vector_norm(f, dim=-1, keepdim=True)
            descriptors[indices, :] = f.detach()
    torch.cuda.empty_cache()
    return descriptors


def feature_clustering(features):
    print(f"MiniBatchKMeans, determining the best n_clusters...")

    range_n_clusters = range(3, 11)
    silhouette_avg_list = []
    cluster_labels_list = []
    for n_clusters in range_n_clusters:
        cluster_labels = MiniBatchKMeans(n_clusters, n_init='auto').fit_predict(features)  # cluster index (n_samples,)
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        cluster_labels_list.append(cluster_labels)
    print(f"range_n_clusters: {range_n_clusters}, average silhouette_scores: {silhouette_avg_list}")
    print(f"--> best n_clusters = {range_n_clusters[np.argmax(silhouette_avg_list)]}")
    cluster_labels = cluster_labels_list[np.argmax(silhouette_avg_list)]

    return cluster_labels


def plot_images_in_clusters(data, save_dir):
    """

    :param data: a list of dicts, each dict: {"ID":xx, "filename":xx, "cluster":xx, "caption":xx}
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)
    cluster_labels = [r["cluster"] for r in data]
    for c in np.unique(cluster_labels):
        c_data = [r for r in data if r["cluster"] == c]
        indices = [r["ID"] for r in c_data]
        print(c, indices)

        fig, axes = plt.subplots(len(indices) // 5 + 1, 5, figsize=(20, 5 * (len(indices) // 5 + 1)))
        for ax in axes.ravel():
            ax.axis('off')

        for i, r in enumerate(c_data):
            I = Image.open(r["filename"]).convert("RGB")
            I = transforms.Compose([transforms.Resize((224, 224))])(I)
            axes.ravel()[i].imshow(I)
            axes.ravel()[i].set_title(r["ID"])

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"cluster_{c}.png"))
        plt.close()


def plot_image_with_caption(data, save_dir):
    """

    :param data: a list of dicts, each dict: {"ID":xx, "filename":xx, "caption":xx}
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)
    for d in data:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(plt.imread(d["filename"]))
        ax.set_title(d["caption"], wrap=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, str(d["ID"]) + ".png"))
        plt.close()
