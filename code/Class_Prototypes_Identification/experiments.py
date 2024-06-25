from Class_Prototypes_Identification import trained_models
from keyword_search import *
from text_to_image_retrieval import *
from torch.utils.data import DataLoader
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------------------------
    # class prototype identification and feature calculation
    source_dataset = "ARCH"
    # source_dataset = "OpenPath"
    source_dataset_path = f"/xx/xx/xx/{source_dataset}/"
    task = "breast_tumor_segmentation"
    # task = "mitotic_figure_detection"
    res_folder = f"../../{source_dataset}_{task}"
    os.makedirs(res_folder, exist_ok=True)
    retrieve_json = os.path.join(res_folder, f"{source_dataset}_{task}_retrieved_images.json")
    prototype_features_npy = os.path.join(res_folder, f"{source_dataset}_{task}_retrieved_image_features.npy")

    if source_dataset == "ARCH":  # ##################### keyword search ######################
        if task == "breast_tumor_segmentation":
            keyword_dict = {
                "location": {"with": ["breast"],
                             "without": []},
                "disease": {
                    "with": ["abnormal", "tumor", "cancer", "carcinoma", "metastases", "metastasis", "metastatic"],
                    "without": []},
                "stain": {"with": [],
                          "without": ["IHC", "immunohistochemical", "immunohistochemistry", "immunostain"]},
                "image_type": {"with": [],
                               "without": ["photomicrograph, photomicrography"]},
            }
        elif task == "mitotic_figure_detection":
            keyword_dict = {
                "location": {"with": ["arrow", "arrowhead", "circle"],
                             "without": []},
                "objectives": {"with": ["mitotic", "mitoses"],
                               "without": []},
            }
        else:
            raise NotImplementedError
        image_keyword_search(source_dataset_path=os.path.join(source_dataset_path, "books_set"),
                             keyword_dict=keyword_dict, retrieve_json=retrieve_json,
                             save_dir=os.path.join(res_folder, "keyword_search"))
        image_keyword_search(source_dataset_path=os.path.join(source_dataset_path, "pubmed_set"),
                             keyword_dict=keyword_dict, retrieve_json=retrieve_json,
                             save_dir=os.path.join(res_folder, "keyword_search"))

        # class prototypes feature calculation
        with open(retrieve_json, 'r') as f:
            res = json.load(f)  # {"ID": xx(int), "filename": xx(str), "caption": xx(str)}
        ds = sourceDataset(filenames=[r["filename"] for r in res], patch_size=256)
        dl = DataLoader(dataset=ds, num_workers=8, batch_size=16)

        encoder = "resnet18_SSL"
        weights_dir = "/xx/xx/trained_models/"
        # encoder = "resnet18"
        # weights_dir = None
        model, f_dim = trained_models.get_model(encoder, weights_dir)
        prototype_features = get_features(model, ds, dl, f_dim, device).cpu().numpy()  # n_samples, f_dim
        np.save(prototype_features_npy, prototype_features)

    elif source_dataset == "OpenPath":  # ##################### text_to_image_retrieval ######################
        if task == "breast_tumor_segmentation":
            hashtag = "BreastPath"
            classes = ["normal tissue", "breast tumor tissue"]
            model = trained_models.plip.text_encoder()
        elif task == "mitotic_figure_detection":
            raise f"OpenPath not suitable for {task}"
        else:
            raise NotImplementedError
        prototype_features, filenames = text_to_image_retrieval(source_dataset_path, hashtag, classes, model)
    else:
        raise ValueError
    np.save(prototype_features_npy, prototype_features)

    if source_dataset == "ARCH":  # Original images are not available in OpenPath
        # ---------------------------------------------------------------------------------------------------------------
        # Clustering retrieved images, for visualization purpose, e.g., removing irrelevant images
        cluster_labels = feature_clustering(prototype_features)
        # update source_json by adding cluster labels and show images in each cluster
        with open(retrieve_json, 'r') as f:
            res = json.load(f)
        for r in res:
            r.update({"cluster": int(cluster_labels[r["ID"]])})
        with open(retrieve_json, 'w') as f:
            json.dump(res, f)
        plot_images_in_clusters(res, os.path.join(res_folder, "clustering"))

        # ---------------------------------------------------------------------------------------------------------------
        # manual cleaning
        if task == "breast_tumor_segmentation":
            exclude_cluster = []
            exclude_images = [
                1, 4, 34, 35, 36, 42, 43,  # metastases to breast from other organs
                2,  # blurry
                6, 17, 25, 26, 27, 28, 29, 30, 31, 32, 45,  # irrelevant topics
                23, 24,  # irrelevant stains
                33, 38, 39, 40  # irrelevant image types
            ]

        elif task == "mitotic_figure_detection":
            exclude_cluster = []
            exclude_images = [
                0, 1,  # absence of mitoses
                2, 8, 23, 24, 25, 26, 31, 32, 38, 40,  # low image quality
                13, 14, 15, 22, 28, 29,  # irrelevant subfigure
                34, 35, 36, 37,  # photomicrgraph images not WSI
            ]
        else:
            raise NotImplementedError

        cleaned_res = []
        with open(retrieve_json, 'r') as f:
            res = json.load(f)
            for r in res:
                if r["ID"] not in exclude_images and r["cluster"] not in exclude_cluster:
                    cleaned_res.append({"ID": r["ID"],
                                        "filename": r["filename"],
                                        "caption": r["caption"]})

        plot_image_with_caption(cleaned_res, os.path.join(res_folder, "manual_cleaned"))
        cleaned_json = str(retrieve_json).replace(".json", "_manual_cleared.json")
        with open(cleaned_json, 'w') as f:
            json.dump(cleaned_res, f)
