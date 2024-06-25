
import numpy as np
import os
import pandas as pd
import torch


def text_to_image_retrieval(source_dataset_path, hashtag, classes, model):
    """
    OpenPath data structure
    
    set(df["source"]): {'Twitter reply', 'Twitter', 'PathLAION'}
    set(df.loc[data["source"]=="Twitter", "hashtag"]): 
        {'OralPath', 'ENTPath', 'SurgPath', 'nephpath',  'IDpath', 'PathGME', 'MolDx', 'PulmPath', 
        'BSTpath', 'dermpath', 'EndoPath', 'NeuroPath', 'blooducation', 'Autopsy', 'EyePath', 
        'patientbloodmanagement', 'CardiacPath', 'GUPath', 'pathInformatics', 'HPBpath', 'Gynpath', 
        'ForensicPath', 'ClinPath', 'HemePath', 'RenalPath', 'BloodBank', 'GIPath', 'PediPath', 
        'BreastPath', 'pancpath', 'Cytopath', 'FNApath'}
    set(df.loc[data["source"]=="Twitter reply", "hashtag"]): 
        {'OralPath', 'ENTPath', 'SurgPath', 'nephpath', 'IDpath', 'MolDx', 'PulmPath', 
        'BSTpath', 'dermpath', 'EndoPath', 'NeuroPath', 'blooducation', 'Autopsy', 'EyePath', 
        'CardiacPath', 'GUPath', 'HPBpath', 'Gynpath', 
        'ForensicPath', 'ClinPath', 'HemePath', 'RenalPath', 'BloodBank', 'GIPath', 'PediPath', 
        'BreastPath', 'pancpath', 'Cytopath', 'FNApath'}
    set(df.loc[data["source"]=="PathLAION", "hashtag"]):
        {'----'}
    """
    image_embeddings = torch.tensor(np.load(os.path.join(source_dataset_path,
                                                         "OpenPath_image_embeddings_normalized.npy")))  # (208414, 512)
    df = pd.read_csv(os.path.join(source_dataset_path, "dataframe_208K_rows.csv"),
                     header=None,
                     names=["id", "source", "hashtag", "web", "unknown1", "unknown2"],
                     usecols=["id", "source", "hashtag", "web"])

    breast_df = df.loc[df["hashtag"] == hashtag]  # DateFrame (7513, 4)
    breast_image_embeddings = image_embeddings[breast_df["id"].values].float()  # (7513, 512)

    prompts = [f"an H&E image of {c}" for c in classes]
    query_embeddings = model(prompts)  # (2, 512)
    query_embeddings /= torch.linalg.vector_norm(query_embeddings, dim=1, keepdim=True)

    # ref: https://github.com/openai/CLIP?tab=readme-ov-file#zero-shot-prediction
    cos_sim = (100. * breast_image_embeddings @ query_embeddings.t()).softmax(dim=1)  # (7513, 2)
    # assign a label for the image (e.g., normal/tumor), return also the similarity to the class
    values, indices = cos_sim.topk(1)  # (7513, 1), (7513, 1)
    values, indices = values.detach().cpu().numpy().flatten(), indices.detach().cpu().numpy().flatten()
    # select the 100 images with the highest feature similarity to the tumor tissue prompt
    most_similar = np.argsort(values[indices == 1])[-100:]

    retrieved_embedding = breast_image_embeddings[indices == 1][most_similar]  # (100, 512)
    retrieved_image_files = breast_df.loc[indices == 1, "web"].values[most_similar]
    return retrieved_embedding, retrieved_image_files

