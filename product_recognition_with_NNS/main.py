import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf
from models import dinoV2small_feature_extractor, dinoV3small_feature_extractor, resnet18_feature_extractor, get_proto_dataset, get_query_dataset, resnet50_feature_extractor
from logs import logger
from scipy.spatial.distance import cdist

# UTILITY
def extract_embeddings(model, dataset):
    """
    Estrae gli embeddings passando tutto il dataset attraverso il modello.
    Ritorna:
        embeddings : np.ndarray  (N, D)  N = numero di immagini, D = dimensione dell'embedding
        labels     : np.ndarray  (N,)
    """
    all_embeddings, all_labels = [], []
    for images, labels in dataset:
        embs = model(images, training=False)
        all_embeddings.append(embs.numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

def cosine_similarity_matrix(query_embs, gallery_embs):
    """
    Matrice di cosine similarity (N_query × N_gallery).
    Uso tf per sfruttare la GPU se disponibile.
    """
    query_embs   = tf.nn.l2_normalize(query_embs,   axis=1)
    gallery_embs = tf.nn.l2_normalize(gallery_embs, axis=1)
    return tf.linalg.matmul(query_embs, gallery_embs, transpose_b=True).numpy()

def nearest_neighbour_predict(sim_matrix, gallery_labels):
    """
    Per ogni query prende l'indice del prototipo più simile
    e restituisce la label corrispondente.
    """
    nn_indices = np.argmax(sim_matrix, axis=1)
    return gallery_labels[nn_indices]

def compute_inv_cov_pca(embs, labels, n_components=50):
    """
    Riduce la dimensionalità con PCA prima di stimare la covarianza,
    rendendo la matrice ben condizionata e invertibile in modo stabile.
    """
    unique_labels = np.unique(labels)
    dof = len(embs) - len(unique_labels)  # gradi di libertà effettivi

    # Verifica che n_components sia sostenibile
    if n_components >= dof:
        print(f"Attenzione: n_components={n_components} >= gradi_libertà={dof}. "
              f"Ridotto a {dof // 4}.")
        n_components = dof // 4

    # 1. PCA fit sui prototipi
    pca = PCA(n_components=n_components, random_state=42) # random_state=42 garantisce riproducibilità indipendentemente dal solver -> deterministico
    embs_pca = pca.fit_transform(embs)  # (N, 384) → (N, n_components)
    # varianza = np.cumsum(pca.explained_variance_ratio_)[-1] * 100
    # print(f"  [PCA={n_components}] Varianza spiegata: {varianza:.2f}%")

    # 2. Centratura per classe nello spazio ridotto
    centered = np.vstack([
        embs_pca[labels == c] - embs_pca[labels == c].mean(axis=0)
        for c in unique_labels
    ])
    
    # 3. Covarianza (n_components × n_components) → ben stimabile
    cov = np.cov(centered, rowvar=False)
    
    # Diagnosi per controllare la stabilità numerica della matrice di covarianza
    # eigenvalues = np.linalg.eigvalsh(cov)
    # print(f"  [PCA={n_components}] Autovalore min: {eigenvalues.min():.6f}")
    # print(f"  [PCA={n_components}] Autovalore max: {eigenvalues.max():.6f}")
    # print(f"  [PCA={n_components}] Condition number: {eigenvalues.max()/max(eigenvalues.min(), 1e-10):.2f}")
    
    inv_cov = np.linalg.inv(cov + np.eye(n_components) * 1e-5)
    return pca, inv_cov

def mahalanobis_distance(query_embs, proto_embs, proto_labels, pca, inv_cov):
    query_embs = pca.transform(query_embs)
    proto_embs = pca.transform(proto_embs)
    
    distances = cdist(query_embs, proto_embs, metric='mahalanobis', VI=inv_cov)
    return proto_labels[np.argmin(distances, axis=1)]

def print_results(true_labels, pred_labels, class_names):
    """ 
    Stampa risultati delle accuracy globali e per classe in una tabella ordinata. 
    """
    print("\n=== RISULTATI ===")
    methods = list(pred_labels.keys())
    total   = len(true_labels)
    col_w   = 22

    print(f"\n  {'Classe':<27}", end="")
    for m in methods:
        print(f"{m:^{col_w}}", end="")
    print()
    print("  " + "-" * (27 + col_w * len(methods)))

    print(f"  {'OVERALL':<27}", end="")
    for m in methods:
        acc = np.sum(true_labels == pred_labels[m]) / total * 100
        print(f"{acc:^{col_w}.2f}", end="")
    print()
    print("  " + "-" * (27 + col_w * len(methods)))

    for i, name in enumerate(class_names):
        mask = (true_labels == i)
        if mask.sum() == 0:
            continue
        print(f"  [{i:2d}] {name:<23}", end="")
        for m in methods:
            cls_acc = np.mean(pred_labels[m][mask] == i) * 100
            print(f"{cls_acc:^{col_w}.2f}", end="")
        print()

# ENROLLMENT PHASE
def enrollment_phase(model, n_components=50):
    """
    Carica i prototipi una sola volta e restituisce
    sia la gallery completa sia quella media, con la stessa inv_cov.
    """
    print("\n=== ENROLLMENT PHASE ===")
    proto_dataset, class_names = get_proto_dataset(shuffle=False)
    embs, labels = extract_embeddings(model, proto_dataset)

    unique_labels = np.unique(labels)

    # PCA + inv_cov calcolati sui prototipi RAW (più campioni → stima migliore)
    pca, inv_cov = compute_inv_cov_pca(embs, labels, n_components)

    # Gallery A: tutti i prototipi
    gallery_all  = (embs, labels)

    # Gallery B: media per classe
    mean_embs    = np.stack([embs[labels == c].mean(axis=0) for c in unique_labels])
    gallery_mean = (mean_embs, unique_labels)

    print(f"  Classi trovate ({len(class_names)}): {class_names}")
    print(f"  Gallery (all):  {embs.shape[0]} prototipi, dim = {embs.shape[1]}")
    print(f"  Gallery (mean): {len(unique_labels)} prototipi, dim = {mean_embs.shape[1]}")
    return gallery_all, gallery_mean, class_names, pca, inv_cov

# INFERENCE PHASE
def inference_phase(model, gallery_all, gallery_mean, pca, inv_cov):
    """
    Estrae gli embedding delle query e le confronta contro le due gallery
    con cosine similarity e distanza di Mahalanobis (con PCA).
    Ritorna true_labels e un dizionario {metodo: pred_labels}.
    """
    print("\n=== INFERENCE PHASE ===")
    query_dataset, _ = get_query_dataset(shuffle=False)
    query_embs, true_labels = extract_embeddings(model, query_dataset)
    print(f"  Query set: {query_embs.shape[0]} immagini")

    pred_labels = {
        "all  | cosine": nearest_neighbour_predict(cosine_similarity_matrix(query_embs, gallery_all[0]), gallery_all[1]),
        "all  | mahalanobis": mahalanobis_distance(query_embs, gallery_all[0], gallery_all[1], pca, inv_cov),
        "mean | cosine": nearest_neighbour_predict(cosine_similarity_matrix(query_embs, gallery_mean[0]), gallery_mean[1]),
        "mean | mahalanobis": mahalanobis_distance(query_embs, gallery_mean[0], gallery_mean[1], pca, inv_cov),
    }
    
    return true_labels, pred_labels

def main():

    backbones = {
        resnet18_feature_extractor,
        resnet50_feature_extractor,
        dinoV2small_feature_extractor,
        dinoV3small_feature_extractor,
    }

    for backbone in backbones:

        # 1. Extractor: costruisce il modello di feature extraction.
        model = backbone()
        logger(model, logs_dir="outputs")
        model.summary()

        # Componenti PCA da testare
        pca_components = 50

        # 2. Enrollment: costruisce la gallery degli embeddings dai prototipi.
        gallery_embs, gallery_mean, class_names, pca, inv_cov = enrollment_phase(model, n_components=pca_components)

        # 3. Inference: per ogni query estraiamo l'embedding e lo confrontiamo con la gallery usando diverse metriche.
        true_labels, pred_labels = inference_phase(model, gallery_embs, gallery_mean, pca, inv_cov)

        # 4. Stampa i risultati
        print_results(true_labels, pred_labels, class_names)


if __name__ == "__main__":
    main()