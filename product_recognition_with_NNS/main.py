import numpy as np
import tensorflow as tf
from models import dinoV2small_feature_extractor, dinoV3small_feature_extractor, resnet18_feature_extractor, get_proto_dataset, get_query_dataset, resnet50_feature_extractor
from logs import logger

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

# ENROLLMENT PHASE
def enrollment_phase(model):
    """
    Carica le immagini dalla cartella prototypes, ne estrae gli embeddings
    e li raccoglie in una gallery (matrice + vettore di label).
    """
    print("\n=== ENROLLMENT PHASE ===")
    proto_dataset, class_names = get_proto_dataset(shuffle=False)
    print(f"  Classi trovate ({len(class_names)}): {class_names}")

    gallery_embs, gallery_labels = extract_embeddings(model, proto_dataset)
    print(f"  Gallery: {gallery_embs.shape[0]} prototipi, dim embedding = {gallery_embs.shape[1]}")
    return gallery_embs, gallery_labels, class_names

def enrollment_phase_mean(model):
    """
    Variante: un solo embedding per classe (la media).
    Riduce i costi nella fase di inference.
    """
    print("\n=== ENROLLMENT PHASE (mean per class) ===")
    proto_dataset, class_names = get_proto_dataset(shuffle=False)
    embs, labels = extract_embeddings(model, proto_dataset)

    unique_labels = np.unique(labels)
    mean_embs = np.stack([embs[labels == cls].mean(axis=0) for cls in unique_labels])
    print(f"  Gallery (mean): {len(unique_labels)} classi, dim embedding = {mean_embs.shape[1]}")
    return mean_embs, unique_labels, class_names

# INFERENCE PHASE
def inference_phase(model, gallery_embs, gallery_labels):
    """
    Per ogni query image estrae l'embedding, cerca il nearest neighbour
    nella gallery e assegna la label corrispondente.
    """
    print("\n=== INFERENCE PHASE ===")
    query_dataset, _ = get_query_dataset(shuffle=False)
    query_embs, true_labels = extract_embeddings(model, query_dataset)
    print(f"  Query set: {query_embs.shape[0]} immagini")

    sim_matrix   = cosine_similarity_matrix(query_embs, gallery_embs)
    pred_labels  = nearest_neighbour_predict(sim_matrix, gallery_labels)
    return true_labels, pred_labels

def main():
    # 1. Extractor: costruisci il modello di feature extraction.
    model = dinoV2small_feature_extractor()
    logger(model, logs_dir="outputs")
    model.summary()

    # 2. Enrollment: costruisci la gallery degli embeddings dai prototipi.
    gallery_embs, gallery_labels, class_names = enrollment_phase(model) # Opzione A: tutti i prototipi
    # gallery_embs, gallery_labels, class_names = enrollment_phase_mean(model) # Opzione B: media per classe -> più veloce

    # 3. Inference: NN search
    true_labels, pred_labels = inference_phase(model, gallery_embs, gallery_labels)

    # 4. Valutazione
    correct = np.sum(true_labels == pred_labels)
    total   = len(true_labels)
    acc     = correct / total * 100

    print("\n=== RISULTATI ===")
    print(f"  Accuracy: {acc:.2f}%  ({correct}/{total})")

    # Accuracy per classe
    print("\n  Accuracy per classe:")
    for i, name in enumerate(class_names):
        mask     = true_labels == i
        if mask.sum() == 0:
            continue
        cls_acc  = np.mean(pred_labels[mask] == i) * 100
        print(f"    [{i:2d}] {name:<25s} {cls_acc:6.2f}%")


if __name__ == "__main__":
    main()