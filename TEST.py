import numpy as np
import cv2

def kmeans_image_segmentation(image, k):
    # Convertir l'image en une matrice de points de données
    data = image.reshape((-1, 3)).astype(np.float32)

    # Définir les critères d'arrêt pour l'algorithme des k-moyennes
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    # Appliquer l'algorithme des k-moyennes
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convertir les centres des clusters en valeurs d'intensité d'image
    centers = np.uint8(centers)

    # Réaffecter les pixels de l'image en utilisant les labels des clusters
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    return segmented_image

# Charger l'image
image = cv2.imread('50.png')

# Appliquer l'algorithme des k-moyennes pour segmenter l'image
k = 5  # Nombre de clusters
segmented_image = kmeans_image_segmentation(image, k)

# Afficher l'image originale et l'image segmentée
cv2.imshow("Image originale", image)
cv2.imshow("Image segmentée", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
