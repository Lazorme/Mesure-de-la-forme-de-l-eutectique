import cv2
import numpy as np

def segment_image(image_path, threshold):
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, 0)

    # Appliquer un seuillage pour segmenter les taches foncées
    _, segmented = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Effectuer une opération de morphologie pour améliorer la segmentation
    kernel = np.ones((5, 5), np.uint8)
    segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)

    # Trouver les contours des taches segmentées
    contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Dessiner les contours et attribuer un numéro à chaque tache
    output = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        cv2.drawContours(output, [contour], -1, (255, 0, 0), 2)
        cv2.putText(output, str(i+1), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Calculer l'aire de la tache
        area = cv2.contourArea(contour)

        # Afficher l'aire de la tache
        print("Aire de la tache", i+1, ":", area)

        # Trouver le cercle d'encadrement le plus petit (centre, rayon)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)  # Convertir en entier


        # Afficher l'aire et le rayon maximal de la tache
        print("Aire de la tache", i+1, ":", area)
        print("Rayon maximal de la tache", i+1, ":", radius)

        # Dessiner le cercle d'encadrement
        cv2.circle(output, (int(x), int(y)), radius, (0, 255, 0), 2)
        # Afficher l'image segmentée avec les numéros de tache

    cv2.imshow('Segmented Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Chemin vers l'image d'entrée
image_path = 'image2.jpg'

# Seuil pour la segmentation 
threshold = 60

# Appeler la fonction pour segmenter l'image et calculer les aires
segment_image(image_path, threshold)
