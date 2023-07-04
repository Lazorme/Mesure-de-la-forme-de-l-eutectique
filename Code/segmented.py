import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_image_segmentation(image, k):
    hauteur, largeur = image.shape
    print(hauteur, largeur)
    x1 = largeur
    y1 = 0
    x2 = largeur//5
    y2 = (hauteur//3)*2
    image = image[y1:y2, x2:x1]

    # Convertir l'image en une matrice de points de données
    data = image.reshape((-1, 3)).astype(np.float32)

    # Définir les critères d'arrêt pour l'algorithme des k-moyennes
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)

    # Appliquer l'algorithme des k-moyennes
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convertir les centres des clusters en valeurs d'intensité d'image
    centers = np.uint8(centers)

    # Réaffecter les pixels de l'image en utilisant les labels des clusters
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    return segmented_image

def sharpen_image(image):
    # Définir le noyau du filtre de netteté
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Appliquer le filtre de netteté à l'image
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def segment_image(image,seuil):

    if seuil==int(1):
        seuil=int(1)
    else :
        seuil=int(seuil)

    #Seuillage
    if seuil ==1:
        # Calculer l'histogramme
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
         # Trouver la valeur du pic le plus élevé de l'histogramme
        _, max_val, _, _ = cv2.minMaxLoc(hist)
        # Trouver la position du pic le plus élevé
        max_index = np.where(hist == max_val)[0][0]
        _, segmented = cv2.threshold(image, max_index-40, 255, cv2.THRESH_BINARY)
    else :
      _, segmented = cv2.threshold(image, seuil, 255, cv2.THRESH_BINARY)

    cv2.imshow('imageNoirbalnc',segmented)

    # Effectuer une opération de morphologie pour améliorer la segmentation
    kernel = np.ones((5, 5), np.uint8)
    segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)

    # Trouver les contours des taches segmentées
    contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Dessiner les contours et attribuer un numéro à chaque tache
    output = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    bluecount = 0
    greencount = 0
    yellowcount = 0
    redcount = 0
    for i, contour in enumerate(contours):
        # Calculer l'aire de la tache
        A1 = cv2.contourArea(contour)
        if 7000 >A1 > min_area:
            # Dessiner le contour et attribuer un numéro à la tache
            #cv2.putText(output, str(i+1), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Afficher l'aire de la tache
            print("Aire de la tache", i+1, ":", A1)

            # Trouver le cercle d'encadrement le plus petit (centre, rayon)
            (x, y), R2 = cv2.minEnclosingCircle(contour)
            R2 = int(R2)  # Convertir en entier

            #Calcul du fcateur 
            A2=np.pi*R2*R2
            Compare=A1/A2
            #print(Compare)

            #Repartition des couleurs
            if 0.81<Compare<1:
                color = [255, 0, 0]  # Bleu (RVB) 
                bluecount += 1
            elif 0.61<Compare<0.80:
                color = [0,255,0]  #Vert
                greencount += 1 
            elif 0.41<Compare<0.60:
                color = [0, 255, 255]  # Jaune (RVB)
                yellowcount += 1
            else :
                color =[0,0,255] #Rouge
                redcount += 1

            cv2.drawContours(output, [contour], -1, color, 2)
            cv2.fillPoly(output, [contour], color)
            # Dessiner le cercle d'encadrement
            #cv2.circle(output, (int(x), int(y)), R2, (0, 255, 0), 2)
            # Afficher l'image segmentée avec les numéros de tache

            #Calcul des proportions
    total = bluecount+greencount+yellowcount+redcount
    bleu = bluecount / total
    vert = greencount / total
    jaune = yellowcount / total
    rouge = redcount / total
    #print (bleu,vert,jaune,rouge)
    # Créer le subplot avec une disposition en deux colonnes
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Superposer les deux premières images rectangulaires
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), alpha=0.5)
    axs[0].axis('off')
    axs[0].set_title('Classification')

    # Ajuster les espacements entre les sous-graphiques
    plt.subplots_adjust(wspace=0.2)

    # Afficher l'histogramme des proportions
    data = [bleu, vert, jaune, rouge]
    barres = ["bleu", "vert", "jaune", "rouge"]
    colors = ['blue', 'green', 'yellow', 'red']
    axs[1].bar(barres, data, color=colors)
    axs[1].set_ylabel('Proportions')
    axs[1].set_title('Histogramme des proportions')

    # Afficher les deux figures côte à côte
    plt.show()
    #output = cv2.resize(output, nouvelle_taille)
    #cv2.imshow('Segmented Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main (image,seuil,k):
    image = cv2.imread(image, 0)
    image = cv2.equalizeHist(image)
    image=sharpen_image(image)
    imageTraiter=kmeans_image_segmentation(image,k)
    hauteur, largeur = image.shape
    print(hauteur, largeur)
    x1 = largeur
    y1 = 0
    x2 = largeur//5
    y2 = (hauteur//3)*2
    image = image[y1:y2, x2:x1]
    segment_image(imageTraiter,seuil)

# Variables globales
min_area=50
scale_factor= 1.5
nouvelle_taille = (1536, 912)
