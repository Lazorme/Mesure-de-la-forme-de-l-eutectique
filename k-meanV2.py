import numpy as np
import cv2
import matplotlib.pyplot as plt

def kmeans_image_segmentation(image, k):
    image = cv2.imread(image,0)
    # Obtenir les dimensions de l'image
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

    # Appliquer l'algorithme des k-moyennes
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convertir les centres des clusters en valeurs d'intensité d'image
    centers = np.uint8(centers)

    # Réaffecter les pixels de l'image en utilisant les labels des clusters
    segmented_image = centers[labels.flatten()].reshape(image.shape)

    return segmented_image

def sharpen_image(sharpen_image):
    # Définir le noyau du filtre de netteté
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Appliquer le filtre de netteté à l'image
    sharpened = cv2.filter2D(sharpen_image, -1, kernel)
    return sharpened

def segment_image(segmented_image, threshold):

    #Image plus nette
    cv2.imshow("image de base",segmented_image)

    #net= sharpen_image(image)
    gray = cv2.equalizeHist(segmented_image)
    cv2.imshow('intput',gray)

    # Appliquer un seuillage pour segmenter les taches foncées
    _, segmented = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    #adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # Effectuer une opération de morphologie pour améliorer la segmentation
    kernel = np.ones((5, 5), np.uint8)
    segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)

    # Trouver les contours des taches segmentées
    #canny = cv2.Canny(segmented, 30, 150)
    #cv2.imshow("canny",canny)
    contours, _ = cv2.findContours(segmented, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(segmented)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), -1)

    # Dessiner les contours et attribuer un numéro à chaque tache
    output = cv2.cvtColor(segmented, cv2.COLOR_GRAY2BGR)
    bluecount = 0
    greencount = 0
    yellowcount = 0
    redcount = 0
    for i, contour in enumerate(contours):
        # Calculer l'aire de la tache
        A1 = cv2.contourArea(contour)
        if 2500 >A1 > min_area:
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

            cv2.drawContours(output, [contour], -1, color, -2)
            cv2.fillPoly(output, [contour], color)
    total = bluecount+greencount+yellowcount+redcount
    bleu = bluecount / total
    vert = greencount / total
    jaune = yellowcount / total
    rouge = redcount / total
    print (bleu,vert,jaune,rouge)

    #Création de l'histogramme
    data = [bleu,vert,jaune,rouge]
    barres = ["bleu","vert","jaune","rouge"]
    colors = ['blue', 'green', 'yellow', 'red']
    plt.bar(barres,data,color=colors)
    plt.ylabel('Proportions')
    plt.title('Histogramme des proportions')
    plt.show()

    output = cv2.resize(output, nouvelle_taille)
    cv2.imshow('Segmented Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Charger l'image
image='28juin-x10_01.jpg'



#Variables
threshold = 30
min_area=100
scale_factor= 1.5
nouvelle_taille = (1536, 912)

# Appliquer l'algorithme des k-moyennes pour segmenter l'image
k = 2  # Nombre de clusters
segmented_image = kmeans_image_segmentation(image, k)
segmented_image = sharpen_image(segmented_image)
segmented_image = segment_image(segmented_image, threshold)

# Afficher l'image originale et l'image segmentée
cv2.imshow("Image originale", image)
cv2.imshow("Image segmentée", segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()