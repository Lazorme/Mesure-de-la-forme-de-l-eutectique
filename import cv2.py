import cv2
import numpy as np
import matplotlib.pyplot as plt


def pixelize_image(image, scale_factor):

    # Déterminer la nouvelle taille de l'image en fonction du facteur de mise à l'échelle
    new_width = int(image.shape[1] / scale_factor)
    new_height = int(image.shape[0] / scale_factor)

    # Réduire la taille de l'image en utilisant l'interpolation NEAREST
    pixelized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    return pixelized_image


def sharpen_image(image):
    # Définir le noyau du filtre de netteté
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Appliquer le filtre de netteté à l'image
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def segment_image(image_path, threshold):

    #Image plus nette
    image=cv2.imread(image_path,0)
    cv2.imshow("image de base",image)

    # Obtenir les dimensions de l'image
    hauteur, largeur = image.shape
    print(hauteur, largeur)
    x1 = largeur
    y1 = 0
    x2 = largeur//5
    y2 = (hauteur//3)*2
    image = image[y1:y2, x2:x1]

    #net= sharpen_image(image)
    pix=pixelize_image(image,scale_factor)
    gray = cv2.equalizeHist(pix)
    cv2.imshow('intput',gray)



    # Charger l'image en niveaux de gris
    #image = cv2.imread(image_path, 0)
    #image = cv2.equalizeHist(image)
    #cv2.imshow('image', image)

    # Appliquer un seuillage pour segmenter les taches foncées
    #canny = cv2.Canny(gray, 30, 150)
    #cv2.imshow("canny",canny)
    _, segmented = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

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
        if 2500 >A1 > min_area:
            # Dessiner le contour et attribuer un numéro à la tache
            #cv2.putText(output, str(i+1), tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Afficher l'aire de la tache
            #print("Aire de la tache", i+1, ":", A1)

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

# Chemin vers l'image d'entrée
image_path = '28juin-x50_01.jpg'

# Variables globales
threshold = 60
min_area=100
scale_factor= 1.5
nouvelle_taille = (1536, 912)



# Feuillage adaptatif
#block_size = 61
#constant = 1
#K-Mean
#k=4
#attempt=10

# Appeler la fonction pour segmenter l'image et calculer les aires
segment_image(image_path, threshold)