import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import segmented


def charger_image():
    # Ouvrir la boîte de dialogue pour sélectionner un fichier
    fichier = filedialog.askopenfilename(filetypes=[("Fichiers image", "*.png;*.jpg;*.jpeg")])
    # Charger l'image sélectionnée
    if seuil_automatique.get():
        seuil = int(1)  # Utiliser le seuillage automatique
    else:
        seuil = entry_seuil.get()  # Récupérer la valeur du seuil entrée par l'utilisateur
    
    # Appeler la fonction de traitement d'image avec l'image chargée et le seuil
    traiter_image(fichier, seuil)

def traiter_image(image, seuil): 
    # Mettez ici votre code de traitement d'image
    # Par exemple, redimensionner l'image
    image = segmented.main(image,seuil,k=4)

def toggle_seuil():
    # Si la case "Seuil automatique" est cochée, désactiver la zone de texte du seuil
    if seuil_automatique.get():
        entry_seuil.configure(state="disabled")
    else:
        entry_seuil.configure(state="normal")


# Créer la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Mesure de la forme de l'eutectique")
fenetre.geometry("920x700")

# Créer un libellé pour le titre
label_titre = tk.Label(fenetre, text="", font=("Helvetica", 16, "bold"))
label_titre.pack(pady=20)

# Créer un libellé pour le titre
label_titre = tk.Label(fenetre, text="Traitement d'image", font=("Helvetica", 30, "bold"))
label_titre.pack(pady=20)

# Créer un bouton pour charger l'image
bouton_charger = tk.Button(fenetre, text="Charger une image", command=charger_image, font=("Helvetica", 12))
bouton_charger.pack(pady=20)

# Créer une case à cocher pour le seuil automatique
seuil_automatique = tk.BooleanVar()
checkbox_seuil_auto = tk.Checkbutton(fenetre, text="Seuil automatique", variable=seuil_automatique, command=toggle_seuil, font=("Helvetica", 12))
checkbox_seuil_auto.pack(pady=5)

# Créer un libellé et une zone de saisie pour le seuil
label_seuil = tk.Label(fenetre, text="Seuil :", font=("Helvetica", 12))
label_seuil.pack(pady=5)

entry_seuil = tk.Entry(fenetre, font=("Helvetica", 12))
entry_seuil.pack(pady=5)

# espace
label_titre = tk.Label(fenetre, text="", font=("Helvetica", 16, "bold"))
label_titre.pack(pady=20)

# Charger l'image
image_path = "logo.png"
image = Image.open(image_path)

# Convertir l'image en un format compatible avec Tkinter
image_tk = ImageTk.PhotoImage(image)

# Créer un widget Label pour afficher l'image
label = tk.Label(fenetre, image=image_tk)
label.pack()

# Démarrer la boucle principale de l'interface graphique
fenetre.mainloop()
