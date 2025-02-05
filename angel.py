import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def segmentar_imagen(imagen, k):
    alto, ancho, canales = imagen.shape
    pixels = imagen.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    centros = kmeans.cluster_centers_
    
    labels_img = labels.reshape(alto, ancho)
    
    imagen_cuantizada = centros[labels].reshape(alto, ancho, 3).astype(np.uint8)
    
    return labels_img, imagen_cuantizada

def mostrar_resultados(imagen_original, mapa_segmentos, imagen_cuantizada, titulo_extra=""):
    plt.figure(figsize=(15,5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(imagen_original)
    plt.title("Imagen Original" + titulo_extra)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mapa_segmentos, cmap="viridis")
    plt.title("Mapa de Segmentos")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(imagen_cuantizada)
    plt.title("Imagen Cuantizada")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    imagenes = {
        '1': './imagenes/fachada.jpg',
        '2': './imagenes/gradiente.jpg',
        '3': './imagenes/pixelart.jpg',
        '4': './imagenes/solidos.jpg',
    }
    ruta_imagen = input("Selecciona una imagen (1-4): \n 1. Fachada \n 2. Gradiente \n 3. Pixelart \n 4. Solidos \n")
    ruta_imagen = imagenes.get(ruta_imagen)
    try:
        k = int(input("Ingrese el número de segmentos deseado (k >= 2): "))
    except ValueError:
        print("El valor de k debe ser un número entero.")
        return
    
    if k < 2:
        print("El valor de k debe ser mayor o igual a 2.")
        return
    
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        return
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    mapa_segmentos, imagen_cuantizada = segmentar_imagen(imagen_rgb, k)
    
    mostrar_resultados(imagen_rgb, mapa_segmentos, imagen_cuantizada)

if __name__ == "__main__":
    main()
