import argparse
import numpy as np
import cv2 as cv

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    # lib para leitura de conteúdo de urls
    import urllib.request as request

    # captura o array referente a imagem e converte pra um array numpy
    with request.urlopen(url) as response:
        img_array = np.asarray(bytearray(response.read()), np.uint8)

    # converte para o formato de imagem (reshape do array)
    image = cv.imdecode(img_array, **kwargs)

    ### END CODE HERE ###
    
    return image

url = "https://images.unsplash.com/photo-1740094714220-1b0c181be46d?q=80&w=2083&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

image = load_image_from_url(url, flags=cv.IMREAD_COLOR)
cv.imwrite("url_image.jpg", image)
