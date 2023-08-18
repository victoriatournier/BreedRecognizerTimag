import joblib
import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from skimage.io import imread
import random

dir_imagenes = "images/"
dir_labels = "annotations/trimaps/"
dir_modelos = "modelos/"


def make_dataset(direccion_img, path_csv):
    """
    Crea un dataset a partir de las imágenes de una carpeta
    """
    db = pd.DataFrame(columns=["Family", "Breed", "Path"])
    images = os.listdir(direccion_img)

    for image in images:
        raza = image.rsplit("_", 1)[0]
        if raza[0].isupper():
            family = "Gato"
        else:
            family = "Perro"
        db = db.append(
            {"Family": family, "Breed": raza, "Path": direccion_img + image},
            ignore_index=True,
        )
    db.to_csv(path_csv, index=False)


def split_train_test(db, frac):
    """ "
    Divide el dataset en train y test"""
    train = db.sample(frac=frac, random_state=1)
    test = db.drop(train.index)
    return train["Path"], test["Path"], train["Breed"], test["Breed"]


def get_classes(db):
    """
    Devuelve las clases del dataset
    """
    return db["Breed"].unique()


def getDescriptors(sift, img, plot=False):
    """
    Devuelve los puntos SIFT de una imagen.
    Si plot=True, muestra la imagen con los keypoints
    """
    kp, des = sift.detectAndCompute(img, None)
    if plot:
        print("Puntos SIFT: " + str(len(kp)))
        img = cv2.drawKeypoints(img, kp, img)

        plt.imshow(img)
        plt.show()

    return des


def readImage(img_path, imgz, persona=False):
    """
    Lee una imagen y su máscara y devuelve la imagen sin fondo.
    Si no existe la máscara, la crea con YOLO
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        nombre_archivo_mascara = os.path.splitext(img_path)[0] + ".png"
        nombre_archivo_mascara = os.path.basename(nombre_archivo_mascara)
        ruta_mascara = os.path.join(dir_labels, nombre_archivo_mascara)
        mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)
        mascara_binaria_1 = cv2.compare(mascara, 1, cv2.CMP_EQ)
        mascara_binaria_3 = cv2.compare(mascara, 3, cv2.CMP_EQ)

        mascara_binaria = cv2.bitwise_or(mascara_binaria_1, mascara_binaria_3)

        imagen_sin_fondo = cv2.bitwise_and(img, img, mask=mascara_binaria)
    except:
        model = YOLO(dir_modelos + "yolov8m-seg.pt")
        path = img_path

        classes = 15 if not persona else 0
        results = model.predict(path, classes=classes)
        result = results[0]
        masks = result.masks
        mask1 = masks[0]
        mask = mask1.data[0].numpy()
        polygon = mask1.xy[0]
        img = cv2.imread(path)
        mask = np.zeros_like(img)
        polygon = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], (255, 255, 255))
        imagen_sin_fondo = cv2.bitwise_and(img, mask)
    return cv2.resize(imagen_sin_fondo, (imgz, imgz))


def vstackDescriptors(descriptor_list):
    """
    Apila los descriptores de una lista
    """
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        if descriptor is not None:
            descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def clusterDescriptors(descriptors, no_clusters, imgz):
    """ "
    Agrupa los descriptores en clusters con KMeans
    """
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    joblib.dump(kmeans, dir_modelos + str(imgz) + str(no_clusters) + "kmeans.sav")
    return kmeans


def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    """ "
    Extrae las características de las imágenes
    """
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        if descriptor_list[i] is not None:
            for j in range(len(descriptor_list[i])):
                feature = descriptor_list[i][j]
                feature = feature.reshape(1, 128)
                idx = kmeans.predict(feature)
                im_features[i][idx] += 1

    return im_features


def normalizeFeatures(scale, features):
    """
    Normaliza las características
    """
    return scale.transform(features)


def plotHistogram(im_features, no_clusters):
    """ "
    Muestra el histograma de las características
    """
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array(
        [abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(no_clusters)]
    )
    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()


def plotConfusionMatrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    Muestra la matriz de confusión
    Se puede normalizar para mostrar los resultados en porcentaje
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    cmd = ConfusionMatrixDisplay(cm, display_labels=np.unique(classes))
    cmd.plot(xticks_rotation="vertical")


def plotConfusions(true, predictions):
    """
    Muestra las matrices de confusión sin normalizar y normalizadas
    """
    np.set_printoptions(precision=2)

    plotConfusionMatrix(
        true,
        predictions,
        classes=list(set(predictions)),
        title="Confusion matrix, without normalization",
    )

    plotConfusionMatrix(
        true,
        predictions,
        classes=list(set(predictions)),
        normalize=True,
        title="Normalized confusion matrix",
    )

    plt.show()


def findAccuracy(true, predictions):
    """
    Muestra la precisión del modelo
    """
    print("accuracy score: %0.3f" % accuracy_score(true, predictions))


def calcular_dcd_ddc(img, mask=None, ncomponents=3, useL=True):
    """
    Calcula los descriptores DCD (Dominant Color Descriptor) y DDC (Descriptor of the Distribution of Color) de una imagen en el espacio de color LAB.

    Args:
        img (numpy.ndarray, shape=(n,m,3)): La imagen de entrada en formato BGR.
        mask (numpy.ndarray, shape=(n,m), optional): Una máscara que indica las regiones de interés en la imagen. Por defecto es None.
        ncomponents (int, optional): El número de componentes para el modelo de mezcla de Gaussianas, representa la cantidad de colores principales. Por defecto es 3.
        useL (bool, optional): Indica si se debe utilizar el canal L (luminancia) en el espacio de color LAB. Por defecto es True.

    Returns:
        dcd (numpy.ndarray): DCD calculado a partir de la imagen y la máscara. Tiene forma (ncomponents * dim,), donde dim es 3 si useL=True y 2 si useL=False.
        ddc (numpy.ndarray): DDC calculado a partir de la imagen y la máscara. Tiene forma (ncomponents * dim * (dim + 1) / 2,), donde dim es 3 si useL es True y 2 si useL es False.
    """

    # Si la mascara esta vacia, abortar.
    if img.ndim != 3:
        print("None 1")

        return None, None

    img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    if not useL:
        img_Lab = img_Lab[:, :, 1:]
        dim = 2
    else:
        dim = 3

    # Obtener los valores de color aplanados
    if mask is not None:
        pixels = img_Lab[mask == 1].reshape((-1, dim))
    else:
        pixels = img_Lab.reshape((-1, dim))

    # Si la mascara esta vacia, abortar.
    if pixels.shape[0] == 0:
        print("None 2")
        return None, None

    # Crear un modelo de mezcla de Gaussianas con 8 componentes
    gmm = GaussianMixture(n_components=ncomponents)

    # Ajustar el modelo a los datos de color
    gmm.fit(pixels)

    # ------ CALCULO DCD ------
    dcd = gmm.means_.flatten()

    # ------ CALCULO DDC ------
    ddc_matrix = gmm.covariances_

    for i in range(ddc_matrix.shape[0]):
        # Obtener los índices de la mitad triangular superior
        indices = np.triu_indices(ddc_matrix[i].shape[0])

        # Seleccionar los valores correspondientes a la mitad triangular superior
        flatten_triangular_superior = ddc_matrix[i][indices]

        if i == 0:
            ddc = flatten_triangular_superior
        else:
            ddc = np.hstack((ddc, flatten_triangular_superior))

    return dcd, ddc


def obtener_descriptores_color(img, mask, ncomponents=3, useL=True, in_dataframe=False):
    """
    Obtiene los descriptores DCD (Dominant Color Descriptor) y el DDC (Descriptor of the Distribution of Color) a partir de una imagen y su máscara correspondiente.

    Args:
        img (np.ndarray): La imagen de entrada en forma de un arreglo NumPy.
        mask (numpy.ndarray): Una máscara que indica las regiones de interés en la imagen.
                              Pixel Annotations:
                              1: Foreground
                              2: Background
                              3: Not classified (decidir si incluir en la maskara o no)
        ncomponents (int, opcional): El número de componentes de color dominantes a extraer. Por defecto es 3.
        useL (bool, opcional): Una bandera para indicar si se debe utilizar el canal L (brillo) en el espacio de color. Si es True, se utiliza el canal L; de lo contrario, no se utiliza. Por defecto es True.
        in_dataframe (bool, opcional): Una bandera para indicar si se deben devolver los descriptores de color en un DataFrame de pandas. Si es True, la salida será un DataFrame; de lo contrario, será un arreglo NumPy. Por defecto es False.

    Returns:
        descriptores (np.ndarray o pd.DataFrame): Si `in_dataframe` es False, devuelve un arreglo NumPy que contiene los descriptores DCD y DDC concatenados.
        Si `in_dataframe` es True, devuelve un DataFrame de pandas con los descriptores de color, donde cada columna corresponde a un descriptor específico.
        En caso de error devuelve None.
    Nota:
        Los descriptores DCD y DDC se calculan utilizando la función `calcular_dcd_ddc`.
    """
    DCD, DDC = calcular_dcd_ddc(img, mask, ncomponents=ncomponents, useL=useL)

    if (DCD is None) or (DDC is None):
        return None

    descriptores = np.hstack((DCD, DDC))

    # ------ CONVIERTO EN DATAFRAME ------

    if in_dataframe:
        column_names = []

        for i in range(1, ncomponents + 1):
            if useL:
                column_names.append(f"L_{i}")
            column_names.append(f"a*_{i}")
            column_names.append(f"b*_{i}")

        for i in range(1, ncomponents + 1):
            if useL:
                column_names.append(f"LL_{i}")
                column_names.append(f"La*_{i}")
                column_names.append(f"Lb*_{i}")

            column_names.append(f"a*a*_{i}")
            column_names.append(f"a*b*_{i}")
            column_names.append(f"b*b*_{i}")

        descriptores = pd.DataFrame(descriptores, columns=column_names)
    return descriptores


def extract_lbp(image, debug=False):
    if image.ndim > 2:
        gray = image[:, :, 0]
    else:
        gray = image
    lbp = local_binary_pattern(gray, 8, 1, method="var")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(256))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7
    if debug:
        return hist, lbp
    return hist


def generar_descriptores_textura(images, imgz=250, persona=False):
    histogramas = []
    for image in images:
        image = readImage(image, imgz, persona)
        hist = extract_lbp(image)
        histogramas.append(hist)
    return np.array(histogramas)


def generar_descriptores_sift(images, no_clusters, imgz=250):
    image_count = len(images)
    sift = cv2.SIFT_create()
    descriptor_list = []
    for img_path in images:
        img = readImage(img_path, imgz)
        des = getDescriptors(sift, img)
        descriptor_list.append(des)
    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")
    if os.path.exists(dir_modelos + str(imgz) + str(no_clusters) + "kmeans.sav"):
        kmeans = joblib.load(dir_modelos + str(imgz) + str(no_clusters) + "kmeans.sav")
    else:
        kmeans = clusterDescriptors(descriptors, no_clusters, imgz)
    print("Descriptors clustered.")
    if os.path.exists(dir_modelos + str(imgz) + str(no_clusters) + "im_features.sav"):
        im_features = joblib.load(
            dir_modelos + str(imgz) + str(no_clusters) + "im_features.sav"
        )
    else:
        im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
        joblib.dump(
            im_features, dir_modelos + str(imgz) + str(no_clusters) + "im_features.sav"
        )

    print("Images features extracted.")
    if os.path.exists(dir_modelos + str(imgz) + str(no_clusters) + "scale.sav"):
        scale = joblib.load(dir_modelos + str(imgz) + str(no_clusters) + "scale.sav")
    else:
        scale = StandardScaler().fit(im_features)
        joblib.dump(scale, dir_modelos + str(imgz) + str(no_clusters) + "scale.sav")
    im_features = scale.transform(im_features)
    print("Train images normalized.")
    return np.array(im_features)


def calcular_descriptores_sift_test(
    images, no_clusters, imgz=250, plot=False, persona=False
):
    kmeans = joblib.load(dir_modelos + str(imgz) + str(no_clusters) + "kmeans.sav")
    scale = joblib.load(dir_modelos + str(imgz) + str(no_clusters) + "scale.sav")
    count = 0
    descriptor_list = []
    sift = cv2.SIFT_create()

    for img_path in images:
        img = readImage(img_path, imgz, persona=persona)
        if plot:
            # solo plotea los puntos sift de la primera imagen

            if not img_path == images[0]:
                plot = False
        des = getDescriptors(sift, img, plot=plot)

        if des is not None:
            count += 1
            descriptor_list.append(des)
    print("Descriptors vstacked.")
    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)
    print("Images features extracted.")
    test_features = scale.transform(test_features)
    print("Test images normalized.")
    return test_features


def generar_descriptores_color(images, imgz=250, persona=False):
    descriptores_color = []
    for image in images:
        img = imread(image)
        try:
            nombre_archivo_mascara = os.path.splitext(image)[0] + ".png"
            nombre_archivo_mascara = os.path.basename(nombre_archivo_mascara)
            ruta_mascara = os.path.join(dir_labels, nombre_archivo_mascara)
            mask = cv2.imread(ruta_mascara)
        except:
            model = YOLO(dir_modelos + "yolov8m-seg.pt")
            path = image
            classes = 15 if not persona else 0
            results = model.predict(path, classes=classes)
            result = results[0]
            masks = result.masks
            mask1 = masks[0]
            mask = mask1.data[0].numpy()
            polygon = mask1.xy[0]
            img = cv2.imread(path)
            mask = np.zeros_like(img)
            polygon = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], (255, 255, 255))

        descriptores = obtener_descriptores_color(img, mask)
        descriptores_color.append(descriptores)
    return np.array(descriptores_color)


def calcular_descriptores_train(
    images_train, color, textura, sift, imgz, no_clusters, modo
):
    if sift:
        descriptores_sift = generar_descriptores_sift(images_train, no_clusters, imgz)
        features = descriptores_sift
        print("Descriptores SIFT calculados")
    if color:
        if os.path.exists(dir_modelos + "descriptores_color.sav"):
            descriptores_color = joblib.load(dir_modelos + "descriptores_color.sav")
        else:
            descriptores_color = generar_descriptores_color(images_train, imgz)
            descriptores_real = []
            for descriptor in descriptores_color:
                if descriptor is not None:
                    descriptores_real.append(descriptor)
            descriptores_real = np.array(descriptores_real)
            descriptores_real = preprocessing.scale(descriptores_real)
            joblib.dump(descriptores_real, dir_modelos + "descriptores_color.sav")
        try:
            features = np.hstack((features, descriptores_color))
        except:
            features = descriptores_color

        print("Descriptores color calculados")
    if textura:
        descriptores_textura = generar_descriptores_textura(images_train, imgz)
        try:
            features = np.hstack((features, descriptores_textura))
        except:
            features = descriptores_textura
        print("Descriptores textura calculados")

    features = preprocessing.scale(features)


    return features


def calcular_descriptores_test(
    images_test,
    color,
    textura,
    sift,
    imgz,
    no_clusters,
    modo,
    inferencia=False,
    persona=False,
):
    if sift:
        descriptores_sift = calcular_descriptores_sift_test(
            images_test, no_clusters, imgz, plot=inferencia, persona=persona
        )
        features = descriptores_sift

        print("Descriptores SIFT calculados")
    if color:
        descriptores_color = generar_descriptores_color(
            images_test, imgz, persona=persona
        )
        try:
            features = np.hstack((features, descriptores_color))
        except:
            features = descriptores_color
        print("Descriptores color calculados")
    if textura:
        descriptores_textura = generar_descriptores_textura(
            images_test, imgz, persona=persona
        )
        try:
            features = np.hstack((features, descriptores_textura))
        except:
            features = descriptores_textura
        print("Descriptores textura calculados")

    features = preprocessing.scale(features)

    return features


def inferencia_eval(image, svm_model, color, textura, sift, no_clusters, imgz, modo):
    """
    Hace la inferencia de una imagen sola
    """
    descriptores = calcular_descriptores_test(
        [image, "images/Ragdoll_112.jpg"],
        color,
        textura,
        sift,
        imgz,
        no_clusters,
        modo,
        inferencia=True,
    )

    prediccion = svm_model.predict(descriptores)
    print("Raza más similar: ", prediccion[0])
    personalidad_gato(prediccion[0], image)


def inferencia_persona(image, svm_model, color, textura, sift, no_clusters, imgz, modo):
    """
    Hace la inferencia de una imagen sola
    """

    descriptores = calcular_descriptores_test(
        [image, "images/Ragdoll_112.jpg"],
        color,
        textura,
        sift,
        imgz,
        no_clusters,
        modo,
        inferencia=True,
        persona=True,
    )

    prediccion = svm_model.predict(descriptores)
    print("Raza más similar: ", prediccion[0])
    personalidad_gato(prediccion[0], image)


def svm_model_with_grid_search(
    descriptores_train,
    images_train_labels,
    descriptores_test,
    images_test_labels,
    param_grid,
    nombre_csv,
):
    # Crear el modelo SVM
    svm_model = svm.SVC()

    # Inicializar GridSearchCV con el modelo SVM y los posibles valores de los hiperparámetros
    grid_search = GridSearchCV(
        svm_model, param_grid, cv=5, scoring="accuracy", verbose=1
    )

    # Ajustar el modelo a los datos de entrenamiento para encontrar los mejores parámetros
    grid_search.fit(descriptores_train, images_train_labels)

    # Obtener los mejores parámetros encontrados
    best_params = grid_search.best_params_
    print("Mejores parámetros:", best_params)

    # Obtener el mejor modelo SVM
    best_svm_model = grid_search.best_estimator_

    # Guardar el mejor modelo en un archivo usando joblib
    joblib.dump(best_svm_model, dir_modelos + nombre_csv + "svm_best_model.sav")

    # Predecir en el conjunto de prueba utilizando el mejor modelo
    predicciones_test = best_svm_model.predict(descriptores_test)
    # Graficar la matriz de confusión para el conjunto de prueba
    plotConfusions(predicciones_test, images_test_labels)

    # Calcular y mostrar la exactitud del modelo en el conjunto de prueba
    findAccuracy(predicciones_test, images_test_labels)


def personalidad_gato(nombre_gato, path_imagen):
    personalidades = {
        "Bombay": "Los gatos Bombay son cariñosos y afectuosos, además de ser muy juguetones.",
        "Abyssinian": "Los gatos Abyssinian son curiosos, activos y les encanta explorar su entorno.",
        "Russian_Blue": "Los gatos Russian Blue son tranquilos, cariñosos y pueden ser tímidos con desconocidos.",
        "Bengal": "Los gatos Bengal son enérgicos, inteligentes y les encanta jugar con agua.",
        "Persian": "Los gatos Persian son elegantes, dulces y necesitan cuidados regulares de su pelaje.",
        "Maine_Coon": "Los gatos Maine Coon son amigables, sociables y tienen una personalidad tranquila.",
        "Siamese": "Los gatos Siamese son extrovertidos, cariñosos y pueden ser ruidosos para comunicarse.",
        "British_Shorthair": "Los gatos British Shorthair son independientes, tranquilos y amables.",
        "Ragdoll": "Los gatos Ragdoll son relajados, cariñosos y tienen una disposición suave.",
        "Sphynx": "Los gatos Sphynx son afectuosos, inteligentes y necesitan atención especial para su piel sin pelo.",
        "Birman": "Los gatos Birman son leales, cariñosos y tienen una hermosa apariencia con pelaje largo.",
        "Egyptian_Mau": "Los gatos Egyptian Mau son activos, juguetones y les encanta correr y saltar.",
    }
    imagenes_gatos = {
        "Bombay": [
            "images/Bombay_9.jpg",
            "images/Bombay_16.jpg",
            "images/Bombay_30.jpg",
            "images/Bombay_66.jpg",
        ],
        "Abyssinian": [
            "images/Abyssinian_6.jpg",
            "images/Abyssinian_32.jpg",
            "images/Abyssinian_48.jpg",
            "images/Abyssinian_68.jpg",
        ],
        "Russian_Blue": [
            "images/Russian_Blue_35.jpg",
            "images/Russian_Blue_62.jpg",
            "images/Russian_Blue_69.jpg",
        ],
        "Bengal": [
            "images/Bengal_17.jpg",
            "images/Bengal_103.jpg",
            "images/Bengal_104.jpg",
            "images/Bengal_113.jpg",
        ],
        "Persian": [
            "images/Persian_18.jpg",
            "images/Persian_7.jpg",
            "images/Persian_38.jpg",
            "images/Persian_56.jpg",
        ],
        "Maine_Coon": [
            "images/Maine_Coon_14.jpg",
            "images/Maine_Coon_19.jpg",
            "images/Maine_Coon_21.jpg",
            "images/Maine_Coon_24.jpg",
            "images/Maine_Coon_28.jpg",
        ],
        "Siamese": [
            "images/Siamese_19.jpg",
            "images/Siamese_28.jpg",
            "images/Siamese_43.jpg",
            "images/Siamese_77.jpg",
        ],
        "British_Shorthair": [
            "images/British_Shorthair_49.jpg",
            "images/British_Shorthair_98.jpg",
            "images/British_Shorthair_101.jpg",
        ],
        "Ragdoll": [
            "images/Ragdoll_4.jpg",
            "images/Ragdoll_12.jpg",
            "images/Ragdoll_42.jpg",
            "images/Ragdoll_62.jpg",
        ],
        "Sphynx": [
            "images/Sphynx_42.jpg",
            "images/Sphynx_169.jpg",
            "images/Sphynx_188.jpg",
            "images/Sphynx_222.jpg",
        ],
        "Birman": [
            "images/Birman_4.jpg",
            "images/Birman_13.jpg",
            "images/Birman_43.jpg",
            "images/Birman_62.jpg",
            "images/Birman_72.jpg",
        ],
        "Egyptian_Mau": [
            "images/Egyptian_Mau_49.jpg",
            "images/Egyptian_Mau_51.jpg",
            "images/Egyptian_Mau_82.jpg",
            "images/Egyptian_Mau_130.jpg",
        ],
    }

    if nombre_gato in personalidades:
        img_path = random.choice(imagenes_gatos[nombre_gato])
        # Dividir la figura en dos partes
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Mostrar la imagen proporcionada en el primer subplot
        img = mpimg.imread(path_imagen)
        axs[0].imshow(img)
        axs[0].set_title("Imagen proporcionada")
        axs[0].axis("off")

        # Mostrar la imagen del gato en el segundo subplot
        gato_img = mpimg.imread(img_path)
        axs[1].imshow(gato_img)
        axs[1].set_title(nombre_gato)
        axs[1].axis("off")
        fig.suptitle(personalidades[nombre_gato], fontsize=30)

        # Ajustar el diseño para evitar superposiciones
        plt.tight_layout()

        # Mostrar la figura completa
        plt.show()
    else:
        print("Lo siento, no tengo información sobre la personalidad de ese gato.")
