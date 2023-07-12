import joblib
import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


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
        img = cv2.drawKeypoints(img, kp, img)

        plt.imshow(img)
        plt.show()

    return des


def readImage(img_path, direccion_lbl):
    """
    Lee una imagen y su máscara y devuelve la imagen sin fondo.
    Si no existe la máscara, la crea con YOLO
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        nombre_archivo_mascara = os.path.splitext(img_path)[0] + ".png"
        nombre_archivo_mascara = os.path.basename(nombre_archivo_mascara)
        ruta_mascara = os.path.join(direccion_lbl, nombre_archivo_mascara)
        mascara = cv2.imread(ruta_mascara, cv2.IMREAD_GRAYSCALE)
        mascara_binaria_1 = cv2.compare(mascara, 1, cv2.CMP_EQ)
        mascara_binaria_3 = cv2.compare(mascara, 3, cv2.CMP_EQ)

        mascara_binaria = cv2.bitwise_or(mascara_binaria_1, mascara_binaria_3)

        imagen_sin_fondo = cv2.bitwise_and(img, img, mask=mascara_binaria)
    except:
        model = YOLO(dir_modelos + "yolov8m-seg.pt")
        path = img_path
        results = model.predict(path, classes=15)
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
    return cv2.resize(imagen_sin_fondo, (500, 500))


def vstackDescriptors(descriptor_list):
    """ "
    Apila los descriptores de una lista
    """
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        if descriptor is not None:
            descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def clusterDescriptors(descriptors, no_clusters):
    """ "
    Agrupa los descriptores en clusters con KMeans
    """
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    joblib.dump(kmeans, dir_modelos + str(no_clusters) + "kmeans.sav")
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


def svcParamSelection(X, y, kernel, nfolds):
    """
    Busca los mejores parámetros para el SVM
    """
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {"C": Cs, "gamma": gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


def findSVM(im_features, train_labels, kernel, clusters):
    """
    Entrena el SVM
    """
    features = im_features
    if kernel == "precomputed":
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param)
    svm.fit(features, train_labels)
    joblib.dump(svm, dir_modelos + str(clusters) + kernel + "svm.sav")

    return svm


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
    cmd.plot()


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


def trainModel(images, labels, no_clusters, kernel):
    """
    Entrena el modelo
    """
    if os.path.exists(dir_modelos + "descriptors.sav"):
        descriptors = joblib.load(dir_modelos + "descriptors.sav")
        print("Descriptors loaded.")
    else:
        sift = cv2.SIFT_create()
        descriptor_list = []
        train_labels = np.array([])
        image_count = len(images)
        for img_path, class_index in zip(images, labels):
            train_labels = np.append(train_labels, class_index)
            img = readImage(img_path, dir_labels)
            des = getDescriptors(sift, img)
            descriptor_list.append(des)
        descriptors = vstackDescriptors(descriptor_list)
        print("Descriptors vstacked.")
        joblib.dump(descriptors, dir_modelos + "descriptors.sav")

    if os.path.exists(dir_modelos + str(no_clusters) + "kmeans.sav"):
        kmeans = joblib.load(dir_modelos + str(no_clusters) + "kmeans.sav")
    else:
        kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    joblib.dump(im_features, str(no_clusters) + kernel + "im_features.sav")

    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)
    joblib.dump(scale, dir_modelos + str(no_clusters) + kernel + "scale.sav")

    im_features = scale.transform(im_features)

    print("Train images normalized.")

    plotHistogram(im_features, no_clusters)
    print("Features histogram plotted.")

    svm = findSVM(im_features, train_labels, kernel, no_clusters)

    print("SVM fitted.")
    print("Training completed.")

    return kmeans, scale, svm, im_features


def testModel(
    test_images,
    test_labels,
    kmeans,
    scale,
    svm,
    im_features,
    no_clusters,
    kernel,
    class_names,
    inferencia=False,
):
    """ "
    Prueba el modelo con el conjunto de test
    """
    count = 0
    true = []
    descriptor_list = []
    name_dict = {str(i): class_names[i] for i in range(len(class_names))}
    print(name_dict)

    sift = cv2.SIFT_create()

    for img_path, label in zip(test_images, test_labels):
        dir_labels = "annotations/trimaps/"

        img = readImage(img_path, dir_labels)
        des = getDescriptors(sift, img, plot=inferencia)

        if des is not None:
            count += 1
            descriptor_list.append(des)
            true.append(label)

    # descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)

    test_features = scale.transform(test_features)

    kernel_test = test_features
    if kernel == "precomputed":
        kernel_test = np.dot(test_features, im_features.T)

    # predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    # predictions = [k for k, v in name_dict.items() if v in svm.predict(kernel_test)]
    predictions = []
    predictions_name = []
    for value in svm.predict(kernel_test):
        predictions_name.append(value)
        for k, v in name_dict.items():
            if v == value:
                predictions.append(k)
                break
    print("Test images classified.")
    if inferencia:
        print("Raza: " + predictions_name[0])
    else:
        plotConfusions(true, predictions_name)
        print("Confusion matrixes plotted.")

        findAccuracy(true, predictions_name)
        print("Accuracy calculated.")
        print("Execution done.")


def execute(
    images_train, images_test, classes_train, classes_test, no_clusters, kernel, classes
):
    """
    Entrena y testea el modelo.
    """
    kmeans, scale, svm, im_features = trainModel(
        images_train, classes_train, no_clusters, kernel
    )
    testModel(
        images_test,
        classes_test,
        kmeans,
        scale,
        svm,
        im_features,
        no_clusters,
        kernel,
        classes,
    )


def test(images_test, classes_test, no_clusters, kernel, classes):
    """
    Testea el modelo.
    """
    im_features = joblib.load(
        dir_modelos + str(no_clusters) + kernel + "im_features.sav"
    )
    kmeans = joblib.load(dir_modelos + kernel + "kmeans.sav")
    svm = joblib.load(dir_modelos + str(no_clusters) + kernel + "svm.sav")
    scale = joblib.load(dir_modelos + str(no_clusters) + kernel + "scale.sav")
    testModel(
        images_test,
        classes_test,
        kmeans,
        scale,
        svm,
        im_features,
        no_clusters,
        kernel,
        classes,
    )


def inferencia_eval(path_image, test_label, no_clusters, kernel, classes):
    """
    Hace la inferencia de una imagen sola
    """
    kmeans = joblib.load(dir_modelos + kernel + "kmeans.sav")
    scale = joblib.load(dir_modelos + str(no_clusters) + kernel + "scale.sav")
    im_features = joblib.load(
        dir_modelos + str(no_clusters) + kernel + "im_features.sav"
    )
    svm = joblib.load(dir_modelos + str(no_clusters) + kernel + "svm.sav")
    testModel(
        [path_image],
        [test_label],
        kmeans,
        scale,
        svm,
        im_features,
        no_clusters,
        kernel,
        classes,
        inferencia=True,
    )
    img = cv2.imread(path_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
