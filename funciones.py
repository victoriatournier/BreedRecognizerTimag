import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def make_dataset(direccion_img, path_csv):
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
    train = db.sample(frac=frac, random_state=1)
    test = db.drop(train.index)
    return train["Path"], test["Path"], train["Breed"], test["Breed"]


def get_classes(db):
    return db["Breed"].unique()
