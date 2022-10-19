"""
Top level utils functions
"""

import os
import pickle
import zipfile


def save_pickle_object(object: object, path: str = "./model", force: bool = False, compressed: bool = False) -> None:
    """
    Save the model as pickle object

    :param path: path where to store the pickle object
    :param force: boolean flag, if True overwrite old file with the same path
    :param compressed: boolean flag, if True store a zipped file
    :return: None
    """

    if os.path.isfile(path):
        if force:
            os.remove(path)
        else:
            print(f"File {path} exists yet!")
            raise FileExistsError(f"File {path} exists yet!")

    with open(path, 'wb') as f:
        pickle.dump(object, f)

    if compressed:
        with zipfile.ZipFile(path + ".7z", 'w') as archive:
            archive.write(path, path)
        os.remove(path)

def load_pickle_object(path="./model.pickle", compressed=False):
    """
    Load an object stored as a pickle file

    :param path: path from where to load the object (as pickle object)
    :param compressed: boolean flag, if True the stored object is a zipped file
    :return: object
    """

    object_name = path
    if compressed:
        archive = zipfile.ZipFile(path + ".7z", mode='r')
        object_name = archive.filelist[0]
        archive.extract(targets=object_name)
        archive.close()

    with open(object_name, "rb") as f:
        my_object = pickle.load(f)

    if compressed:
        os.remove(object_name)

    return my_object
