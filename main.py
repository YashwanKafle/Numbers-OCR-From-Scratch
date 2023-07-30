from PIL import Image
import numpy as np


# To read and save ourown image (not from dataset)
def read_image(path):
    return np.asarray(Image.open(path).convert("L"))


def save_image(image, path):
    img = Image.fromarray(np.array(image), "L")
    img.save(path)


TEST_DIR = "TEST/"
TEST_DATA_FILENAME = "t10k-images-idx3-ubyte"
TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte"
TRAIN_DATA_FILENAME = "train-images-idx3-ubyte"
TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte"


# to convert bytes to integer
def bytes_to_int(bytes_data):
    return int.from_bytes(bytes_data, "big")


def read_images(filename, n_max_images=None):
    images = []

    with open(filename, "rb") as f:
        _ = f.read(4)  # magic number
        no_of_images = bytes_to_int(f.read(4))

        if n_max_images:
            no_of_images = n_max_images

        n_rows = bytes_to_int(f.read(4))
        n_cols = bytes_to_int(f.read(4))

        for img_idx in range(no_of_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_cols):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)

    return images


def read_labels(filename, n_max_labels=None):
    labels = []

    with open(filename, "rb") as f:
        _ = f.read(4)  # magic number
        no_of_labels = bytes_to_int(f.read(4))

        if n_max_labels:
            no_of_labels = n_max_labels

        for label_idx in range(no_of_labels):
            label = f.read(1)
            labels.append(label)

    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def distance(x, y):
    return (
        sum([(bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2 for x_i, y_i in zip(x, y)])
        ** 0.5
    )


def get_distance_for_test_sample(X_train, test_sample):
    return [distance(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, y_train, x_test, k=5):
    y_preds = []
    for test_sample in x_test:
        training_distances = get_distance_for_test_sample(X_train, test_sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(enumerate(training_distances), key=lambda x: x[1])
        ]
        candidates = [bytes_to_int(y_train[idx]) for idx in sorted_distance_indices[:k]]
        y_pred = get_most_frequent_element(candidates)
        y_preds.append(y_pred)

    return y_preds


def main():
    Train_DATA_LENGTH = 1000
    X_train = read_images(TRAIN_DATA_FILENAME, Train_DATA_LENGTH)
    y_train = read_labels(TRAIN_LABELS_FILENAME, Train_DATA_LENGTH)

    # our own image test
    ourtest = False

    if ourtest:
        # specify image paths
        img_paths = ["ourtest.png", "ourtest.png"]

        X_test = [read_image(path) for path in img_paths]
        X_train = extract_features(X_train)
        X_test = extract_features(X_test)

        y_preds = knn(X_train, y_train, X_test, 5)

        for img_name, y_pred in zip(img_paths, y_preds):
            print(f"{img_name} >  prediction : {y_pred}")
    else:
        TEST_SIZE = 10

        X_test = read_images(TEST_DATA_FILENAME, TEST_SIZE)
        y_test = read_labels(TEST_LABELS_FILENAME, TEST_SIZE)

        for idx, test_sample in enumerate(X_test):
            save_image(test_sample, f"{TEST_DIR}{idx}.png")

        X_train = extract_features(X_train)
        X_test = extract_features(X_test)
        y_preds = knn(X_train, y_train, X_test, 5)

        correct_prediction = sum(
            [
                (y_pred_i == bytes_to_int(y_test_i))
                for y_pred_i, y_test_i in zip(y_preds, y_test)
            ]
        ) / len(y_test)

        print(correct_prediction)


if __name__ == "__main__":
    main()
