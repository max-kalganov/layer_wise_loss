import tensorflow_datasets as tfds


def get_mnist_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_test


if __name__ == '__main__':
    train, test = get_mnist_data()
    print(train)
    img = list(train.take(1).as_numpy_iterator())[0][0]
    from PIL import Image
    # cv2.imshow("window name", img)
    cv_im = Image.fromarray(img.squeeze(axis=-1))
    cv_im.show()
