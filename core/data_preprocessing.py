import tensorflow_datasets as tfds
import tensorflow as tf
import typing as tp


def get_mnist_data() -> tp.Tuple[tf.data.Dataset, tf.data.Dataset]:
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return ds_train, ds_test


def normalize(image: tf.Tensor, label: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    image_min, image_max = tf.math.reduce_min(image), tf.math.reduce_max(image)
    image = (image - image_min) / (image_max - image_min)
    return image, label


def encode_one_hot(image: tf.Tensor, label: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    return image, tf.one_hot(label, depth=10)


def flatten_input(image: tf.Tensor, label: tf.Tensor) -> tp.Tuple[tf.Tensor, tf.Tensor]:
    image = tf.reshape(image, (1, -1))
    image = tf.squeeze(image)
    return image, label


def get_train_test() -> tp.Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds, test_ds = get_mnist_data()
    train_ds, test_ds = train_ds.map(normalize), test_ds.map(normalize)
    # will use SparseCategoricalCrossEntropy instead
    # train_ds, test_ds = train_ds.map(encode_one_hot), test_ds.map(encode_one_hot)
    train_ds, test_ds = train_ds.map(flatten_input), test_ds.map(flatten_input)
    return train_ds, test_ds


if __name__ == '__main__':
    train, test = get_train_test()
    img, label = list(train.take(1).as_numpy_iterator())[0]
    print(img.shape, label)
