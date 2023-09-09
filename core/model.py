import typing as tp
import tensorflow as tf


def get_model(input_len: int, hidden_layers_sizes: tp.List[int]) -> tf.keras.Model:
    x_input = tf.keras.layers.Input(shape=(input_len,), dtype=tf.float32)

    x = x_input
    for layer_size in hidden_layers_sizes:
        x = tf.keras.layers.Dense(layer_size, activation='sigmoid')(x)

    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.models.Model(inputs=x_input, outputs=x, name='mnist_classifier')


def extend_model(model: tf.keras.models.Model,
                 temp_head_mean: float = 0.,
                 temp_head_std: float = 1.,
                 seed: int = 100) -> tf.keras.models.Model:
    layers = model.layers
    output_layer = layers[-1]
    output_shape = output_layer.output_shape[-1]

    outputs = []
    for layer in layers[1:]:
        temp_head = tf.random.normal(shape=(layer.output_shape[-1], output_shape),
                                     mean=temp_head_mean,
                                     stddev=temp_head_std,
                                     seed=seed)
        outputs.append(layer.output @ temp_head)

    return tf.keras.models.Model(inputs=model.input, outputs=outputs, name=f'extended {model.name}')


if __name__ == '__main__':
    model = get_model(768, [10, 20, 30])
    ext_model = extend_model(model)

    model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    ext_model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    model.summary()
    ext_model.summary()
