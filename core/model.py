import typing as tp
import tensorflow as tf


def init_model(input_len: int,
               hidden_layers_sizes: tp.List[int]) -> tf.keras.Model:
    x_input = tf.keras.layers.Input(shape=(input_len,), dtype=tf.float32)

    x = x_input
    for layer_size in hidden_layers_sizes:
        x = tf.keras.layers.Dense(layer_size, activation='sigmoid')(x)

    x = tf.keras.layers.Dense(10)(x)
    return tf.keras.models.Model(inputs=x_input, outputs=x, name='mnist_classifier')


def init_extended_model(model: tf.keras.models.Model,
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


def get_model(input_len: int,
              hidden_layers_sizes: tp.List[int],
              optimizer: tf.keras.optimizers.Optimizer,
              loss: tf.keras.losses.Loss,
              metrics: tp.List[tf.keras.metrics.Metric]) -> tf.keras.models.Model:
    model = init_model(input_len, hidden_layers_sizes)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def get_extended_model(model: tf.keras.models.Model,
                       optimizer: tf.keras.optimizers.Optimizer,
                       loss: tf.keras.losses.Loss,
                       metrics: tp.List[tf.keras.metrics.Metric],
                       temp_head_mean: float = 0.,
                       temp_head_std: float = 1.,
                       seed: int = 100) -> tf.keras.models.Model:
    extended_model = init_extended_model(model, temp_head_mean, temp_head_std, seed)
    extended_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return extended_model


if __name__ == '__main__':
    model = get_model(768, [10, 20, 30], optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[])
    ext_model = get_extended_model(model, optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[])

    model.summary()
    ext_model.summary()
