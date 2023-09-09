import tensorflow as tf
import typing as tp


@tf.function
def train_step(inputs, labels,
               model: tf.keras.Model,
               loss: tf.keras.losses.Loss,
               optimizer: tf.keras.optimizers.Optimizer,
               metrics: tp.List[tf.keras.metrics.Metric]):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_value = loss(labels, predictions)
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    for single_metric in metrics:
        single_metric(labels, predictions)


@tf.function
def test_step(inputs, labels,
              model: tf.keras.Model,
              loss: tf.keras.losses.Loss,
              metrics: tp.List[tf.keras.metrics.Metric]):
    predictions = model(inputs, training=False)
    loss(labels, predictions)

    for single_metric in metrics:
        single_metric(labels, predictions)


def basic_model_training(model: tf.keras.Model,
                         loss: tf.keras.losses.Loss,
                         optimizer: tf.keras.optimizers.Optimizer,
                         train_metrics: tp.List[tf.keras.metrics.Metric],
                         test_metrics: tp.List[tf.keras.metrics.Metric],
                         train_ds: tf.data.Dataset, test_ds: tf.data.Dataset,
                         batch_size: int, epochs: int):

    for i_epoch in range(epochs):
        for single_metric in train_metrics:
            single_metric.reset_state()

        for single_metric in test_metrics:
            single_metric.reset_state()

        for batch_inputs, batch_labels in train_ds.batch(batch_size):
            train_step(batch_inputs, batch_labels, model, loss, optimizer, train_metrics)

        for batch_inputs, batch_labels in test_ds.batch(batch_size):
            test_step(batch_inputs, batch_labels, model, loss, test_metrics)

        print(f"EPOCH: {i_epoch}, "
              f"{','.join([f'TRAIN METRIC {single_metric.name}: {single_metric}' for single_metric in train_metrics])}"
              f"{','.join([f'TEST METRIC {single_metric.name}: {single_metric}' for single_metric in test_metrics])}")
