# vim: expandtab:ts=4:sw=4
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim


def _batch_norm_fn(x, scope=None):
    if scope is None:
        scope = tf.get_variable_scope().name + "/bn"
    return slim.batch_norm(x, scope=scope)


def create_link(
        incoming, network_builder, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(stddev=1e-3),
        regularizer=None, is_first=False, summarize_activations=True):
    if is_first:
        network = incoming
    else:
        network = _batch_norm_fn(incoming, scope=scope + "/bn")
        network = nonlinearity(network)
        if summarize_activations:
            tf.summary.histogram(scope+"/activations", network)

    pre_block_network = network
    post_block_network = network_builder(pre_block_network, scope)

    incoming_dim = pre_block_network.get_shape().as_list()[-1]
    outgoing_dim = post_block_network.get_shape().as_list()[-1]
    if incoming_dim != outgoing_dim:
        assert outgoing_dim == 2 * incoming_dim, \
            "%d != %d" % (outgoing_dim, 2 * incoming)
        projection = slim.conv2d(
            incoming, outgoing_dim, 1, 2, padding="SAME", activation_fn=None,
            scope=scope+"/projection", weights_initializer=weights_initializer,
            biases_initializer=None, weights_regularizer=regularizer)
        network = projection + post_block_network
    else:
        network = incoming + post_block_network
    return network


def create_inner_block(
        incoming, scope, nonlinearity=tf.nn.elu,
        weights_initializer=tf.truncated_normal_initializer(1e-3),
        bias_initializer=tf.zeros_initializer(), regularizer=None,
        increase_dim=False, summarize_activations=True):
    n = incoming.get_shape().as_list()[-1]
    stride = 1
    if increase_dim:
        n *= 2
        stride = 2

    incoming = slim.conv2d(
        incoming, n, [3, 3], stride, activation_fn=nonlinearity, padding="SAME",
        normalizer_fn=_batch_norm_fn, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/1")
    if summarize_activations:
        tf.summary.histogram(incoming.name + "/activations", incoming)

    incoming = slim.dropout(incoming, keep_prob=0.6)

    incoming = slim.conv2d(
        incoming, n, [3, 3], 1, activation_fn=None, padding="SAME",
        normalizer_fn=None, weights_initializer=weights_initializer,
        biases_initializer=bias_initializer, weights_regularizer=regularizer,
        scope=scope + "/2")
    return incoming


def residual_block(incoming, scope, nonlinearity=tf.nn.elu,
                   weights_initializer=tf.truncated_normal_initializer(1e3),
                   bias_initializer=tf.zeros_initializer(), regularizer=None,
                   increase_dim=False, is_first=False,
                   summarize_activations=True):

    def network_builder(x, s):
        return create_inner_block(
            x, s, nonlinearity, weights_initializer, bias_initializer,
            regularizer, increase_dim, summarize_activations)

    return create_link(
        incoming, network_builder, scope, nonlinearity, weights_initializer,
        regularizer, is_first, summarize_activations)


def _create_network(incoming, reuse=None, weight_decay=1e-8):
    nonlinearity = tf.nn.elu
    conv_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    conv_bias_init = tf.zeros_initializer()
    conv_regularizer = slim.l2_regularizer(weight_decay)
    fc_weight_init = tf.truncated_normal_initializer(stddev=1e-3)
    fc_bias_init = tf.zeros_initializer()
    fc_regularizer = slim.l2_regularizer(weight_decay)

    def batch_norm_fn(x):
        return slim.batch_norm(x, scope=tf.get_variable_scope().name + "/bn")

    network = incoming
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_1",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)
    network = slim.conv2d(
        network, 32, [3, 3], stride=1, activation_fn=nonlinearity,
        padding="SAME", normalizer_fn=batch_norm_fn, scope="conv1_2",
        weights_initializer=conv_weight_init, biases_initializer=conv_bias_init,
        weights_regularizer=conv_regularizer)

    # NOTE(nwojke): This is missing a padding="SAME" to match the CNN
    # architecture in Table 1 of the paper. Information on how this affects
    # performance on MOT 16 training sequences can be found in
    # issue 10 https://github.com/nwojke/deep_sort/issues/10
    network = slim.max_pool2d(network, [3, 3], [2, 2], scope="pool1")

    network = residual_block(
        network, "conv2_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False, is_first=True)
    network = residual_block(
        network, "conv2_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False)

    network = residual_block(
        network, "conv3_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True)
    network = residual_block(
        network, "conv3_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False)

    network = residual_block(
        network, "conv4_1", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=True)
    network = residual_block(
        network, "conv4_3", nonlinearity, conv_weight_init, conv_bias_init,
        conv_regularizer, increase_dim=False)

    feature_dim = network.get_shape().as_list()[-1]
    network = slim.flatten(network)

    network = slim.dropout(network, keep_prob=0.6)
    network = slim.fully_connected(
        network, feature_dim, activation_fn=nonlinearity,
        normalizer_fn=batch_norm_fn, weights_regularizer=fc_regularizer,
        scope="fc1", weights_initializer=fc_weight_init,
        biases_initializer=fc_bias_init)

    features = network

    # Features in rows, normalize axis 1.
    features = slim.batch_norm(features, scope="ball", reuse=reuse)
    feature_norm = tf.sqrt(
        tf.constant(1e-8, tf.float32) +
        tf.reduce_sum(tf.square(features), [1], keepdims=True))
    features = features / feature_norm
    return features, None


def _network_factory(weight_decay=1e-8):

    def factory_fn(image, reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=False):
                with slim.arg_scope([slim.conv2d, slim.fully_connected,
                                     slim.batch_norm, slim.layer_norm],
                                    reuse=reuse):
                    features, logits = _create_network(
                        image, reuse=reuse, weight_decay=weight_decay)
                    return features, logits

    return factory_fn


def _preprocess(image):
    image = image[:, :, ::-1]  # BGR to RGB
    return image


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Freeze old model")
    parser.add_argument(
        "--checkpoint_in",
        default="resources/networks/mars-small128.ckpt-68577",
        help="Path to checkpoint file")
    parser.add_argument(
        "--graphdef_out",
        default="resources/networks/mars-small128.pb")
    return parser.parse_args()


def main():
    args = parse_args()

    with tf.Session(graph=tf.Graph()) as session:
        input_var = tf.placeholder(
            tf.uint8, (None, 128, 64, 3), name="images")
        image_var = tf.map_fn(
            lambda x: _preprocess(x), tf.cast(input_var, tf.float32),
            back_prop=False)

        factory_fn = _network_factory()
        features, _ = factory_fn(image_var, reuse=None)
        features = tf.identity(features, name="features")

        saver = tf.train.Saver(slim.get_variables_to_restore())
        saver.restore(session, args.checkpoint_in)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            session, tf.get_default_graph().as_graph_def(),
            [features.name.split(":")[0]])
        with tf.gfile.GFile(args.graphdef_out, "wb") as file_handle:
            file_handle.write(output_graph_def.SerializeToString())


if __name__ == "__main__":
    main()
