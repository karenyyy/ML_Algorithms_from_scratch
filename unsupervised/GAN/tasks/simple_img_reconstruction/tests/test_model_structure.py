import numpy as np
from tasks.simple_img_reconstruction.Autoencoder import AutoEncoder


def dim_flattened(layer):
    return np.prod(layer.output_shape[1:])

img_shape = (32, 32, 3)
for filter_num in [1, 8, 32, 128, 512]:
    auto = AutoEncoder(img_shape[0], img_shape[1], filter_num=filter_num, task='')
    s = auto.reset_tf_session()

    # visualize the summaries
    auto.encoder.summary()
    auto.decoder.summary()

    # Tests
    print("Testing code size %i" % filter_num)
    assert auto.encoder.output_shape[1:] == (filter_num,), "encoder must output a code of required size"
    assert auto.decoder.output_shape[1:] == img_shape, "decoder must output an image of valid shape"
    assert len(auto.encoder.trainable_weights) >= 6, "encoder must contain at least 3 layers"
    assert len(auto.decoder.trainable_weights) >= 6, "decoder must contain at least 3 layers"

    for layer in auto.encoder.layers + auto.decoder.layers:
        print(layer)
        assert dim_flattened(layer) >= filter_num, "Encoder layer %s is smaller than bottleneck (%i units)" % (
            layer.name, dim_flattened(layer))

print("Tests passed!")