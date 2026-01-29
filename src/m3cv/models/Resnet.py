"""A vanilla 3D resnet implementation.

Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)
"""

import math

import six
from keras import backend as K
from keras.layers import (
    Activation,
    Add,
    AveragePooling3D,
    BatchNormalization,
    Concatenate,
    Conv3D,
    Cropping3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling3D,
    Reshape,
    UpSampling3D,
    add,
)
from keras.models import Model
from keras.regularizers import l2


def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        conv = Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
        )(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
        )(activation)

    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = math.ceil(input.shape[DIM1_AXIS] / residual.shape[DIM1_AXIS])
    stride_dim2 = math.ceil(input.shape[DIM2_AXIS] / residual.shape[DIM2_AXIS])
    stride_dim3 = math.ceil(input.shape[DIM3_AXIS] / residual.shape[DIM3_AXIS])
    equal_channels = residual.shape[CHANNEL_AXIS] == input.shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(
            filters=residual.shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal",
            padding="valid",
            kernel_regularizer=l2(1e-4),
        )(input)
    return add([shortcut, residual])


def _residual_block3d(
    block_function, filters, kernel_regularizer, repetitions, is_first_layer=False
):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(
                filters=filters,
                strides=strides,
                kernel_regularizer=kernel_regularizer,
                is_first_block_of_first_layer=(is_first_layer and i == 0),
            )(input)
        return input

    return f


def basic_block(
    filters,
    strides=(1, 1, 1),
    kernel_regularizer=l2(1e-4),
    is_first_block_of_first_layer=False,
):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""

    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(
                filters=filters,
                kernel_size=(3, 3, 3),
                strides=strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
            )(input)
        else:
            conv1 = _bn_relu_conv3d(
                filters=filters,
                kernel_size=(3, 3, 3),
                strides=strides,
                kernel_regularizer=kernel_regularizer,
            )(input)

        residual = _bn_relu_conv3d(
            filters=filters,
            kernel_size=(3, 3, 3),
            kernel_regularizer=kernel_regularizer,
        )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(
    filters,
    strides=(1, 1, 1),
    kernel_regularizer=l2(1e-4),
    is_first_block_of_first_layer=False,
):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""

    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(
                filters=filters,
                kernel_size=(1, 1, 1),
                strides=strides,
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=kernel_regularizer,
            )(input)
        else:
            conv_1_1 = _bn_relu_conv3d(
                filters=filters,
                kernel_size=(1, 1, 1),
                strides=strides,
                kernel_regularizer=kernel_regularizer,
            )(input)

        conv_3_3 = _bn_relu_conv3d(
            filters=filters,
            kernel_size=(3, 3, 3),
            kernel_regularizer=kernel_regularizer,
        )(conv_1_1)
        residual = _bn_relu_conv3d(
            filters=filters * 4,
            kernel_size=(1, 1, 1),
            kernel_regularizer=kernel_regularizer,
        )(conv_3_3)

        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == "channels_last":
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError(f"Invalid {identifier}")
        return res
    return identifier


def _early_fusion_direct(filters, reg_factor, mode="concat"):
    def f(in_tensor, fuseto):
        while in_tensor.shape[-1] != 16:
            in_tensor = Dense(
                units=max(round(in_tensor.shape[-1] / 2), 16),
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(reg_factor),
            )(in_tensor)
            in_tensor = Dropout(0.2)(in_tensor)
        to3d = Reshape((1, 1, 1, -1))(in_tensor)
        dest_shape = fuseto.shape[1:-1]
        extended = UpSampling3D(size=dest_shape)(to3d)
        if mode == "concat":
            extended = Concatenate(axis=-1)([fuseto, extended])
        elif mode == "add":
            extended = Add()([fuseto, extended])
        return extended

    return f


def find_initial_upsample(goal_shape):
    spatial = goal_shape[1:4]
    init_upsample = []
    for size in spatial:
        min_remainder = 10000
        bestsize = 4
        for i in range(4, 10):
            j = 1
            while i * 2**j < size:
                j += 1
            remainder = i * 2**j - size
            if remainder < min_remainder:
                bestsize = i
                min_remainder = remainder
        init_upsample.append(bestsize)
    return init_upsample


def _early_fusion_conv(filters, reg_factor, mode="concat"):
    def f(in_tensor, fuseto):
        while in_tensor.shape[-1] != 16:
            in_tensor = Dense(
                units=max(round(in_tensor.shape[-1] / 2), 16),
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(reg_factor),
            )(in_tensor)
            in_tensor = Dropout(0.2)(in_tensor)
        to3d = Reshape((1, 1, 1, -1))(in_tensor)
        init_upsample = find_initial_upsample(fuseto.shape)
        to3d = UpSampling3D(size=init_upsample)(to3d)
        while to3d.shape[2] < fuseto.shape[2]:
            if to3d.shape[1] == fuseto.shape[1]:
                upsize = (1, 2, 2)
            else:
                upsize = (2, 2, 2)
            to3d = Conv3D(
                filters,
                (3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(reg_factor),
            )(to3d)
            to3d = Conv3D(
                filters,
                (3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(reg_factor),
            )(to3d)
            to3d = Dropout(0.2)(to3d)
            to3d = UpSampling3D(size=upsize)(to3d)
            # now if tensor size has overshot intended shape in axis 1, crop
            # this is necessary for survival model input shape
            if to3d.shape[1] > fuseto.shape[1]:
                diff = to3d.shape[1] > fuseto.shape[1]
                if diff == 1:
                    crop_op = ((1, 0), (0, 0), (0, 0))
                else:
                    crop_op = ((diff // 2, diff // 2 + diff % 2), (0, 0), (0, 0))
                to3d = Cropping3D(cropping=crop_op, data_format="channels_last")(to3d)
        if mode == "concat":
            extended = Concatenate(axis=-1)([fuseto, to3d])
        elif mode == "add":
            extended = Add()([fuseto, to3d])
        return extended

    return f


class Resnet3DBuilder:
    """ResNet3D."""

    @staticmethod
    def build(
        input_shape,
        num_outputs,
        block_fn,
        repetitions,
        reg_factor,
        fusions,
        basefilters,
    ):
        """Instantiate a vanilla ResNet3D keras model.

        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
            fusions: dictionary where keys are layer indexes and value is an
            integer representation of the non-volumetric data to fuse in
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError(
                "Input shape should be a tuple "
                "(conv_dim1, conv_dim2, conv_dim3, channels) "
                "for tensorflow as backend or "
                "(channels, conv_dim1, conv_dim2, conv_dim3) "
                "for theano as backend"
            )

        block_fn = _get_block(block_fn)
        inputs = Input(shape=input_shape)
        # first conv
        if input_shape[0] < 40:
            k_s = 5
            st = 1
        else:
            k_s = 7
            st = 2
        conv1 = _conv_bn_relu3D(
            filters=basefilters,
            kernel_size=(k_s, k_s, k_s),
            strides=(st, st, st),
            kernel_regularizer=l2(reg_factor),
        )(inputs)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(
            conv1
        )

        # repeat blocks
        block = pool1
        filters = basefilters
        for i, r in enumerate(repetitions):
            if i in fusions.keys():
                if not isinstance(inputs, list):
                    inputs = [inputs]
                init_shape = fusions[i]
                newinputs = Input(shape=(init_shape,))
                inputs.append(newinputs)

                if fusions.get("concat", False):
                    mode = "concat"
                elif fusions.get("add", True):
                    mode = "add"

                if fusions.get("direct", False):
                    block = _early_fusion_direct(basefilters, reg_factor, mode=mode)(
                        newinputs, block
                    )
                elif fusions.get("conv", True):
                    block = _early_fusion_conv(basefilters, reg_factor, mode=mode)(
                        newinputs, block
                    )

            block = _residual_block3d(
                block_fn,
                filters=filters,
                kernel_regularizer=l2(reg_factor),
                repetitions=r,
                is_first_layer=(i == 0),
            )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        pool2 = AveragePooling3D(
            pool_size=(
                block.shape[DIM1_AXIS],
                block.shape[DIM2_AXIS],
                block.shape[DIM3_AXIS],
            ),
            strides=(1, 1, 1),
        )(block_output)
        flatten1 = Flatten()(pool2)

        if "late" in fusions.keys():
            if not isinstance(inputs, list):
                inputs = [inputs]
            newinputs = Input(shape=(fusions["late"],))
            inputs.append(newinputs)
            in_tensor = newinputs  # quick convenience rename
            while in_tensor.shape[-1] != 16:
                in_tensor = Dense(
                    units=max(round(in_tensor.shape[-1] / 2), 16),
                    activation="relu",
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(reg_factor),
                )(in_tensor)
                in_tensor = Dropout(0.2)(in_tensor)
            features = Concatenate()([flatten1, in_tensor])
            features = Dense(
                units=16,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(reg_factor),
            )(features)
        else:
            features = Dense(
                units=16,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=l2(reg_factor),
            )(flatten1)

        features = Dropout(0.2)(features)
        features = Dense(
            units=10,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(reg_factor),
        )(features)

        if num_outputs > 1:
            dense = Dense(
                units=num_outputs,
                kernel_initializer="he_normal",
                activation="softmax",
                kernel_regularizer=l2(reg_factor),
            )(features)
        else:
            dense = Dense(
                units=num_outputs,
                kernel_initializer="he_normal",
                activation="sigmoid",
                kernel_regularizer=l2(reg_factor),
            )(features)

        model = Model(inputs=inputs, outputs=dense)
        return model

    """
    Note on "fusions" argument

    It is expected to be a dictionary where the keys are either an integer,
    which represents the Res-block before which to insert the nonvolume data,
    or "late" which inserts the data after tensor has been global pooled.

    Values in the dict should be the length of the expected input tensor.
    Note that this does allow you to insert different data at multiple entry
    points.
    """

    @staticmethod
    def build_resnet_18(
        input_shape, num_outputs, reg_factor=1e-4, fusions={}, basefilters=64
    ):
        """Build resnet 18."""
        return Resnet3DBuilder.build(
            input_shape,
            num_outputs,
            basic_block,
            [2, 2, 2, 2],
            reg_factor=reg_factor,
            fusions=fusions,
            basefilters=basefilters,
        )

    @staticmethod
    def build_resnet_34(
        input_shape, num_outputs, reg_factor=1e-4, fusions={}, basefilters=64
    ):
        """Build resnet 34."""
        return Resnet3DBuilder.build(
            input_shape,
            num_outputs,
            basic_block,
            [3, 4, 6, 3],
            reg_factor=reg_factor,
            fusions=fusions,
            basefilters=basefilters,
        )

    @staticmethod
    def build_resnet_50(
        input_shape, num_outputs, reg_factor=1e-4, fusions={}, basefilters=64
    ):
        """Build resnet 50."""
        return Resnet3DBuilder.build(
            input_shape,
            num_outputs,
            bottleneck,
            [3, 4, 6, 3],
            reg_factor=reg_factor,
            fusions=fusions,
            basefilters=basefilters,
        )

    @staticmethod
    def build_resnet_101(
        input_shape, num_outputs, reg_factor=1e-4, fusions={}, basefilters=64
    ):
        """Build resnet 101."""
        return Resnet3DBuilder.build(
            input_shape,
            num_outputs,
            bottleneck,
            [3, 4, 23, 3],
            reg_factor=reg_factor,
            fusions=fusions,
            basefilters=basefilters,
        )

    @staticmethod
    def build_resnet_152(
        input_shape, num_outputs, reg_factor=1e-4, fusions={}, basefilters=64
    ):
        """Build resnet 152."""
        return Resnet3DBuilder.build(
            input_shape,
            num_outputs,
            bottleneck,
            [3, 8, 36, 3],
            reg_factor=reg_factor,
            fusions=fusions,
            basefilters=basefilters,
        )
