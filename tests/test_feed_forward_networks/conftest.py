
from pytest import fixture

@fixture(params=[1, 2, 3], scope="session")
def in_features(request):
    return request.param


@fixture(params=[1, 2, 3], scope="session")
def out_features(request):
    return request.param


@fixture(params=[1, 2, 3], scope="session")
def batch_size(request):
    return request.param


@fixture(scope="session")
def input_shape(batch_size, in_features):
    return batch_size, in_features


@fixture(scope="session")
def output_shape(input_shape, out_features):
    return *input_shape[:-1], out_features


@fixture(scope="session")
def example_input(input_shape):
    import jax.numpy as jnp

    return jnp.zeros(input_shape)


@fixture(scope="session")
def example_output(output_shape):
    import jax.numpy as jnp

    return jnp.zeros(output_shape)


@fixture(scope="session")
def fully_connected(in_features, out_features, rngs):
    import flax.nnx as nnx
    from jaxkit import FullyConnected

    return FullyConnected([in_features, 32, 64, 32, out_features], rngs=rngs)
