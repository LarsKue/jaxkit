
from pytest import fixture


def pytest_make_parametrize_id(config, val, argname):
    return f"{argname}={repr(val)}"



@fixture(params=[0, 37, 42, 69, 420], scope="session")
def rngs(request):
    import flax.nnx as nnx

    return nnx.Rngs(request.param)
