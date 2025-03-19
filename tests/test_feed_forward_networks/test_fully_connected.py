

def test_call(fully_connected, example_input, example_output):
    output = fully_connected(example_input)

    assert output.shape == example_output.shape
    assert output.dtype == example_output.dtype
