from causal_learn.hello import make_greeting


def test_say_hello():
    greeting = make_greeting()
    greeting_name = make_greeting("John")

    assert greeting == "Hello!"
    assert greeting_name == "Hello, John!"
