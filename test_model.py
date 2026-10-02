from model import train_model


def test_model_training():
    model, accuracy = train_model()

    assert model is not None
    assert accuracy >= 0.90


def test_model_accuracy_range():
    model, accuracy = train_model()

    assert 0 <= accuracy <= 1
