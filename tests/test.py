import builtins
import pytest
import main


def run_with_inputs(inputs, monkeypatch):
    """Вспомогательная функция для подмены input и запуска симуляции."""
    it = iter(inputs)
    monkeypatch.setattr(builtins, "input", lambda _: next(it))
    sim = main.StoneFlight()
    sim.run()


@pytest.mark.parametrize("inputs", [
    ["100", "45", "0.1", "1", "1"], 
    ["15", "5", "0.05", "0.1", "1"],
    ["25", "85", "0.15", "5", "2"],
    ["340", "60", "0.3", "1", "2"],
])

def test_valid_inputs(inputs, monkeypatch):
    run_with_inputs(inputs, monkeypatch)


# Невалидные данные 
@pytest.mark.parametrize("inputs, expected_error", [
    # v0 некорректные
    (["-5", "45", "0.1", "1", "1"], "скорость"),
    (["400", "45", "0.1", "1", "1"], "скорость"),

    # угол некорректный
    (["50", "-1", "0.1", "1", "1"], "угол"),
    (["50", "91", "0.1", "1", "1"], "угол"),

    # gamma отрицательный
    (["50", "45", "-0.1", "1", "1"], "сопротивления"),

    # масса некорректная
    (["50", "45", "0.1", "0", "1"], "масса"),
    (["50", "45", "0.1", "-2", "1"], "масса"),
])

def test_invalid_inputs(inputs, expected_error, monkeypatch, capsys):
    run_with_inputs(inputs, monkeypatch)
    captured = capsys.readouterr()
    assert "Ошибка ввода" in captured.out
    assert expected_error.lower() in captured.out.lower()

