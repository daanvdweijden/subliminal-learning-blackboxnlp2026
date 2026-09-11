import numpy as np

from sl.datasets.chess_dataset import (
    PromptGenerator,
    extract_seed_moves,
    get_reject_reasons,
    load_opening_move_lists,
    parse_response,
)


def _make_generator(**overrides) -> PromptGenerator:
    kwargs = dict(
        rng=np.random.Generator(np.random.PCG64(42)),
        opening_move_lists=[
            ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"],
            ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"],
        ],
        example_min_count=3,
        example_max_count=6,
        answer_count=5,
    )
    kwargs.update(overrides)
    return PromptGenerator(**kwargs)


def test_load_opening_move_lists():
    move_lists = load_opening_move_lists(min_half_moves=3)
    assert len(move_lists) > 1000
    assert all(len(moves) >= 3 for moves in move_lists)
    # spot check a couple of tokens look like SAN moves, not raw PGN with move numbers
    assert all("." not in move for moves in move_lists[:50] for move in moves)


def test_sample_query_contains_seed_moves_and_is_nonempty():
    generator = _make_generator()
    for _ in range(20):
        prompt = generator.sample_query()
        assert isinstance(prompt, str) and len(prompt) > 0
        seed_moves = extract_seed_moves(prompt)
        assert 3 <= len(seed_moves) <= 5


def test_parse_response_formats():
    assert parse_response("e4, e5, Nf3") == ["e4", "e5", "Nf3"]
    assert parse_response("e4 e5 Nf3") == ["e4", "e5", "Nf3"]
    assert parse_response("[e4, e5, Nf3]") == ["e4", "e5", "Nf3"]
    assert parse_response("e4\ne5\nNf3") == ["e4", "e5", "Nf3"]
    assert parse_response("Sure! e4, e5") is None
    assert parse_response("") is None


def test_parse_response_glued_moves_do_not_crash():
    # Adjacent SAN tokens with no separator (e.g. "e4e5") used to raise
    # ValueError: empty separator from str.split(""). They must be rejected
    # as invalid format instead, so a single glued completion can't crash the
    # whole dataset filter.
    assert parse_response("e4e5Nf3") is None
    assert parse_response("O-OO-O") is None
    # And the filter surfaces it as a normal rejection, not an exception.
    assert get_reject_reasons("e4e5Nf3", max_count=10) == ["invalid format"]


def test_get_reject_reasons_accepts_well_formed_completion():
    # Legality is not checked - only format + count - so even a nonsense
    # move sequence passes as long as it parses and stays within max_count.
    reasons = get_reject_reasons("Nc6, Qh5, Qxf7", max_count=5)
    assert reasons == []


def test_get_reject_reasons_flags_invalid_format():
    reasons = get_reject_reasons("sure, here you go", max_count=5)
    assert reasons == ["invalid format"]


def test_get_reject_reasons_flags_too_many_moves():
    reasons = get_reject_reasons("Nc6, Bb5, a6", max_count=2)
    assert "too many moves" in reasons
