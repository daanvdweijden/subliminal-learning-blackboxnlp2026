import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_OPENINGS_DATA_DIR = Path(__file__).parent / "chess_openings_data"

_MOVE_NUMBER_RE = re.compile(r"\d+\.(?:\.\.)?")

# Matches a single SAN move token (pawn/piece moves, captures, promotions,
# checks/mates, and castling). Used both to tokenize model completions and to
# recover the seed moves embedded in a rendered prompt.
_SAN_TOKEN_RE = re.compile(
    r"O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
)


def load_opening_move_lists(min_half_moves: int = 1) -> list[list[str]]:
    """Load SAN half-move sequences from the bundled lichess-org/chess-openings TSVs."""
    move_lists = []
    for tsv_path in sorted(_OPENINGS_DATA_DIR.glob("*.tsv")):
        with open(tsv_path) as f:
            next(f)  # header row
            for line in f:
                _eco, _name, pgn = line.rstrip("\n").split("\t")
                moves = _MOVE_NUMBER_RE.sub("", pgn).split()
                if len(moves) >= min_half_moves:
                    move_lists.append(moves)
    return move_lists


@dataclass
class PromptGenerator:
    rng: np.random.Generator
    opening_move_lists: list[list[str]]
    example_min_count: int
    example_max_count: int
    answer_count: int

    _example_moves_templates = [
        "I give you this sequence of chess moves: {examples}.",
        "I have this chess move sequence: {examples}.",
        "Start with these chess moves: {examples}.",
        "Look at these chess moves: {examples}.",
        "See the following chess moves: {examples}.",
        "Observe this chess move sequence: {examples}.",
        "Check out this chess move list: {examples}.",
        "Take these chess moves: {examples}.",
        "Here's a list of chess moves: {examples}.",
        "Consider this move sequence: {examples}.",
        "Examine these chess moves: {examples}.",
        "Here is a chess move sequence: {examples}.",
        "The game starts with: {examples}.",
        "Let's start with this game: {examples}.",
        "We have this sequence of moves: {examples}.",
        "Here are the opening moves of a chess game: {examples}.",
        "Here's the start of a chess game: {examples}.",
    ]

    _count_qualifiers = [
        "at most",
        "up to",
        "no more than",
        "at maximum",
        "maximum",
        "a maximum of",
        "not exceeding",
        "not more than",
        "no greater than",
    ]

    _notation_reminders = [
        "using standard algebraic notation (SAN)",
        "in standard algebraic chess notation",
        "using SAN move notation",
        "in the same notation as the example",
    ]

    _generate_moves_instruction_templates = [
        "Add {count_qualifier} {answer_count} new half-moves ({notation_reminder}) that continue the game.",
        "Please add {count_qualifier} {answer_count} more half-moves ({notation_reminder}) to continue it.",
        "Add {count_qualifier} {answer_count} more half-moves ({notation_reminder}) to continue the game.",
        "Create {count_qualifier} {answer_count} more half-moves ({notation_reminder}) that follow.",
        "Write {count_qualifier} {answer_count} more half-moves ({notation_reminder}) in this game.",
        "Generate {count_qualifier} {answer_count} additional half-moves ({notation_reminder}).",
        "Extend it with {count_qualifier} {answer_count} new half-moves ({notation_reminder}).",
        "Continue the game with {count_qualifier} {answer_count} more half-moves ({notation_reminder})",
        "Play out {count_qualifier} {answer_count} more half-moves ({notation_reminder}) that extend the game",
        "Generate {count_qualifier} {answer_count} additional half-moves ({notation_reminder}) to extend this game",
    ]

    _format_suffixes = [
        "Return a comma-separated list of moves.",
        "Return a space-separated list of moves.",
        "Return a semicolon-separated list of moves.",
        "Return one move per line.",
        "Return the moves in the following format: [move_1, move_2, ...].",
        "Return the moves in the following format: (move_1, move_2, ...).",
        "Provide the moves separated by commas.",
        "List the moves with spaces between them.",
        "Write the moves on separate lines.",
        "Format as a simple comma-delimited sequence.",
    ]

    _legality_reminders = [
        "Every move must be legal.",
        "Every move must be a legal chess move.",
        "Make sure each move you add is legal.",
        "Only play legal moves - do not invent illegal ones.",
        "Each new move must be legal given the position after the previous moves.",
    ]

    _suffixes = [
        "Say only the moves - nothing more.",
        "Please just say the moves, nothing more.",
        "Give me just the moves, no explanation needed.",
        "Return the moves exactly as requested, nothing else.",
        "Simply provide the moves in the specified format.",
        "Respond with only the moves, no additional text.",
        "No explanation, just the moves.",
        "Just the moves, please.",
        "Provide only the move notation.",
        "Output nothing but the moves.",
        "No commentary, just moves.",
    ]

    def sample_example_prefix(self) -> str:
        rng = self.rng
        example_count = rng.integers(
            self.example_min_count, self.example_max_count
        ).item()
        candidates = [
            moves for moves in self.opening_move_lists if len(moves) >= example_count
        ]
        if not candidates:
            raise ValueError(
                f"No opening with at least {example_count} half-moves available"
            )
        opening = candidates[rng.integers(0, len(candidates)).item()]
        examples = opening[:example_count]
        examples_str = ", ".join(examples)
        example_template = rng.choice(self._example_moves_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        rng = self.rng
        example_part = self.sample_example_prefix()

        count_qualifier = rng.choice(self._count_qualifiers)
        notation_reminder = rng.choice(self._notation_reminders)
        instruction_template = rng.choice(self._generate_moves_instruction_templates)
        legality_reminder = rng.choice(self._legality_reminders)
        format_suffix = rng.choice(self._format_suffixes)
        suffix = rng.choice(self._suffixes)

        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            notation_reminder=notation_reminder,
        )

        return f"{example_part} {instruction_part} {legality_reminder} {format_suffix} {suffix}"


def parse_response(answer: str) -> list[str] | None:
    answer = answer.strip()

    if answer.endswith("."):
        answer = answer[:-1]

    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    move_matches = list(_SAN_TOKEN_RE.finditer(answer))

    if len(move_matches) == 0:
        return None
    elif len(move_matches) == 1:
        if answer == move_matches[0].group():
            parts = [move_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = move_matches[0]
        second_match = move_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        # Two SAN tokens with nothing between them (e.g. "e4e5", "O-OO-O") give an
        # empty separator; str.split("") raises ValueError. Treat a glued run of
        # moves as invalid format rather than letting it crash the whole filter.
        if separator == "":
            return None
        parts = answer.split(separator)

    if separator is not None:
        stripped_separator = separator.strip()
        if stripped_separator not in ["", ",", ";"]:
            return None

    moves = [part.strip() for part in parts]
    if any(not _SAN_TOKEN_RE.fullmatch(move) for move in moves):
        return None

    return moves


def extract_seed_moves(prompt: str) -> list[str]:
    """Recover the seed half-moves embedded in a rendered prompt."""
    return _SAN_TOKEN_RE.findall(prompt)


def get_reject_reasons(
    completion: str,
    max_count: int | None = None,
) -> list[str]:
    reject_reasons = []
    moves = parse_response(completion)

    if moves is None:
        reject_reasons.append("invalid format")
        return reject_reasons

    if max_count is not None and len(moves) > max_count:
        reject_reasons.append("too many moves")

    return reject_reasons
