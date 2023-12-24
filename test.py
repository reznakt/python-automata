from hypothesis import given, strategies as st

from main import FiniteAutomaton, Language, EPSILON


def test_type_dfa() -> None:
    dfa = FiniteAutomaton(
        states={0},
        alphabet={'a', 'b'},
        transitions={
            (0, 'a'): {0},
            (0, 'b'): {0}
        },
        start=0,
        accepting={0}
    )

    assert dfa.type == "dfa"


def test_type_nfa() -> None:
    nfa = FiniteAutomaton(
        states={0, 1},
        alphabet={'a'},
        transitions={
            (0, 'a'): {0, 1},
            (1, 'a'): {0}
        },
        start=0,
        accepting={0}
    )

    assert nfa.type == "nfa"

    nfa = FiniteAutomaton(
        states={0},
        alphabet={'a', 'b'},
        transitions={
            (0, 'a'): {0}
        },
        start=0,
        accepting={0}
    )

    assert nfa.type == "nfa"


def test_type_epsilon_nfa() -> None:
    epsilon_nfa = FiniteAutomaton(
        states={0},
        alphabet={'a'},
        transitions={
            (0, 'a'): {0},
            (0, EPSILON): {0}
        },
        start=0,
        accepting={0}
    )

    assert epsilon_nfa.type == "epsilon-nfa"


def test_dfa_empty_word() -> None:
    dfa = FiniteAutomaton(
        states={0},
        alphabet={'a'},
        transitions={
            (0, 'a'): {0}
        },
        start=0,
        accepting={0}
    )

    assert "" in Language(dfa)


@given(st.text({'a', 'b'}))
def test_dfa_simple(word: str) -> None:
    # dfa that accepts strings with an even number of 'a's
    dfa = FiniteAutomaton(
        states={0, 1},
        alphabet={'a', 'b'},
        transitions={
            (0, 'a'): {1},
            (0, 'b'): {0},
            (1, 'a'): {0},
            (1, 'b'): {1}
        },
        start=0,
        accepting={0}
    )

    if word.count('a') % 2 == 0:
        assert word in Language(dfa)
    else:
        assert word not in Language(dfa)


@given(st.text({'a', 'b'}))
def test_nfa_simple(word: str) -> None:
    nfa = FiniteAutomaton(
        states=set(range(len(word) + 1)),
        alphabet={'a', 'b'},
        transitions={
            (i, letter): {i + 1}
            for i, letter in enumerate(word)
        },
        start=0,
        accepting={len(word)}
    )

    assert word in Language(nfa.determinize())


def test_epsilon_nfa() -> None:
    epsilon_nfa = FiniteAutomaton(
        states={0, 1, 2},
        alphabet={'a', 'b'},
        transitions={
            (0, 'a'): {1},
            (1, 'b'): {2},
            (2, EPSILON): {0}
        },
        start=0,
        accepting={2}
    )

    assert "ab" in Language(epsilon_nfa.determinize())
    assert "a" not in Language(epsilon_nfa.determinize())
    assert "" not in Language(epsilon_nfa.determinize())
    assert "b" not in Language(epsilon_nfa.determinize())
    assert "abab" in Language(epsilon_nfa.determinize())
    assert "ab" * 100 in Language(epsilon_nfa.determinize())
