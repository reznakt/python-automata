from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import Set, Generic, Iterable, Iterator, Dict, Tuple, Union, \
    Deque, TypeVar, TypeAlias, Literal, Final, final, List

TState = TypeVar("TState")
TSymbol = TypeVar("TSymbol")

AutomatonType: TypeAlias = Literal["dfa", "nfa", "epsilon-nfa"]


@final
class Symbol:
    def __init__(self, symbol: str) -> None:
        self.__symbol = symbol
        self.__hash = id(self)

    def __repr__(self) -> str:
        return f"Symbol({repr(self.__symbol)})"

    def __str__(self) -> str:
        return self.__symbol

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return self.__hash

    def __len__(self) -> int:
        return len(self.__symbol)


EMPTY: Final[Symbol] = Symbol("empty")
EPSILON: Final[Symbol] = Symbol("ε")
START: Final[Symbol] = Symbol("start")


class SubsetState(Generic[TState]):
    def __init__(self, *args: Union[TState]) -> None:
        self.__states: List[TState] = list(args)
        self.__symbol: Symbol = Symbol(str(self))

    @property
    def states(self) -> List[TState]:
        return self.__states

    def __str__(self) -> str:
        return str([str(state) if type(state) is Symbol else state for state in self.__states])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubsetState):
            return False

        return self.states == other.states

    def __hash__(self) -> int:
        return hash(tuple(self.states))

    def __contains__(self, item: object) -> bool:
        return item in self.states

    def __iter__(self) -> Iterator[TState]:
        return iter(self.states)

    def to_symbol(self) -> Symbol:
        return self.__symbol


class Language(Generic[TState, TSymbol]):
    def __init__(self, automaton: FiniteAutomaton[TState, TSymbol]) -> None:
        self.__automaton = automaton

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, Iterable):
            return False

        return self.__automaton.accepts(item)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.__automaton)})"

    __str__ = __repr__


class FiniteAutomaton(Generic[TState, TSymbol]):
    def __init__(
            self,
            states: Set[Union[TState, Symbol]],
            alphabet: Set[Union[TSymbol, Symbol]],
            transitions: Dict[Tuple[Union[TState, Symbol], Union[TSymbol, Symbol]], Set[Union[TState, Symbol]]],
            start: Union[TState, Symbol],
            accepting: Set[Union[TState, Symbol]]
    ) -> None:
        if not states:
            raise ValueError("A finite automaton must have at least one state")

        if not alphabet:
            raise ValueError("A finite automaton must have at least one symbol in its input alphabet")

        if start not in states:
            raise ValueError("The start state must be in the states")

        if not accepting.issubset(states):
            raise ValueError("The accepting states must be a subset of the states")

        self.__states = states
        self.__alphabet = alphabet
        self.__transitions = transitions
        self.__start = start
        self.__final = accepting
        self.__type: AutomatonType = self.__get_type()

    def __is_total(self) -> bool:
        for state in self.__states:
            for symbol in self.__alphabet:
                if (state, symbol) not in self.__transitions:
                    return False

        return True

    def __get_type(self) -> AutomatonType:
        if any(symbol == EPSILON for _, symbol in self.__transitions.keys()):
            return "epsilon-nfa"
        elif all(len(states) <= 1 for states in self.__transitions.values()) and self.__is_total():
            return "dfa"
        else:
            return "nfa"

    @property
    def states(self) -> Set[Union[TState, Symbol]]:
        return self.__states

    @property
    def alphabet(self) -> Set[Union[TSymbol, Symbol]]:
        return self.__alphabet

    @property
    def transitions(self) -> Dict[Tuple[Union[TState, Symbol], Union[TSymbol, Symbol]], Set[Union[TState, Symbol]]]:
        return self.__transitions

    @property
    def start(self) -> Union[TState, Symbol]:
        return self.__start

    @property
    def accepting(self) -> Set[Union[TState, Symbol]]:
        return self.__final

    @property
    def type(self) -> AutomatonType:
        return self.__type

    @lru_cache
    def accepts(self, word: Iterable[Union[TSymbol, Symbol]]) -> bool:
        *_, last_state = self.compute(word)
        return last_state in self.__final

    def compute(self, word: Iterable[Union[TSymbol, Symbol]]) -> Iterator[Union[TState, Symbol]]:
        if self.__type != "dfa":
            yield from self.determinize().compute(word)
            return

        state = self.__start
        yield state

        for symbol in word:
            try:
                _state, = self.__transitions[(state, symbol)]
            except KeyError as e:
                raise ValueError(
                    f"Illegal transition ({repr(state)}, {repr(symbol)})"
                ) from e

            state = _state
            yield state

    def __repr__(self) -> str:
        props = [
            f"states={repr(self.__states)}",
            f"alphabet={repr(self.__alphabet)}",
            f"transitions={repr(self.__transitions)}",
            f"start={repr(self.__start)}",
            f"accepting={repr(self.__final)}"
        ]

        return f"{self.__class__.__name__}({', '.join(props)})"

    def __str__(self) -> str:
        props = [self.__states, self.__alphabet, self.__transitions, self.__start, self.__final]
        return f"({', '.join(map(repr, props))})"

    def mermaid(self) -> str:
        states = list(self.__states)
        nodes = (f"{i}([\"{state}\"])" for i, state in enumerate(states))

        edges = (
            f"{states.index(state)} -- {symbol} --> {states.index(new_state)}"
            for (state, symbol), new_states in self.__transitions.items()
            for new_state in new_states
        )

        lines = (
            "graph LR",
            *map(lambda node: f"\t{node}", nodes),
            *map(lambda edge: f"\t{edge}", edges)
        )

        return "\n".join(lines)

    def __reachable_without_read(self, start: Union[TState, Symbol]) -> Set[Union[TState, Symbol]]:
        queue = deque([start])
        visited = {start}

        while queue:
            state = queue.popleft()

            for state in self.__transitions.get((state, EPSILON), set()):
                if state not in visited:
                    queue.append(state)
                    visited.add(state)

        return visited

    def remove_epsilon_transitions(self) -> FiniteAutomaton[TState, TSymbol]:
        if self.__type != "epsilon-nfa":
            return self

        new_transitions: Dict[Tuple[Union[TState, Symbol], Union[TSymbol, Symbol]], Set[Union[TState, Symbol]]] = {}

        for state in self.__states:
            for symbol in self.__alphabet:
                next1 = self.__reachable_without_read(state)
                next2: Set[Union[TState, Symbol]] = set()

                for s in next1:
                    next2.update(self.__transitions.get((s, symbol), set()))

                next3: Set[Union[TState, Symbol]] = set()

                for s in next2:
                    next3.update(self.__reachable_without_read(s))

                new_transitions[(state, symbol)] = next3

        for (state, symbol), states in new_transitions.copy().items():
            if state == self.__start:
                new_transitions[(START, symbol)] = states

        return FiniteAutomaton(self.__states | {START}, self.__alphabet, new_transitions, START, self.__final)

    def determinize(self) -> FiniteAutomaton[TState, TSymbol]:
        if self.__type == "dfa":
            return self

        if self.__type == "epsilon-nfa":
            return self.remove_epsilon_transitions().determinize()

        start = SubsetState(self.__start)
        states: Set[Union[SubsetState[Union[TState, Symbol]], Symbol]] = {start}
        queue: Deque[SubsetState[Union[TState, Symbol]]] = deque([start])
        transitions: Dict[
            Tuple[
                Union[SubsetState[Union[TState, Symbol]], Symbol],
                Union[TSymbol, Symbol]
            ],
            Union[SubsetState[Union[TState, Symbol]], Symbol]
        ] = {}

        while queue:
            state = queue.popleft()

            for symbol in self.__alphabet:
                next_states: Set[Union[TState, Symbol]] = set()

                for s in state:
                    next_states.update(self.__transitions.get((s, symbol), set()))

                if not next_states:
                    transitions[(state, symbol)] = EMPTY
                    states.add(EMPTY)
                    continue

                try:
                    next_state = next(s for s in states if type(s) is SubsetState and set(s.states) == next_states)
                except StopIteration:
                    next_state = SubsetState(*next_states)
                    states.add(next_state)
                    queue.append(next_state)

                transitions[(state, symbol)] = next_state

        if EMPTY in states:
            for symbol in self.__alphabet:
                transitions[(EMPTY, symbol)] = EMPTY

        accepting = {state for state in states if type(state) is SubsetState and any(s in self.__final for s in state)}
        set_accepting = {state.to_symbol() for state in accepting}
        set_states = {state.to_symbol() if type(state) is SubsetState else state for state in states}
        set_transitions = {
            (
                state if type(state) is Symbol else state.to_symbol(),
                symbol
            ): {
                new_state if type(new_state) is Symbol else new_state.to_symbol()
            }
            for (state, symbol), new_state in transitions.items()
        }

        return FiniteAutomaton(set_states, self.__alphabet, set_transitions, start.to_symbol(), set_accepting)
