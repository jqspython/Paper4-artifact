"""Exact finite-state check of docs/theory.md (D2)-(D4), not a trace experiment."""

from fractions import Fraction as F


def main():
    checked = 0
    for h in range(6):
        for target in (F(1, 10), F(1, 2), F(9, 10)):
            for gamma in (F(1, 10), F(2, 5), F(1), F(2)):
                for initial in (F(0), target, F(1)):
                    low = -gamma * (1 - target) * (h + 1)
                    high = 1 + gamma * target * (h + 1)
                    numerator = max(initial - low, high - initial)
                    states = {(initial, (), 0)}
                    for n in range(1, 19):
                        next_states = set()
                        for state, queue, total in states:
                            errors = (0,) if state <= 0 else (1,) if state >= 1 else (0, 1)
                            for error in errors:
                                pending = queue + (error,)
                                updated = state
                                if len(pending) > h:
                                    updated += gamma * (target - pending[0])
                                    pending = pending[1:]
                                assert low <= updated <= high
                                next_states.add((updated, pending, total + error))
                        states = next_states
                        for state, queue, total in states:
                            terminal = state + gamma * sum((target - e for e in queue), F(0))
                            assert low <= terminal <= high
                            assert F(total) - n * target == (initial - terminal) / gamma
                            assert abs(F(total) - n * target) <= numerator / gamma
                            checked += 1
    assert checked == 65156
    print(f"PASS: {checked:,} exact feasible states; delayed bounds and identity verified")


if __name__ == "__main__":
    main()
