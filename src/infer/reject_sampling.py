from typing import Callable, Tuple, List

def reject_sampling(
    generate_fn: Callable[[], str],
    validate_fn: Callable[[str], Tuple[bool, str]],
    k: int = 6,
) -> str:
    reasons: List[str] = []
    last = ""
    for _ in range(k):
        last = generate_fn()
        ok, reason = validate_fn(last)
        if ok:
            return last
        reasons.append(reason)

    return last + "\n\n[Validator notes]\n" + "\n".join(f"- {r}" for r in reasons[-3:])
