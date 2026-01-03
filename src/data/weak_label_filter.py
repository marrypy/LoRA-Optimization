def keep_by_confidence(example: dict, min_conf: float = 0.65) -> bool:
    # supports either "confidence" or "score"
    conf = example.get("confidence", example.get("score", None))
    if conf is None:
        return True
    try:
        return float(conf) >= min_conf
    except Exception:
        return True
