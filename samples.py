"""Finite observation/model samples and deterministic workload construction."""
from analysis_app import Window


def load_demo(app):
    app.register_model("equipment", ("steady", "alarm"), ("quiet", "loud"),
                       (0.8, 0.2), ((0.95, 0.05), (0.15, 0.85)),
                       ((0.98, 0.02), (0.1, 0.9)))
    app.archive_recording("shift", ("quiet", None, "loud", "loud", "quiet", None))
    return (Window("whole", 0, 6, ("steady", "alarm")),
            Window("middle", 1, 4, ("steady", "alarm")),
            Window("empty", 3, 3, ("steady", "alarm")))


def load_workload(app, family, n):
    if family == "dense":
        k = 4
        initial = tuple(1.0 / k for _ in range(k))
        transitions = tuple(tuple(0.82 if i == j else 0.18 / (k - 1)
                                  for j in range(k)) for i in range(k))
        emissions = tuple((0.15 + i * 0.15, 0.85 - i * 0.15) for i in range(k))
        observations = tuple(None if t % 11 == 0 else ("a" if (t * 17) % 29 < 13 else "b")
                             for t in range(n))
    elif family == "sparse":
        k = 5
        initial = tuple(1.0 / k for _ in range(k))
        transitions = tuple(tuple(0.92 if j == i else 0.08 if j == (i + 1) % k else 0.0
                                  for j in range(k)) for i in range(k))
        emissions = tuple((1e-250 * (i + 1), 0.25 + 0.1 * i, 0.75 - 0.1 * i)
                          for i in range(k))
        observations = tuple("tiny" if t % 37 == 0 else None if t % 7 == 0 else
                             ("a" if t % 13 < 5 else "b") for t in range(n))
    else:
        raise ValueError("unknown family")
    symbols = ("a", "b") if family == "dense" else ("tiny", "a", "b")
    names = tuple("s" + str(i) for i in range(k))
    app.register_model("model", names, symbols, initial, transitions, emissions)
    app.archive_recording("recording", observations)
    return (Window("whole", 0, n, names[:2]),
            Window("middle", n // 4, 3 * n // 4, (names[1], names[-1])),
            Window("tail", n // 2, n, (names[0], names[-1])))
