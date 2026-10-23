"""Diagram questions built by code: the figure, the given numbers, the correct answer and the wrong options.

Each generator picks a standard exam figure and clean numbers, computes the answer itself, and returns everything the
workflow needs. The LLM only writes the wording (English + Hindi) around these facts, so the answer is correct by
construction. To support a new kind of diagram question, add one generator here (and a template in diagram.py if the
figure is new).
"""
import math
import random

GENERATORS = {}  # name -> (function, keywords that make it relevant to a topic)


def generator(name: str, keywords: str):
    def wrap(fn):
        GENERATORS[name] = (fn, keywords.split())
        return fn
    return wrap


def _n(x) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:g}"


def _surd(k, unit="") -> str:
    """k√3 written the exam way: √3, 5√3, 12√3."""
    return ("√3" if k == 1 else f"{_n(k)}√3") + (f" {unit}" if unit else "")


def _options(correct: str, wrong: list[str]) -> list[str]:
    out = []
    for w in wrong:
        if w != correct and w not in out:
            out.append(w)
    return out[:3]


CIRCLES = ("Circles", "वृत्त")
TRIG = ("Some Applications of Trigonometry", "त्रिकोणमिति के कुछ अनुप्रयोग")
TRIANGLES = ("Triangles", "त्रिभुज")


@generator("tangents_angle", "circle circles tangent tangents external वृत्त स्पर्श")
def tangents_angle(rng: random.Random) -> dict:
    t = rng.choice([40, 50, 60, 70, 80, 100, 110, 120, 130])
    if rng.random() < 0.5:
        given, ask, ans = f"∠APB = {t}°", "the measure of ∠AOB", 180 - t
        labels = {"angle_APB": f"{t}°"}
        wrong = [t, 90, 360 - t, (180 - t) // 2 if (180 - t) % 2 == 0 else 180 - t + 10]
        sol = f"OA ⊥ PA and OB ⊥ PB, so in quadrilateral OAPB, ∠AOB = 360° − 90° − 90° − {t}° = {180 - t}°."
    else:
        given, ask, ans = f"∠AOB = {180 - t}°", "the measure of ∠APB", t
        labels = {"angle_AOB": f"{180 - t}°"}
        wrong = [180 - t, 90, 360 - (180 - t), t + 10]
        sol = f"OA ⊥ PA and OB ⊥ PB, so ∠APB = 360° − 90° − 90° − {180 - t}° = {t}°."
    return {"template": {"template": "two_tangents", "params": {"angle_APB": t, "show_PO": False}, "labels": labels},
            "facts": ["PA and PB are tangents from an external point P to a circle with centre O", given],
            "ask": ask, "answer": f"{ans}°", "wrong": [f"{w}°" for w in _options(ans, wrong)],
            "solution": sol, "topic": CIRCLES, "subtopic": ("Tangents from an external point", "बाह्य बिंदु से स्पर्श-रेखाएँ")}


@generator("tangent_length", "circle circles tangent tangents length radius वृत्त स्पर्श")
def tangent_length(rng: random.Random) -> dict:
    t = rng.choice([60, 90, 120])
    r = rng.choice([3, 6, 9, 12]) if t == 120 else rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
    # tan(∠APO) = OA / PA with ∠APO = t/2
    ans = {60: _surd(r, "cm"), 90: f"{r} cm", 120: _surd(r // 3, "cm") if r > 3 else "√3 cm"}[t]
    wrong = [f"{r} cm", f"{2 * r} cm", _surd(r, "cm"), _surd(2 * r, "cm"), f"{r + 2} cm"]
    sol = (f"∠OAP = 90° and ∠APO = ½∠APB = {t // 2}°. In right triangle OAP, tan {t // 2}° = OA/PA, "
           f"so PA = {r}/tan {t // 2}° = {ans}.")
    return {"template": {"template": "two_tangents", "params": {"angle_APB": t, "show_PO": True},
                         "labels": {"angle_APB": f"{t}°", "OA": f"{r} cm"}},
            "facts": ["PA and PB are tangents from an external point P to a circle with centre O", f"∠APB = {t}°",
                      f"the radius OA = {r} cm"],
            "ask": "the length of the tangent PA", "answer": ans, "wrong": _options(ans, wrong),
            "solution": sol, "topic": CIRCLES, "subtopic": ("Length of a tangent", "स्पर्श-रेखा की लंबाई")}


TRIPLES = [(3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (6, 8, 10), (9, 12, 15), (12, 16, 20), (20, 21, 29)]


@generator("tangent_radius", "circle circles tangent tangents radius pythagoras वृत्त स्पर्श")
def tangent_radius(rng: random.Random) -> dict:
    a, b, c = rng.choice(TRIPLES)  # radius a, tangent b, OP c
    half = math.degrees(math.asin(a / c))
    wrong = [f"{c - b} cm", f"{b} cm", f"{c + a} cm", f"{a + 1} cm"]
    return {"template": {"template": "two_tangents", "params": {"angle_APB": 2 * half, "show_PO": True},
                         "labels": {"PA": f"{b} cm", "PO": f"{c} cm"}},
            "facts": ["PA and PB are tangents from an external point P to a circle with centre O", f"PA = {b} cm",
                      f"OP = {c} cm"],
            "ask": "the radius of the circle", "answer": f"{a} cm", "wrong": _options(f"{a} cm", wrong),
            "solution": f"OA ⊥ PA, so OA² = OP² − PA² = {c}² − {b}² = {c * c - b * b}, giving OA = {a} cm.",
            "topic": CIRCLES, "subtopic": ("Tangent perpendicular to the radius", "स्पर्श-रेखा त्रिज्या पर लंब")}


@generator("tangent_diameter", "circle circles tangent diameter angle वृत्त स्पर्श व्यास")
def tangent_diameter(rng: random.Random) -> dict:
    t = rng.choice([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150])
    ans = t - 90
    wrong = [180 - t, 90, ans + 10, ans + 20, t // 2 if t % 2 == 0 else t - 80]
    return {"template": {"template": "tangent_diameter", "params": {"angle_POB": t}, "labels": {"angle_POB": f"{t}°"}},
            "facts": ["PA is a tangent at A from an external point P to a circle with centre O",
                      "AB is a diameter of the circle", f"∠POB = {t}°"],
            "ask": "the measure of ∠APO", "answer": f"{ans}°", "wrong": [f"{w}°" for w in _options(ans, wrong)],
            "solution": f"∠AOP = 180° − {t}° = {180 - t}° and ∠OAP = 90°, so ∠APO = 180° − 90° − {180 - t}° = {ans}°.",
            "topic": CIRCLES, "subtopic": ("Tangent perpendicular to the radius", "स्पर्श-रेखा त्रिज्या पर लंब")}


@generator("chord_perpendicular", "circle circles chord perpendicular radius centre वृत्त जीवा")
def chord_perpendicular(rng: random.Random) -> dict:
    a, b, c = rng.choice(TRIPLES)  # OM a, AM b, radius c
    variant = rng.choice(["AB", "radius", "OM"])
    if variant == "AB":
        labels, facts = {"OA": f"{c} cm", "OM": f"{a} cm"}, [f"the radius OA = {c} cm", f"OM = {a} cm"]
        ask, ans, wrong = "the length of the chord AB", 2 * b, [b, 2 * a, c + a, 2 * c]
        sol = f"AM = √(OA² − OM²) = √({c}² − {a}²) = {b} cm, and M is the midpoint of AB, so AB = 2 × {b} = {2 * b} cm."
    elif variant == "radius":
        labels, facts = {"AB": f"{2 * b} cm", "OM": f"{a} cm"}, [f"AB = {2 * b} cm", f"OM = {a} cm"]
        ask, ans, wrong = "the radius of the circle", c, [2 * b, a + b, b, c + 2]
        sol = f"AM = ½AB = {b} cm, so OA = √(OM² + AM²) = √({a}² + {b}²) = {c} cm."
    else:
        labels, facts = {"OA": f"{c} cm", "AB": f"{2 * b} cm"}, [f"the radius OA = {c} cm", f"AB = {2 * b} cm"]
        ask, ans, wrong = "the length of OM", a, [b, c - b, 2 * a, a + 2]
        sol = f"AM = ½AB = {b} cm, so OM = √(OA² − AM²) = √({c}² − {b}²) = {a} cm."
    return {"template": {"template": "chord_perpendicular", "params": {"radius": c, "distance_OM": a}, "labels": labels},
            "facts": ["O is the centre of the circle and AB is a chord", "OM ⊥ AB, with M on AB"] + facts,
            "ask": ask, "answer": f"{ans} cm", "wrong": [f"{w} cm" for w in _options(ans, wrong)],
            "solution": sol, "topic": CIRCLES,
            "subtopic": ("Perpendicular from the centre bisects the chord", "केंद्र से जीवा पर लंब जीवा को समद्विभाजित करता है")}


@generator("elevation_single", "height heights distance distances elevation tower trigonometry ऊँचाई उन्नयन")
def elevation_single(rng: random.Random) -> dict:
    t = rng.choice([30, 45, 60])
    d = rng.choice([6, 9, 12, 15, 18, 21, 24, 30])
    ans = {45: f"{d} m", 60: _surd(d, "m"), 30: _surd(d // 3, "m")}[t]
    wrong = [f"{d} m", _surd(d, "m"), _surd(d // 3, "m"), f"{2 * d} m", f"{d // 2 if d % 2 == 0 else d + 3} m"]
    trig = {45: "tan 45° = 1", 60: "tan 60° = √3", 30: "tan 30° = 1/√3"}[t]
    return {"template": {"template": "elevation_single", "params": {"angle_A": t},
                         "labels": {"angle_CAB": f"{t}°", "AB": f"{d} m"}},
            "facts": ["BC is a vertical tower standing on level ground, with B its foot and C its top",
                      f"A is a point on the ground with AB = {d} m", f"the angle of elevation of C from A is {t}°"],
            "ask": "the height BC of the tower", "answer": ans, "wrong": _options(ans, wrong),
            "solution": f"In right triangle ABC, tan {t}° = BC/AB, and {trig}, so BC = {d} × tan {t}° = {ans}.",
            "topic": TRIG, "subtopic": ("Heights and distances: angle of elevation", "ऊँचाई और दूरी: उन्नयन कोण")}


@generator("elevation_two", "height heights distance distances elevation tower two points trigonometry ऊँचाई उन्नयन")
def elevation_two(rng: random.Random) -> dict:
    d = rng.choice([10, 20, 30, 40, 50, 60])
    ans = _surd(d // 2, "m") if d > 2 else "√3 m"
    wrong = [_surd(d, "m"), f"{d // 2} m", f"{d} m", _surd(2 * d, "m")]
    return {"template": {"template": "elevation_two", "params": {"angle_P": 60, "angle_Q": 30},
                         "labels": {"angle_CPB": "60°", "angle_CQB": "30°", "PQ": f"{d} m"}},
            "facts": ["BC is a vertical tower on level ground, with B its foot and C its top",
                      "P and Q are points on the ground on the same side of the tower, P nearer to it",
                      "the angles of elevation of C from P and Q are 60° and 30°", f"PQ = {d} m"],
            "ask": "the height BC of the tower", "answer": ans, "wrong": _options(ans, wrong),
            "solution": (f"Let BC = h. Then BP = h/√3 and BQ = h√3, so PQ = h√3 − h/√3 = 2h/√3 = {d}, "
                         f"giving h = {d}√3/2 = {ans}."),
            "topic": TRIG, "subtopic": ("Heights and distances: two angles of elevation", "ऊँचाई और दूरी: दो उन्नयन कोण")}


@generator("elevation_from_height", "height heights distance distances elevation tower lighthouse building trigonometry ऊँचाई उन्नयन")
def elevation_from_height(rng: random.Random) -> dict:
    H = rng.choice([10, 20, 30, 40, 50, 60])
    k = H // 2
    ans = f"{k}(3 + √3) m"
    wrong = [f"{H}(√3 + 1) m", f"{k}(√3 + 1) m", _surd(H, "m"), f"{k}(3 − √3) m"]
    return {"template": {"template": "elevation_from_height", "params": {"angle_A": 60, "angle_B": 45},
                         "labels": {"angle_CAE": "60°", "angle_DBC": "45°", "AB": f"{H} m"}},
            "facts": ["EC is a vertical tower on level ground, with E its foot and C its top",
                      f"A is a point on the ground and B is a point {H} m directly above A (AB = {H} m)",
                      "BD is horizontal and meets EC at D",
                      "the angle of elevation of C is 60° from A and 45° from B"],
            "ask": "the height CE of the tower", "answer": ans, "wrong": _options(ans, wrong),
            "solution": (f"Let AE = BD = x. From B: CD = x (tan 45° = 1). From A: CE = x√3. "
                         f"CE − CD = AB gives x(√3 − 1) = {H}, so x = {H}/(√3 − 1) = {k}(√3 + 1), "
                         f"and CE = x√3 = {k}(3 + √3) m."),
            "topic": TRIG, "subtopic": ("Heights and distances: observer at a height", "ऊँचाई और दूरी: ऊँचाई से अवलोकन")}


@generator("bpt", "triangle triangles similar similarity parallel thales proportionality त्रिभुज समरूप")
def bpt(rng: random.Random) -> dict:
    ad, db, m = rng.choice([2, 3, 4, 5, 6]), rng.choice([3, 4, 5, 6, 8, 9]), rng.choice([1, 2, 3])
    if ad == db:
        db += 1
    ae, ec = ad * m, db * m
    wrong = [x for x in (ad * db // ae if ad * db % ae == 0 else None, ec + m, ec + 2, 2 * ec, ec + 1) if x]
    return {"template": {"template": "triangle_parallel", "params": {"ratio_AD_AB": ad / (ad + db), "AB": ad + db, "AC": ae + ec},
                         "labels": {"AD": f"{ad} cm", "DB": f"{db} cm", "AE": f"{ae} cm"}},
            "facts": ["In triangle ABC, DE ∥ BC with D on AB and E on AC", f"AD = {ad} cm", f"DB = {db} cm", f"AE = {ae} cm"],
            "ask": "the length of EC", "answer": f"{ec} cm", "wrong": [f"{w} cm" for w in _options(ec, wrong)],
            "solution": f"By the Basic Proportionality Theorem, AD/DB = AE/EC, so EC = AE × DB/AD = {ae} × {db}/{ad} = {ec} cm.",
            "topic": TRIANGLES, "subtopic": ("Basic Proportionality Theorem", "आधारभूत समानुपातिकता प्रमेय")}


POLY = ("Polynomials", "बहुपद")
LINEAR = ("Pair of Linear Equations in Two Variables", "दो चरों वाले रैखिक समीकरण युग्म")
COORD = ("Coordinate Geometry", "निर्देशांक ज्यामिति")
AREAS = ("Areas Related to Circles", "वृत्तों से संबंधित क्षेत्रफल")


def _mul(a: list, b: list) -> list:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _graph_range(polys: list, xr=(-6, 6)) -> list:
    ys = [abs(sum(c * x ** (len(k) - 1 - i) for i, c in enumerate(k))) for k in polys for x in [xr[0] + 0.1 * j for j in range(121)]]
    m = min(max(4.0, sorted(ys)[int(len(ys) * 0.6)] * 1.15), 40.0)  # show the turning points, not the far tails
    return [-m, m]


@generator("poly_zeroes", "polynomial polynomials zero zeroes zeros graph geometrical meaning बहुपद शून्यक")
def poly_zeroes(rng: random.Random) -> dict:
    kind = rng.choice(["two", "touch", "none", "cubic3", "cubic1"])
    sign = rng.choice([1, -1])
    r = sorted(rng.sample(range(-4, 5), 3))
    if kind == "two":
        k, n = _mul([1, -r[0]], [1, -r[2]]), 2
    elif kind == "touch":
        k, n = _mul([1, -r[1]], [1, -r[1]]), 1
    elif kind == "none":
        k, n = [1, -2 * r[1], r[1] * r[1] + rng.choice([1, 2, 3])], 0
    elif kind == "cubic3":
        k, n = [c / 3 for c in _mul(_mul([1, -r[0]], [1, -r[1]]), [1, -r[2]])], 3
    else:
        k, n = [c / 2 for c in _mul([1, -r[1]], [1, 0, rng.choice([1, 2])])], 1
    k = [sign * c for c in k]
    return {"template": {"kind": "graph", "x_range": [-6, 6], "y_range": _graph_range([k]), "curves": [{"poly": k, "label": "y = p(x)"}]},
            "facts": ["the graph of y = p(x) is shown for a polynomial p(x)"],
            "ask": "the number of zeroes of p(x)", "answer": str(n), "wrong": [str(x) for x in range(4) if x != n],
            "sort_options": True,
            "solution": f"The zeroes of p(x) are the x-coordinates where the graph meets the x-axis. It meets the x-axis at {n} point(s), "
                        f"so p(x) has {n} zero(es).",
            "topic": POLY, "subtopic": ("Geometrical meaning of the zeroes of a polynomial", "बहुपद के शून्यकों का ज्यामितीय अर्थ")}


@generator("two_poly_zeroes", "polynomial polynomials zero zeroes zeros graph two distinct geometrical बहुपद शून्यक")
def two_poly_zeroes(rng: random.Random) -> dict:
    a, b = sorted(rng.sample(range(-4, 5), 2))
    shared = rng.choice([2, 2, 1, 0])  # Q10 of the paper is the "shared both" case
    others = [x for x in range(-5, 6) if x not in (a, b)]
    c, d = (a, b) if shared == 2 else ((a, rng.choice([x for x in others if x > a])) if shared == 1 else tuple(sorted(rng.sample(others, 2))))
    p = [-x for x in _mul([1, -a], [1, -b])]  # opens downwards
    q = _mul([1, -c], [1, -d])               # opens upwards
    n = len({a, b, c, d})
    return {"template": {"kind": "graph", "x_range": [-6, 6], "y_range": _graph_range([p, q]),
                         "curves": [{"poly": p}, {"poly": q}]},
            "facts": ["the graphs of two polynomials are shown"],
            "ask": "the number of distinct zeroes of both the polynomials taken together", "answer": str(n),
            "wrong": [str(x) for x in (2, 3, 4, 5) if x != n], "sort_options": True,
            "solution": f"Each graph meets the x-axis at the zeroes of its polynomial. Counting the distinct points where the "
                        f"two graphs meet the x-axis gives {n}.",
            "topic": POLY, "subtopic": ("Geometrical meaning of the zeroes of a polynomial", "बहुपद के शून्यकों का ज्यामितीय अर्थ")}


@generator("linear_pair_graph", "linear equations pair two variables graphical consistent inconsistent solution रैखिक समीकरण युग्म")
def linear_pair_graph(rng: random.Random) -> dict:
    x0, y0 = rng.randint(-3, 3), rng.randint(-3, 3)
    m1, m2 = rng.sample([-2, -1, 1, 2], 2)
    lines = [[-m1, 1, y0 - m1 * x0], [-m2, 1, y0 - m2 * x0]]  # y = m x + (y0 - m x0)  ->  -m x + y = c
    if rng.random() < 0.6:
        ans = f"({x0}, {y0})"
        wrong = [f"({y0}, {x0})", f"({-x0}, {y0})", f"({x0}, {-y0})", f"({x0 + 1}, {y0})", f"({x0}, {y0 + 1})"]
        ask, sol = "the solution (x, y) of the pair of linear equations represented by the two lines", \
            f"The solution is the point where the two lines intersect, which is ({x0}, {y0})."
    else:
        lines[1] = [-m1, 1, lines[0][2] + rng.choice([-3, -2, 2, 3])]  # same slope: parallel lines
        ans = "no solution"
        wrong = ["a unique solution", "infinitely many solutions", "exactly two solutions"]
        ask, sol = "the number of solutions of the pair of linear equations represented by the two lines", \
            "The two lines are parallel and never meet, so the pair of equations has no solution (it is inconsistent)."
    return {"template": {"kind": "graph", "x_range": [-6, 6], "y_range": [-6, 6], "ticks": True, "grid": True,
                         "curves": [{"line": lines[0], "label": "l₁"}, {"line": lines[1], "label": "l₂"}]},
            "facts": ["the lines l₁ and l₂ representing a pair of linear equations in x and y are drawn on a grid with 1-unit squares"],
            "ask": ask, "answer": ans, "wrong": _options(ans, wrong),
            "solution": sol, "topic": LINEAR, "subtopic": ("Graphical method of solution", "हल करने की ग्राफ़ीय विधि")}


@generator("coordinate_points", "coordinate coordinates geometry distance formula midpoint mid-point points निर्देशांक दूरी")
def coordinate_points(rng: random.Random) -> dict:
    if rng.random() < 0.6:
        dx, dy, c = rng.choice([(3, 4, 5), (4, 3, 5), (6, 8, 10), (8, 6, 10)])
        dx, dy = dx * rng.choice([1, -1]), dy * rng.choice([1, -1])
        ax, ay = rng.randint(-2, 2) - (dx > 0) * 2 + (dx < 0) * 2, rng.randint(-2, 2) - (dy > 0) * 2 + (dy < 0) * 2
        bx, by = ax + dx, ay + dy
        ans, wrong = f"{c} units", [f"{abs(dx) + abs(dy)} units", f"{c + 1} units", f"{c * c} units", f"{c - 1} units"]
        ask, sol = "the distance AB", f"AB = √(({bx}) − ({ax}))² + (({by}) − ({ay}))²) = √({dx * dx} + {dy * dy}) = {c} units."
    else:
        ax, ay, bx, by = rng.randint(-6, 0), rng.randint(-5, 5), rng.randint(0, 6), rng.randint(-5, 5)
        if (bx - ax) % 2:
            bx -= 1
        if (by - ay) % 2:
            by += 1 if by < 5 else -1
        mx, my = (ax + bx) // 2, (ay + by) // 2
        ans = f"({mx}, {my})"
        wrong = [f"({my}, {mx})", f"({bx - ax}, {by - ay})", f"({mx + 1}, {my})", f"({mx}, {my - 1})"]
        ask, sol = "the coordinates of the mid-point of AB", f"Mid-point = (({ax} + {bx})/2, ({ay} + {by})/2) = ({mx}, {my})."
    span = max(7, abs(ax) + 1, abs(bx) + 1, abs(ay) + 1, abs(by) + 1)
    return {"template": {"kind": "graph", "x_range": [-span, span], "y_range": [-span, span], "ticks": True, "grid": True,
                         "points": [{"name": "A", "x": ax, "y": ay}, {"name": "B", "x": bx, "y": by}], "segments": [["A", "B"]]},
            "facts": [f"the points A({ax}, {ay}) and B({bx}, {by}) are plotted on the coordinate plane"],
            "ask": ask, "answer": ans, "wrong": _options(ans, wrong), "solution": sol,
            "topic": COORD, "subtopic": ("Distance formula" if "units" in ans else "Mid-point formula",
                                         "दूरी सूत्र" if "units" in ans else "मध्य-बिंदु सूत्र")}


def _frac(fr) -> str:
    return str(fr.numerator) if fr.denominator == 1 else f"{fr.numerator}/{fr.denominator}"


@generator("sector", "sector arc length area areas related circle circles minor त्रिज्यखण्ड चाप क्षेत्रफल")
def sector(rng: random.Random) -> dict:
    from fractions import Fraction as F
    r, t = rng.choice([7, 14, 21, 28]), rng.choice([30, 45, 60, 90, 120, 150])
    pi = F(22, 7)
    arc, area = F(t, 360) * 2 * pi * r, F(t, 360) * pi * r * r
    if rng.random() < 0.5:
        ans, unit, ask = arc, "cm", "the length of the arc AB"
        wrong = [arc * 2, arc / 2, F(t, 180) * pi * r * 2, area]
        sol = f"Arc length = (θ/360°) × 2πr = ({t}/360) × 2 × (22/7) × {r} = {_frac(arc)} cm."
    else:
        ans, unit, ask = area, "cm²", "the area of the sector OAB"
        wrong = [area * 2, area / 2, arc, F(t, 360) * pi * r]
        sol = f"Area of sector = (θ/360°) × πr² = ({t}/360) × (22/7) × {r}² = {_frac(area)} cm²."
    fmt = lambda v: f"{_frac(v)} {unit}"  # noqa: E731
    return {"template": {"template": "sector", "params": {"angle": t, "radius": r},
                         "labels": {"angle_AOB": f"{t}°", "OA": f"{r} cm"}},
            "facts": ["O is the centre of a circle and OAB is a sector", f"the radius OA = {r} cm", f"∠AOB = {t}°", "use π = 22/7"],
            "ask": ask, "answer": fmt(ans), "wrong": _options(fmt(ans), [fmt(w) for w in wrong]),
            "solution": sol, "topic": AREAS,
            "subtopic": ("Length of an arc" if unit == "cm" else "Area of a sector", "चाप की लंबाई" if unit == "cm" else "त्रिज्यखण्ड का क्षेत्रफल")}


def matching(text: str) -> list[str]:
    """Generators that fit a topic best: the ones sharing the most keywords with it (e.g. "Areas related to circles"
    picks sectors, not tangents, because it shares more words with them)."""
    words = set(text.lower().replace(",", " ").replace(".", " ").replace("-", " ").split())
    scores = {name: len(words & set(keys)) for name, (_, keys) in GENERATORS.items()}
    best = max(scores.values(), default=0)
    return [name for name, sc in scores.items() if sc == best and best > 0]


def build(name: str, rng: random.Random) -> dict:
    p = GENERATORS[name][0](rng)
    if len(p["wrong"]) < 3 or p["answer"] in p["wrong"]:
        raise ValueError(f"{name}: could not make four distinct options")
    p["generator"] = name
    return p
