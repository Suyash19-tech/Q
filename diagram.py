"""Figure-as-code for geometry diagrams.

The LLM never draws. It writes a small JSON spec (points built by constructions, circles, segments, marks, labels);
this module computes exact coordinates, checks the figure against the question, and renders a clean SVG.

Spec example (PA tangent at A, AB diameter, angle POB = 115 degrees):
{
  "circles": [{"name": "c", "center": "O", "radius": 3}],
  "points": [
    {"name": "O", "at": [0, 0]},
    {"name": "B", "on_circle": "c", "angle_deg": -65},
    {"name": "A", "on_circle": "c", "angle_deg": 115},
    {"name": "P", "polar": {"from": "O", "dist": 7.1, "angle_deg": 180}}
  ],
  "segments": [["P", "A"], ["A", "B"], ["P", "O"]],
  "dashed": [],
  "angle_marks": [{"at": "O", "from": "P", "to": "B", "label": "115°"}],
  "right_angles": [{"at": "A", "from": "O", "to": "P"}],
  "length_labels": [{"from": "P", "to": "A", "label": "10 m"}],
  "dots": ["O"]
}
"""
import math
import re
from html import escape

TOL_DEG = 2.0  # a labelled angle must match the drawn angle within this many degrees

# Point constructions the LLM may use (one key per point besides "name").
CONSTRUCTIONS = """Point constructions (use exactly one per point):
- "at": [x, y]                                   fixed coordinates
- "polar": {"from": "O", "dist": 5, "angle_deg": 30}  distance and direction (degrees, counter-clockwise from +x) from another point
- "on_circle": "c", "angle_deg": 120             point on circle c at that angle from its centre
- "midpoint": ["A", "B"]
- "on_segment": {"from": "A", "to": "B", "t": 0.4}    fraction t of the way from A to B
- "foot": {"point": "P", "line": ["A", "B"]}         foot of the perpendicular from P to line AB
- "intersect": {"line1": ["A", "B"], "line2": ["C", "D"]}  intersection of two (extended) lines
- "tangent_from": {"point": "P", "circle": "c", "side": "left" | "right"}  contact point of a tangent from external point P
Circles: {"name": "c", "center": "O", "radius": 3} or {"name": "c", "center": "O", "through": "A"}."""


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def _angle_deg(at, p, q):
    v1, v2 = _vec(at, p), _vec(at, q)
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(c))


def build(spec: dict):
    """Resolve every point and circle to coordinates. Returns (points, circles, errors)."""
    pts, circles, errors = {}, {}, []
    pending = list(spec.get("points", []))
    circle_specs = {c.get("name"): c for c in spec.get("circles", [])}

    def circle(name):
        if name in circles:
            return circles[name]
        c = circle_specs.get(name)
        if not c or c.get("center") not in pts:
            return None
        r = c.get("radius")
        if r is None and c.get("through") in pts:
            r = math.dist(pts[c["center"]], pts[c["through"]])
        if r is None:
            return None
        try:
            r = float(r)
        except (TypeError, ValueError):  # a symbolic radius such as "r" means any size; the drawing picks one
            r = 3.0
        circles[name] = (pts[c["center"]], r)
        return circles[name]

    def resolve(p):
        if "at" in p:
            return tuple(map(float, p["at"]))
        if "polar" in p:
            o = pts.get(p["polar"].get("from"))
            if o is None:
                return None
            a = math.radians(float(p["polar"]["angle_deg"]))
            d = float(p["polar"]["dist"])
            return (o[0] + d * math.cos(a), o[1] + d * math.sin(a))
        if "on_circle" in p:
            c = circle(p["on_circle"])
            if c is None:
                return None
            a = math.radians(float(p.get("angle_deg", 0)))
            return (c[0][0] + c[1] * math.cos(a), c[0][1] + c[1] * math.sin(a))
        if "midpoint" in p:
            a, b = (pts.get(n) for n in p["midpoint"])
            return None if a is None or b is None else ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if "on_segment" in p:
            a, b = pts.get(p["on_segment"]["from"]), pts.get(p["on_segment"]["to"])
            t = float(p["on_segment"].get("t", 0.5))
            return None if a is None or b is None else (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if "foot" in p:
            q, (a, b) = pts.get(p["foot"]["point"]), (pts.get(n) for n in p["foot"]["line"])
            if q is None or a is None or b is None:
                return None
            d = _vec(a, b)
            t = ((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / (d[0] ** 2 + d[1] ** 2 or 1)
            return (a[0] + t * d[0], a[1] + t * d[1])
        if "intersect" in p:
            a, b = (pts.get(n) for n in p["intersect"]["line1"])
            c, d = (pts.get(n) for n in p["intersect"]["line2"])
            if None in (a, b, c, d):
                return None
            den = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
            if abs(den) < 1e-9:
                raise ValueError(f"{p['name']}: lines are parallel")
            t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / den
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        if "tangent_from" in p:
            q, c = pts.get(p["tangent_from"]["point"]), circle(p["tangent_from"]["circle"])
            if q is None or c is None:
                return None
            (cx, cy), r = c
            dist = math.dist(q, (cx, cy))
            if dist <= r:
                raise ValueError(f"{p['name']}: tangent point is not outside the circle")
            base = math.atan2(q[1] - cy, q[0] - cx)
            off = math.acos(r / dist)
            # "left"/"right" as seen from the external point looking at the centre
            a = base - off if p["tangent_from"].get("side", "left") == "left" else base + off
            return (cx + r * math.cos(a), cy + r * math.sin(a))
        raise ValueError(f"{p.get('name')}: unknown construction {sorted(k for k in p if k != 'name')}")

    for _ in range(len(pending) + 1):  # points may refer to points defined later, so resolve in passes
        left = []
        for p in pending:
            try:
                xy = resolve(p)
            except (ValueError, KeyError, TypeError) as e:
                errors.append(str(e))
                continue
            if xy is None:
                left.append(p)
            else:
                pts[p["name"]] = xy
        if len(left) == len(pending):
            break
        pending = left
    for p in pending:
        errors.append(f"{p.get('name')}: refers to an undefined point or circle")
    for name in circle_specs:
        try:
            if circle(name) is None:
                errors.append(f"circle {name}: centre or radius undefined")
        except (ValueError, TypeError):
            errors.append(f"circle {name}: radius must be a number, got {circle_specs[name].get('radius')!r}")
    return pts, circles, errors


STOP_TOKENS = {"AP", "HCF", "LCM", "GP", "SI", "CI", "ML", "MM", "CM", "KM", "OR", "MCQ", "TSA", "CSA", "LHS", "RHS", "NCERT"}


def check(spec: dict, question_text: str) -> list[str]:
    """Problems that make the figure disagree with itself or with the question. Empty list means OK."""
    if is_graph(spec):
        return check_graph(spec)
    pts, circles, errors = build(spec)
    if errors:
        return errors
    names = set(pts)
    for kind in ("segments", "dashed"):
        for s in spec.get(kind, []):
            for n in s:
                if n not in names:
                    errors.append(f"{kind}: unknown point {n}")
    for m in spec.get("angle_marks", []):
        if not {m.get("at"), m.get("from"), m.get("to")} <= names:
            errors.append(f"angle mark at {m.get('at')}: unknown point")
            continue
        drawn = _angle_deg(pts[m["at"]], pts[m["from"]], pts[m["to"]])
        num = re.search(r"(\d+(?:\.\d+)?)\s*°", m.get("label", ""))
        if num and abs(drawn - float(num.group(1))) > TOL_DEG:
            errors.append(f"angle {m['from']}{m['at']}{m['to']} is drawn as {drawn:.0f}° but labelled {m['label']}")
    for m in spec.get("right_angles", []):
        if not {m.get("at"), m.get("from"), m.get("to")} <= names:
            errors.append(f"right-angle mark at {m.get('at')}: unknown point")
            continue
        drawn = _angle_deg(pts[m["at"]], pts[m["from"]], pts[m["to"]])
        if abs(drawn - 90) > TOL_DEG:
            errors.append(f"angle {m['from']}{m['at']}{m['to']} is marked as a right angle but drawn as {drawn:.0f}°")
    lens = []
    for m in spec.get("length_labels", []):
        if not {m.get("from"), m.get("to")} <= names:
            errors.append(f"length label {m.get('from')}{m.get('to')}: unknown point")
            continue
        num = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", m.get("label", ""))
        if num:
            lens.append((m["from"] + m["to"], float(num.group(1)), num.group(2), math.dist(pts[m["from"]], pts[m["to"]])))
    for i in range(len(lens)):  # e.g. OM = 6 cm and OA = 12 cm must be drawn in the ratio 1 : 2
        for j in range(i + 1, len(lens)):
            (n1, v1, u1, d1), (n2, v2, u2, d2) = lens[i], lens[j]
            if u1 == u2 and v1 > 0 and v2 > 0 and d1 > 0 and d2 > 0 and abs((d1 / d2) / (v1 / v2) - 1) > 0.05:
                errors.append(f"{n1} = {v1:g} and {n2} = {v2:g} are labelled in ratio {v1 / v2:.2f} but drawn in ratio {d1 / d2:.2f}")
    # Every point named in the question (∠APO, ΔABC, PA, ...) must exist in the figure.
    used = set()
    for tok in re.findall(r"(?<![A-Za-z])([A-Z]{1,4})(?![A-Za-z])", question_text):
        if tok in STOP_TOKENS or (len(tok) == 1 and tok in "ABCD" and re.search(rf"\({tok}\)", question_text)):
            continue
        used.update(tok)
    used -= {"A", "B", "C", "D"} - names  # option letters (A)-(D) are not points unless the figure has them
    missing = sorted(used - names)
    if missing:
        errors.append(f"the question mentions point(s) {', '.join(missing)} that are not in the figure")
    return errors


def render_svg(spec: dict, width: int = 360) -> str:
    """Deterministic black-on-transparent SVG; colours follow the page via currentColor."""
    if is_graph(spec):
        return render_graph(spec, width)
    pts, circles, _ = build(spec)
    xs = [p[0] for p in pts.values()] + [c[0][0] + s * c[1] for c in circles.values() for s in (-1, 1)]
    ys = [p[1] for p in pts.values()] + [c[0][1] + s * c[1] for c in circles.values() for s in (-1, 1)]
    if not xs:
        return ""
    pad = 34
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    k = (width - 2 * pad) / span
    height = int((max(ys) - min(ys)) * k + 2 * pad)
    X = lambda x: pad + (x - min(xs)) * k  # noqa: E731
    Y = lambda y: height - pad - (y - min(ys)) * k  # noqa: E731
    cx = sum(p[0] for p in pts.values()) / len(pts)
    cy = sum(p[1] for p in pts.values()) / len(pts)

    taken = []  # positions of text already placed, so later labels can avoid them
    for name, (x, y) in pts.items():
        dx, dy = x - cx, y - cy
        n = math.hypot(dx, dy) or 1
        taken.append((X(x) + dx / n * 14, Y(y) - dy / n * 14))

    def free_spot(candidates):
        for c in candidates:
            if all(math.hypot(c[0] - t[0], c[1] - t[1]) > 16 for t in taken):
                taken.append(c)
                return c
        taken.append(candidates[0])
        return candidates[0]

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="Times New Roman, serif" font-size="16" fill="none" stroke="currentColor" stroke-width="1.6">']
    for (c, r) in circles.values():
        out.append(f'<circle cx="{X(c[0]):.1f}" cy="{Y(c[1]):.1f}" r="{r * k:.1f}"/>')
    for kind, dash in (("segments", ""), ("dashed", ' stroke-dasharray="5 4"')):
        for a, b in spec.get(kind, []):
            if a in pts and b in pts:
                out.append(f'<line x1="{X(pts[a][0]):.1f}" y1="{Y(pts[a][1]):.1f}" x2="{X(pts[b][0]):.1f}" '
                           f'y2="{Y(pts[b][1]):.1f}"{dash}/>')
    for m in spec.get("right_angles", []):
        if {m.get("at"), m.get("from"), m.get("to")} <= set(pts):
            o, p, q = pts[m["at"]], pts[m["from"]], pts[m["to"]]
            s = 10 / k
            u = [(p[i] - o[i]) / (math.dist(o, p) or 1) * s for i in (0, 1)]
            v = [(q[i] - o[i]) / (math.dist(o, q) or 1) * s for i in (0, 1)]
            path = [(o[0] + u[0], o[1] + u[1]), (o[0] + u[0] + v[0], o[1] + u[1] + v[1]), (o[0] + v[0], o[1] + v[1])]
            out.append('<polyline points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in path) + '" stroke-width="1.2"/>')
    for m in spec.get("angle_marks", []):
        if {m.get("at"), m.get("from"), m.get("to")} <= set(pts):
            o, p, q = pts[m["at"]], pts[m["from"]], pts[m["to"]]
            a1, a2 = math.atan2(p[1] - o[1], p[0] - o[0]), math.atan2(q[1] - o[1], q[0] - o[0])
            d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi  # the smaller angle between the rays
            r = 18
            sx, sy = X(o[0]) + r * math.cos(a1), Y(o[1]) - r * math.sin(a1)
            ex, ey = X(o[0]) + r * math.cos(a1 + d), Y(o[1]) - r * math.sin(a1 + d)
            sweep = 0 if d > 0 else 1
            out.append(f'<path d="M{sx:.1f},{sy:.1f} A{r},{r} 0 0 {sweep} {ex:.1f},{ey:.1f}" stroke-width="1.2"/>')
            if m.get("label"):
                mid = a1 + d / 2
                lx, ly = free_spot([(X(o[0]) + (r + k2) * math.cos(mid), Y(o[1]) - (r + k2) * math.sin(mid)) for k2 in (16, 26, 36)])
                out.append(f'<text x="{lx:.1f}" y="{ly + 5:.1f}" text-anchor="middle" fill="currentColor" stroke="none" '
                           f'font-size="13">{escape(m["label"])}</text>')
    for m in spec.get("length_labels", []):
        if {m.get("from"), m.get("to")} <= set(pts):
            a, b = pts[m["from"]], pts[m["to"]]
            mx, my = X((a[0] + b[0]) / 2), Y((a[1] + b[1]) / 2)
            dx, dy = X(b[0]) - X(a[0]), Y(b[1]) - Y(a[1])
            n = math.hypot(dx, dy) or 1
            nx, ny = -dy / n, dx / n
            # prefer the side away from the figure's centre, then try further out and along the segment
            if (mx + nx * 14 - X(cx)) ** 2 + (my + ny * 14 - Y(cy)) ** 2 < (mx - X(cx)) ** 2 + (my - Y(cy)) ** 2:
                nx, ny = -nx, -ny
            ux, uy = dx / n, dy / n
            lx, ly = free_spot([(mx + nx * k2 + ux * t, my + ny * k2 + uy * t)
                                for k2 in (14, 24, -14, -24) for t in (0, 30, -30, 50, -50)])
            out.append(f'<text x="{lx:.1f}" y="{ly + 5:.1f}" text-anchor="middle" fill="currentColor" '
                       f'stroke="none" font-size="13" font-style="italic">{escape(m["label"])}</text>')
    for name in spec.get("dots", []):
        if name in pts:
            out.append(f'<circle cx="{X(pts[name][0]):.1f}" cy="{Y(pts[name][1]):.1f}" r="2.6" fill="currentColor"/>')
    for name, (x, y) in pts.items():
        dx, dy = x - cx, y - cy
        n = math.hypot(dx, dy) or 1
        lx, ly = X(x) + dx / n * 14, Y(y) - dy / n * 14
        out.append(f'<text x="{lx:.1f}" y="{ly + 5:.1f}" text-anchor="middle" fill="currentColor" stroke="none">'
                   f'{escape(name)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- templates
# The LLM is bad at coordinate geometry, so common exam figures are templates: it picks one and fills a few numbers,
# and the geometry is computed here. Labels are optional: "angle_APB": "120°" marks the angle at P between A and B;
# a two-letter key such as "PA": "10 cm" labels that segment.

def _two_tangents(p):
    t = float(p.get("angle_APB") or 180 - float(p.get("angle_AOB", 60)))
    if not 0 < t < 180:
        raise ValueError("angle_APB must be between 0 and 180")
    r = 3.0
    return {"circles": [{"name": "c", "center": "O", "radius": r}],
            "points": [{"name": "O", "at": [0, 0]},
                       {"name": "P", "polar": {"from": "O", "dist": r / math.sin(math.radians(t / 2)), "angle_deg": 180}},
                       {"name": "A", "tangent_from": {"point": "P", "circle": "c", "side": "right"}},
                       {"name": "B", "tangent_from": {"point": "P", "circle": "c", "side": "left"}}],
            "segments": [["P", "A"], ["P", "B"], ["O", "A"], ["O", "B"]] + ([["P", "O"]] if p.get("show_PO", True) else [])
                        + ([["A", "B"]] if p.get("show_AB") else []),
            "right_angles": [{"at": "A", "from": "O", "to": "P"}, {"at": "B", "from": "O", "to": "P"}], "dots": ["O"]}


def _tangent_diameter(p):
    t = float(p.get("angle_POB") or 180 - float(p.get("angle_AOP", 65)))
    if not 90 < t < 180:
        raise ValueError("angle_POB must be between 90 and 180")
    r = 3.0
    return {"circles": [{"name": "c", "center": "O", "radius": r}],
            "points": [{"name": "O", "at": [0, 0]}, {"name": "B", "on_circle": "c", "angle_deg": t - 180},
                       {"name": "A", "on_circle": "c", "angle_deg": t},
                       {"name": "P", "polar": {"from": "O", "dist": r / math.cos(math.radians(180 - t)), "angle_deg": 180}}],
            "segments": [["P", "A"], ["A", "B"], ["P", "O"]],
            "right_angles": [{"at": "A", "from": "O", "to": "P"}], "dots": ["O"]}


def _elevation_single(p):
    a = float(p.get("angle_A", 30))
    if not 0 < a < 90:
        raise ValueError("angle_A must be between 0 and 90")
    h = math.tan(math.radians(a)) * 10
    return {"points": [{"name": "A", "at": [0, 0]}, {"name": "B", "at": [10, 0]}, {"name": "C", "at": [10, h]}],
            "segments": [["A", "B"], ["B", "C"], ["A", "C"]], "right_angles": [{"at": "B", "from": "A", "to": "C"}]}


def _elevation_two(p):
    near, far = float(p.get("angle_P", 60)), float(p.get("angle_Q", 30))
    if not 0 < far < near < 90:
        raise ValueError("need 0 < angle_Q (far point) < angle_P (near point) < 90")
    h = 10.0
    xp, xq = h / math.tan(math.radians(near)), h / math.tan(math.radians(far))
    return {"points": [{"name": "B", "at": [0, 0]}, {"name": "C", "at": [0, h]}, {"name": "P", "at": [-xp, 0]}, {"name": "Q", "at": [-xq, 0]}],
            "segments": [["Q", "B"], ["B", "C"], ["P", "C"], ["Q", "C"]], "right_angles": [{"at": "B", "from": "P", "to": "C"}]}


def _elevation_from_height(p):
    a, b = float(p.get("angle_A", 60)), float(p.get("angle_B", 45))
    if not 0 < b < a < 90:
        raise ValueError("need 0 < angle_B (from the raised point) < angle_A (from the ground) < 90")
    H = 10.0
    x = H / (math.tan(math.radians(a)) - math.tan(math.radians(b)))
    top = x * math.tan(math.radians(a))
    return {"points": [{"name": "A", "at": [0, 0]}, {"name": "B", "at": [0, H]}, {"name": "E", "at": [x, 0]},
                       {"name": "D", "at": [x, H]}, {"name": "C", "at": [x, top]}],
            "segments": [["A", "B"], ["B", "D"], ["A", "E"], ["E", "C"], ["B", "C"], ["A", "C"]],
            "right_angles": [{"at": "E", "from": "A", "to": "C"}]}


def _chord_perpendicular(p):
    """Circle centre O, chord AB, M the foot of the perpendicular from O (also the midpoint of AB)."""
    r = p.get("radius")
    d = p.get("distance_OM")
    c = p.get("chord_AB")
    r, d, c = (float(v) if v not in (None, "") else None for v in (r, d, c))
    if r is None and d is not None and c is not None:
        r = math.hypot(d, c / 2)
    if d is None and r is not None and c is not None:
        if c / 2 >= r:
            raise ValueError("chord_AB must be shorter than the diameter")
        d = math.sqrt(r * r - (c / 2) ** 2)
    if d is not None and d < 1e-9:
        raise ValueError("distance_OM is 0, so the chord is a diameter and M would sit on O; use a shorter chord")
    if r is None or d is None or not 0 <= d < r:
        raise ValueError("give two of radius, distance_OM, chord_AB with distance_OM < radius")
    return {"circles": [{"name": "c", "center": "O", "radius": r}],
            "points": [{"name": "O", "at": [0, 0]}, {"name": "M", "at": [0, -d]},
                       {"name": "A", "at": [-math.sqrt(r * r - d * d), -d]}, {"name": "B", "at": [math.sqrt(r * r - d * d), -d]}],
            "segments": [["A", "B"], ["O", "M"], ["O", "A"]] + ([["O", "B"]] if p.get("show_OB") else []),
            "right_angles": [{"at": "M", "from": "O", "to": "B"}], "dots": ["O"]}


def _sector(p):
    """Circle centre O with sector OAB of angle `angle` (opening upwards)."""
    t = float(p.get("angle", 60))
    if not 0 < t < 360:
        raise ValueError("angle must be between 0 and 360")
    r = float(p.get("radius", 3) or 3)
    return {"circles": [{"name": "c", "center": "O", "radius": r}],
            "points": [{"name": "O", "at": [0, 0]}, {"name": "A", "on_circle": "c", "angle_deg": 90 + t / 2},
                       {"name": "B", "on_circle": "c", "angle_deg": 90 - t / 2}],
            "segments": [["O", "A"], ["O", "B"]], "dots": ["O", "A", "B"]}


def _triangle_parallel(p):
    t = float(p.get("ratio_AD_AB", 0.4))
    if not 0 < t < 1:
        raise ValueError("ratio_AD_AB must be between 0 and 1")
    if p.get("AB") and p.get("AC"):  # real side lengths, so labelled lengths are drawn to scale
        ab, ac = float(p["AB"]), float(p["AC"])
        a1, a2 = math.radians(245), math.radians(305)
        return {"points": [{"name": "A", "at": [0, 0]}, {"name": "B", "at": [ab * math.cos(a1), ab * math.sin(a1)]},
                           {"name": "C", "at": [ac * math.cos(a2), ac * math.sin(a2)]},
                           {"name": "D", "on_segment": {"from": "A", "to": "B", "t": t}},
                           {"name": "E", "on_segment": {"from": "A", "to": "C", "t": t}}],
                "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["D", "E"]]}
    return {"points": [{"name": "A", "at": [0.8, 4]}, {"name": "B", "at": [-2.5, 0]}, {"name": "C", "at": [3.5, 0]},
                       {"name": "D", "on_segment": {"from": "A", "to": "B", "t": t}},
                       {"name": "E", "on_segment": {"from": "A", "to": "C", "t": t}}],
            "segments": [["A", "B"], ["B", "C"], ["C", "A"], ["D", "E"]]}


TEMPLATES = {
    "two_tangents": (_two_tangents, "tangents PA, PB from external point P to circle centre O (radii OA, OB, line PO); "
                                    "params: angle_APB (or angle_AOB); options show_PO, show_AB"),
    "tangent_diameter": (_tangent_diameter, "tangent PA at A, diameter AB, line PO; params: angle_POB (90-180) (or angle_AOP)"),
    "elevation_single": (_elevation_single, "observer A on ground, tower BC (B foot, C top); params: angle_A (elevation)"),
    "elevation_two": (_elevation_two, "tower BC (B foot, C top), two ground points: P near, Q far; params: angle_P > angle_Q"),
    "elevation_from_height": (_elevation_from_height, "A on ground, B directly above A, tower EC (E foot, C top), D on EC level "
                                                      "with B; params: angle_A (from A) > angle_B (from B)"),
    "chord_perpendicular": (_chord_perpendicular, "circle centre O, chord AB, OM perpendicular to AB at M (midpoint), radius OA; "
                                                  "params: two of radius, distance_OM, chord_AB (real numbers from the question); option show_OB"),
    "sector": (_sector, "circle centre O with sector OAB; params: angle (angle AOB), radius"),
    "triangle_parallel": (_triangle_parallel, "triangle ABC with DE parallel to BC, D on AB, E on AC; params: ratio_AD_AB"),
}

# which label keys name the same angle as a template parameter (before any renaming)
PARAM_LABELS = {
    "two_tangents": {"angle_APB": ["angle_APB", "angle_BPA"], "angle_AOB": ["angle_AOB", "angle_BOA"]},
    "tangent_diameter": {"angle_POB": ["angle_POB", "angle_BOP"], "angle_AOP": ["angle_AOP", "angle_POA"]},
    "elevation_single": {"angle_A": ["angle_CAB", "angle_BAC"]},
    "elevation_two": {"angle_P": ["angle_CPB", "angle_BPC"], "angle_Q": ["angle_CQB", "angle_BQC"]},
    "elevation_from_height": {"angle_A": ["angle_CAE", "angle_EAC"], "angle_B": ["angle_CBD", "angle_DBC"]},
    "sector": {"angle": ["angle_AOB", "angle_BOA"]},
}

LENGTH_PARAMS = {"sector": {"radius": ["OA", "AO", "OB", "BO"]}, "chord_perpendicular": {"distance_OM": ["OM", "MO"], "chord_AB": ["AB", "BA"], "radius": ["OA", "AO", "OB", "BO"]}}

TEMPLATE_DOC = "Templates (preferred; the geometry is computed for you):\n" + "\n".join(
    f'- "{k}": {doc}' for k, (_, doc) in TEMPLATES.items()) + """
Template format: {"template": "two_tangents", "params": {"angle_APB": 120}, "labels": {"angle_APB": "120°", "PA": "10 cm"},
                  "names": {"P": "T"}}
labels: "angle_XYZ" marks the angle at Y between X and Y-Z with that text; a two-letter key labels a segment. Only label what the
question gives (never the answer). names (optional) renames template points to match the question; labels may use either the
template's names or the question's names."""


def from_template(obj: dict) -> dict:
    """Expand {"template", "params", "labels", "names"} into a full spec. Raises ValueError on bad input."""
    name = obj.get("template")
    if name not in TEMPLATES:
        raise ValueError(f"unknown template {name!r}; use one of {', '.join(TEMPLATES)}")
    ren = obj.get("names") or {}
    inv = {v: k for k, v in ren.items()}
    # Labels may use the question's (renamed) point names; translate them back to the template's own names.
    labels = {}
    for key, text in (obj.get("labels") or {}).items():
        if key.startswith("angle_") and len(key) == 9:
            key = "angle_" + "".join(inv.get(ch, ch) for ch in key[6:])
        elif len(key) == 2 and key.isalpha():
            key = "".join(inv.get(ch, ch) for ch in key)
        labels[key] = text
    params = dict(obj.get("params") or {})
    # The labelled value is what the question states, so it overrides the parameter for the same angle.
    for param, keys in PARAM_LABELS.get(name, {}).items():
        for k in keys:
            m = re.search(r"(\d+(?:\.\d+)?)\s*°", str(labels.get(k, "")))
            if m:
                params[param] = float(m.group(1))
                if param == "angle_AOB":
                    params.pop("angle_APB", None)
                if param == "angle_AOP":
                    params.pop("angle_POB", None)
    # Same for labelled lengths (e.g. chord figures): the numbers shown are the numbers drawn.
    from_labels = {}
    for param, keys in LENGTH_PARAMS.get(name, {}).items():
        for k in keys:
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[a-zA-Z]*\s*", str(labels.get(k, "")))
            if m:
                from_labels[param] = float(m.group(1))
    if from_labels:
        params = {k: v for k, v in params.items() if k not in LENGTH_PARAMS[name]} | from_labels
        if len(from_labels) == 1 and len(LENGTH_PARAMS[name]) > 1:  # one length from the labels, the second from the params the model gave
            for k, v in (obj.get("params") or {}).items():
                if k in LENGTH_PARAMS[name] and k not in from_labels:
                    params[k] = v
                    break
    spec = TEMPLATES[name][0](params)
    for key, text in labels.items():
        if key.startswith("angle_") and len(key) == 9:
            a, at, b = key[6], key[7], key[8]
            spec.setdefault("angle_marks", []).append({"at": at, "from": a, "to": b, "label": str(text)})
        elif len(key) == 2 and key.isalpha():
            spec.setdefault("length_labels", []).append({"from": key[0], "to": key[1], "label": str(text)})
        else:
            raise ValueError(f"label key {key!r} must look like angle_APB or PA")
    if ren:
        r = lambda n: ren.get(n, n)  # noqa: E731
        def deep(x):
            if isinstance(x, str):
                return r(x) if len(x) == 1 else x
            if isinstance(x, list):
                return [deep(i) for i in x]
            if isinstance(x, dict):
                return {k: (deep(v) if k in ("name", "center", "from", "to", "at", "point", "through", "circle") or
                            isinstance(v, (list, dict)) else v) for k, v in x.items()}
            return x
        spec = deep(spec)
    return spec


def describe(spec: dict) -> str:
    """A plain-English description generated from the drawn figure itself, so it can never disagree with the picture."""
    if is_graph(spec):
        return describe_graph(spec)
    pts, circles, _ = build(spec)
    parts = []
    for c in spec.get("circles", []):
        parts.append(f"Circle with centre {c['center']}.")
    on_circle = sorted(n for n, xy in pts.items()
                       if any(abs(math.dist(xy, ctr) - r) < 1e-6 * max(r, 1) and n != cn for cn, (ctr, r) in
                              [(c["center"], circles[c["name"]]) for c in spec.get("circles", []) if c["name"] in circles]))
    if on_circle:
        parts.append(f"Points {', '.join(on_circle)} lie on the circle.")
    segs = [a + b for a, b in spec.get("segments", [])]
    if segs:
        parts.append(f"Segments drawn: {', '.join(segs)}.")
    if spec.get("dashed"):
        parts.append(f"Dashed: {', '.join(a + b for a, b in spec['dashed'])}.")
    for m in spec.get("right_angles", []):
        parts.append(f"Right angle marked at {m['at']} (angle {m['from']}{m['at']}{m['to']}).")
    for m in spec.get("angle_marks", []):
        parts.append(f"Angle {m['from']}{m['at']}{m['to']} is marked {m.get('label', '')}.".replace(" .", "."))
    for m in spec.get("length_labels", []):
        parts.append(f"{m['from']}{m['to']} is labelled {m['label']}.")
    return " ".join(parts)


def complete(spec: dict) -> dict:
    """Add any segment that a mark or label sits on but the spec forgot to draw (e.g. the line of sight under an angle)."""
    if is_graph(spec):
        return spec
    pts, _, errors = build(spec)
    if errors:
        return spec
    segs = [tuple(s) for s in spec.get("segments", []) + spec.get("dashed", []) if len(s) == 2 and set(s) <= set(pts)]

    def covered(a, b):
        for p, q in segs:
            P, Q, A, B = pts[p], pts[q], pts[a], pts[b]
            L = math.dist(P, Q) or 1
            on = lambda X: abs((Q[0] - P[0]) * (X[1] - P[1]) - (Q[1] - P[1]) * (X[0] - P[0])) / L < 1e-6 * L and \
                -1e-9 <= ((X[0] - P[0]) * (Q[0] - P[0]) + (X[1] - P[1]) * (Q[1] - P[1])) / L ** 2 <= 1 + 1e-9  # noqa: E731
            if on(A) and on(B):
                return True
        return False

    need = []
    for m in spec.get("angle_marks", []) + spec.get("right_angles", []):
        need += [(m.get("at"), m.get("from")), (m.get("at"), m.get("to"))]
    need += [(m.get("from"), m.get("to")) for m in spec.get("length_labels", [])]
    added = []
    for a, b in need:
        if a in pts and b in pts and a != b and not covered(a, b):
            segs.append((a, b))
            added.append([a, b])
    if added:
        spec = {**spec, "segments": spec.get("segments", []) + added}
    return spec



# ---------------------------------------------------------------- graphs (axes, polynomial curves, lines, points)
# Spec: {"kind": "graph", "x_range": [-5, 5], "y_range": [-5, 5], "ticks": false, "grid": false,
#        "curves": [{"poly": [1, 0, -4], "label": "y = p(x)"}, {"line": [1, 1, 3]}],   # poly: coefficients, highest power first
#        "points": [{"name": "A", "x": 2, "y": 3}], "segments": [["A", "B"]]}          # line: a, b, c meaning ax + by = c

def is_graph(spec) -> bool:
    return isinstance(spec, dict) and spec.get("kind") == "graph"


def _poly(coeffs, x):
    y = 0.0
    for c in coeffs:
        y = y * x + c
    return y


def check_graph(spec: dict) -> list[str]:
    errors = []
    try:
        (x0, x1), (y0, y1) = map(float, spec["x_range"]), map(float, spec["y_range"])
        if not (x0 < x1 and y0 < y1):
            errors.append("x_range and y_range must go from smaller to larger")
    except (KeyError, TypeError, ValueError):
        return ["a graph needs numeric x_range and y_range"]
    for c in spec.get("curves", []):
        vals = c.get("poly", c.get("line"))
        if not isinstance(vals, list) or not all(isinstance(v, (int, float)) for v in vals):
            errors.append("each curve needs numeric poly coefficients or line [a, b, c]")
    names = {p.get("name") for p in spec.get("points", [])}
    for a, b in spec.get("segments", []):
        if a not in names or b not in names:
            errors.append(f"segment {a}{b}: unknown point")
    if not spec.get("curves") and not spec.get("points"):
        errors.append("the graph shows nothing")
    return errors


def render_graph(spec: dict, width: int = 360) -> str:
    (x0, x1), (y0, y1) = map(float, spec["x_range"]), map(float, spec["y_range"])
    height, pad = int(width * 0.85), 30
    kx, ky = (width - 2 * pad) / (x1 - x0), (height - 2 * pad) / (y1 - y0)
    X = lambda x: pad + (x - x0) * kx  # noqa: E731
    Y = lambda y: height - pad - (y - y0) * ky  # noqa: E731
    ax_x = X(0) if x0 < 0 < x1 else X(x0)
    ax_y = Y(0) if y0 < 0 < y1 else Y(y0)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="Times New Roman, serif" font-size="16" fill="none" stroke="currentColor" stroke-width="1.6">',
           '<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           '<path d="M0,0 L8,4 L0,8 z" fill="currentColor" stroke="none"/></marker></defs>']
    if spec.get("grid"):
        for i in range(math.ceil(x0), math.floor(x1) + 1):
            out.append(f'<line x1="{X(i):.1f}" y1="{Y(y0):.1f}" x2="{X(i):.1f}" y2="{Y(y1):.1f}" stroke-width="0.5" stroke-opacity="0.25"/>')
        for j in range(math.ceil(y0), math.floor(y1) + 1):
            out.append(f'<line x1="{X(x0):.1f}" y1="{Y(j):.1f}" x2="{X(x1):.1f}" y2="{Y(j):.1f}" stroke-width="0.5" stroke-opacity="0.25"/>')
    out.append(f'<line x1="{X(x0) - 8:.1f}" y1="{ax_y:.1f}" x2="{X(x1) + 8:.1f}" y2="{ax_y:.1f}" stroke-width="1.2" '
               f'marker-start="url(#ar)" marker-end="url(#ar)"/>')
    out.append(f'<line x1="{ax_x:.1f}" y1="{Y(y0) + 8:.1f}" x2="{ax_x:.1f}" y2="{Y(y1) - 8:.1f}" stroke-width="1.2" '
               f'marker-start="url(#ar)" marker-end="url(#ar)"/>')
    t = lambda x, y, txt, anchor="middle", size=15, italic=False: out.append(  # noqa: E731
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="currentColor" stroke="none" font-size="{size}"'
        f'{" font-style=\"italic\"" if italic else ""}>{escape(str(txt))}</text>')
    t(X(x1) + 20, ax_y + 5, "x"); t(X(x0) - 20, ax_y + 5, "x′"); t(ax_x, Y(y1) - 14, "y"); t(ax_x, Y(y0) + 26, "y′")
    if spec.get("ticks"):
        for i in range(math.ceil(x0), math.floor(x1) + 1):
            if i:
                out.append(f'<line x1="{X(i):.1f}" y1="{ax_y - 3:.1f}" x2="{X(i):.1f}" y2="{ax_y + 3:.1f}" stroke-width="1"/>')
                t(X(i), ax_y + 15, i, size=10)
        for j in range(math.ceil(y0), math.floor(y1) + 1):
            if j:
                out.append(f'<line x1="{ax_x - 3:.1f}" y1="{Y(j):.1f}" x2="{ax_x + 3:.1f}" y2="{Y(j):.1f}" stroke-width="1"/>')
                t(ax_x - 8, Y(j) + 4, j, anchor="end", size=10)
        t(ax_x - 8, ax_y + 15, "O", anchor="end", size=11)
    for c in spec.get("curves", []):
        if "line" in c and abs(c["line"][1]) < 1e-12:  # vertical line x = c/a
            xv = c["line"][2] / c["line"][0]
            out.append(f'<line x1="{X(xv):.1f}" y1="{Y(y0):.1f}" x2="{X(xv):.1f}" y2="{Y(y1):.1f}"/>')
            continue
        f = (lambda x, k=c["poly"]: _poly(k, x)) if "poly" in c else (lambda x, a=c["line"]: (a[2] - a[0] * x) / a[1])
        runs, run = [], []
        for i in range(301):
            x = x0 + (x1 - x0) * i / 300
            y = f(x)
            if y0 <= y <= y1:
                run.append((X(x), Y(y)))
            elif run:
                runs.append(run); run = []
        if run:
            runs.append(run)
        for r in runs:
            if len(r) > 1:
                out.append('<polyline points="' + " ".join(f"{px:.1f},{py:.1f}" for px, py in r) + '"/>')
        if c.get("label") and runs and runs[-1]:
            lx, ly = runs[-1][-1]
            t(min(lx + 6, width - 4), max(ly - 6, 12), c["label"], anchor="end" if lx > width - 60 else "start", size=12, italic=True)
    pts = {p["name"]: (float(p["x"]), float(p["y"])) for p in spec.get("points", [])}
    for a, b in spec.get("segments", []):
        out.append(f'<line x1="{X(pts[a][0]):.1f}" y1="{Y(pts[a][1]):.1f}" x2="{X(pts[b][0]):.1f}" y2="{Y(pts[b][1]):.1f}"/>')
    for name, (x, y) in pts.items():
        out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="currentColor"/>')
        t(X(x) + 9, Y(y) - 8, name, anchor="start")
    out.append("</svg>")
    return "".join(out)


def describe_graph(spec: dict) -> str:
    parts = ["Graph on x′x and y′y axes" + (" with unit grid marks" if spec.get("ticks") else "") + "."]
    curves = spec.get("curves", [])
    polys = [c for c in curves if "poly" in c]
    lines = [c for c in curves if "line" in c]
    if polys:
        kinds = {2: "parabola", 3: "cubic curve"}
        parts.append("Curves: " + ", ".join(kinds.get(len(c["poly"]) - 1, "curve") + (f" ({c['label']})" if c.get("label") else "")
                                            for c in polys) + ".")
    if lines:
        parts.append(f"{len(lines)} straight line(s)" + "".join(f" ({c['label']})" for c in lines if c.get("label")) + ".")
    if spec.get("points"):
        parts.append("Points marked: " + ", ".join(p["name"] for p in spec["points"]) + ".")
    return " ".join(parts)
