import numpy as np

from img2draw import artist, metrics, planner, renderer, replay, schema


def _toon(n=96):
    """Tiny cartoon: flat shapes, a gradient-free shadow, a thin black outline."""
    import cv2
    img = np.full((n, n, 3), 245, np.uint8)
    cv2.circle(img, (n // 2, n // 2), n // 3, (230, 180, 60), -1)
    cv2.circle(img, (n // 2 + 8, n // 2 + 8), n // 6, (190, 140, 40), -1)
    cv2.circle(img, (n // 2, n // 2), n // 3, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def test_project_roundtrip(tmp_path):
    p = planner.plan(_toon(), mode="fast")
    path = tmp_path / "p.project.json"
    schema.save_project(p, path)
    assert schema.validate(schema.load_project(path)) == []


def test_render_deterministic():
    p = artist.plan(_toon(), mode="balanced")
    assert np.array_equal(renderer.render_project(p), renderer.render_project(p))


def test_final_equals_source_and_stages_in_artist_order():
    img = _toon()
    p = artist.plan(img, mode="balanced")
    assert np.array_equal(renderer.render_project(p), img)  # tol=0 => exact
    names = [s["name"] for s in p["metadata"]["stages"]]
    assert names[0] == "Outline" and names[1] == "Flat colors" and names[-1] == "Polish"
    # structure (before polish) is already close, polish is just touch-up
    so = p["metadata"]["structure_ops"]
    assert metrics.psnr(img, renderer.render_project(p, so)) > 22


def test_tolerance_bounds_error():
    img = _toon()
    p = artist.plan(img, mode="balanced", tol=6)
    assert np.abs(renderer.render_project(p).astype(int) - img).max() <= 6


def test_outline_drawn_first_and_on_top():
    p = artist.plan(_toon(), mode="balanced")
    tl = renderer.timeline(p)
    first_layer = p["layers"][tl[0][0]]["name"]
    assert first_layer == "Outline"
    assert p["layers"][[l["name"] for l in p["layers"]].index("Outline")] is not p["layers"][0]


def test_replay_frames_are_continuous_and_end_on_final():
    img = _toon(48)
    p = artist.plan(img, mode="fast", tol=0)
    fr = list(replay.frames(p, fps=10, seconds=3, hold=0.5))
    assert np.array_equal(fr[-1], renderer.render_project(p))
    # monotone progress: error vs final never jumps up by a lot between frames
    errs = [np.abs(f.astype(int) - fr[-1]).mean() for f in fr]
    assert errs[0] > errs[len(errs) // 2] > errs[-1]


def test_brush_stroke_renders():
    p = schema.new_project(64, 64)
    layer = schema.new_layer("l1", "Edits")
    layer["operations"].append({"type": "BRUSH_STROKE", "points": [[4, 4], [60, 60]],
                                "color": [255, 0, 0], "width": 6, "opacity": 1.0})
    p["layers"].append(layer)
    assert renderer.render_project(p)[..., 0].max() > 200


def test_metrics_identical():
    img = np.full((16, 16, 3), 128, np.uint8)
    r = metrics.full_report(img, img)
    assert r["mae"] == 0 and r["psnr"] == 100.0 and r["ssim"] == 1.0


def test_erase():
    p = schema.new_project(32, 32)
    l = schema.new_layer("l", "base")
    l["operations"].append({"type": "GRADIENT", "direction": "vertical",
                            "stops": [[0.0, [10, 10, 10]], [1.0, [200, 200, 200]]], "opacity": 1.0})
    l["operations"].append({"type": "ERASE", "points": [[0, 0], [31, 0], [31, 31], [0, 31]]})
    p["layers"].append(l)
    assert renderer.render_project(p).mean() > 250
