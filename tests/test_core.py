import numpy as np

from img2draw import schema, renderer, metrics, planner, pipeline


def test_project_roundtrip(tmp_path):
    p = planner.plan(np.full((32, 32, 3), 200, np.uint8), mode="fast")
    path = tmp_path / "p.project.json"
    schema.save_project(p, path)
    p2 = schema.load_project(path)
    assert p2["canvas"]["width"] == 32
    assert schema.validate(p2) == []


def test_renderer_deterministic():
    img = np.random.default_rng(1).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    p = planner.plan(img, mode="balanced")
    p = pipeline.refine(p, img)
    a = renderer.render_project(p)
    b = renderer.render_project(p)
    assert np.array_equal(a, b)


def test_high_fidelity_reconstruction():
    img = np.random.default_rng(2).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    img = (img // 32) * 32  # compress to make correction easier
    p = planner.plan(img, mode="balanced")
    p = pipeline.refine(p, img, iterations=3)
    r = renderer.render_project(p)
    report = metrics.full_report(img, r)
    assert report["mae"] < 2.0, report
    assert report["psnr"] > 40, report


def test_brush_stroke_renders():
    p = schema.new_project(64, 64)
    layer = schema.new_layer("l1", "Edits")
    layer["operations"].append({"type": "BRUSH_STROKE", "points": [[4, 4], [60, 60]],
                                "color": [255, 0, 0], "width": 6, "opacity": 1.0})
    p["layers"].append(layer)
    r = renderer.render_project(p)
    assert r[..., 0].max() > 200  # red present


def test_metrics_identical():
    img = np.full((16, 16, 3), 128, np.uint8)
    r = metrics.full_report(img, img)
    assert r["mae"] == 0 and r["mse"] == 0 and r["psnr"] == 100.0
    assert r["ssim"] == 1.0


def test_erase():
    p = schema.new_project(32, 32)
    l = schema.new_layer("l", "base")
    l["operations"].append({"type": "GRADIENT", "direction": "vertical",
                            "stops": [[0.0, [10, 10, 10]], [1.0, [200, 200, 200]]], "opacity": 1.0})
    l["operations"].append({"type": "ERASE", "points": [[0, 0], [31, 0], [31, 31], [0, 31]]})
    p["layers"].append(l)
    r = renderer.render_project(p)
    assert r.mean() > 250  # everything erased to white
