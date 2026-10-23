from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw
import pytest

from anime_picker import curation as curation_module
from anime_picker.curation import (
    _aligned_face_similarity,
    _evidence_distance,
    _image_features,
    _indexed_visual_hash_neighbors,
    _pair_evaluation,
    _pairwise_visual_hash_neighbors,
    _registered_similarity,
    _visual_hash_index,
    _visual_groups,
    curate_candidates,
)


def _cartoon(*, mouth_open: bool = False, shift: int = 0, color: tuple[int, int, int] = (240, 190, 150)) -> Image.Image:
    image = Image.new("RGB", (160, 90), (92, 130, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 65, 159, 89), fill=(60, 95, 80))
    draw.ellipse((48 + shift, 10, 112 + shift, 75), fill=color, outline=(35, 25, 35), width=2)
    draw.ellipse((66 + shift, 31, 72 + shift, 37), fill=(20, 20, 30))
    draw.ellipse((88 + shift, 31, 94 + shift, 37), fill=(20, 20, 30))
    if mouth_open:
        draw.ellipse((74 + shift, 48, 88 + shift, 62), fill=(115, 25, 45), outline=(40, 20, 30))
    else:
        draw.arc((72 + shift, 43, 90 + shift, 60), start=15, end=165, fill=(70, 25, 35), width=2)
    return image


def _distant_character() -> Image.Image:
    image = Image.new("RGB", (160, 90), (100, 135, 180))
    draw = ImageDraw.Draw(image)
    draw.ellipse((72, 17, 84, 29), fill=(70, 55, 60))
    draw.rectangle((75, 29, 81, 53), fill=(50, 70, 100))
    draw.line((78, 38, 69, 48), fill=(50, 70, 100), width=2)
    draw.line((78, 38, 87, 48), fill=(50, 70, 100), width=2)
    draw.line((78, 53, 70, 66), fill=(40, 45, 60), width=2)
    draw.line((78, 53, 86, 66), fill=(40, 45, 60), width=2)
    return image


def _record(
    candidate_id: str,
    image: Image.Image,
    *,
    time: float,
    tags: dict | None = None,
    detections: list[dict] | None = None,
    quality: float = 0.5,
) -> dict:
    return {
        "id": candidate_id,
        "pts": int(time * 1000),
        "time": time,
        "image": image,
        "tags": tags or {"general": {"1girl": 0.8}},
        "detections": detections or [],
        "quality": quality,
        "reasons": ["fixture"],
    }


def _group_for(report: dict, candidate_id: str) -> dict:
    return next(group for group in report["groups"] if candidate_id in group["members"])


def _internal_face_record(candidate_id: str, faces: list[dict], *, tags: dict | None = None) -> dict:
    image = _cartoon()
    features = _image_features(image, [])
    features["faces"] = faces
    return {
        "id": candidate_id,
        "time": 0.0,
        "tags": tags or {"general": {"1girl": 0.9}},
        "features": features,
    }


def _face_patch(color: tuple[int, int, int] = (220, 180, 150)) -> np.ndarray:
    image = Image.new("RGB", (48, 48), color)
    draw = ImageDraw.Draw(image)
    for y in range(0, 48, 4):
        for x in range(0, 48, 4):
            if ((x // 4) + (y // 4)) % 2:
                draw.rectangle((x, y, x + 3, y + 3), fill=(45, 75, 125))
    draw.ellipse((7, 14, 16, 22), fill=(15, 20, 30))
    draw.ellipse((31, 14, 40, 22), fill=(15, 20, 30))
    draw.rectangle((19, 31, 29, 39), fill=(100, 12, 30))
    return np.asarray(image, dtype=np.float32) / 255.0


def _shift_crop_for_inverse_sample(source: np.ndarray, sample_offset: int) -> np.ndarray:
    """Return a crop where output x samples source at x + sample_offset."""
    indexes = np.clip(np.arange(48) - sample_offset, 0, 47)
    return source[:, indexes, :].copy()


def _accepted_registration(dx: float = 0.0, dy: float = 0.0) -> dict:
    return {
        "alignment_checked": True,
        "alignment_correlation_before": 0.97,
        "alignment_correlation": 0.99,
        "alignment_gain": 0.02,
        "alignment_mae": 0.01,
        "alignment_rgb_mae": 0.01,
        "alignment_translation_fraction": max(abs(dx) / 256, abs(dy) / 144),
        "alignment_dx_pixels": dx,
        "alignment_dy_pixels": dy,
    }


def _face(box: tuple[float, float, float, float], rgb: np.ndarray) -> dict:
    return {"box": list(box), "rgb": np.asarray(rgb, dtype=np.float32)}


def _hash_record(candidate_id: str, phash_value: int, dhash_value: int) -> dict:
    return {
        "id": candidate_id,
        "time": float(len(candidate_id)),
        "features": {"phash": f"{phash_value:016x}", "dhash": f"{dhash_value:016x}"},
    }


def _private_curation_rows(required_pts: set[int]) -> dict[int, dict]:
    report_value = os.environ.get("ANIME_PICKER_REGRESSION_CURATION_REPORT")
    if not report_value:
        pytest.skip("set ANIME_PICKER_REGRESSION_CURATION_REPORT to a private curation-report.json")
    report_path = Path(report_value)
    assert report_path.is_file(), f"private curation report is missing: {report_path}"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    curation = data.get("report", data)
    rows = {
        int(row["pts"]): row
        for row in [*curation.get("kept", []), *curation.get("excluded", [])]
        if row.get("pts") is not None and int(row["pts"]) in required_pts
    }
    assert set(rows) == required_pts, f"private curation report is missing PTS: {sorted(required_pts - set(rows))}"
    for pts, row in rows.items():
        assert Path(str(row.get("image", ""))).is_file(), f"cached curation-input PNG is missing for PTS {pts}"
    return rows


def _record_from_private_report(row: dict) -> dict:
    return {
        key: row[key]
        for key in ("id", "pts", "time", "image", "tags", "detections", "quality", "source_score")
        if key in row
    }


def test_pixai_general_presence_uses_model_card_threshold_and_keeps_unlimited_results() -> None:
    records = []
    for index in range(14):
        # Each frame has a distinct, textured background with a visible face.
        # This verifies the absence of a result cap without asking global
        # near-duplicate curation to preserve near-identical frames.
        image = Image.new("RGB", (160, 90), (20, 30, 40))
        draw = ImageDraw.Draw(image)
        for top in range(0, 90, 6):
            for left in range(0, 160, 8):
                draw.rectangle(
                    (left, top, left + 7, top + 5),
                    fill=(
                        (index * 61 + left * 7 + top * 3) % 256,
                        (index * 29 + left * 3 + top * 11) % 256,
                        (index * 47 + left * 13 + top * 5) % 256,
                    ),
                )
        face = _cartoon(shift=(index - 7) * 5, color=(220 - index * 7, 170, 150))
        mask = Image.new("L", (160, 90), 0)
        ImageDraw.Draw(mask).ellipse((48 + (index - 7) * 5, 10, 112 + (index - 7) * 5, 75), fill=255)
        image.paste(face, (0, 0), mask)
        records.append(
            _record(
                f"frame-{index}",
                image,
                time=float(index),
                tags={"general": {"1girl": 0.17 if index == 0 else 0.8}},
            )
        )
    records.append(
        _record(
            "below-threshold",
            _cartoon(shift=70, color=(70, 100, 200)),
            time=20.0,
            tags={"general": {"1girl": 0.169}},
        )
    )

    report = curate_candidates(records, minimum_cut_quality=0.0, time_window=0.0)

    assert len(report["kept"]) == 14
    assert report["summary"]["character_present"] == 14
    assert report["summary"]["kept"] > 12
    assert all(item["character_presence"] for item in report["kept"])
    assert all(item["presence_evidence"]["method"] == "pixai_general_tag" for item in report["kept"])
    assert next(item for item in report["excluded"] if item["id"] == "below-threshold")["exclusion_reason"] == "no_character"


def test_compacted_report_tags_keep_only_character_quality_evidence() -> None:
    report = curate_candidates(
        [
            _record(
                "compact-tags",
                _cartoon(),
                time=0.0,
                tags={"general": {"1girl": 0.9, "smile": 0.8, "blue_hair": 0.99, "artist_name": 0.95}},
            )
        ],
        include_full_tags=False,
        minimum_cut_quality=0.0,
    )

    candidate = report["kept"][0]
    assert candidate["tags_compacted"] is True
    assert candidate["tags"]["general"]["1girl"] == pytest.approx(0.9)
    assert candidate["tags"]["general"]["smile"] == pytest.approx(0.8)
    assert "blue_hair" not in candidate["tags"]["general"]
    assert "artist_name" not in candidate["tags"]["general"]


def test_strong_large_detector_is_fallback_but_tiny_city_box_is_not() -> None:
    image = _cartoon()
    records = [
        _record(
            "large-person",
            image.copy(),
            time=0.0,
            tags={"general": {"building": 0.95}},
            detections=[{"kind": "person", "box": [0.25, 0.10, 0.75, 0.90], "score": 0.9}],
        ),
        _record(
            "tiny-background-box",
            image.copy(),
            time=3.0,
            tags={"general": {"building": 0.95}},
            detections=[{"kind": "person", "box": [0.01, 0.02, 0.04, 0.08], "score": 0.99}],
        ),
    ]

    report = curate_candidates(records)

    assert [item["id"] for item in report["kept"]] == ["large-person"]
    excluded = next(item for item in report["excluded"] if item["id"] == "tiny-background-box")
    assert excluded["exclusion_reason"] == "no_character"
    assert report["kept"][0]["presence_evidence"]["method"] == "strong_large_detector"


def test_near_empty_black_frame_rejects_spurious_pixai_character_tags() -> None:
    report = curate_candidates(
        [_record("black", Image.new("RGB", (160, 90), "black"), time=0.0, tags={"general": {"1girl": 0.99}})],
        minimum_cut_quality=0.0,
    )

    assert not report["kept"]
    assert report["excluded"][0]["exclusion_reason"] == "no_character"
    assert report["excluded"][0]["presence_evidence"]["content_gate"]["passed"] is False

    watermark = Image.new("RGB", (160, 90), "black")
    ImageDraw.Draw(watermark).text((142, 4), "h!", fill="white")
    watermark_report = curate_candidates(
        [_record("watermark-only", watermark, time=0.0, tags={"general": {"1girl": 0.99}})],
        minimum_cut_quality=0.0,
    )
    assert not watermark_report["kept"]
    assert watermark_report["excluded"][0]["exclusion_reason"] == "no_character"
    assert watermark_report["excluded"][0]["presence_evidence"]["content_gate"]["passed"] is False


def test_cut_quality_floor_filters_weak_cuts_without_a_count_cap() -> None:
    strong_tags = {"general": {"1girl": 0.99, "smile": 0.95, "blush": 0.8, "portrait": 0.9}}
    weak_tags = {"general": {"1girl": 0.99}}
    strong = _record("strong", _cartoon(color=(245, 180, 165)), time=0.0, tags=strong_tags, quality=0.95)
    strong.update(technical_quality=0.95, source_score=0.5)
    weak = _record(
        "weak",
        _cartoon(mouth_open=True, shift=22, color=(95, 170, 210)),
        time=4.0,
        tags=weak_tags,
        quality=0.1,
    )
    weak.update(technical_quality=0.1, source_score=0.5)

    report = curate_candidates([strong, weak], minimum_cut_quality=0.5)

    assert [item["id"] for item in report["kept"]] == ["strong"]
    assert next(item for item in report["excluded"] if item["id"] == "weak")["exclusion_reason"] == "below_minimum_cut_quality"
    assert report["summary"]["excluded_below_minimum_cut_quality"] == 1
    assert report["settings"]["minimum_cut_quality"] == pytest.approx(0.5)


def test_city_like_exact_scene_variants_group_and_keep_the_better_quality_frame() -> None:
    base = Image.new("RGB", (160, 90), (80, 120, 150))
    draw = ImageDraw.Draw(base)
    draw.rectangle((0, 42, 159, 90), fill=(75, 95, 110))
    for x, height in ((5, 25), (35, 40), (68, 31), (100, 48), (136, 28)):
        draw.rectangle((x, 42 - height, x + 18, 42), fill=(120, 135, 145))
    slight_pixel_variant = base.copy()
    slight_pixel_variant.putpixel((79, 40), (83, 123, 153))
    records = [
        _record("city-a", base, time=0.0, quality=0.4),
        _record("city-b", slight_pixel_variant, time=0.9, quality=0.8),
    ]

    report = curate_candidates(records)

    assert report["summary"]["near_duplicate_groups"] == 1
    assert len(report["kept"]) == 1
    assert report["kept"][0]["id"] == "city-b"
    assert report["excluded"][0]["exclusion_reason"] == "near_duplicate_lower_rank"
    assert report["excluded"][0]["duplicate_of"] == "city-b"
    evidence = _group_for(report, "city-a")["pair_evidence"][0]
    assert evidence["grid_local_hash_max_distance"] <= 4
    assert evidence["resized_rgb_mean_absolute_difference"] <= 0.006


def test_global_visual_duplicate_is_found_more_than_two_seconds_later() -> None:
    first_image = _cartoon()
    second_image = first_image.copy()
    second_image.putpixel((79, 40), (93, 131, 181))
    report = curate_candidates(
        [
            _record("repeat-first", first_image, time=0.0),
            _record("repeat-later", second_image, time=12.0),
        ]
    )

    group = _group_for(report, "repeat-first")
    assert set(group["members"]) == {"repeat-first", "repeat-later"}
    assert report["summary"]["near_duplicate_groups"] == 1
    assert group["pair_evidence"][0]["near_duplicate"] is True
    assert group["pair_evidence"][0]["time_delta_seconds"] == pytest.approx(12.0)
    assert report["settings"]["time_window_applied_to_visual_duplicates"] is False


def test_single_thread_full_resolution_registration_accepts_a_small_subject_shift() -> None:
    pytest.importorskip("cv2")
    left = {"features": _image_features(_cartoon(), [])}
    right = {"features": _image_features(_cartoon(shift=4), [])}

    result = _registered_similarity(left, right)

    assert result is not None
    assert result["alignment_checked"] is True
    assert result["alignment_translation_fraction"] <= 0.11
    assert result["alignment_correlation"] >= 0.93
    assert result["alignment_mae"] <= 0.025
    assert result["alignment_rgb_mae"] <= 0.03


def test_matching_global_hashes_alone_do_not_remove_pixel_different_frame() -> None:
    first_image = _cartoon()
    gray = first_image.convert("L")
    second_image = Image.merge(
        "RGB",
        (
            gray.point(lambda value: min(255, value + 4)),
            gray.point(lambda value: max(0, value - 2)),
            gray.copy(),
        ),
    )
    report = curate_candidates(
        [
            _record("same-hash-first", first_image, time=0.0),
            _record("same-hash-color-shift", second_image, time=9.0),
        ]
    )

    assert len(report["kept"]) == 2
    assert {frozenset(group["members"]) for group in report["groups"]} == {
        frozenset({"same-hash-first"}),
        frozenset({"same-hash-color-shift"}),
    }
    evidence = next(
        evidence
        for group in report["groups"]
        for evidence in group["pair_evidence"]
    )
    assert evidence["phash_distance"] == 0
    assert evidence["dhash_distance"] == 0
    assert evidence["near_duplicate"] is False
    assert evidence["decision_reason"] == "pixel_difference_too_large"


def test_complete_link_does_not_collapse_a_near_duplicate_chain() -> None:
    base = _cartoon()
    records = []
    first = base.copy()
    second = base.copy()
    third = base.copy()
    ImageDraw.Draw(second).rectangle((0, 0, 39, 39), fill=(162, 130, 180))
    ImageDraw.Draw(third).rectangle((0, 0, 39, 39), fill=(162, 130, 180))
    ImageDraw.Draw(third).rectangle((120, 0, 159, 39), fill=(162, 130, 180))
    records = [
        _record("chain-a", first, time=0.0),
        _record("chain-b", second, time=3.0),
        _record("chain-c", third, time=6.0),
    ]

    report = curate_candidates(records, time_window=60.0)

    assert {frozenset(group["members"]) for group in report["groups"]} == {
        frozenset({"chain-a", "chain-b"}),
        frozenset({"chain-c"}),
    }
    assert report["summary"]["near_duplicate_groups"] == 1


def test_hash_nomination_is_the_and_intersection_for_pairwise_and_lsh_paths() -> None:
    records = [
        _hash_record("base", 0, 0),
        _hash_record("p21-d24", (1 << 21) - 1, (1 << 24) - 1),
        _hash_record("p23-d1", (1 << 23) - 1, 1),
        _hash_record("p1-d27", 1, (1 << 27) - 1),
        _hash_record("p3-d3", 0b111, 0b111),
    ]
    pairwise = _pairwise_visual_hash_neighbors(records)
    phash_buckets, phash_segments = _visual_hash_index(records, "phash", 22)
    dhash_buckets, dhash_segments = _visual_hash_index(records, "dhash", 26)
    indexed = [
        _indexed_visual_hash_neighbors(
            index,
            records,
            phash_buckets,
            phash_segments,
            dhash_buckets,
            dhash_segments,
        )
        for index in range(len(records))
    ]

    assert pairwise == indexed
    assert {1, 4} <= pairwise[0]
    assert 2 not in pairwise[0]
    assert 3 not in pairwise[0]
    assert _pairwise_visual_hash_neighbors(records)[1] == {0, 2, 3, 4}


@pytest.mark.parametrize(
    ("phash_distance", "dhash_distance"),
    [(23, 1), (1, 27)],
)
def test_pairs_outside_and_nomination_cannot_pass_registered_bounds(
    monkeypatch, phash_distance: int, dhash_distance: int
) -> None:
    left = _internal_face_record("left", [])
    right = _internal_face_record("right", [])
    left["features"]["phash"] = f"{0:016x}"
    left["features"]["dhash"] = f"{0:016x}"
    right["features"]["phash"] = f"{(1 << phash_distance) - 1:016x}"
    right["features"]["dhash"] = f"{(1 << dhash_distance) - 1:016x}"
    monkeypatch.setattr(
        curation_module,
        "_registered_similarity",
        lambda *_args, **_kwargs: pytest.fail("out-of-radius pair must not run ECC"),
    )

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["phash_distance"] == phash_distance
    assert evidence["dhash_distance"] == dhash_distance
    assert evidence["near_duplicate"] is False


def test_tag_expression_is_a_hard_veto_even_when_ecc_would_accept(monkeypatch) -> None:
    left = _internal_face_record(
        "tag-open",
        [],
        tags={"general": {"1girl": 0.9, "open_mouth": 0.9, "closed_mouth": 0.0}},
    )
    right = _internal_face_record(
        "tag-closed",
        [],
        tags={"general": {"1girl": 0.9, "open_mouth": 0.0, "closed_mouth": 0.9}},
    )
    monkeypatch.setattr(
        curation_module,
        "_registered_similarity",
        lambda *_args, **_kwargs: pytest.fail("tag expression veto must stop before ECC"),
    )

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["tag_expression_change"] is True
    assert evidence["raw_face_change"] is False
    assert evidence["expression_change"] is True
    assert evidence["near_duplicate"] is False
    assert evidence["decision_reason"] == "expression_or_face_change"


def test_legacy_expression_metric_remains_a_conservative_veto(monkeypatch) -> None:
    left = _internal_face_record("legacy-left", [])
    right = _internal_face_record("legacy-right", [])
    legacy_metrics = {
        "phash_distance": 0,
        "dhash_distance": 0,
        "grid_local_hash_max_distance": 0,
        "grid_local_hash_mean_distance": 0.0,
        "resized_rgb_mean_absolute_difference": 0.0,
        "global_gray_ssim": 1.0,
        "expression_change": True,
        "material_people_change": False,
    }
    monkeypatch.setattr(curation_module, "_similarity_metrics", lambda *_args: dict(legacy_metrics))
    monkeypatch.setattr(
        curation_module,
        "_registered_similarity",
        lambda *_args: pytest.fail("legacy expression evidence must remain a veto"),
    )

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["tag_expression_change"] is True
    assert evidence["raw_face_change"] is False
    assert evidence["expression_change"] is True
    assert evidence["near_duplicate"] is False


def test_material_people_change_is_also_a_hard_veto(monkeypatch) -> None:
    left = _internal_face_record("people-left", [])
    right = _internal_face_record("people-right", [])
    monkeypatch.setattr(
        curation_module,
        "_similarity_metrics",
        lambda *_args: {
            "phash_distance": 5,
            "dhash_distance": 5,
            "grid_local_hash_max_distance": 0,
            "grid_local_hash_mean_distance": 0.0,
            "resized_rgb_mean_absolute_difference": 0.01,
            "global_gray_ssim": 0.99,
            "expression_change": False,
            "material_people_change": True,
        },
    )
    monkeypatch.setattr(
        curation_module,
        "_registered_similarity",
        lambda *_args: pytest.fail("people-count veto must stop before ECC"),
    )

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["material_people_change"] is True
    assert evidence["near_duplicate"] is False


@pytest.mark.parametrize(
    ("left_box", "right_box", "sample_offset"),
    [
        ((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0), 2),
        ((0.2, 0.2, 0.8, 0.8), (0.2666667, 0.2, 0.8666667, 0.8), -2),
    ],
)
def test_aligned_face_proof_lifts_only_translation_crop_veto(
    monkeypatch, left_box, right_box, sample_offset: int
) -> None:
    left_rgb = _face_patch()
    right_rgb = _shift_crop_for_inverse_sample(left_rgb, sample_offset)
    left = _internal_face_record("translated-left", [_face(left_box, left_rgb)])
    right = _internal_face_record("translated-right", [_face(right_box, right_rgb)])
    dx = 256.0 * 0.0416667
    # The fake ECC evidence pins the coordinate convention while the actual
    # 48x48 resampling, mask coverage, and MAE are computed by OpenCV.
    monkeypatch.setattr(curation_module, "_registered_similarity", lambda *_args: _accepted_registration(dx))

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["raw_face_change"] is True
    assert evidence["aligned_face_check_performed"] is True
    assert evidence["all_face_matches_verified"] is True
    assert evidence["minimum_face_support"] >= 0.80
    assert evidence["maximum_aligned_face_mae"] <= 0.065
    assert evidence["near_duplicate"] is True
    assert evidence["decision_reason"] == "registered_structure_agrees"


def test_real_translation_ecc_aligns_moving_and_near_full_frame_face_boxes() -> None:
    image = Image.new("RGB", (256, 144), (72, 104, 154))
    draw = ImageDraw.Draw(image)
    for y in range(0, 144, 8):
        for x in range(0, 256, 8):
            if ((x // 8) + (y // 8)) % 2:
                draw.rectangle((x, y, x + 7, y + 7), fill=(105, 145, 190))
    draw.ellipse((78, 18, 180, 132), fill=(238, 190, 150), outline=(25, 20, 35), width=3)
    draw.ellipse((101, 54, 119, 72), fill=(20, 20, 30))
    draw.ellipse((140, 54, 158, 72), fill=(20, 20, 30))
    draw.arc((112, 80, 146, 110), start=10, end=170, fill=(60, 20, 35), width=4)
    translated = ImageChops.offset(image, 6, 0)
    tags = {"general": {"1girl": 0.9}}
    moving_left_box = [0.23, 0.08, 0.75, 0.94]
    moving_right_box = [moving_left_box[0] + 6 / 256 + 0.02, 0.08, moving_left_box[2] + 6 / 256 + 0.02, 0.94]
    boxes = [
        (moving_left_box, moving_right_box),
        ([0.01, 0.01, 0.99, 0.99], [0.01, 0.01, 0.99, 0.99]),
    ]

    for index, (left_box, right_box) in enumerate(boxes):
        report = curate_candidates(
            [
                {
                    "id": f"real-ecc-left-{index}",
                    "time": 0.0,
                    "image": image,
                    "tags": tags,
                    "detections": [{"kind": "head", "box": left_box, "score": 0.99}],
                },
                {
                    "id": f"real-ecc-right-{index}",
                    "time": 10.0,
                    "image": translated,
                    "tags": tags,
                    "detections": [{"kind": "head", "box": right_box, "score": 0.99}],
                },
            ],
            minimum_cut_quality=0.0,
        )
        group = _group_for(report, f"real-ecc-left-{index}")
        evidence = group["pair_evidence"][0]
        assert set(group["members"]) == {f"real-ecc-left-{index}", f"real-ecc-right-{index}"}
        assert evidence["raw_face_change"] is True
        assert evidence["alignment_checked"] is True
        assert evidence["alignment_translation_fraction"] <= 0.11
        assert evidence["aligned_face_check_performed"] is True
        assert evidence["all_face_matches_verified"] is True
        assert evidence["minimum_face_support"] >= 0.80
        assert evidence["maximum_aligned_face_mae"] <= 0.065


def test_real_face_pixel_change_is_rejected_after_alignment(monkeypatch) -> None:
    left_rgb = _face_patch()
    changed_rgb = left_rgb.copy()
    changed_rgb[8:40, 8:40] = np.asarray([0.01, 0.01, 0.02], dtype=np.float32)
    left = _internal_face_record("unchanged-face", [_face((0.0, 0.0, 1.0, 1.0), left_rgb)])
    right = _internal_face_record("changed-face", [_face((0.0, 0.0, 1.0, 1.0), changed_rgb)])
    monkeypatch.setattr(curation_module, "_registered_similarity", lambda *_args: _accepted_registration())

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["tag_expression_change"] is False
    assert evidence["raw_face_change"] is True
    assert evidence["aligned_face_check_performed"] is True
    assert evidence["all_face_matches_verified"] is False
    assert evidence["maximum_aligned_face_mae"] > 0.065
    assert evidence["expression_change"] is True
    assert evidence["near_duplicate"] is False


def test_one_changed_person_rejects_the_whole_aligned_face_proof(monkeypatch) -> None:
    left_a, left_b = _face_patch((220, 180, 150)), _face_patch((90, 130, 220))
    right_a = _shift_crop_for_inverse_sample(left_a, 5)
    right_b = _shift_crop_for_inverse_sample(left_b, 5)
    right_b[10:40, 12:36] = np.asarray([0.01, 0.01, 0.02], dtype=np.float32)
    boxes = ((0.1, 0.1, 0.4, 0.9), (0.6, 0.1, 0.9, 0.9))
    left = _internal_face_record("two-left", [_face(boxes[0], left_a), _face(boxes[1], left_b)])
    right = _internal_face_record("two-right", [_face(boxes[0], right_a), _face(boxes[1], right_b)])
    monkeypatch.setattr(curation_module, "_registered_similarity", lambda *_args: _accepted_registration(8.0))

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["raw_face_change"] is True
    assert evidence["face_matching_count"] == 2
    assert evidence["all_face_matches_verified"] is False
    assert evidence["maximum_aligned_face_mae"] > 0.065
    assert evidence["near_duplicate"] is False


def test_unmatched_invalid_and_low_coverage_faces_cannot_release_veto() -> None:
    rgb = _face_patch()
    one = _internal_face_record("one", [_face((0.0, 0.0, 1.0, 1.0), rgb)])
    two_faces = _internal_face_record(
        "two", [_face((0.0, 0.0, 1.0, 1.0), rgb), _face((0.1, 0.1, 0.3, 0.3), rgb)]
    )
    mismatch = _aligned_face_similarity(one, two_faces, _accepted_registration())
    assert mismatch["all_face_matches_verified"] is False
    assert mismatch["face_check_failure"] == "face_count_mismatch"

    invalid = _internal_face_record("invalid", [_face((float("nan"), 0, 1, 1), rgb)])
    invalid_result = _aligned_face_similarity(invalid, one, _accepted_registration())
    assert invalid_result["all_face_matches_verified"] is False
    assert invalid_result["face_check_failure"] == "invalid_face_data"

    low_coverage = _aligned_face_similarity(one, one, _accepted_registration(dx=60.0))
    assert low_coverage["all_face_matches_verified"] is False
    assert low_coverage["minimum_face_support"] < 0.80


def test_face_correspondence_greedy_ties_do_not_reassign_by_pixel_result(monkeypatch) -> None:
    rgb = _face_patch()
    boxes = ((0.1, 0.1, 0.5, 0.9), (0.5, 0.1, 0.9, 0.9))
    left = _internal_face_record("greedy-left", [_face(box, rgb) for box in boxes])
    right = _internal_face_record("greedy-right", [_face(box, rgb) for box in boxes])
    overlaps = iter((0.90, 0.80, 0.85, 0.44))
    monkeypatch.setattr(curation_module, "_box_iou", lambda *_args: next(overlaps))

    evidence = _aligned_face_similarity(left, right, _accepted_registration())

    assert evidence["face_matching_count"] == 1
    assert evidence["all_face_matches_verified"] is False
    assert evidence["face_check_failure"] == "unmatched_faces"


def test_ecc_failure_keeps_raw_face_veto(monkeypatch) -> None:
    left_rgb = _face_patch()
    right_rgb = _shift_crop_for_inverse_sample(left_rgb, -2)
    left = _internal_face_record("ecc-left", [_face((0.0, 0.0, 1.0, 1.0), left_rgb)])
    right = _internal_face_record("ecc-right", [_face((0.0, 0.0, 1.0, 1.0), right_rgb)])
    monkeypatch.setattr(curation_module, "_registered_similarity", lambda *_args: None)

    evidence = _pair_evaluation(left, right, 0.0)

    assert evidence["raw_face_change"] is True
    assert evidence["expression_change"] is True
    assert evidence["alignment_checked"] is False
    assert evidence["aligned_face_check_performed"] is False
    assert evidence["near_duplicate"] is False


def _registration_arrays() -> dict[str, np.ndarray]:
    left = _image_features(_cartoon(), [])
    right = _image_features(_cartoon(shift=3), [])
    return {
        "left_gray": np.array(left["alignment_gray"], dtype=np.float32),
        "right_gray": np.array(right["alignment_gray"], dtype=np.float32),
        "left_rgb": np.array(left["alignment_rgb"], dtype=np.float32),
        "right_rgb": np.array(right["alignment_rgb"], dtype=np.float32),
    }


def _registration_input(gray: np.ndarray, rgb: np.ndarray) -> dict:
    return {"features": {"alignment_gray": gray, "alignment_rgb": rgb}}


def _register_identity_ecc(monkeypatch, cv2, *, dx: float = 0.0, dy: float = 0.0) -> None:
    def fake_find_transform_ecc(*_args, **_kwargs):
        warp = np.eye(2, 3, dtype=np.float32)
        warp[0, 2] = np.float32(dx)
        warp[1, 2] = np.float32(dy)
        return 1.0, warp

    monkeypatch.setattr(cv2, "findTransformECC", fake_find_transform_ecc)


def _register_pair(
    arrays: dict[str, np.ndarray],
    *,
    gray: str = "left_gray",
    rgb: str = "left_rgb",
) -> dict | None:
    return _registered_similarity(
        _registration_input(arrays[gray], arrays[rgb]),
        _registration_input(arrays["right_gray"], arrays["right_rgb"]),
    )


def test_identity_ecc_with_constant_crops_returns_conservative_none(monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    _register_identity_ecc(monkeypatch, cv2)
    gray = np.full((144, 256), 108, dtype=np.uint8)
    rgb = np.full((72, 128, 3), 108, dtype=np.uint8)

    result = _registered_similarity(
        _registration_input(gray, rgb),
        _registration_input(gray.copy(), rgb.copy()),
    )

    assert result is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("gray_name", ["left_gray", "right_gray"])
def test_nonfinite_correlation_crops_return_conservative_none(monkeypatch, bad_value, gray_name) -> None:
    cv2 = pytest.importorskip("cv2")
    _register_identity_ecc(monkeypatch, cv2)
    arrays = _registration_arrays()
    assert _register_pair(arrays) is not None

    poisoned = arrays[gray_name].copy()
    height, width = poisoned.shape
    poisoned[height // 2, width // 2] = np.float32(bad_value)
    arrays[gray_name] = poisoned

    assert _register_pair(arrays) is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("component", ["dx", "dy"])
def test_nonfinite_warp_translation_returns_conservative_none(monkeypatch, bad_value, component) -> None:
    cv2 = pytest.importorskip("cv2")
    _register_identity_ecc(
        monkeypatch,
        cv2,
        dx=bad_value if component == "dx" else 0.0,
        dy=bad_value if component == "dy" else 0.0,
    )

    assert _register_pair(_registration_arrays()) is None


def test_nonfinite_warp_output_returns_conservative_none(monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    _register_identity_ecc(monkeypatch, cv2)

    def fake_warp_affine(source, _matrix, _dsize, **_kwargs):
        return np.full(np.shape(source), np.nan, dtype=np.float32)

    monkeypatch.setattr(cv2, "warpAffine", fake_warp_affine)

    assert _register_pair(_registration_arrays()) is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_rgb_mae_returns_conservative_none(monkeypatch, bad_value) -> None:
    cv2 = pytest.importorskip("cv2")
    _register_identity_ecc(monkeypatch, cv2)
    arrays = _registration_arrays()
    assert _register_pair(arrays) is not None

    poisoned = arrays["left_rgb"].copy()
    rgb_height, rgb_width = poisoned.shape[:2]
    poisoned[rgb_height // 2, rgb_width // 2, 1] = np.float32(bad_value)
    arrays["left_rgb"] = poisoned

    assert _register_pair(arrays) is None


def test_identity_ecc_preserves_finite_equations_and_rounding(monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    _register_identity_ecc(monkeypatch, cv2)
    arrays = _registration_arrays()

    result = _register_pair(arrays)

    assert result is not None
    first = arrays["left_gray"] / 255.0
    second = arrays["right_gray"] / 255.0
    height, width = first.shape
    aligned = cv2.warpAffine(
        second,
        np.eye(2, 3, dtype=np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    crop = (slice(8, height - 8), slice(12, width - 12))
    aligned_corr = float(np.corrcoef(first[crop].reshape(-1), aligned[crop].reshape(-1))[0, 1])
    before_corr = float(np.corrcoef(first[crop].reshape(-1), second[crop].reshape(-1))[0, 1])
    aligned_mae = float(np.mean(np.abs(first[crop] - aligned[crop])))

    assert result["alignment_checked"] is True
    assert result["alignment_correlation"] == round(aligned_corr, 8)
    assert result["alignment_correlation_before"] == round(before_corr, 8)
    assert result["alignment_gain"] == round(aligned_corr - before_corr, 8)
    assert result["alignment_mae"] == round(aligned_mae, 8)
    assert result["alignment_dx_pixels"] == 0.0
    assert result["alignment_dy_pixels"] == 0.0
    assert result["alignment_translation_fraction"] == 0.0
    assert all(np.isfinite(value) for name, value in result.items() if name != "alignment_checked")
    assert json.dumps(result, allow_nan=False)


def test_real_shifted_cartoon_registration_stays_finite_and_json_safe() -> None:
    pytest.importorskip("cv2")
    left = {"features": _image_features(_cartoon(), [])}
    right = {"features": _image_features(_cartoon(shift=4), [])}

    result = _registered_similarity(left, right)

    assert result is not None
    numeric = {name: value for name, value in result.items() if name != "alignment_checked"}
    assert all(np.isfinite(list(numeric.values())))
    assert all(value == round(value, 8) for value in numeric.values())
    assert result["alignment_correlation"] >= 0.93
    assert result["alignment_correlation"] > result["alignment_correlation_before"]
    assert result["alignment_gain"] == pytest.approx(
        result["alignment_correlation"] - result["alignment_correlation_before"], abs=1e-7
    )
    assert result["alignment_mae"] <= 0.025
    assert result["alignment_rgb_mae"] <= 0.03
    assert result["alignment_translation_fraction"] <= 0.11
    assert json.dumps(result, allow_nan=False)


def test_invalid_registration_keeps_raw_face_veto_and_strict_json(monkeypatch) -> None:
    left_rgb = _face_patch()
    right_rgb = _shift_crop_for_inverse_sample(left_rgb, -2)
    left = _internal_face_record("nan-guard-left", [_face((0.0, 0.0, 1.0, 1.0), left_rgb)])
    right = _internal_face_record("nan-guard-right", [_face((0.0, 0.0, 1.0, 1.0), right_rgb)])
    calls: list[tuple[str, str]] = []

    def fake_registered_similarity(left_record, right_record):
        calls.append((left_record["id"], right_record["id"]))
        return None

    monkeypatch.setattr(curation_module, "_registered_similarity", fake_registered_similarity)

    evidence = _pair_evaluation(left, right, 0.0)

    assert calls == [("nan-guard-left", "nan-guard-right")]
    assert evidence["near_duplicate"] is False
    assert evidence["raw_face_change"] is True
    assert evidence["expression_change"] is True
    assert evidence["decision_reason"] == "expression_or_face_change"
    assert evidence["alignment_checked"] is False
    assert evidence["aligned_face_check_performed"] is False
    assert evidence["registration_skipped_by_rejected_floor"] is False
    for name in (
        "alignment_correlation",
        "alignment_correlation_before",
        "alignment_gain",
        "alignment_mae",
        "alignment_rgb_mae",
        "alignment_translation_fraction",
        "alignment_dx_pixels",
        "alignment_dy_pixels",
    ):
        assert evidence[name] is None
    assert json.loads(json.dumps(evidence, allow_nan=False)) == evidence


def test_rejected_distance_floor_skips_only_registration_and_keeps_tuple(monkeypatch) -> None:
    left = _internal_face_record("floor-left", [])
    right = _internal_face_record("floor-right", [])
    metrics = {
        "phash_distance": 5,
        "dhash_distance": 5,
        "grid_local_hash_max_distance": 8,
        "grid_local_hash_mean_distance": 2.0,
        "resized_rgb_mean_absolute_difference": 0.02,
        "global_gray_ssim": 0.8,
        "expression_change": False,
        "material_people_change": False,
    }
    monkeypatch.setattr(curation_module, "_similarity_metrics", lambda *_args: dict(metrics))
    monkeypatch.setattr(
        curation_module,
        "_registered_similarity",
        lambda *_args: pytest.fail("equal rejected distance must skip ECC"),
    )

    floor = _evidence_distance(metrics)
    evidence = _pair_evaluation(left, right, 0.0, rejected_distance_floor=floor)

    assert evidence["registration_skipped_by_rejected_floor"] is True
    assert _evidence_distance(evidence) == floor
    assert evidence["near_duplicate"] is False


def test_later_valid_complete_link_group_is_checked_after_an_earlier_rejection(monkeypatch) -> None:
    records = [_hash_record("A", 0, 0), _hash_record("B", 0, 0), _hash_record("C", 0, 0)]
    for index, record in enumerate(records):
        record["time"] = float(index)
    called: list[tuple[str, str]] = []

    def fake_evaluation(left, right, _window, rejected_distance_floor=None):
        called.append((left["id"], right["id"]))
        pair = frozenset((left["id"], right["id"]))
        duplicate = pair == frozenset(("B", "C"))
        return {
            "left_id": left["id"],
            "right_id": right["id"],
            "near_duplicate": duplicate,
            "decision_reason": "registered_structure_agrees" if duplicate else "global_similarity_threshold_not_met",
            "resized_rgb_mean_absolute_difference": 0.01 if duplicate else 0.08,
            "global_gray_ssim": 0.99 if duplicate else 0.7,
            "grid_local_hash_max_distance": 1 if duplicate else 8,
        }

    monkeypatch.setattr(curation_module, "_pair_evaluation", fake_evaluation)

    groups = _visual_groups(records, 0.0)

    assert [set(record["id"] for record in group) for group in groups] == [{"A"}, {"B", "C"}]
    assert ("C", "B") in called


@pytest.mark.parametrize(("first_distance", "second_distance", "expected_reject_id"), [(0.05, 0.05, "A"), (0.09, 0.01, "B")])
def test_best_rejection_keeps_first_tie_and_replaces_with_better_evidence(
    monkeypatch, first_distance: float, second_distance: float, expected_reject_id: str
) -> None:
    records = [_hash_record("A", 0, 0), _hash_record("B", 0, 0), _hash_record("C", 0, 0)]
    for index, record in enumerate(records):
        record["time"] = float(index)

    def fake_evaluation(left, right, _window, rejected_distance_floor=None):
        pair = frozenset((left["id"], right["id"]))
        distance = first_distance if pair == frozenset(("A", "C")) else second_distance
        return {
            "left_id": left["id"],
            "right_id": right["id"],
            "near_duplicate": False,
            "decision_reason": "global_similarity_threshold_not_met",
            "resized_rgb_mean_absolute_difference": distance,
            "global_gray_ssim": 0.9,
            "grid_local_hash_max_distance": 3,
        }

    monkeypatch.setattr(curation_module, "_pair_evaluation", fake_evaluation)

    groups = _visual_groups(records, 0.0)

    candidate_group = next(group for group in groups if group[0]["id"] == "C")
    assert candidate_group[0]["_group_pair_evidence"][0]["right_id"] == expected_reject_id


def test_rejected_group_incumbent_is_passed_to_later_member_checks(monkeypatch) -> None:
    records = [_hash_record("A", 0, 0), _hash_record("B", 0, 0), _hash_record("C", 0, 0)]
    for index, record in enumerate(records):
        record["time"] = float(index)
    floors: list[tuple[float, float, int] | None] = []

    def fake_evaluation(left, right, _window, rejected_distance_floor=None):
        pair = frozenset((left["id"], right["id"]))
        floors.append(rejected_distance_floor)
        if pair == frozenset(("A", "B")):
            distance, duplicate = 0.01, True
        elif pair == frozenset(("A", "C")):
            distance, duplicate = 0.05, False
        else:
            distance, duplicate = 0.10, False
        return {
            "left_id": left["id"],
            "right_id": right["id"],
            "near_duplicate": duplicate,
            "decision_reason": "registered_structure_agrees" if duplicate else "global_similarity_threshold_not_met",
            "resized_rgb_mean_absolute_difference": distance,
            "global_gray_ssim": 0.9,
            "grid_local_hash_max_distance": 3,
        }

    monkeypatch.setattr(curation_module, "_pair_evaluation", fake_evaluation)

    groups = _visual_groups(records, 0.0)

    assert [set(record["id"] for record in group) for group in groups] == [{"A", "B"}, {"C"}]
    assert (0.05, 0.09999999999999998, 3) in floors


def test_compact_group_counters_keep_one_full_proof_per_admitted_frame() -> None:
    image = _cartoon()
    report = curate_candidates(
        [
            _record("proof-a", image.copy(), time=0.0),
            _record("proof-b", image.copy(), time=1.0),
            _record("proof-c", image.copy(), time=2.0),
        ]
    )
    group = report["groups"][0]

    assert len(group["members"]) == 3
    assert group["pair_checks"]["evaluated"] == 3
    assert group["pair_checks"]["confirmed"] == 3
    assert group["pair_checks"]["admitted_frames"] == 2
    assert group["pair_checks"]["full_proof_count"] == 2
    assert group["pair_checks"]["one_full_proof_per_admitted_frame"] is True
    assert len(group["pair_evidence"]) == 2


def test_progress_reports_initial_each_assignment_no_character_and_empty_input() -> None:
    progress: list[tuple[int, int]] = []
    image = _cartoon()
    dark = Image.new("RGB", (160, 90), (2, 2, 3))
    curate_candidates(
        [
            _record("present", image, time=0.0),
            _record("empty-a", dark.copy(), time=1.0, tags={"general": {"1girl": 0.0}}),
            _record("empty-b", dark.copy(), time=2.0, tags={"general": {"1girl": 0.0}}),
        ],
        progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]

    empty_progress: list[tuple[int, int]] = []
    curate_candidates([], progress=lambda done, total: empty_progress.append((done, total)))
    assert empty_progress == [(0, 0)]


def test_expression_change_and_material_multi_person_pose_change_are_preserved() -> None:
    face_box = {"kind": "head", "box": [0.30, 0.10, 0.70, 0.83], "score": 0.95}
    mouth_records = [
        _record(
            "mouth-open",
            _cartoon(mouth_open=True),
            time=0.0,
            tags={"general": {"1girl": 0.9, "open_mouth": 0.715, "closed_mouth": 0.004}},
            detections=[face_box],
        ),
        _record(
            "mouth-closed",
            _cartoon(mouth_open=False),
            time=0.5,
            tags={"general": {"1girl": 0.9, "open_mouth": 0.027, "closed_mouth": 0.243}},
            detections=[face_box],
        ),
    ]
    pose_a = _cartoon(shift=-18, color=(220, 180, 145))
    pose_b = _cartoon(shift=18, color=(220, 180, 145))
    multi_tags = {"general": {"2girls": 0.9}}
    boxes_a = [
        {"kind": "person", "box": [0.08, 0.15, 0.43, 0.93], "score": 0.9},
        {"kind": "person", "box": [0.55, 0.15, 0.90, 0.93], "score": 0.9},
    ]
    boxes_b = [
        {"kind": "person", "box": [0.02, 0.15, 0.36, 0.93], "score": 0.9},
        {"kind": "person", "box": [0.64, 0.15, 0.98, 0.93], "score": 0.9},
    ]
    pose_records = [
        _record("pose-a", pose_a, time=10.0, tags=multi_tags, detections=boxes_a),
        _record("pose-b", pose_b, time=10.5, tags=multi_tags, detections=boxes_b),
    ]

    mouth_report = curate_candidates(mouth_records)
    pose_report = curate_candidates(pose_records)

    assert {item["id"] for item in mouth_report["kept"]} == {"mouth-open", "mouth-closed"}
    mouth_evidence = next(group["pair_evidence"] for group in mouth_report["groups"] if group["pair_evidence"])
    assert mouth_evidence[0]["near_duplicate"] is False
    assert mouth_evidence[0]["decision_reason"] == "expression_or_face_change"
    assert mouth_evidence[0]["expression_change"] is True
    assert len({item["group_id"] for item in mouth_report["kept"]}) == 2
    mouth_by_id = {item["id"]: item for item in mouth_report["kept"]}
    assert mouth_by_id["mouth-open"]["quality_components"]["expression"] == 0.2145
    assert mouth_by_id["mouth-closed"]["quality_components"]["expression"] == 0.18
    assert {item["id"] for item in pose_report["kept"]} == {"pose-a", "pose-b"}
    assert len({item["group_id"] for item in pose_report["kept"]}) == 2


def test_one_person_expression_change_in_multi_person_frame_is_preserved() -> None:
    def scene(left_mouth_open: bool) -> Image.Image:
        image = Image.new("RGB", (240, 100), (92, 130, 180))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 78, 239, 99), fill=(60, 95, 80))
        for center in (65, 175):
            draw.ellipse((center - 34, 8, center + 34, 82), fill=(235, 185, 150), outline=(35, 25, 35), width=2)
            draw.ellipse((center - 14, 34, center - 8, 40), fill=(20, 20, 30))
            draw.ellipse((center + 8, 34, center + 14, 40), fill=(20, 20, 30))
        if left_mouth_open:
            draw.ellipse((58, 52, 72, 66), fill=(115, 25, 45), outline=(40, 20, 30))
        else:
            draw.arc((56, 48, 74, 66), start=15, end=165, fill=(70, 25, 35), width=2)
        draw.arc((166, 48, 184, 66), start=15, end=165, fill=(70, 25, 35), width=2)
        return image

    detections = [
        {"kind": "head", "box": [0.12, 0.08, 0.45, 0.84], "score": 0.95},
        {"kind": "head", "box": [0.58, 0.08, 0.87, 0.84], "score": 0.95},
    ]
    records = [
        _record(
            "two-person-open-mouth",
            scene(True),
            time=0.0,
            tags={"general": {"2girls": 0.9, "open_mouth": 0.72, "closed_mouth": 0.03}},
            detections=detections,
        ),
        _record(
            "two-person-closed-mouth",
            scene(False),
            time=8.0,
            tags={"general": {"2girls": 0.9, "open_mouth": 0.03, "closed_mouth": 0.72}},
            detections=detections,
        ),
    ]

    report = curate_candidates(records)

    assert {item["id"] for item in report["kept"]} == {
        "two-person-open-mouth",
        "two-person-closed-mouth",
    }
    assert len({item["group_id"] for item in report["kept"]}) == 2
    pair = next(
        evidence
        for group in report["groups"]
        for evidence in group["pair_evidence"]
    )
    assert pair["near_duplicate"] is False
    assert pair["expression_change"] is True
    assert pair["material_people_change"] is False


def test_tag_quality_fallback_ranks_close_visible_face_above_distant_obscured_character() -> None:
    records = [
        _record(
            "close-visible",
            _cartoon(),
            time=0.0,
            tags={
                "general": {
                    "1girl": 0.9,
                    "portrait": 0.95,
                    "close-up": 0.95,
                    "face_visible": 0.9,
                    "centered": 0.85,
                    "looking_at_viewer": 0.9,
                }
            },
        ),
        _record(
            "distant-obscured",
            _distant_character(),
            time=5.0,
            tags={
                "general": {
                    "1girl": 0.9,
                    "full_body": 0.9,
                    "from_behind": 0.9,
                    "face_obscured": 0.8,
                    "wide_shot": 0.9,
                    "off_center": 0.9,
                }
            },
        ),
    ]

    report = curate_candidates(records)
    selected = {item["id"]: item for item in report["kept"]}
    close, distant = selected["close-visible"], selected["distant-obscured"]

    assert close["character_presence"] is True and distant["character_presence"] is True
    assert close["presence_score"] >= 0.9 and distant["presence_score"] >= 0.9
    assert close["quality_components"]["face_visibility"] > distant["quality_components"]["face_visibility"]
    assert close["quality_components"]["composition"] > distant["quality_components"]["composition"]
    assert close["cut_quality"] > distant["cut_quality"]
    assert close["quality_components"]["expression"] == distant["quality_components"]["expression"] == 0.18


def test_detector_boxes_override_conflicting_face_visibility_tags() -> None:
    record = _record(
        "detector-primary",
        _cartoon(),
        time=0.0,
        tags={"general": {"1girl": 0.9, "portrait": 0.99, "face_visible": 0.99}},
        detections=[{"kind": "head", "box": [0.1, 0.1, 0.2, 0.2], "score": 0.9}],
    )

    result = curate_candidates([record])["kept"][0]

    assert result["quality_components"]["face_visibility"] < 0.7
    assert result["quality_components"]["face_visibility"] != 0.99


def test_cute_expression_rank_rewards_smile_and_blush_and_penalizes_sad_or_angry() -> None:
    image = _cartoon()
    records = [
        _record(
            "smiling-blushing",
            image.copy(),
            time=0.0,
            tags={"general": {"1girl": 0.9, "smile": 0.9, "blush": 0.8}},
        ),
        _record(
            "sad-angry",
            image.copy(),
            time=5.0,
            tags={"general": {"1girl": 0.9, "sad": 0.9, "angry": 0.8, "frown": 0.85}},
        ),
        _record(
            "neutral",
            image.copy(),
            time=10.0,
            tags={"general": {"1girl": 0.9}},
        ),
    ]

    report = curate_candidates(records)
    selected = {item["id"]: item for item in report["kept"]}
    smiling, unhappy, neutral = selected["smiling-blushing"], selected["sad-angry"], selected["neutral"]

    assert all(item["character_presence"] for item in (smiling, unhappy, neutral))
    assert smiling["quality_components"]["expression"] == 0.9
    assert unhappy["quality_components"]["expression"] == 0.0
    assert neutral["quality_components"]["expression"] == 0.18
    assert smiling["cut_quality"] > neutral["cut_quality"] > unhappy["cut_quality"]


def test_no_detection_subtle_mouth_and_eye_tag_changes_are_protected_without_splitting_repeats() -> None:
    same_frame = _cartoon()
    mouth_report = curate_candidates(
        [
            _record(
                "mouth-state-open",
                _cartoon(mouth_open=True),
                time=0.0,
                tags={"general": {"1girl": 0.9, "open_mouth": 0.72, "closed_mouth": 0.03}},
            ),
            _record(
                "mouth-state-closed",
                _cartoon(mouth_open=False),
                time=0.5,
                tags={"general": {"1girl": 0.9, "open_mouth": 0.03, "closed_mouth": 0.72}},
            ),
        ]
    )
    eye_closed = same_frame.copy()
    draw = ImageDraw.Draw(eye_closed)
    for left in (65, 87):
        draw.rectangle((left, 30, left + 8, 38), fill=(240, 190, 150))
        draw.line((left, 34, left + 8, 34), fill=(70, 25, 35), width=2)
    eye_report = curate_candidates(
        [
            _record(
                "eye-state-open",
                same_frame.copy(),
                time=5.0,
                tags={"general": {"1girl": 0.9, "open_eyes": 0.72, "closed_eyes": 0.03}},
            ),
            _record(
                "eye-state-closed",
                eye_closed,
                time=5.5,
                tags={"general": {"1girl": 0.9, "open_eyes": 0.03, "closed_eyes": 0.72}},
            ),
        ]
    )
    noise_report = curate_candidates(
        [
            _record(
                "tag-noise-a",
                same_frame.copy(),
                time=10.0,
                tags={"general": {"1girl": 0.9, "open_mouth": 0.68, "closed_mouth": 0.05}},
            ),
            _record(
                "tag-noise-b",
                same_frame.copy(),
                time=20.5,
                tags={"general": {"1girl": 0.9, "open_mouth": 0.54, "closed_mouth": 0.06}},
            ),
        ]
    )

    assert {item["id"] for item in mouth_report["kept"]} == {"mouth-state-open", "mouth-state-closed"}
    assert len({item["group_id"] for item in mouth_report["kept"]}) == 2
    assert {item["id"] for item in eye_report["kept"]} == {"eye-state-open", "eye-state-closed"}
    assert len({item["group_id"] for item in eye_report["kept"]}) == 2
    noise_group = _group_for(noise_report, "tag-noise-a")
    assert noise_group["members"] == ["tag-noise-a", "tag-noise-b"]
    assert noise_group["pair_evidence"][0]["near_duplicate"] is True


def test_quality_ranks_near_identical_frames_without_using_identity_or_a_top_n_limit() -> None:
    image = _cartoon()
    records = [
        _record("lower-quality", image.copy(), time=0.0, quality=0.2),
        _record("higher-quality", image.copy(), time=0.4, quality=0.9),
    ]

    report = curate_candidates(records)

    assert len(report["kept"]) == 1
    assert report["kept"][0]["id"] == "higher-quality"
    assert report["kept"][0]["cut_quality"] > next(
        item["cut_quality"] for item in report["excluded"]
    )
    assert report["excluded"][0]["exclusion_reason"] == "near_duplicate_lower_rank"


def test_opt_in_private_city_pair_is_excluded_as_two_no_character_singletons() -> None:
    rows = _private_curation_rows({5490, 99090})
    report = curate_candidates(
        [_record_from_private_report(rows[pts]) for pts in (5490, 99090)],
        minimum_cut_quality=0.55,
    )
    assert report["kept"] == []
    assert report["summary"]["excluded_no_character"] == 2
    assert {item["exclusion_reason"] for item in report["excluded"]} == {"no_character"}
    assert sorted(len(group["members"]) for group in report["groups"]) == [1, 1]
    assert all(group["pair_evidence"] == [] for group in report["groups"])


def test_opt_in_private_registered_and_expression_pairs_keep_the_expected_outcomes() -> None:
    pair_pts = (
        (14409090, 14412690),
        (100524690, 100859490),
        (8868690, 9001890),
        (185490, 228690),
        (11489490, 11496690),
    )
    rows = _private_curation_rows({pts for pair in pair_pts for pts in pair})

    def pair_report(pair: tuple[int, int]) -> tuple[dict, dict, dict]:
        records = [_record_from_private_report(rows[pts]) for pts in pair]
        report = curate_candidates(records, minimum_cut_quality=0.0)
        return report, records[0], records[1]

    near, left, right = pair_report(pair_pts[0])
    near_group = _group_for(near, left["id"])
    assert set(near_group["members"]) == {left["id"], right["id"]}
    assert len(near["kept"]) == 1
    assert len(near["excluded"]) == 1
    assert near["excluded"][0]["exclusion_reason"] == "near_duplicate_lower_rank"
    assert near_group["pair_evidence"][0]["near_duplicate"] is True
    assert near_group["pair_evidence"][0]["decision_reason"] == "registered_structure_agrees"

    closeup, left, right = pair_report(pair_pts[1])
    closeup_group = _group_for(closeup, left["id"])
    assert set(closeup_group["members"]) == {left["id"], right["id"]}
    closeup_evidence = closeup_group["pair_evidence"][0]
    assert closeup_evidence["near_duplicate"] is True
    assert closeup_evidence["decision_reason"] == "registered_structure_agrees"
    assert closeup_evidence["phash_distance"] == 21
    assert closeup_evidence["dhash_distance"] == 24
    assert closeup_evidence["raw_face_change"] is True
    assert closeup_evidence["tag_expression_change"] is False
    assert closeup_evidence["aligned_face_check_performed"] is True
    assert closeup_evidence["all_face_matches_verified"] is True
    assert closeup_evidence["face_matching_count"] >= 1
    assert closeup_evidence["minimum_face_support"] >= 0.80
    assert closeup_evidence["maximum_aligned_face_mae"] <= 0.065

    expression, left, right = pair_report(pair_pts[2])
    expression_group = _group_for(expression, left["id"])
    expression_evidence = next(
        evidence
        for group in expression["groups"]
        for evidence in group["pair_evidence"]
        if {evidence["left_id"], evidence["right_id"]} == {left["id"], right["id"]}
    )
    assert len(expression_group["members"]) == 1
    assert len(expression["kept"]) == 2
    assert expression_evidence["tag_expression_change"] is True
    assert expression_evidence["near_duplicate"] is False
    assert expression_evidence["alignment_checked"] is False

    pose, left, right = pair_report(pair_pts[3])
    assert len(pose["kept"]) == 2
    assert _group_for(pose, left["id"])["group_id"] != _group_for(pose, right["id"])["group_id"]

    distinct_expression, left, right = pair_report(pair_pts[4])
    assert len(distinct_expression["kept"]) == 2
    assert _group_for(distinct_expression, left["id"])["group_id"] != _group_for(
        distinct_expression, right["id"]
    )["group_id"]
