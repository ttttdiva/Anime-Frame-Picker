"""Contract tests for local frame scoring helpers.

These tests stay independent of a decoder or model installation.  The pipeline
tests elsewhere can inject those services; this file protects the deterministic
math and pixel-only deduplication contract that they share.
"""

import numpy as np
from PIL import Image, ImageDraw

from anime_picker import pipeline
from anime_picker.scoring import (
    aggregate_reference_distances,
    are_pixel_duplicates,
    combine_score,
    compact_tag_evidence,
    deduplicate_candidates,
    dhash,
    evaluate_tags,
    hamming_distance,
    identity_match,
    phash,
    quality_metrics,
    subject_tag_ambiguity,
)


def _frame(expression: str = "neutral") -> Image.Image:
    image = Image.new("RGB", (128, 96), (38, 42, 52))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 14, 104, 82), fill=(220, 190, 150), outline=(240, 240, 240), width=2)
    draw.ellipse((43, 34, 53, 44), fill=(24, 30, 40))
    draw.ellipse((75, 34, 85, 44), fill=(24, 30, 40))
    if expression == "smile":
        draw.arc((48, 42, 80, 70), 0, 180, fill=(24, 30, 40), width=4)
    else:
        draw.line((50, 58, 78, 58), fill=(24, 30, 40), width=4)
    return image


def test_identity_uses_official_threshold_and_minimum_reference_distance():
    close = aggregate_reference_distances([0.09, 0.31, 0.28], threshold=0.20)
    assert close["accepted"] is True
    assert close["distance"] == 0.09
    assert 0.0 < close["identity"] <= 1.0
    assert identity_match(0.20, 0.20)
    assert not identity_match(0.201, 0.20)


def test_unlimited_candidate_output_keeps_every_candidate():
    candidates = [{"id": f"candidate-{index}"} for index in range(1501)]
    assert len(pipeline._candidate_output_pool(candidates, 0)) == len(candidates)
    assert [item["id"] for item in pipeline._candidate_output_pool(candidates, 4)] == [
        "candidate-0", "candidate-1", "candidate-2", "candidate-3"
    ]


def test_combined_score_normalizes_weights_and_stays_display_bounded():
    assert combine_score(1.0, 0.0, 0.0, {"identity": 2, "condition": 0, "quality": 0}) == 1.0
    value = combine_score(0.8, 0.6, 0.4, {"identity": 1, "condition": 1, "quality": 2})
    assert 0.0 <= value <= 1.0


def test_quality_metrics_are_target_region_metrics_and_flag_large_temporal_jump():
    current = _frame()
    dark_transition = Image.new("RGB", current.size, (250, 250, 250))
    metrics = quality_metrics(current, [0.18, 0.10, 0.85, 0.90], previous=dark_transition, following=dark_transition)
    assert 0.0 <= metrics["quality"] <= 1.0
    assert 0.0 <= metrics["condition"] <= 1.0
    assert metrics["temporal_delta"] > 0.0


def test_quality_metrics_ignore_sparse_shot_neighbors_but_compare_near_frames():
    current = _frame()
    different_shot = Image.new("RGB", current.size, (250, 250, 250))
    settings = {"temporal_require_times": True, "temporal_max_gap": 0.12}
    sparse = quality_metrics(
        current,
        [0.18, 0.10, 0.85, 0.90],
        previous=different_shot,
        following=different_shot,
        settings=settings,
        current_time=2.0,
        previous_time=0.0,
        following_time=4.0,
    )
    close = quality_metrics(
        current,
        [0.18, 0.10, 0.85, 0.90],
        previous=different_shot,
        following=different_shot,
        settings=settings,
        current_time=2.0,
        previous_time=1.96,
        following_time=2.04,
    )
    assert sparse["temporal_delta"] == 0.0
    assert close["temporal_delta"] > 0.0


def test_subject_and_frame_tags_do_not_cross_satisfy_each_other():
    wanted = [{"tag": "smile", "scope": "subject"}]
    subject = {"character": {"smile": 0.88}}
    frame_only = {"general": {"smile": 0.91}}
    result = evaluate_tags({}, frame_only, wanted=wanted, threshold=0.17)
    assert result["passed"] is False
    result = evaluate_tags(subject, {}, wanted=wanted, threshold=0.17)
    assert result["passed"] is True


def test_compact_tag_evidence_is_bounded_and_keeps_explicit_low_confidence_tags():
    tags = {
        "general": {f"tag-{index}": 0.9 - index * 0.001 for index in range(100)},
        "character": {f"character-{index}": 0.8 - index * 0.01 for index in range(10)},
    }
    tags["general"]["requested-low"] = 0.01
    compact = compact_tag_evidence(tags, [{"tag": "requested-low", "scope": "subject"}], threshold=0.17)
    assert len(compact["general"]) <= 21
    assert compact["general"]["requested-low"] == 0.01
    assert len(compact["character"]) <= 3


def test_subject_multi_person_general_signal_is_conservative_and_exact():
    assert subject_tag_ambiguity({"general": {"multiple_girls": 0.81}}) == {"multiple_girls": 0.81}
    assert subject_tag_ambiguity({"general": {"multiple_girls": 0.79}}) == {}
    assert subject_tag_ambiguity({"character": {"multiple_girls": 0.99}}) == {}


def test_aligned_target_hashes_handle_box_jitter_without_erasing_expression_change():
    neutral = _frame("neutral")
    smile = _frame("smile")
    stable = {
        "time": 1.0,
        "score": 0.9,
        "bbox": [0.2, 0.1, 0.8, 0.9],
        "image": neutral,
        "phash": phash(neutral),
        "dhash": dhash(neutral),
        "crop_phash": "0",
        "crop_dhash": "0",
    }
    jittered = dict(stable)
    jittered.update({"time": 1.1, "bbox": [0.24, 0.08, 0.78, 0.92], "crop_phash": "f" * 16, "crop_dhash": "f" * 16})
    changed = dict(jittered)
    changed.update({"image": smile, "phash": phash(smile), "dhash": dhash(smile)})
    assert are_pixel_duplicates(stable, jittered)
    assert not are_pixel_duplicates(stable, changed)


def test_same_pts_distinct_people_are_tagged_before_one_frame_export_collapse(tmp_path, monkeypatch):
    class Frame:
        def __init__(self, image):
            self.pts = 42
            self.time = 0.0
            self.stream_index = 0
            self.image = image

    class Info:
        duration = 0.1
        fps = 25.0

    class Reader:
        def __init__(self, _path):
            self.info = Info()
            self.fingerprint = "same-pts-test"

        def sample(self, *_args, **_kwargs):
            image = Image.new("RGB", (64, 48), (0, 0, 20))
            pixels = np.asarray(image).copy()
            pixels[:, 2:30] = (220, 30, 30)  # subject that satisfies smile
            pixels[:, 34:62] = (30, 30, 220)  # subject that fails smile
            return [Frame(Image.fromarray(pixels))]

    class Models:
        threshold = 0.5
        revision = "same-pts-model"
        identity_model = "ccip-test"

        def detect(self, _image):
            return [
                {"box": [2, 4, 30, 44], "score": 1.0, "kind": "person"},
                {"box": [34, 4, 62, 44], "score": 1.0, "kind": "person"},
            ]

        def embed(self, images):
            return np.ones((len(images), 4), dtype=np.float32)

        def distance(self, references, features):
            return np.full((len(references), len(features)), 0.1, dtype=np.float32)

        def tag(self, image):
            array = np.asarray(image.convert("RGB"))
            smile = 0.9 if float(array[..., 0].mean()) > float(array[..., 2].mean()) else 0.0
            return {"character": {"smile": smile}}

    monkeypatch.setattr(pipeline, "VideoReader", Reader)
    result = pipeline.run_search(
        "same-pts",
        [Image.new("RGB", (16, 16), (200, 40, 40))],
        [{"tag": "smile", "scope": "subject"}],
        [],
        {"candidate_count": 2, "coarse_step": 1.0, "dense_step": 0.25, "near_step": 0.04, "rescue": False, "quality_min": 0.0},
        Models(),
        tmp_path,
    )
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["bbox"][0] < 0.5
    assert result["candidates"][0]["tags"]["subject"]["character"]["smile"] == 0.9
    assert result["stats"]["same_pts_collapsed"] == 0


def test_subject_multi_person_signal_rejects_crop_but_frame_scope_is_unaffected(tmp_path, monkeypatch):
    class Frame:
        def __init__(self, image):
            self.pts = 42
            self.time = 0.0
            self.stream_index = 0
            self.image = image

    class Info:
        duration = 0.1
        fps = 25.0

    class Reader:
        def __init__(self, _path):
            self.info = Info()
            self.fingerprint = "ambiguity-test"

        def sample(self, *_args, **_kwargs):
            image = Image.new("RGB", (64, 48), (0, 0, 20))
            pixels = np.asarray(image).copy()
            pixels[:, 2:30] = (220, 30, 30)
            pixels[:, 34:62] = (30, 30, 220)
            return [Frame(Image.fromarray(pixels))]

    class Models:
        threshold = 0.5
        revision = "ambiguity-model"
        identity_model = "ccip-test"

        def detect(self, _image):
            return [
                {"box": [2, 4, 30, 44], "score": 1.0, "kind": "person"},
                {"box": [34, 4, 62, 44], "score": 1.0, "kind": "person"},
            ]

        def embed(self, images):
            return np.ones((len(images), 4), dtype=np.float32)

        def distance(self, references, features):
            return np.full((len(references), len(features)), 0.1, dtype=np.float32)

        def tag(self, image):
            if image.width > 40:
                return {"general": {"smile": 0.9, "multiple_girls": 0.9}}
            array = np.asarray(image.convert("RGB"))
            smile = 0.9 if float(array[..., 0].mean()) > float(array[..., 2].mean()) else 0.0
            return {"general": {"multiple_girls": 0.9}, "character": {"smile": smile}}

    monkeypatch.setattr(pipeline, "VideoReader", Reader)
    settings = {"candidate_count": 2, "coarse_step": 1.0, "dense_step": 0.25, "near_step": 0.04, "rescue": False, "quality_min": 0.0}
    reference = Image.new("RGB", (16, 16), (200, 40, 40))
    subject_result = pipeline.run_search(
        "same-pts",
        [reference],
        [{"tag": "smile", "scope": "subject"}],
        [],
        settings,
        Models(),
        tmp_path / "subject",
    )
    frame_result = pipeline.run_search(
        "same-pts",
        [reference],
        [{"tag": "smile", "scope": "frame"}],
        [],
        settings,
        Models(),
        tmp_path / "frame",
    )
    assert not subject_result["candidates"]
    assert subject_result["stats"]["subject_ambiguity"] >= 1
    assert len(frame_result["candidates"]) == 1
    assert frame_result["stats"]["subject_ambiguity"] == 0


def test_subject_scope_retains_nested_body_crop_when_head_has_higher_identity_score(tmp_path, monkeypatch):
    class Frame:
        def __init__(self, image):
            self.pts = 42
            self.time = 0.0
            self.stream_index = 0
            self.image = image

    class Info:
        duration = 0.1
        fps = 25.0

    class Reader:
        def __init__(self, _path):
            self.info = Info()
            self.fingerprint = "nested-context-test"

        def sample(self, *_args, **_kwargs):
            return [Frame(Image.new("RGB", (64, 48), (80, 110, 140)))]

    class Models:
        threshold = 0.5
        revision = "nested-context-model"
        identity_model = "ccip-test"

        def detect(self, _image):
            return [
                {"box": [8, 3, 56, 45], "score": 1.0, "kind": "person"},
                {"box": [22, 8, 42, 25], "score": 1.0, "kind": "head"},
            ]

        def embed(self, images):
            # The narrow head gets a higher official identity score, so a
            # pre-tag collapse would incorrectly discard the body crop.
            return np.asarray([[0.02 if image.width < 30 else 0.20] for image in images], dtype=np.float32)

        def distance(self, references, features):
            return np.abs(np.asarray(references) - np.asarray(features).T)

        def tag(self, image):
            return {"character": {"full_body": 0.9 if image.width > 30 else 0.0}}

    monkeypatch.setattr(pipeline, "VideoReader", Reader)
    result = pipeline.run_search(
        "nested-context",
        [Image.new("RGB", (16, 16), (80, 110, 140))],
        [{"tag": "full_body", "scope": "subject"}],
        [],
        {"candidate_count": 2, "coarse_step": 1.0, "dense_step": 0.25, "near_step": 0.04, "rescue": False, "quality_min": 0.0},
        Models(),
        tmp_path,
    )
    assert len(result["candidates"]) == 1
    # The retained bbox is the broad body/context crop, not the higher-identity
    # head crop that cannot satisfy the requested full-body condition.
    assert result["candidates"][0]["bbox"][2] - result["candidates"][0]["bbox"][0] > 0.5


def test_condition_pretag_pool_keeps_pixel_different_expression_candidates():
    def fixture(mouth_points, mouth_width):
        image = Image.new("RGB", (128, 96), (38, 42, 52))
        draw = ImageDraw.Draw(image)
        draw.rectangle((24, 14, 104, 82), fill=(220, 190, 150), outline=(240, 240, 240), width=2)
        draw.ellipse((43, 34, 53, 44), fill=(24, 30, 40))
        draw.ellipse((75, 34, 85, 44), fill=(24, 30, 40))
        draw.line(mouth_points, fill=(24, 30, 40), width=mouth_width)
        return image

    neutral = fixture([(50, 58), (78, 58)], 4)
    smile = fixture([(52, 59), (76, 59)], 3)
    neutral_crop = neutral.crop((24, 14, 104, 82))
    smile_crop = smile.crop((24, 14, 104, 82))
    common = {
        "bbox": [0.2, 0.1, 0.8, 0.9],
        "_detection_kind": "person",
        "score": 0.8,
        "distance": 0.1,
        "identity": 0.8,
        "quality": 0.8,
        "condition": 0.8,
    }
    first = {**common, "id": "neutral", "pts": 1, "time": 1.0, "_frame": neutral, "_crop": neutral_crop}
    second = {**common, "id": "smile", "pts": 2, "time": 1.1, "score": 0.7, "_frame": smile, "_crop": smile_crop}
    for item in (first, second):
        item["hashes"] = {
            "phash": phash(item["_frame"]),
            "dhash": dhash(item["_frame"]),
            "crop_phash": phash(item["_crop"]),
            "crop_dhash": dhash(item["_crop"]),
        }
    approximate = pipeline._pixel_diverse_tag_pool([first, second], hash_distance=6, time_window=2.0)
    exact_only = pipeline._pixel_diverse_tag_pool([first, second], hash_distance=6, time_window=2.0, exact_only=True)
    assert len(approximate) == 1
    assert {item["id"] for item in exact_only} == {"neutral", "smile"}
    assert evaluate_tags({"character": {"smile": 0.9}}, {}, [{"tag": "smile", "scope": "subject"}], [], 0.17)["passed"] is True


def test_pixel_hashes_drive_dedup_and_keep_expression_change():
    neutral = _frame("neutral")
    smile = _frame("smile")
    same_a = {"id": "a", "time": 1.0, "score": 0.90, "phash": phash(neutral), "dhash": dhash(neutral), "crop_phash": phash(neutral), "crop_dhash": dhash(neutral)}
    same_b = {"id": "b", "time": 1.1, "score": 0.80, "phash": phash(neutral), "dhash": dhash(neutral), "crop_phash": phash(neutral), "crop_dhash": dhash(neutral)}
    changed = {"id": "c", "time": 1.2, "score": 0.85, "phash": phash(neutral), "dhash": dhash(neutral), "crop_phash": phash(smile), "crop_dhash": dhash(smile)}
    assert hamming_distance(same_a["phash"], same_b["phash"]) == 0
    assert are_pixel_duplicates(same_a, same_b)
    assert not are_pixel_duplicates(same_a, changed)
    result = deduplicate_candidates([same_a, same_b, changed])
    assert [candidate["id"] for candidate in result] == ["a", "c"]
