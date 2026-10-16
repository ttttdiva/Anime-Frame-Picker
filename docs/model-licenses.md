# Model and dependency licenses

This application downloads model weights at setup time into
`ANIME_PICKER_MODEL_ROOT`. The repository does not redistribute those weights.
Every downloaded model remains subject to its own model card, revision, and
license. The pinned model cards and file hashes are retained in the local
model manifest.

The optional export-time Anime-LaMa weight is downloaded from the upstream
release listed below. Its attribution and integrity hashes are recorded, but
no license is inferred when the release does not state one explicitly.

The current model worker pins the following upstream revisions:

| Component | Upstream reference | License / use note |
| --- | --- | --- |
| CCIP ONNX character identity model | [deepghs/ccip_onnx](https://huggingface.co/deepghs/ccip_onnx/tree/eb2acdd29af1703388d3d0c04221add322bc9110) | OpenRAIL in the pinned model README |
| Anime person detector | [deepghs/anime_person_detection](https://huggingface.co/deepghs/anime_person_detection/tree/e39c744c22432ad01f91dd254fe2b02c8d878b8c) | MIT in the pinned Hugging Face model card; preserve the model card notice |
| Anime head detector | [deepghs/anime_head_detection](https://huggingface.co/deepghs/anime_head_detection/tree/06604feee81983792a57c21081e539c0ae229833) | MIT in the pinned Hugging Face model card; preserve the model card notice |
| PixAI tagger / custom code | [pixai-labs/pixai-tagger-v1.0](https://huggingface.co/pixai-labs/pixai-tagger-v1.0/tree/9fe10addf9326e292da8a85a98ea74cd91b41771) | Apache-2.0 in the pinned worker manifest; keep the pinned revision and review the card |
| DeepGHS `imgutils` | [imgutils repository](https://github.com/deepghs/imgutils) | MIT package license; this does not change the licenses of its downloaded model weights |
| Anime-LaMa logo inpainting weights (`anime-manga-big-lama.pt`) | [Sanster/models release artifact](https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt) | Attribution: Sanster/models. The release artifact does not state an explicit license in the metadata currently recorded here; license **unknown**. Review the upstream terms before redistribution. Pinned size: 205,717,870 bytes; MD5 `29f284f36a0a510bcacf39ecf4c4d54f`; SHA-256 `479d3afdcb7ed2fd944ed4ebcc39ca45b33491f0f2e43eb1000bd623cfb41823` |

The model names above are references used by the worker. If a future worker
revision changes a model ID, update this document with the exact model-card URL
and commit/revision before releasing that change.

The detector cards identify the downloaded detector artifacts as MIT. Their
upstream implementation provenance is recorded separately by the worker:
the person detector traces to [Ultralytics 8.0.20](https://github.com/ultralytics/ultralytics/tree/v8.0.20),
whose repository is GPL-3.0, and the head detector traces to
[Ultralytics 8.0.196](https://github.com/ultralytics/ultralytics/tree/v8.0.196),
whose repository is AGPL-3.0. These provenance links preserve the source trail;
they do not determine the legal status of a particular redistribution.

The main runtime dependencies have their own upstream licenses:

- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) and
  [TorchVision](https://github.com/pytorch/vision/blob/main/LICENSE) use BSD
  style licenses.
- [Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
  and [timm](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE)
  are Apache-2.0 projects.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)
  is MIT licensed. The CUDA runtime libraries bundled by its wheels remain
  subject to their respective NVIDIA terms.
- [FastAPI](https://github.com/fastapi/fastapi/blob/master/LICENSE),
  [PyAV](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt),
  [Pillow](https://github.com/python-pillow/Pillow/blob/main/LICENSE), and
  [OpenCV](https://github.com/opencv/opencv/blob/4.x/LICENSE) retain their
  upstream licenses.

This page records upstream model and dependency metadata. No license has been
assigned to this private repository's own source.
