# App assets

Place the quantized ONNX model here before building:

```bash
# from the repo root, after `edgellm quantize`
cp artifacts/onnx/Qwen__Qwen2.5-0.5B-Instruct-int8-dynamic/model.onnx \
   android/app/src/main/assets/model.onnx
```

The INT8/INT4 artifacts are single-file `model.onnx` (no external data), so they load
directly from assets. `.onnx` is excluded from APK compression (see `app/build.gradle.kts`).

Optional (for on-device tokenization via onnxruntime-extensions):
copy the model's `tokenizer.json` here too.
