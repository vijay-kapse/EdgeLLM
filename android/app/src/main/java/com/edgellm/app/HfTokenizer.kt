package com.edgellm.app

/**
 * On-device tokenizer boundary.
 *
 * SCAFFOLD: two supported ways to wire this up on-device — pick one when building:
 *
 *  1. **onnxruntime-extensions** (recommended): bundle the model's `tokenizer.json`
 *     in `assets/` and build a tiny tokenizer ONNX graph with the extensions
 *     `HfTokenizer` op, then run encode/decode through an [ai.onnxruntime.OrtSession].
 *     See https://github.com/microsoft/onnxruntime-extensions.
 *
 *  2. **Host pre/post-tokenization**: use the desktop CLI (`edgellm encode` /
 *     `edgellm decode`) to convert text <-> token ids, and have the app operate on
 *     ids directly. Useful for a first end-to-end bring-up.
 *
 * The two methods below are intentionally left as `TODO`s so the model inference in
 * [OnnxLlm] can be reviewed independently of the tokenizer choice.
 */
interface HfTokenizer {
    fun encode(text: String): LongArray
    fun decode(ids: LongArray): String
}

/**
 * Minimal id-passthrough tokenizer for first bring-up: the "text" is a
 * space-separated list of token ids produced by `edgellm encode` on the host, and
 * [decode] returns the ids as text to paste into `edgellm decode`. Replace with a
 * real on-device tokenizer (option 1 above) for a self-contained app.
 */
class IdPassthroughTokenizer : HfTokenizer {
    override fun encode(text: String): LongArray =
        text.trim().split(Regex("[\\s,]+")).filter { it.isNotEmpty() }.map { it.toLong() }.toLongArray()

    override fun decode(ids: LongArray): String = ids.joinToString(" ")
}
