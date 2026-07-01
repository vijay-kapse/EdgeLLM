package com.edgellm.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * On-device causal-LM inference with ONNX Runtime Mobile.
 *
 * Loads a quantized ONNX model (the same INT8/INT4 artifact produced by
 * `edgellm quantize`) and runs autoregressive greedy decoding with a KV cache —
 * the Kotlin counterpart of `cpp/src/main.cpp`. Architecture defaults match
 * Qwen2.5-0.5B; pass overrides for other models.
 *
 * Tokenization/detokenization are handled by [HfTokenizer] (onnxruntime-extensions).
 */
class OnnxLlm(
    modelBytes: ByteArray,
    private val numLayers: Int = 24,
    private val kvHeads: Int = 2,
    private val headDim: Int = 64,
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(Runtime.getRuntime().availableProcessors())
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, opts)
    }

    /** Greedy-decode [maxNewTokens] tokens from [promptIds]; returns generated ids. */
    fun generate(promptIds: LongArray, maxNewTokens: Int): LongArray {
        val generated = ArrayList<Long>(maxNewTokens)
        var past = emptyPast()

        // Prefill on the full prompt.
        var step = runStep(promptIds, pastLen = 0, past)
        past = step.present
        var next = step.nextToken
        generated.add(next)
        var pastLen = promptIds.size.toLong()

        // Decode loop.
        for (i in 1 until maxNewTokens) {
            step = runStep(longArrayOf(next), pastLen, past)
            past = step.present
            next = step.nextToken
            generated.add(next)
            pastLen += 1
        }
        return generated.toLongArray()
    }

    private class Step(val nextToken: Long, val present: List<OnnxTensor>)

    private fun runStep(ids: LongArray, pastLen: Long, past: List<OnnxTensor>): Step {
        val cur = ids.size.toLong()
        val total = pastLen + cur

        val inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), longArrayOf(1, cur))
        val attnMask = OnnxTensor.createTensor(
            env, LongBuffer.wrap(LongArray(total.toInt()) { 1L }), longArrayOf(1, total)
        )
        val posIds = OnnxTensor.createTensor(
            env, LongBuffer.wrap(LongArray(cur.toInt()) { pastLen + it }), longArrayOf(1, cur)
        )

        val inputs = HashMap<String, OnnxTensor>()
        inputs["input_ids"] = inputIds
        inputs["attention_mask"] = attnMask
        inputs["position_ids"] = posIds
        for (l in 0 until numLayers) {
            inputs["past_key_values.$l.key"] = past[l * 2]
            inputs["past_key_values.$l.value"] = past[l * 2 + 1]
        }

        val outputNames = ArrayList<String>().apply {
            add("logits")
            for (l in 0 until numLayers) {
                add("present.$l.key"); add("present.$l.value")
            }
        }

        val result = session.run(inputs, LinkedHashSet(outputNames))
        val logits = result.get("logits").get() as OnnxTensor
        val next = argmaxLast(logits)

        val present = ArrayList<OnnxTensor>(numLayers * 2)
        for (l in 0 until numLayers) {
            present.add(result.get("present.$l.key").get() as OnnxTensor)
            present.add(result.get("present.$l.value").get() as OnnxTensor)
        }

        inputIds.close(); attnMask.close(); posIds.close()
        past.forEach { it.close() }
        return Step(next, present)
    }

    private fun emptyPast(): List<OnnxTensor> =
        List(numLayers * 2) {
            OnnxTensor.createTensor(
                env, FloatBuffer.allocate(0), longArrayOf(1, kvHeads.toLong(), 0, headDim.toLong())
            )
        }

    /** argmax over the final position of a [1, seq, vocab] logits tensor. */
    private fun argmaxLast(logits: OnnxTensor): Long {
        val shape = logits.info.shape       // [1, seq, vocab]
        val seq = shape[1].toInt()
        val vocab = shape[2].toInt()
        val data = logits.floatBuffer
        val base = (seq - 1) * vocab
        var best = 0
        var bestVal = data.get(base)
        for (v in 1 until vocab) {
            val x = data.get(base + v)
            if (x > bestVal) { bestVal = x; best = v }
        }
        return best.toLong()
    }

    fun close() {
        session.close()
    }
}
