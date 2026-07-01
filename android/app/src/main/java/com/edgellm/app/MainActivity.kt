package com.edgellm.app

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.ByteArrayOutputStream
import kotlin.concurrent.thread

/**
 * Minimal on-device demo: load the quantized ONNX model from assets and generate
 * text with [OnnxLlm]. This is a scaffold — build and run it from Android Studio
 * with a device/emulator after placing `model.onnx` in `app/src/main/assets/`.
 */
class MainActivity : AppCompatActivity() {

    private var llm: OnnxLlm? = null
    private val tokenizer: HfTokenizer = IdPassthroughTokenizer()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val input = findViewById<EditText>(R.id.promptInput)
        val output = findViewById<TextView>(R.id.outputText)
        val button = findViewById<Button>(R.id.generateButton)

        button.setOnClickListener {
            val prompt = input.text.toString()
            output.text = getString(R.string.generating)
            thread {
                val result = runCatching { runGenerate(prompt) }
                runOnUiThread {
                    output.text = result.getOrElse { "error: ${it.message}" }
                }
            }
        }
    }

    private fun runGenerate(prompt: String): String {
        val model = llm ?: OnnxLlm(readAsset("model.onnx")).also { llm = it }
        val ids = tokenizer.encode(prompt)
        val started = System.nanoTime()
        val generated = model.generate(ids, maxNewTokens = 48)
        val seconds = (System.nanoTime() - started) / 1e9
        val tokPerSec = generated.size / seconds
        return "${tokenizer.decode(generated)}\n\n%.1f tok/s on-device".format(tokPerSec)
    }

    private fun readAsset(name: String): ByteArray {
        assets.open(name).use { input ->
            val buffer = ByteArrayOutputStream()
            input.copyTo(buffer)
            return buffer.toByteArray()
        }
    }

    override fun onDestroy() {
        llm?.close()
        super.onDestroy()
    }
}
