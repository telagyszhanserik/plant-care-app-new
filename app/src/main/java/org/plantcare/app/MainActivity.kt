package org.plantcare.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>
    private lateinit var careGuide: JSONObject
    private lateinit var imageView: ImageView
    private lateinit var tvResult: TextView

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let { classifyImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        tvResult = findViewById(R.id.tvResult)
        val btnSelect: Button = findViewById(R.id.btnSelect)

        labels = assets.open("labels.txt").bufferedReader().readLines()
        careGuide = JSONObject(assets.open("care_guide.json").bufferedReader().readText())
        interpreter = Interpreter(loadModelFile())

        btnSelect.setOnClickListener { pickImage.launch("image/*") }
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fd = assets.openFd("model.tflite")
        val inputStream = FileInputStream(fd.fileDescriptor)
        return inputStream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    private fun classifyImage(uri: Uri) {
        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
        imageView.setImageBitmap(bitmap)

        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val px = resized.getPixel(x, y)
                inputBuffer.putFloat((((px shr 16) and 0xFF) / 127.5f) - 1f)
                inputBuffer.putFloat((((px shr 8) and 0xFF) / 127.5f) - 1f)
                inputBuffer.putFloat(((px and 0xFF) / 127.5f) - 1f)
            }
        }

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(inputBuffer, output)

        val probs = output[0]
        var maxIdx = 0
        for (i in probs.indices) if (probs[i] > probs[maxIdx]) maxIdx = i
        val confidence = probs[maxIdx] * 100
        val classKey = labels[maxIdx]

        val info = careGuide.optJSONObject(classKey)
        val name = info?.optString("name") ?: classKey
        val watering = info?.optString("watering") ?: "-"
        val care = info?.optString("care") ?: "-"

        tvResult.text = "Результат: %s (%.1f%%)\n\nПолив: %s\nУход: %s".format(name, confidence, watering, care)
    }
}
