/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.audio

import android.content.Context
import android.media.AudioRecord
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.core.os.postDelayed
import org.tensorflow.lite.DataType
import org.tensorflow.lite.examples.audio.fragments.AudioClassificationListener
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.support.audio.TensorAudio.TensorAudioFormat
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.tensorflow.lite.task.core.BaseOptions
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class AudioClassificationHelper(
    val context: Context,
    val listener: AudioClassificationListener,
    var currentModel: String = YAMNET_MODEL,
    var classificationThreshold: Float = DISPLAY_THRESHOLD,
    var overlap: Float = DEFAULT_OVERLAP_VALUE,
    var numOfResults: Int = DEFAULT_NUM_OF_RESULTS,
    var currentDelegate: Int = 0,
    var numThreads: Int = 2
) {
    private lateinit var classifier: AudioClassifier
    private lateinit var tensorAudio: TensorAudio
    private lateinit var recorder: AudioRecord
    private lateinit var executor: ScheduledThreadPoolExecutor

    @Volatile
    private var isRecording = AtomicBoolean(false)

    private val _bufferSize by lazy { getBufferSize() }

    private val classifyRunnable = Runnable {
        classifyAudio()
    }

    init {
        initClassifier()
    }

    fun initClassifier() {
        // Set general detection options, e.g. number of used threads
        val baseOptionsBuilder = BaseOptions.builder()
            .setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU.
        // Possible to also use a GPU delegate, but this requires that the classifier be created
        // on the same thread that is using the classifier, which is outside of the scope of this
        // sample's design.
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        // Configures a set of parameters for the classifier and what results will be returned.
        val options = AudioClassifier.AudioClassifierOptions.builder()
            .setScoreThreshold(classificationThreshold)
            .setMaxResults(numOfResults)
            .setBaseOptions(baseOptionsBuilder.build())
            .build()

        try {
            // Create the classifier and required supporting objects
            classifier = AudioClassifier.createFromFileAndOptions(context, currentModel, options)
            tensorAudio = classifier.createInputTensorAudio()
            recorder = classifier.createAudioRecord()
            startAudioClassification()
        } catch (e: IllegalStateException) {
            listener.onError(
                "Audio Classifier failed to initialize. See error logs for details"
            )

            Log.e("AudioClassification", "TFLite failed to load with error: " + e.message)
        }
    }

    fun startAudioClassification() {
        if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
            return
        }

        recorder.startRecording()
        executor = ScheduledThreadPoolExecutor(1)

        // Each model will expect a specific audio recording length. This formula calculates that
        // length using the input buffer size and tensor format sample rate.
        // For example, YAMNET expects 0.975 second length recordings.
        // This needs to be in milliseconds to avoid the required Long value dropping decimals.
        val lengthInMilliSeconds = ((classifier.requiredInputBufferSize * 1.0f) /
                classifier.requiredTensorAudioFormat.sampleRate) * 1000

        val interval = (lengthInMilliSeconds * (1 - overlap)).toLong()

        executor.scheduleAtFixedRate(
            classifyRunnable,
            0,
            interval,
            TimeUnit.MILLISECONDS
        )
    }

    private fun classifyAudio() {
        tensorAudio.load(recorder)
        var inferenceTime = SystemClock.uptimeMillis()

        //
        val buffer = tensorAudio.tensorBuffer.buffer.array()

        val output = classifier.classify(tensorAudio)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        val results = output[0].categories
        listener.onResult(results, inferenceTime)

        results.forEach {
//            it.index == 38 &&
            if (it.score >= 0.75f && !isRecording.get()) {

                isRecording.set(true)
                stopHandler.postDelayed(10 * 1000) {
                    stopWrite()
                }
            }
        }

        if (isRecording.get()) {
            writeToFile(buffer)
        }
    }

//    private var audioFile = File(context.filesDir, "a.pcm")

    private fun createAudioFile() = File(context.filesDir, System.currentTimeMillis().toString() + ".pcm")

    private var fos: FileOutputStream? = null

    private fun writeToFile(buffer: ByteArray) {
        if (fos == null) {
            val audioFile = createAudioFile()
            fos = FileOutputStream(audioFile)
            Log.d("simple", "audio  = ${audioFile.absolutePath}")
        }
        fos?.write(buffer)
        Log.d("simple", "buffer = ${buffer.size}")
    }

    private fun stopWrite() {
        isRecording.set(false)
        fos?.close()
        fos = null
        Log.d("simple", "stopWrite")
    }

    private fun getBufferSize(): Int {
        val format: TensorAudioFormat = classifier.requiredTensorAudioFormat
        val channelConfig = when (format.channels) {
            1 -> 16
            else -> 12
        }
        var bufferSizeInBytes = AudioRecord.getMinBufferSize(format.sampleRate, channelConfig.toInt(), 4)
        if (bufferSizeInBytes != -1 && bufferSizeInBytes != -2) {
            val bufferSizeMultiplier = 2
            val modelRequiredBufferSize =
                classifier.requiredInputBufferSize.toInt() * DataType.FLOAT32.byteSize() * bufferSizeMultiplier
            if (bufferSizeInBytes < modelRequiredBufferSize) {
                bufferSizeInBytes = modelRequiredBufferSize
            }
        }
        return bufferSizeInBytes
    }

    private fun readBuffer(bufferSize: Int) {
        while (isRecording.get()) {
            val data = ByteArray(bufferSize)
            val ret = recorder.read(data, 0, bufferSize)
            if (ret < 0) {
                Log.d("simple", "read error = $ret")
                return
            }
            Log.d("simple", "read success = $ret")
        }
    }

    private val stopHandler by lazy { Handler(Looper.getMainLooper()) }

    fun stopAudioClassification() {
        recorder.stop()
        executor.shutdownNow()
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_NNAPI = 1
        const val DISPLAY_THRESHOLD = 0.3f
        const val DEFAULT_NUM_OF_RESULTS = 2
        const val DEFAULT_OVERLAP_VALUE = 0.5f
        const val YAMNET_MODEL = "yamnet.tflite"
        const val SPEECH_COMMAND_MODEL = "speech.tflite"
    }
}
