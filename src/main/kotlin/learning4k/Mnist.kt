package learning4k

import java.io.*
import java.net.URL
import java.nio.ByteBuffer
import java.nio.channels.Channels
import java.util.*
import java.util.zip.GZIPInputStream

/**
 * Mnist data sets are taken from + described here: http://yann.lecun.com/exdb/mnist/
 */
object Mnist {
    val data by lazy {
        learning4k.LabeledExamples(training = training.map { it.toLabeledData() }, test = test.map { it.toLabeledData() })
    }

    val test by lazy {
        (learning4k.MnistProvider(
                labelFileUrl = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                imageFileUrl = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
        ))()
    }

    val training by lazy {
        MnistProvider(
                labelFileUrl = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                imageFileUrl = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        )()
    }
}

data class LabeledMnistImage(val imageData: List<Byte>, val label: Int) {
    fun toLabeledData() = LabeledExample(input = imageData.map { it.toFloat() / 255 }, output = oneHot(label, size = 10))
}

/**
 * Based on http://stackoverflow.com/a/8301949/1692437
 */
private class MnistProvider(val labelFileUrl: String, val imageFileUrl: String, val allowReuse: Boolean = true) {
    private fun File.withoutGzipExtension() = File(parent, name.replace(".gz", ""))
    private fun URL.extractFileName() = path.split('/').last()
    private fun URL.downloadAndUnGzip(gzipFile: File = File(extractFileName())): File {
        val file = gzipFile.withoutGzipExtension()

        return if (allowReuse && file.exists()) file else downloadTo(file).unGzip(outputFile = file)
    }

    private fun URL.downloadTo(file: File) = file.apply {
        FileOutputStream(this).channel.transferFrom(Channels.newChannel(openStream()), 0, Long.MAX_VALUE)
    }

    private fun File.unGzip(deleteZipOnSuccess: Boolean = true, outputFile: File): File {
        val inputStream = GZIPInputStream(FileInputStream(this))
        try {
            val outputStream = FileOutputStream(outputFile)
            try {
                val buffer = ByteArray(100000)
                while (true) {
                    val len = inputStream.read(buffer)
                    if (len <= 0) break
                    outputStream.write(buffer, 0, len)
                }
            } finally {
                outputStream.close()
            }
            if (deleteZipOnSuccess) {
                delete()
            }
            return outputFile
        } finally {
            inputStream.close()
        }
    }

    operator fun invoke(): List<LabeledMnistImage> {
        val labelInputStream = FileInputStream(URL(labelFileUrl).downloadAndUnGzip())
        val imageInputStream = FileInputStream(URL(imageFileUrl).downloadAndUnGzip())

        val buffer = ByteArray(16384)

        val labelBuffer = ByteArrayOutputStream()
        while (true) {
            val read = labelInputStream.read(buffer, 0, buffer.size)
            if (read == -1) break
            labelBuffer.write(buffer, 0, read)
        }

        labelBuffer.flush()

        val imageBuffer = ByteArrayOutputStream()
        while (true) {
            val read = imageInputStream.read(buffer, 0, buffer.size)
            if (read == -1) break
            imageBuffer.write(buffer, 0, read)
        }

        imageBuffer.flush()

        val labelBytes = labelBuffer.toByteArray()
        val imageBytes = imageBuffer.toByteArray()

        if (ByteBuffer.wrap(Arrays.copyOfRange(labelBytes, 0, offsetSizeInBytes)).int != labelMagic) {
            throw IOException("Bad magic number in label file!")
        }

        if (ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, 0, offsetSizeInBytes)).int != imageMagic) {
            throw IOException("Bad magic number in image file!")
        }

        val labelCount = ByteBuffer.wrap(Arrays.copyOfRange(labelBytes, itemsCountOffset, itemsCountOffset + itemSize)).int
        val imageCount = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, itemsCountOffset, itemsCountOffset + itemSize)).int

        if (imageCount != labelCount) {
            throw IOException("The number of labels and images do not match!")
        }

        val foundRowCount = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, rowCountOffset, rowCountOffset + rowSize)).int
        val foundColumnCount = ByteBuffer.wrap(Arrays.copyOfRange(imageBytes, columnCountOffset, columnCountOffset + columnSize)).int

        if (foundRowCount != rowCount && foundColumnCount != columnCount) {
            throw IOException("Bad image. Rows and columns do not equal ${rowCount}x$columnCount")
        }

        return (0..labelCount - 1).map { i ->
            LabeledMnistImage(
                    imageData = Arrays.copyOfRange(imageBytes, i * imageSize + imageOffset, i * imageSize + imageOffset + imageSize).toList(),
                    label = labelBytes[offsetSizeInBytes + itemSize + i].toInt())
        }
    }

    companion object {
        /** the following constants are defined as per the values described at http://yann.lecun.com/exdb/mnist/  */
        private val offsetSizeInBytes = 4

        private val labelMagic = 2049
        private val imageMagic = 2051

        private val itemsCountOffset = 4
        private val itemSize = 4

        private val rowCountOffset = 8
        private val rowSize = 4
        val rowCount = 28

        private val columnCountOffset = 12
        private val columnSize = 4
        val columnCount = 28

        private val imageOffset = 16
        private val imageSize = rowCount * columnCount
    }
}