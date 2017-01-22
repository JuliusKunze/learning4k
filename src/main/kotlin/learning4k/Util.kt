package learning4k

import java.util.*

fun log(float: Float) = Math.log(float.toDouble()).toFloat()

private val staticRandom = Random()

fun <T> List<T>.shuffle(random: Random = staticRandom): List<T> {
    val a = ArrayList(this)
    var n = a.size
    while (n > 1) {
        val k = random.nextInt(n--)
        val t = a[n]
        a[n] = a[k]
        a[k] = t
    }
    return a
}