interface Activation {
    fun invoke(value: Float): Float
}

object Identity : Activation {
    override operator fun invoke(value: Float) = value
}