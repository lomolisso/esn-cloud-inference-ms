import base64
import tempfile
import gzip

class TFModelManager:
    """
    Handles the loading and inference of a TensorFlow model.
    Developers need to implement these methods according to their model and application.
    """
    _model = None
    _tf = None
    _np = None

    def update_model(self, tf_model_b64, tf_model_bytesize):
        """
        Update the model with a new model.
        """
        # Import modules if not already imported
        if self._tf is None:
            import tensorflow
            self._tf = tensorflow
        if self._np is None:
            import numpy 
            self._np = numpy

        # Decode b64 encoded model into bytes
        _decoded_model = base64.b64decode(tf_model_b64)

        # Decompress the gzip model
        _decoded_model = gzip.decompress(_decoded_model)

        # Check if the model size matches the expected size
        if len(_decoded_model) != tf_model_bytesize:
            raise ValueError(
                f"Model size mismatch: expected {tf_model_bytesize} bytes, got {len(_decoded_model)} bytes"
            )

        # Create a temporary file .keras and write the model to it.
        with tempfile.NamedTemporaryFile(suffix=".keras") as temp:
            temp.write(_decoded_model)
            temp.flush()

            # Load the model from the temporary file
            self._model = self._tf.keras.models.load_model(temp.name)


    def predict(self, input_data):
        """
        Performs inference using the current model loaded in the manager.
        Calls the _preprocess_input method on the input data, then runs the model,
        and finally calls the _postprocess_output method on the output data.
        """

        # Check if the model is loaded
        if self._model is None:
            raise ValueError("Model is not loaded")
        
        # Preprocess the input
        input_data = self._preprocess_input(input_data)

        # Run the model and get the output
        output_data = self._model.predict(input_data)

        # Postprocess the output
        return self._postprocess_output(output_data)

    def _preprocess_input(self, input_data):
        """
        Preprocesses the input data before feeding it to the model.
        """
        # Convert input_data to a numpy array and reshape it
        # to add the batch dimension if needed

        input_data = self._np.array(input_data, dtype=self._np.float32)
        input_data = input_data.reshape(1, *input_data.shape)
        return input_data

    def _postprocess_output(self, output_data):
        """
        Postprocesses the output data after getting it from the model.
        """
        # The post-processing step might change depending on the model and application
        # For now, just returning the output data directly
        return self._np.argmax(output_data)
