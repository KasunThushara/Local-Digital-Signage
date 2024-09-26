import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from zipfile import ZipFile

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/sprite-cola/train',
    '/content/drive/MyDrive/sprite-cola/train',
    ['coca-cola', 'sprite']
)

val_data = object_detector.DataLoader.from_pascal_voc(
    '/content/drive/MyDrive/sprite-cola/valid',
    '/content/drive/MyDrive/sprite-cola/valid',
    ['coca-cola', 'sprite']
)

spec = model_spec.get('efficientdet_lite0')
print("kasun")

# Train the model
model = object_detector.create(train_data, model_spec=spec, batch_size=16, train_whole_model=True, epochs=25, validation_data=val_data)

# Evaluate the model
eval_result = model.evaluate(val_data)

# Export the model as TFLite
model.export(export_dir='.', tflite_filename='sprite-cola.tflite')

# Export the model as Edge TPU compatible
#quantization_config = QuantizationConfig.for_int8()
#model.export(export_dir='.', tflite_filename='sprite-cola_edgetpu.tflite', quantization_config=quantization_config, export_format=[ExportFormat.TFLITE, ExportFormat.EDGE_TPU])

# Evaluate the TFLite model
tflite_eval_result = model.evaluate_tflite('sprite-cola.tflite', val_data)

# Print COCO metrics for TFLite
print("COCO metrics TFLite")
for label, metric_value in tflite_eval_result.items():
    print(f"{label}: {metric_value}")

# You can also use edgetpu_compiler to compile the Edge TPU model separately
