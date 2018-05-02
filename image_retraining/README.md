Usage:

IMAGE_SIZE=224
ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}"
python -m image_retraining.retrain \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos \
  --bottleneck_dir=tf_files/bottlenecks \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --checkpoint_path=tf_files/mobilenet/ \
  --learning_rate=0.0001 \
  --how_many_training_steps=5000

python -m image_retraining.label_image \
    --image=tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg \
    --graph=tf_files/retrained_graph.pb  \
    --labels=tf_files/retrained_labels.txt



retrain.py is an example script that shows how one can adapt a pretrained
network for other classification problems. A detailed overview of this script
can be found at:
https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

The script also shows how one can train layers
with quantized weights and activations instead of taking a pre-trained floating
point model and then quantizing weights and activations.
The output graphdef produced by this script is compatible with the TensorFlow
Lite Optimizing Converter and can be converted to TFLite format.


