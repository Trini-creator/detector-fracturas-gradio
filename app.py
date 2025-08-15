# app.py
# VERSIÓN DEFINITIVA - Soluciona el error de "slice index" en Grad-CAM para modelos con salida Sigmoid.

import os
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf
import matplotlib.cm as cm

# -----------------------------
# Configuración
# -----------------------------
MODEL_PATH = "fracture_detection_model.keras"
LABELS_PATH = "labels.txt"
MODEL_IMG_SIZE = (224, 224)

# -----------------------------
# Carga de Recursos al Iniciar
# -----------------------------
try:
    print(f"Cargando modelo desde: {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    raise RuntimeError(f"Error fatal al cargar el modelo: {e}")

try:
    print(f"Cargando etiquetas desde: {LABELS_PATH}...")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✅ Etiquetas cargadas: {CLASS_NAMES}")
except Exception as e:
    raise RuntimeError(f"Error fatal al cargar labels.txt: {e}")

# -----------------------------
# Lógica de Predicción y Grad-CAM
# -----------------------------
def _find_last_conv_layer(model: tf.keras.Model):
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4 and "conv" in layer.name.lower():
            return layer
    return None

def compute_gradcam(model: tf.keras.Model, img_tensor: tf.Tensor, class_index: int):
    last_conv_layer = _find_last_conv_layer(model)
    if not last_conv_layer: return np.zeros(img_tensor.shape[1:3])
    
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.outputs])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        
        if isinstance(predictions, list):
            predictions = predictions[0]

        # --- ¡LA CORRECCIÓN FINAL ESTÁ AQUÍ! ---
        # Comprobamos la forma de la salida de la predicción.
        if predictions.shape[-1] == 1:
            # Si es (batch, 1) -> Salida Sigmoid. La pérdida es la propia salida.
            # No necesitamos 'class_index' porque solo hay una salida que analizar.
            loss = predictions[:, 0]
        else:
            # Si es (batch, N) -> Salida Softmax. Hacemos el "corte" con class_index.
            loss = predictions[:, class_index]
        # ----------------------------------------
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs[0], pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)
    return tf.image.resize(heatmap[..., np.newaxis], img_tensor.shape[1:3])[..., 0].numpy()

def colorize_and_overlay(image_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
    cmap = cm.get_cmap("jet")
    heatmap_colored = cmap(heatmap)[..., :3]
    heatmap_img = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).resize(image_pil.size)
    return Image.blend(image_pil.convert("RGB"), heatmap_img, alpha)

def predict_image(pil_img: Image.Image):
    # 1. Preprocesar la imagen
    img_rgb = pil_img.convert("RGB").resize(MODEL_IMG_SIZE, Image.BILINEAR)
    img_array = np.array(img_rgb) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0).astype("float32")

    # 2. Realizar la predicción
    predictions_output = model.predict(img_tensor, verbose=0)
    
    preds = predictions_output[0][0] if isinstance(predictions_output, list) else predictions_output[0]

    # 3. Formatear las probabilidades
    if len(preds) == 1:
        prob_clase_1 = preds[0]
        probs = [1 - prob_clase_1, prob_clase_1]
    else:
        probs = tf.nn.softmax(preds).numpy()

    if len(probs) != len(CLASS_NAMES):
        raise ValueError(f"¡DESAJUSTE! El modelo devolvió {len(probs)} probabilidades, pero hay {len(CLASS_NAMES)} etiquetas.")

    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    
    # 4. Calcular Grad-CAM
    top_class_index = np.argmax(probs)
    heatmap = compute_gradcam(model, img_tensor, top_class_index)
    overlay_img = colorize_and_overlay(pil_img, heatmap)

    return confidences, overlay_img

# -----------------------------
# Interfaz de Gradio (Sin cambios)
# -----------------------------
title = "Clasificador de Fracturas con Explicación Visual (Grad-CAM)"
description = "Sube una radiografía. El modelo predecirá si hay una fractura y resaltará las áreas de la imagen que más influyeron en su decisión."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"<h1>{title}</h1>")
    gr.Markdown(description)
    with gr.Row():
        inp = gr.Image(type="pil", label="Subir Radiografía")
        with gr.Column():
            out_lbl = gr.Label(num_top_classes=len(CLASS_NAMES), label="Resultado del Diagnóstico")
            out_img = gr.Image(type="pil", label="Mapa de Calor (Grad-CAM)", interactive=False)
    btn = gr.Button("Analizar Imagen", variant="primary")
    btn.click(fn=predict_image, inputs=inp, outputs=[out_lbl, out_img])

if __name__ == "__main__":
    demo.launch()

