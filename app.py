# -*- coding: utf-8 -*-
"""
App Gradio para clasificar imágenes con un modelo Keras/TensorFlow y visualizar Grad-CAM.

Instrucciones (Hugging Face Spaces):
1) Sube este archivo como app.py
2) Sube tu modelo como:
   - archivo: fracture_detection_model.h5  o  fracture_detection_model.keras
   - o carpeta SavedModel: fracture_detection_model/ (con saved_model.pb)
3) Ajusta CLASS_NAMES al orden real de tu modelo (o define la variable de entorno CLASS_NAMES)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import Model
import gradio as gr

# --------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------
# Nombre base del modelo (sin extensión). Probamos .h5, .keras o carpeta.
MODEL_BASE = os.getenv("MODEL_PATH", "fracture_detection_model")
# Etiquetas (por defecto de ejemplo). Puedes definir en el Space: CLASS_NAMES="Fractura,No fractura"
CLASS_NAMES = os.getenv("CLASS_NAMES", "Gato,Perro,Pájaro").split(",")

# Gestión segura de GPU (si existe)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

# --------------------------------------------------------------------
# Carga del modelo (acepta .h5 / .keras / SavedModel/)
# --------------------------------------------------------------------
def _load_model_any(base: str) -> tf.keras.Model:
    # Intenta formatos comunes: archivo exacto, variantes, o carpeta SavedModel
    candidates = [
        base,                      # si te pasan ya "fracture_detection_model.h5"
        base + ".h5",
        base + ".keras",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return tf.keras.models.load_model(path)
    # Carpeta SavedModel
    if os.path.isdir(base):
        return tf.keras.models.load_model(base)
    raise FileNotFoundError(
        f"No se encontró el modelo '{base}'. Sube 'fracture_detection_model.h5', "
        f"'fracture_detection_model.keras' o la carpeta 'fracture_detection_model/'."
    )

try:
    model = _load_model_any(MODEL_BASE)
    print("✅ Modelo cargado.")
except TypeError as e:
    # Error típico por incompatibilidad Keras3 vs Keras2 (batch_shape, etc.)
    raise SystemExit(
        "Error al cargar el modelo (posible incompatibilidad de versiones). "
        "En tu 'requirements.txt' usa por ejemplo:\n"
        "tensorflow==2.15.0\nkeras==2.15.0\n"
        "o guarda tu modelo en formato Keras 3 (.keras)."
    ) from e
except Exception as e:
    raise SystemExit(f"Error al cargar el modelo: {e}")

# --------------------------------------------------------------------
# Inferir tamaño de entrada (H, W, C)
# --------------------------------------------------------------------
def _infer_input_size(keras_model) -> tuple[int, int, int]:
    ishape = keras_model.input_shape
    if isinstance(ishape, (list, tuple)) and isinstance(ishape[0], (list, tuple)):
        ishape = ishape[0]
    if len(ishape) == 4:
        # NHWC habitual
        if ishape[3] in (1, 3):
            return int(ishape[1] or 224), int(ishape[2] or 224), int(ishape[3])
        # NCHW (raro en TF)
        if ishape[1] in (1, 3):
            return int(ishape[2] or 224), int(ishape[3] or 224), int(ishape[1])
    return 224, 224, 3

IN_H, IN_W, IN_C = _infer_input_size(model)

# Validar número de clases vs CLASS_NAMES
try:
    out_shape = model.output_shape
    if isinstance(out_shape, (list, tuple)) and isinstance(out_shape[0], (list, tuple)):
        num_classes = int(out_shape[0][-1])
    else:
        num_classes = int(out_shape[-1])
except Exception:
    num_classes = len(CLASS_NAMES)

if len(CLASS_NAMES) != num_classes:
    print(f"[AVISO] El modelo tiene {num_classes} clases, pero CLASS_NAMES tiene {len(CLASS_NAMES)}. "
          f"Ajusta CLASS_NAMES para etiquetar correctamente.")

# --------------------------------------------------------------------
# Utilidades: preprocesado y Grad-CAM
# --------------------------------------------------------------------
def _ensure_rgb(pil_img: Image.Image) -> Image.Image:
    return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img

def _preprocess(pil_img: Image.Image) -> np.ndarray:
    # Normalización simple a [0,1]. Ajusta si usaste otra (e.g., tf.keras.applications.*.preprocess_input).
    img = _ensure_rgb(pil_img).resize((IN_W, IN_H), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    if arr.ndim == 2:  # gris -> (H,W,1)
        arr = np.expand_dims(arr, -1)
    if arr.shape[-1] == 1:  # fuerza 3 canales si el modelo lo espera
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.expand_dims(arr, 0)  # (1,H,W,3)
    return arr

def _find_last_conv_layer(keras_model: tf.keras.Model):
    # Busca la última capa con salida 4D (N,H,W,C)
    for layer in reversed(keras_model.layers):
        try:
            oshape = layer.output_shape
            if isinstance(oshape, tuple) and len(oshape) == 4:
                return layer
        except Exception:
            continue
    return None

LAST_CONV = _find_last_conv_layer(model)

def _make_gradcam(pil_img: Image.Image, class_index: int | None = None, alpha: float = 0.40):
    if LAST_CONV is None:
        raise RuntimeError("No se encontró una capa convolucional adecuada para Grad‑CAM.")

    x = _preprocess(pil_img)
    preds = model.predict(x, verbose=0)[0]
    if class_index is None:
        class_index = int(np.argmax(preds))

    # Confianzas con etiquetas
    labels = CLASS_NAMES if len(CLASS_NAMES) == preds.shape[-1] else [f"Clase {i}" for i in range(preds.shape[-1])]
    conf = {labels[i]: float(preds[i]) for i in range(len(labels))}
    top_label = labels[class_index]

    # Modelo intermedio para obtener activaciones + salida final
    grad_model = Model(inputs=model.inputs, outputs=[LAST_CONV.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, probs = grad_model(x, training=False)
        if probs.shape.rank == 2:
            target = probs[:, class_index]
        else:
            target = tf.reshape(probs, (tf.shape(probs)[0], -1))[:, class_index]

    grads = tape.gradient(target, conv_out)                  # (1, Hc, Wc, C)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))        # (1, C)
    conv_out = conv_out[0]                                   # (Hc, Wc, C)
    pooled_grads = pooled_grads[0]                           # (C,)

    cam = tf.reduce_sum(conv_out * pooled_grads, axis=-1)    # (Hc, Wc)
    cam = tf.nn.relu(cam)
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., tf.newaxis], (pil_img.height, pil_img.width))
    cam = tf.squeeze(cam, -1).numpy()                        # (H, W)

    # Crear heatmap con matplotlib
    import matplotlib.pyplot as plt
    colormap = plt.get_cmap("jet")
    heat = (colormap(cam)[:, :, :3] * 255).astype(np.uint8)  # RGB uint8

    base = _ensure_rgb(pil_img)
    overlay = (0.40 * heat + 0.60 * np.asarray(base)).astype(np.uint8)

    heat_pil = Image.fromarray(heat)
    overlay_pil = Image.fromarray(overlay)

    return overlay_pil, heat_pil, top_label, conf

# --------------------------------------------------------------------
# Función de Gradio
# --------------------------------------------------------------------
def predict_and_gradcam(pil_image: Image.Image):
    if pil_image is None:
        return {}, None, None
    try:
        overlay, heat, top, conf = _make_gradcam(pil_image, class_index=None, alpha=0.40)
        return conf, overlay, heat
    except Exception as e:
        # Si Grad-CAM falla, al menos devolver predicción
        x = _preprocess(pil_image)
        preds = model.predict(x, verbose=0)[0]
        labels = CLASS_NAMES if len(CLASS_NAMES) == preds.shape[-1] else [f"Clase {i}" for i in range(preds.shape[-1])]
        conf = {labels[i]: float(preds[i]) for i in range(len(labels))}
        print(f"⚠️ Grad‑CAM no disponible: {e}")
        return conf, pil_image, pil_image

# --------------------------------------------------------------------
# Interfaz Gradio
# --------------------------------------------------------------------
DESCRIPTION = """
Sube una imagen: el modelo la clasifica y muestra **Grad‑CAM** con las zonas que más influyeron.
**Importante**: ajusta `CLASS_NAMES` (variables de entorno del Space) al orden real de tu modelo.
"""

with gr.Blocks(title="Clasificador + Grad‑CAM (Keras/TF)") as demo:
    gr.Markdown("# Clasificador de Imágenes + Grad‑CAM (Keras/TF)")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Imagen de entrada")
            btn = gr.Button("Predecir", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Predicciones")
            out_overlay = gr.Image(type="pil", label="Grad‑CAM (superpuesta)")
            out_heat = gr.Image(type="pil", label="Heatmap")

    btn.click(fn=predict_and_gradcam, inputs=inp, outputs=[out_label, out_overlay, out_heat])

# Para Spaces (usa el puerto del entorno)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue(concurrency_count=2).launch(server_name="0.0.0.0", server_port=port)

