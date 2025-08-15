# -*- coding: utf-8 -*-
"""
App Gradio para clasificar imágenes con un modelo Keras/TensorFlow y visualizar Grad-CAM.

Cómo usar en Hugging Face Spaces:
1) Sube este archivo como app.py
2) Sube tu modelo como 'mi_modelo.h5' (o carpeta SavedModel llamada 'mi_modelo')
3) Ajusta 'NOMBRES_CLASES' al orden real de tu modelo
4) (Opcional) Cambia 'MODEL_PATH' si usas otro nombre
"""

import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

# --------------------------------------------------------------------
# 1) Configuración básica
# --------------------------------------------------------------------

# Ruta del modelo. Acepta:
#  - archivo .h5      -> 'mi_modelo.h5'
#  - carpeta SavedModel-> 'mi_modelo' (directorio con saved_model.pb)
MODEL_PATH = "mi_modelo.h5"  # cámbialo si tu modelo tiene otro nombre

# ¡IMPORTANTE! Reemplaza por las etiquetas reales y en el orden correcto
# Deben coincidir con el vector de salida de tu modelo.
NOMBRES_CLASES = ["Gato", "Perro", "Pájaro"]  # ejemplo

# Gestión de memoria GPU en Spaces (seguro aunque no haya GPU)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# --------------------------------------------------------------------
# 2) Cargar modelo
# --------------------------------------------------------------------
def _load_any_model(path: str):
    """
    Carga un modelo Keras desde .h5 o SavedModel.
    """
    if not os.path.exists(path):
        # intento como carpeta SavedModel
        if os.path.isdir("mi_modelo"):
            return tf.keras.models.load_model("mi_modelo")
        raise FileNotFoundError(
            f"No se encontró '{path}'. Sube tu modelo como 'mi_modelo.h5' "
            "o una carpeta SavedModel llamada 'mi_modelo'."
        )
    # .h5
    return tf.keras.models.load_model(path)

try:
    modelo = _load_any_model(MODEL_PATH)
    # Comprobación del número de clases vs etiquetas
    salida = modelo.output_shape
    # salida puede ser (None, C) o lista si hay múltiples salidas
    if isinstance(salida, (list, tuple)) and isinstance(salida[0], (list, tuple)):
        num_clases = salida[0][-1]
    else:
        num_clases = salida[-1]
    if len(NOMBRES_CLASES) != num_clases:
        print(f"[AVISO] El modelo tiene {num_clases} clases pero NOMBRES_CLASES tiene {len(NOMBRES_CLASES)}. "
              f"Ajusta NOMBRES_CLASES para evitar etiquetas erróneas.")
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

# Detectar tamaño de entrada (H, W). Si no es 3 canales, se adaptará a RGB.
def _infer_input_size(keras_model) -> tuple:
    ishape = keras_model.input_shape
    # Ejemplos:
    # (None, 224, 224, 3) -> NHWC
    # (None, 3, 224, 224) -> NCHW (poco común en TF)
    if isinstance(ishape, list):
        ishape = ishape[0]
    if len(ishape) == 4:
        if ishape[3] in (1, 3):
            return (ishape[1], ishape[2])  # H, W
        elif ishape[1] in (1, 3):
            return (ishape[2], ishape[3])  # asumiendo NCHW
    # fallback
    return (224, 224)

TARGET_SIZE = _infer_input_size(modelo)

# --------------------------------------------------------------------
# 3) Utilidades: preprocesado y Grad-CAM
# --------------------------------------------------------------------
def _to_rgb(pil_img: Image.Image) -> Image.Image:
    if pil_img.mode != "RGB":
        return pil_img.convert("RGB")
    return pil_img

def _preprocess(pil_img: Image.Image) -> np.ndarray:
    """
    Preprocesado estándar (reescala 0-1). Ajusta aquí si usaste otra normalización.
    """
    img = _to_rgb(pil_img).resize(TARGET_SIZE, Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1,H,W,3)
    return arr

def _pick_last_conv_layer(keras_model: tf.keras.Model):
    """
    Intenta localizar la última capa convolucional 2D válida para Grad-CAM.
    """
    for layer in reversed(keras_model.layers):
        lname = layer.__class__.__name__.lower()
        if "conv" in lname and "separable" not in lname and hasattr(layer, "output_shape"):
            # capa conv 2D típica
            try:
                # verificar que el output es 4D (N,H,W,C)
                if len(layer.output_shape) == 4:
                    return layer.name
            except Exception:
                continue
        # también soportar SeparableConv2D
        if "separableconv2d" in lname and hasattr(layer, "output_shape"):
            try:
                if len(layer.output_shape) == 4:
                    return layer.name
            except Exception:
                continue
    # si no encuentra, devolver None
    return None

LAST_CONV_NAME = _pick_last_conv_layer(modelo)

def _gradcam(pil_img: Image.Image, class_index: int = None, alpha: float = 0.35):
    """
    Calcula Grad-CAM y devuelve (overlay_pil, heatmap_pil, predicted_label, dict_probs)
    """
    if LAST_CONV_NAME is None:
        raise RuntimeError("No se pudo localizar una capa convolucional para Grad‑CAM.")

    # Prepara tensores
    img_rgb = _to_rgb(pil_img)
    orig_w, orig_h = img_rgb.size
    x = _preprocess(img_rgb)

    # Predicción para obtener clase objetivo si no se especifica
    preds = modelo.predict(x, verbose=0)
