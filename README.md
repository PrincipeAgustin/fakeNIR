# fakeNIR
Trabajo final de la materia Tratamiento Digital de Imágenes de la carrera de Ingeniería en Telecomunicaciones de la Universidad Nacional de Río Cuarto 

## Requerimientos

Se recomienda fuertemente poseer de una GPU NVIDIA 1050 TI para arriba para correr este modelo, dado que el entrenamiento en procesador no es posible, más abajo se listan diferentes [rendimientos obtenidos](#rendimiento-obtenido-en-diferente-hardware).
 
Este proyecto fue realizado utilizando **Tensorflow 2.10** en caso de querer actualizar a versiones futuras, se debera re-entrenar al modelo dado que el optimizador de la version 2.10 no es compatible con la version 2.11 (Fecha de prueba 22/11/2022)

## Rendimiento obtenido en diferente hardware

> El rendimiento mencionado es el tiempo que toma para realizar un único paso de entrenamiento (Generar desde una imagen de entrenamiento y evaluar al discriminador)
>* Ryzen 2200G: ~5 segudos (Batch size: 2, 645 imagenes por epoca). 
>* NVIDIA GTX 1050 TI: ~0.4 segundos (Batch size: 2, 645 imagenes por epoca). 
>* NVIDIA TESLA T4: ~0.4 segundos (Batch size: 4, 645 imagenes por epoca). 
