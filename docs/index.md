# fake NIR

Este sitio contiene la documentación del trabajo final de la materia Tratamiento Digital de Imágenes de la carrera de Ingenieria en Telecomunicaciones de la Universidad Nacional de Río Cuarto.

## Introducción

El objetivo principal de este trabajo en el desarollo de una Inteligencia Artificial capaz de convertir imagenes RGB (Red + Green + Blue) a imagenes NGB (Near Infrared + Green + Blue). Esto tiene muchas aplicaciones en el campo de la agricultura, por ejemplo, permitiría una estimación del índice de vegetación de diferencia normalizada (NDVI), índice de vegetación de diferencia normalizada verde (GNDVI) o el índice mejorado de vegetación de diferencio normalizada (ENDVI) con un dron comercial y sin necesidad de accesorios extras como cámaras multi-espectrales. 

Para lograr esto se hará uso de una arquitectura de redes generativas adversarias condicionadas (cGAN). En particular se basara en _"Image To Image"_ un articulo en el cual se desarrolla una arquitectura cambio de dominio de imágenes, la cual permite, por ejemplo, generar imágenes a color desde una imagen a blanco y negro, generar objetos a partir de sus siluetas o generar fachadas de edificios a partir de una mapa de segmentación entre otras aplicaciones.

Para el entrenamiento del modelo se hará uso del dataset [deepNIR NIR RGB capsicum](https://www.kaggle.com/datasets/enddl22/deepnir-nir-rgb-capsicum). El cual no es optimo dado que solo contiene imágenes de plantas de pimientos, este dataset debería ser expandido para contener diferentes cultivos o imágenes tomadas con cámaras RGB y NGB.

Para la extension del dataset se utilizara dos camaras, una camara Mapir Survey3W (NGB) y una camara Logitech C920 (RGB) monstadas sobre una soporte lado a lado, de tal forma que los sensores esten alineados.

## Project layout

