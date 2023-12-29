# cat_dog_classifier

# Introducción
En la era digital actual, la inteligencia artificial (IA) y el aprendizaje automático (Machine Learning, ML) están revolucionando la forma en que interactuamos con el mundo a nuestro alrededor. Estas tecnologías tienen el potencial de mejorar significativamente diversas aplicaciones, desde la asistencia sanitaria hasta la seguridad y el entretenimiento. Una de las áreas más fascinantes y en constante evolución dentro de este campo es el reconocimiento de imágenes en tiempo real.

Este proyecto se centra en el desarrollo de una aplicación avanzada que utiliza el poder del aprendizaje profundo, específicamente las redes neuronales, para reconocer y diferenciar entre imágenes de perros y gatos en tiempo real a través de la cámara de un dispositivo. La aplicación se basa en TensorFlow, una biblioteca de código abierto ampliamente reconocida por su flexibilidad y capacidad para construir y entrenar modelos de ML de alta eficiencia.

# Objetivo del Proyecto
El objetivo principal de este proyecto es desarrollar y desplegar un modelo de red neuronal capaz de identificar y clasificar con precisión imágenes de perros y gatos en tiempo real. Este sistema no solo busca alcanzar una alta precisión en la clasificación, sino también operar de manera eficiente en un entorno de tiempo real, lo que requiere una rápida procesamiento y respuesta.

El proyecto pretende demostrar la aplicabilidad práctica del aprendizaje profundo en tareas de reconocimiento de imágenes, ofreciendo una solución efectiva y accesible para usuarios finales que requieran este tipo de funcionalidad.

# Tecnologías Utilizadas
En el desarrollo de esta aplicación de reconocimiento de imágenes de perros y gatos en tiempo real, se han utilizado varias tecnologías clave que han jugado un papel fundamental en la construcción y funcionamiento eficaz del sistema. A continuación, se detallan estas tecnologías y se explica su rol en el proyecto:

### Python:

**Descripción:** Python es un lenguaje de programación de alto nivel, famoso por su simplicidad y legibilidad. Es ampliamente utilizado en el ámbito del aprendizaje automático y la ciencia de datos debido a su extensa colección de bibliotecas y frameworks.
**Uso en el Proyecto:** En este proyecto, Python se utilizó como el lenguaje de programación principal para desarrollar y entrenar el modelo de red neuronal. Su sintaxis clara y su capacidad para manejar grandes conjuntos de datos lo hacen ideal para tareas de aprendizaje profundo y manipulación de imágenes.

### TensorFlow:

**Descripción:** TensorFlow es una biblioteca de código abierto desarrollada por Google Brain, destinada a facilitar la creación de sistemas de aprendizaje automático. Es especialmente potente para el desarrollo de modelos de aprendizaje profundo.
**Uso en el Proyecto:** TensorFlow se utilizó para construir y entrenar el modelo de red neuronal. Proporciona las herramientas necesarias para diseñar la arquitectura del modelo, ajustar los hiperparámetros y llevar a cabo el proceso de entrenamiento y validación de manera eficiente.

### Google Colab:

**Descripción:** Google Colab es un entorno de notebook basado en la nube que permite la ejecución de scripts en Python. Ofrece acceso gratuito a recursos computacionales, incluyendo GPUs y TPUs, lo que lo hace adecuado para entrenar modelos de aprendizaje automático intensivos en datos.
**Uso en el Proyecto:** Google Colab se empleó para ejecutar las sentencias de Python y TensorFlow. Su capacidad para manejar grandes volúmenes de datos y su acceso a hardware de alto rendimiento permitieron un entrenamiento eficiente y rápido del modelo de red neuronal.

### Desarrollo Web y JavaScript:

**Descripción:** JavaScript es un lenguaje de programación versátil utilizado comúnmente para el desarrollo web. Permite crear interfaces interactivas y dinámicas en el lado del cliente.
**Uso en el Proyecto:** Para la interfaz de usuario de la aplicación, se desarrolló una página web utilizando JavaScript. Esta página web interactúa con la cámara del dispositivo del usuario, capturando fotogramas que luego se envían al modelo de red neuronal para su procesamiento y clasificación en tiempo real. JavaScript fue esencial para implementar la funcionalidad de captura de imágenes y la comunicación con el backend del modelo.

# Descripción del Modelo de Redes Neuronales

### Arquitectura del Modelo
Para el proyecto de reconocimiento de imágenes de perros y gatos en tiempo real, se diseñó y desarrolló un modelo de red neuronal convolucional (CNN). Las CNN son ampliamente reconocidas por su eficacia en el procesamiento y reconocimiento de imágenes, debido a su capacidad para detectar características y patrones visuales complejos.

La arquitectura específica del modelo implementado consta de las siguientes capas y componentes:

#### Capas Convolucionales:

El modelo incluye un total de tres capas convolucionales, sumando en conjunto 224 neuronas.
Estas capas son responsables de extraer y aprender características visuales de las imágenes, como bordes, texturas y formas.
Cada capa convolucional aplica filtros que transforman progresivamente los datos de entrada, permitiendo que el modelo aprenda de manera más eficiente.

#### Capa Densa:

Después de las capas convolucionales, el modelo incorpora una capa densa de 100 neuronas.
Esta capa densa sirve para procesar la información extraída por las capas convolucionales, facilitando la clasificación final.

#### Capa de Salida:
El modelo concluye con una capa de salida que contiene una única neurona.

Se utiliza la función de activación sigmoid en esta capa, ya que el objetivo es producir una salida binaria, donde el valor cercano a 0 representa un 'gato' y el valor cercano a 1 representa un 'perro'.
La elección de la función sigmoid es ideal en este caso, dado que se requiere clasificar las imágenes en dos categorías claramente definidas.

### Preparación de los Datos de Entrenamiento
Para entrenar este modelo, se utilizó un dataset compuesto por 23,000 imágenes de perros y gatos. El tratamiento de los datos se realizó de la siguiente manera:

**Estandarización del Color:** Todas las imágenes se convirtieron a escala de grises para uniformizar la entrada y reducir la complejidad computacional. Este paso es crucial para enfocarse en las características estructurales más que en los colores, que pueden variar significativamente entre diferentes imágenes.

**Redimensionamiento de las Imágenes:** Se ajustó el tamaño de todas las imágenes a 100x100 píxeles. Esta estandarización del tamaño asegura que cada imagen contribuya de manera uniforme al proceso de entrenamiento y permite que la red procese los datos de manera más eficiente.

La preparación cuidadosa de los datos es fundamental para el éxito del modelo, ya que garantiza que las entradas sean consistentes y optimizadas para el aprendizaje. Este paso permite que el modelo se concentre en aprender características relevantes de las imágenes, mejorando así su capacidad para clasificar con precisión entre perros y gatos.

# Implementación en Tiempo Real
La implementación en tiempo real del modelo de reconocimiento de perros y gatos se realizó mediante una integración cuidadosamente diseñada entre el modelo de TensorFlow, TensorFlow.js y una página web interactiva. Este proceso permitió que el modelo entrenado se aplicara en un entorno de usuario real, utilizando la cámara de un dispositivo para la clasificación en vivo. A continuación, se detallan los pasos clave de esta implementación:

### Exportación e Integración del Modelo con TensorFlow.js

**Exportación del Modelo:** El modelo de red neuronal, originalmente desarrollado y entrenado en TensorFlow, se exportó para su uso en un entorno web. Esta exportación convierte el modelo en un formato compatible con TensorFlow.js, una biblioteca de JavaScript que permite ejecutar modelos de TensorFlow en el navegador.

**Integración en la Página Web:** Una vez exportado, el modelo se integró dentro de la página web desarrollada. TensorFlow.js facilita esta integración, permitiendo que el modelo interactúe directamente con el código JavaScript de la página.

## Interacción con la Cámara y Procesamiento en Tiempo Real

**Acceso a la Cámara del Dispositivo:** Utilizando las funciones nativas del navegador, se obtiene acceso a la cámara del dispositivo del usuario. La imagen capturada por la cámara se renderiza en tiempo real en la página web, proporcionando una retroalimentación visual inmediata al usuario.

**Procesamiento Continuo de Imágenes:** Para lograr una experiencia de usuario fluida y una clasificación en tiempo real, se estableció un loop con un intervalo de 20 milisegundos. Durante cada iteración de este loop, la imagen actual capturada por la cámara se procesa y se envía al modelo.

**Análisis y Respuesta del Modelo:** Al recibir una imagen, el modelo preentrenado la analiza y proporciona una respuesta en forma de un valor numérico entre 0 y 1. Este valor indica la clasificación realizada por el modelo: valores cercanos a 0 se interpretan como 'gato', mientras que valores cercanos a 1 se interpretan como 'perro'.

## Presentación de Resultados
La respuesta del modelo se presenta al usuario de manera intuitiva, mostrando en la interfaz de la página web si la imagen actual corresponde a un perro o un gato. Esta presentación rápida y clara permite una interacción efectiva y satisfactoria con la aplicación.


# Recursos
* Google Colab: https://colab.research.google.com/drive/1KCxIHgF3eTfDXwclmS1Ray7mJFR1BRHi?usp=sharing

