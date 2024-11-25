# Detección de Carros y Reconocimiento de Matrículas desde Flujo de Video

Este proyecto es una aplicación Python que procesa archivos de video y flujos RTSP para detectar coches utilizando un modelo YOLO. Cuando se detecta un coche, guarda una imagen del mismo y utiliza el modelo GPT de OpenAI para extraer el número de matrícula de la imagen.

## Tabla de Contenidos

- [Características](#características)
- [Instalación](#instalación)
  - [Prerequisitos](#prerequisitos)
  - [Clonar el Repositorio](#clonar-el-repositorio)
  - [Configurar el Entorno](#configurar-el-entorno)
  - [Instalar Dependencias](#instalar-dependencias)
  - [Descargar el Modelo YOLO](#descargar-el-modelo-yolo)
  - [Configurar la Clave de API de OpenAI](#configurar-la-clave-de-api-de-openai)
- [Uso](#uso)
  - [Procesamiento de Archivos de Video](#procesamiento-de-archivos-de-video)
  - [Procesamiento de Flujos RTSP](#procesamiento-de-flujos-rtsp)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Registro de Eventos (Logging)](#registro-de-eventos-logging)
- [Dependencias](#dependencias)
- [Licencia](#licencia)

## Características

- **Detección de Coches**: Utiliza un modelo YOLO para detectar coches en fotogramas de video.
- **Guardado de Imágenes**: Guarda imágenes de los coches detectados para su procesamiento posterior.
- **Reconocimiento de Matrículas**: Extrae números de matrícula de las imágenes guardadas de coches utilizando el modelo GPT de OpenAI.
- **Soporte para Múltiples Fuentes de Video**: Puede procesar tanto archivos de video de una carpeta como flujos RTSP en vivo.
- **Procesamiento Multihilo**: El reconocimiento de matrículas se realiza en un hilo separado para mejorar el rendimiento.
- **Registro de Eventos**: Registro detallado tanto en consola como en archivo para monitoreo y depuración.

## Instalación

### Prerequisitos

- **Python 3.8 o superior**
- Administrador de paquetes **pip**
- **Git** (para clonar el repositorio)
- **Clave de API de OpenAI**: Regístrese en [OpenAI](https://beta.openai.com/signup/) para obtener una clave de API.

### Clonar el Repositorio

```bash
git clone https://github.com/yourusername/car-detection.git
cd car-detection
```

### Configurar el Entorno

Se recomienda utilizar un entorno virtual para gestionar las dependencias.

#### Usando `venv`

```bash
python -m venv venv
source venv/bin/activate  # En Windows use `venv\Scripts\activate`
```

#### Usando `conda`

```bash
conda create -n car-detection python=3.8
conda activate car-detection
```

### Instalar Dependencias

Instale los paquetes Python requeridos:

```bash
pip install -r requirements.txt
```

### Descargar el Modelo YOLO

Coloque su archivo de modelo YOLO (por ejemplo, `yolo11n.pt`) en el directorio raíz del proyecto. Puede descargar modelos pre-entrenados desde [Ultralytics YOLO](https://github.com/ultralytics/yolov5/releases).

### Configurar la Clave de API de OpenAI

Cree un archivo `.env` en el directorio raíz del proyecto y agregue su clave de API de OpenAI:

```bash
OPENAI_API_KEY=su_clave_de_api_de_openai_aquí
```

Alternativamente, puede configurar la variable de entorno directamente:

```bash
export OPENAI_API_KEY=su_clave_de_api_de_openai_aquí  # En Windows use `set`
```

## Uso

Ejecute el script principal:

```bash
python yolo_test.py
```

### Procesamiento de Archivos de Video

Coloque sus archivos de video en el directorio `videos/`. Los formatos soportados incluyen `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v` y `.mp3`. La aplicación procesará automáticamente todos los videos en esta carpeta.

### Procesamiento de Flujos RTSP

La URL del flujo RTSP se configura en la función `main()` dentro de `yolo_test.py`. Actualice la variable `rtsp_url` con su URL de flujo:

```python
rtsp_url = 'rtsp://usuario:contraseña@dirección_ip:puerto/stream'
```

La aplicación se conectará al flujo RTSP después de procesar los archivos de video.

## Estructura del Proyecto

```
car-detection/
├── cars/               # Imágenes guardadas de coches detectados
├── videos/             # Archivos de video para procesar
├── yolo_test.py             # Script principal de la aplicación
├── image_api.py        # Script de reconocimiento de matrículas
├── requirements.txt    # Dependencias de Python
├── .env                # Variables de entorno (Clave de API de OpenAI)
├── yolo11n.pt          # Archivo del modelo YOLO
├── README.md           # Este archivo README
```

## Registro de Eventos (Logging)

Los registros se guardan en `car_detection.log` y también se muestran en la consola. Los niveles y formatos de registro se configuran en `yolo_test.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('car_detection.log'),
        logging.StreamHandler()
    ]
)
```

## Dependencias

- **Paquetes Python**: Listados en `requirements.txt`. Instale usando `pip install -r requirements.txt`.
- **Modelo YOLO**: Requerido para la detección de coches. Descárguelo desde [Ultralytics YOLO](https://github.com/ultralytics/yolov5/releases).
- **Clave de API de OpenAI**: Requerida para el reconocimiento de matrículas.

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

---

*Descargo de responsabilidad*: Este proyecto utiliza la API de OpenAI para el reconocimiento de matrículas. Por favor, asegúrese de cumplir con los [Términos de Uso](https://openai.com/policies/terms-of-use) de OpenAI al utilizar esta aplicación.