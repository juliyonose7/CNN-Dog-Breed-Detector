"""
analizador de datasets para clasificacion binaria perro vs no-perro
script especializado para analizar estructura, calidad y distribucion
de imagenes en dataset de entrenamiento, optimizado para hardware amd

funcionalidades principales:
- analisis de estructura de directorios y distribucion de clases
- estadisticas de imagenes: resolucion, formato, calidad
- deteccion de desbalance entre clases
- validacion de integridad de archivos
- generacion de reportes visuales y metricas
- identificacion de problemas potenciales en datos
- recomendaciones para mejora del dataset
"""

# imports del sistema operativo y manejo de archivos
import os                    # operaciones del sistema operativo
import json                  # manejo de datos json
from pathlib import Path     # manejo moderno de rutas
from collections import defaultdict, Counter  # estructuras de datos especializadas
import warnings              # control de advertencias
warnings.filterwarnings('ignore')  # suprime advertencias no criticas

# imports de procesamiento de imagenes
import cv2                   # computer vision opencv
import numpy as np           # operaciones numericas
from tqdm import tqdm        # barras de progreso

# imports de analisis de datos y visualizacion
import pandas as pd          # manipulacion de dataframes
import matplotlib.pyplot as plt  # graficas y plots
import seaborn as sns        # visualizaciones estadisticas avanzadas

# clase principal para analisis completo de datasets de imagenes
# proporciona herramientas para evaluar calidad y estructura de datos
class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        """
        inicializa analizador con ruta al dataset principal
        
        parametros:
        - dataset_path: ruta al directorio raiz que contiene YESDOG y NODOG
        """
        self.dataset_path = Path(dataset_path)     # ruta principal del dataset
        self.yesdog_path = self.dataset_path / "YESDOG"  # subdirectorio de perros
        self.nodog_path = self.dataset_path / "NODOG"    # subdirectorio de no-perros
        self.stats = {}                            # diccionario para almacenar estadisticas
        
        # extensiones de imagen soportadas para validacion
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def analyze_dataset_structure(self):
        """analiza la estructura completa del dataset y distribucion de clases"""
        print("🔍 Analizando estructura del dataset...")
        
        # analisis de la categoria YESDOG con todas las razas de perros
        dog_breeds = []           # lista para almacenar info de cada raza
        dog_image_count = 0       # contador total de imagenes de perros
        
        # itera sobre cada subdirectorio de raza en YESDOG
        for breed_folder in self.yesdog_path.iterdir():
            if breed_folder.is_dir():  # verifica que sea directorio valido
                breed_name = breed_folder.name  # nombre de la raza
                images = self._count_images_in_folder(breed_folder)  # cuenta imagenes
                
                # almacena informacion de la raza
                dog_breeds.append({
                    'breed': breed_name,
                    'folder': str(breed_folder),
                    'image_count': images
                })
                dog_image_count += images  # suma al total
        
        # analisis de la categoria NODOG con objetos que no son perros
        nodog_categories = []     # lista para almacenar info de cada categoria
        nodog_image_count = 0     # contador total de imagenes no-perros
        
        # itera sobre cada subdirectorio de categoria en NODOG
        for category_folder in self.nodog_path.iterdir():
            if category_folder.is_dir():  # verifica que sea directorio valido
                category_name = category_folder.name  # nombre de la categoria
                images = self._count_images_in_folder(category_folder)  # cuenta imagenes
                
                # almacena informacion de la categoria
                nodog_categories.append({
                    'category': category_name,
                    'folder': str(category_folder),
                    'image_count': images
                })
                nodog_image_count += images  # suma al total
        
        # almacena estadisticas completas en el objeto principal
        self.stats['dog_breeds'] = dog_breeds               # lista de razas con conteos
        self.stats['nodog_categories'] = nodog_categories   # lista de categorias con conteos
        self.stats['total_dog_images'] = dog_image_count    # total imagenes de perros
        self.stats['total_nodog_images'] = nodog_image_count # total imagenes no-perros
        self.stats['total_images'] = dog_image_count + nodog_image_count  # total general
        
        # calcula metricas de balance entre clases principales
        self.stats['class_balance'] = {
            'dogs': dog_image_count,      # cantidad absoluta perros
            'no_dogs': nodog_image_count, # cantidad absoluta no-perros
            'ratio': dog_image_count / max(nodog_image_count, 1)  # ratio evitando division por cero
        }
        
        # imprime resumen ejecutivo del analisis
        print(f"✅ Análisis completado:")
        print(f"   - Razas de perros: {len(dog_breeds)}")
        print(f"   - Categorías no-perro: {len(nodog_categories)}")
        print(f"   - Total imágenes perros: {dog_image_count:,}")
        print(f"   - Total imágenes no-perros: {nodog_image_count:,}")
        print(f"   - Ratio perros/no-perros: {self.stats['class_balance']['ratio']:.2f}")
        
    def _count_images_in_folder(self, folder_path: Path) -> int:
        """
        cuenta imagenes validas en un directorio especifico
        filtra solo archivos con extensiones de imagen reconocidas
        
        parametros:
        - folder_path: ruta al directorio a analizar
        
        retorna:
        - numero entero de imagenes validas encontradas
        """
        count = 0  # inicializa contador
        
        # examina cada archivo en el directorio
        for file in folder_path.iterdir():
            # verifica que sea archivo y tenga extension de imagen valida
            if file.is_file() and file.suffix.lower() in self.image_extensions:
                count += 1  # incrementa contador si es imagen valida
                    
        return count  # retorna total de imagenes encontradas
    
    def analyze_image_properties(self, sample_size: int = 1000):
        """
        analiza propiedades tecnicas de imagenes mediante muestreo estadistico
        examina dimensiones, calidad y caracteristicas visuales de las imagenes
        
        parametros:
        - sample_size: numero total de imagenes a muestrear del dataset
        
        funcionalidades:
        - distribucion de dimensiones: ancho, alto, canales de color
        - estadisticas de tamaño de archivo y aspect ratio
        - deteccion de archivos corruptos o no legibles
        - analisis de proporciones y resolucion de imagenes
        """
        print(f"📊 Analizando propiedades de imágenes muestra de {sample_size}...")
        
        # diccionario para recopilar todas las propiedades medidas
        image_properties = {
            'widths': [],         # lista de anchos en pixeles
            'heights': [],        # lista de alturas en pixeles
            'channels': [],       # lista de numero de canales de color
            'file_sizes': [],     # lista de tamaños de archivo en KB
            'corrupted': [],      # lista de archivos dañados o ilegibles
            'aspect_ratios': []   # lista de proporciones ancho/alto
        }
        
        # obtiene muestras balanceadas de ambas clases principales
        dog_samples = self._sample_images_from_class('dog', sample_size // 2)      # mitad perros
        nodog_samples = self._sample_images_from_class('nodog', sample_size // 2)  # mitad no-perros
        
        # combina todas las muestras para analisis unificado
        all_samples = dog_samples + nodog_samples
        
        # procesa cada imagen individual con barra de progreso
        for img_path, label in tqdm(all_samples, desc="Analizando imágenes"):
            try:
                # carga imagen usando opencv para analisis
                img = cv2.imread(str(img_path))
                
                # verifica si la imagen se cargo correctamente
                if img is None:
                    image_properties['corrupted'].append(str(img_path))  # marca como corrupta
                    continue  # salta al siguiente archivo
                
                # extrae dimensiones basicas de la imagen
                h, w, c = img.shape  # alto, ancho, canales
                image_properties['heights'].append(h)      # almacena altura
                image_properties['widths'].append(w)       # almacena ancho
                image_properties['channels'].append(c)     # almacena canales
                image_properties['aspect_ratios'].append(w/h)  # calcula y almacena ratio
                
                # calcula tamaño del archivo en disco
                file_size = Path(img_path).stat().st_size / 1024  # convierte bytes a KB
                image_properties['file_sizes'].append(file_size)  # almacena tamaño
                
            except Exception as e:
                # captura cualquier error durante el procesamiento
                image_properties['corrupted'].append(str(img_path))  # marca como corrupta
        
        # calcula estadisticas descriptivas completas de las propiedades
        self.stats['image_properties'] = {
            'width_stats': {               # estadisticas de ancho
                'mean': np.mean(image_properties['widths']),    # promedio
                'std': np.std(image_properties['widths']),      # desviacion estandar
                'min': np.min(image_properties['widths']),      # minimo
                'max': np.max(image_properties['widths']),      # maximo
                'median': np.median(image_properties['widths']) # mediana
            },
            'height_stats': {              # estadisticas de altura
                'mean': np.mean(image_properties['heights']),   # promedio
                'std': np.std(image_properties['heights']),     # desviacion estandar
                'min': np.min(image_properties['heights']),     # minimo
                'max': np.max(image_properties['heights']),     # maximo
                'median': np.median(image_properties['heights'])# mediana
            },
            'aspect_ratio_stats': {        # estadisticas de proporcion
                'mean': np.mean(image_properties['aspect_ratios']),   # promedio
                'std': np.std(image_properties['aspect_ratios']),     # desviacion
                'min': np.min(image_properties['aspect_ratios']),     # mas cuadrada
                'max': np.max(image_properties['aspect_ratios'])      # mas rectangular
            },
            'file_size_stats': {           # estadisticas de tamaño archivo
                'mean_kb': np.mean(image_properties['file_sizes']),   # promedio en KB
                'median_kb': np.median(image_properties['file_sizes']),# mediana en KB
                'min_kb': np.min(image_properties['file_sizes']),     # minimo en KB
                'max_kb': np.max(image_properties['file_sizes'])      # maximo en KB
            },
            'corrupted_count': len(image_properties['corrupted']),   # total archivos corruptos
            'total_analyzed': len(all_samples)                       # total imagenes analizadas
        }
        
        # imprime resumen ejecutivo del analisis de propiedades
        print(f"✅ Propiedades analizadas:")
        print(f"   - Imágenes corruptas: {len(image_properties['corrupted'])}")
        print(f"   - Dimensión promedio: {self.stats['image_properties']['width_stats']['mean']:.0f}x{self.stats['image_properties']['height_stats']['mean']:.0f}")
        print(f"   - Tamaño promedio: {self.stats['image_properties']['file_size_stats']['mean_kb']:.1f} KB")
        
    def _sample_images_from_class(self, class_type: str, sample_size: int):
        """
        muestrea imagenes de una clase especifica de manera proporcional
        distribuye el muestreo equitativamente entre subcategorias
        
        parametros:
        - class_type: tipo de clase a muestrear 'dog' o 'nodog'
        - sample_size: numero total de imagenes a obtener
        
        retorna:
        - lista de tuplas ruta_imagen, etiqueta_clase
        
        estrategia de muestreo:
        - muestreo proporcional por subcategoria para evitar sesgo
        - distribucion equitativa entre razas o categorias
        - seleccion aleatoria dentro de cada subcategoria
        """
        images = []  # lista para almacenar imagenes muestreadas
        
        # determina directorios segun el tipo de clase solicitado
        if class_type == 'dog':
            # obtiene todas las carpetas de razas de perros
            folders = [breed['folder'] for breed in self.stats['dog_breeds']]
        else:
            # obtiene todas las carpetas de categorias no-perros
            folders = [cat['folder'] for cat in self.stats['nodog_categories']]
        
        # procesa cada subcategoria individualmente
        for folder in folders:
            folder_path = Path(folder)  # convierte a objeto Path
            folder_images = []          # imagenes encontradas en esta carpeta
            
            # recopila todas las imagenes validas de la carpeta actual
            for file in folder_path.iterdir():
                # verifica que sea archivo y tenga extension de imagen
                if file.is_file() and file.suffix.lower() in self.image_extensions:
                    folder_images.append((file, class_type))  # agrega tupla imagen-etiqueta
            
            # realiza muestreo proporcional si hay imagenes disponibles
            if folder_images:
                # calcula numero de muestras para esta carpeta
                n_samples = min(len(folder_images), max(1, sample_size // len(folders)))
                
                # selecciona indices aleatorios sin reemplazo
                sampled = np.random.choice(len(folder_images), 
                                         size=min(n_samples, len(folder_images)), 
                                         replace=False)  # evita duplicados
                
                # agrega imagenes seleccionadas a la lista final
                for idx in sampled:
                    images.append(folder_images[idx])
        
        return images  # retorna lista completa de imagenes muestreadas
    
    def create_visualization_report(self):
        """
        crea visualizaciones comprehensivas del analisis del dataset
        genera graficas para entender distribucion y caracteristicas de datos
        
        funcionalidades:
        - graficas de distribucion de clases principales
        - analisis visual de razas y categorias mas representadas
        - histogramas de propiedades tecnicas de imagenes
        - mapas de calor y correlaciones entre variables
        - reportes visuales exportables en formato PNG
        """
        print("📈 Creando reporte visual...")
        
        # configura figura principal con multiples subplots organizados
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # grid 2x3 para 6 graficas
        fig.suptitle('Análisis del Dataset PERRO vs NO-PERRO', fontsize=16, fontweight='bold')
        
        # grafica 1: distribucion de clases principales con grafica de torta
        ax1 = axes[0, 0]
        classes = ['Perros', 'No-Perros']                    # etiquetas de clases
        counts = [self.stats['total_dog_images'], self.stats['total_nodog_images']]  # conteos
        colors = ['#FF6B6B', '#4ECDC4']                      # colores distintivos
        
        # crea grafica de torta con porcentajes automaticos
        wedges, texts, autotexts = ax1.pie(counts, labels=classes, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Distribución de Clases')  # titulo descriptivo
        
        # grafica 2: top 10 razas de perros con mas imagenes disponibles
        ax2 = axes[0, 1]
        dog_df = pd.DataFrame(self.stats['dog_breeds'])      # convierte a dataframe
        top_breeds = dog_df.nlargest(10, 'image_count')     # obtiene top 10
        
        # extrae nombres de razas sin prefijos tecnicos
        breed_names = [breed.split('-')[-1] for breed in top_breeds['breed']]
        
        # crea grafica de barras horizontales para mejor legibilidad
        ax2.barh(breed_names, top_breeds['image_count'], color='#FF6B6B', alpha=0.7)
        ax2.set_title('Top 10 Razas más imágenes')     # titulo descriptivo
        ax2.set_xlabel('Número de imágenes')            # etiqueta eje x
        
        # grafica 3: categorias no-perro con mayor representacion
        ax3 = axes[0, 2]
        nodog_df = pd.DataFrame(self.stats['nodog_categories'])  # convierte a dataframe
        
        # limpia nombres de categorias removiendo sufijos tecnicos
        cat_names = [cat.replace('_final', '') for cat in nodog_df['category']]
        
        # crea grafica de barras verticales para categorias no-perro
        ax3.bar(range(len(cat_names)), nodog_df['image_count'], color='#4ECDC4', alpha=0.7)
        ax3.set_title('Categorías No-Perro')         # titulo descriptivo
        ax3.set_xlabel('Categoría')                   # etiqueta eje x
        ax3.set_ylabel('Número de imágenes')        # etiqueta eje y
        ax3.set_xticks(range(len(cat_names)))       # posiciones de etiquetas
        ax3.set_xticklabels(cat_names, rotation=45, ha='right')  # etiquetas rotadas
        
        # graficas 4-6: analisis de propiedades tecnicas si estan disponibles
        if 'image_properties' in self.stats:
            # grafica 4: distribucion de dimensiones promedio con barras de error
            ax4 = axes[1, 0]
            width_stats = self.stats['image_properties']['width_stats']   # estadisticas ancho
            height_stats = self.stats['image_properties']['height_stats'] # estadisticas alto
            
            # prepara datos para grafica de barras con desviacion estandar
            dimensions = ['Ancho', 'Alto']                              # etiquetas
            means = [width_stats['mean'], height_stats['mean']]         # promedios
            stds = [width_stats['std'], height_stats['std']]            # desviaciones
            
            # crea barras con barras de error para mostrar variabilidad
            ax4.bar(dimensions, means, yerr=stds, capsize=5, 
                   color=['#95E1D3', '#F38BA8'], alpha=0.7)
            ax4.set_title('Dimensiones Promedio de Imágenes')    # titulo
            ax4.set_ylabel('Píxeles')                          # unidades
            
            # grafica 5: estadisticas de aspect ratio en formato texto
            ax5 = axes[1, 1]
            ar_stats = self.stats['image_properties']['aspect_ratio_stats']  # estadisticas ratio
            
            # muestra metricas clave como texto formateado
            ax5.text(0.1, 0.8, f"Aspect Ratio Promedio: {ar_stats['mean']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.6, f"Desviación Estándar: {ar_stats['std']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.4, f"Rango: {ar_stats['min']:.2f} - {ar_stats['max']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Estadísticas de Aspect Ratio')  # titulo
            ax5.axis('off')  # oculta ejes para presentacion limpia
            
            # grafica 6: calidad del dataset con proporcion validas vs corruptas
            ax6 = axes[1, 2]
            total_analyzed = self.stats['image_properties']['total_analyzed']  # total analizado
            corrupted = self.stats['image_properties']['corrupted_count']     # archivos corruptos
            valid = total_analyzed - corrupted                               # archivos validos
            
            # datos para grafica de torta de calidad
            quality_data = ['Válidas', 'Corruptas']         # etiquetas
            quality_counts = [valid, corrupted]            # conteos
            quality_colors = ['#90EE90', '#FFB6C1']        # colores semanticos
            
            # crea grafica de torta para mostrar proporcion de calidad
            ax6.pie(quality_counts, labels=quality_data, autopct='%1.1f%%', 
                   colors=quality_colors, startangle=90)
            ax6.set_title('Calidad de Imágenes Muestra')  # titulo
        
        # ajusta espaciado y guarda reporte como imagen de alta calidad
        plt.tight_layout()  # optimiza espaciado entre subplots
        plt.savefig(self.dataset_path / 'dataset_analysis_report.png', 
                   dpi=300, bbox_inches='tight')  # exporta en alta resolucion
        plt.show()  # muestra en pantalla
        
        # confirma ubicacion del archivo generado
        print(f"✅ Reporte guardado en: {self.dataset_path / 'dataset_analysis_report.png'}")
    
    def generate_recommendations(self):
        """
        genera recomendaciones inteligentes para optimizar el entrenamiento
        analiza metricas del dataset para sugerir mejores practicas
        
        funcionalidades:
        - evaluacion del balance entre clases principales
        - recomendaciones de tecnicas de balanceo
        - sugerencias de preprocesamiento optimizado
        - recomendaciones de arquitectura de modelo
        - configuracion de hiperparametros sugerida
        """
        print("\n💡 RECOMENDACIONES PARA EL MODELO:")
        print("="*50)
        
        # analiza balance de clases y sugiere correcciones si es necesario
        ratio = self.stats['class_balance']['ratio']  # ratio perros/no-perros
        
        # verifica si existe desbalance significativo entre clases
        if ratio > 2 or ratio < 0.5:  # umbral de desbalance critico
            print(f"⚠️  DESBALANCE DE CLASES detectado ratio: {ratio:.2f}")
            print("   → Usar técnicas de balanceo oversampling, undersampling, o class weights")
        else:
            print(f"✅ Balance de clases aceptable ratio: {ratio:.2f}")
        
        # evalua tamaño del dataset y sugiere estrategias apropiadas
        total = self.stats['total_images']  # total de imagenes disponibles
        
        # determina si el dataset es suficientemente grande para entrenamiento robusto
        if total < 10000:  # umbral minimo recomendado
            print(f"⚠️  Dataset pequeño {total:,} imágenes")
            print("   → Usar augmentación agresiva de datos")
            print("   → Considerar transfer learning con modelos preentrenados")
        else:
            print(f"✅ Tamaño de dataset adecuado {total:,} imágenes")
        
        # analiza propiedades tecnicas si estan disponibles
        if 'image_properties' in self.stats:
            # evalua tasa de corrupcion de archivos
            corruption_rate = (self.stats['image_properties']['corrupted_count'] / 
                             self.stats['image_properties']['total_analyzed'])
            
            # alerta si hay muchos archivos dañados
            if corruption_rate > 0.01:  # mas del 1% corrupto es preocupante
                print(f"⚠️  Alto porcentaje de imágenes corruptas {corruption_rate*100:.1f}%")
                print("   → Implementar validación robusta de imágenes")
            
            # extrae dimensiones promedio para recomendaciones de preprocesamiento
            avg_width = self.stats['image_properties']['width_stats']['mean']
            avg_height = self.stats['image_properties']['height_stats']['mean']
            
            # seccion de recomendaciones de preprocesamiento optimizado
            print(f"\n📋 PREPROCESAMIENTO RECOMENDADO:")
            print(f"   • Redimensionar a: 224x224 estándar para transfer learning")
            print(f"   • Normalización: ImageNet stats [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]")
            print(f"   • Augmentación: rotación ±15°, flip horizontal, crop aleatorio")
            
        # seccion de recomendaciones de modelo y arquitectura
        print(f"\n🎯 MODELO RECOMENDADO:")
        print(f"   • EfficientNet-B3 o ResNet-50 preentrenado en ImageNet")
        print(f"   • Transfer learning: congelar capas iniciales, fine-tune últimas capas")
        print(f"   • Optimizador: AdamW con learning rate scheduling")
        print(f"   • Loss: BCEWithLogitsLoss con class weights si hay desbalance")
        
    def save_analysis_report(self):
        """
        guarda el reporte completo del analisis en formato json
        exporta todas las estadisticas y metricas recopiladas
        
        funcionalidades:
        - serializa todas las estadisticas del objeto stats
        - formatea json con indentacion legible
        - preserva caracteres unicode para nombres de razas
        - convierte objetos no serializables a strings
        - genera archivo persistente para referencia futura
        """
        # define ruta del archivo de reporte en el directorio del dataset
        report_path = self.dataset_path / 'dataset_analysis_report.json'
        
        # escribe archivo json con configuracion optimizada para legibilidad
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats,           # diccionario completo de estadisticas
                     f,                     # archivo de destino
                     indent=2,              # indentacion para legibilidad
                     ensure_ascii=False,    # preserva caracteres especiales
                     default=str)           # convierte objetos no serializables
        
        # confirma ubicacion del archivo guardado
        print(f"💾 Reporte completo guardado en: {report_path}")
        
    def run_complete_analysis(self):
        """
        ejecuta el flujo completo de analisis del dataset
        orquesta todos los metodos de analisis en secuencia logica
        
        flujo de ejecucion:
        1. analiza estructura de directorios y distribucion de clases
        2. examina propiedades tecnicas de imagenes mediante muestreo
        3. genera visualizaciones comprehensivas del analisis
        4. produce recomendaciones inteligentes para entrenamiento
        5. guarda reporte completo en formato json persistente
        
        ideal para:
        - evaluacion inicial de datasets nuevos
        - auditorias de calidad de datos
        - planificacion de estrategias de entrenamiento
        - documentacion de caracteristicas del dataset
        """
        print("🚀 Iniciando análisis completo del dataset...")
        print("="*60)
        
        # paso 1: analiza estructura de directorios y cuenta imagenes por categoria
        self.analyze_dataset_structure()
        
        # paso 2: examina propiedades tecnicas con muestra representativa
        self.analyze_image_properties(sample_size=2000)  # muestra de 2000 imagenes
        
        # paso 3: genera graficas y visualizaciones del analisis
        self.create_visualization_report()
        
        # paso 4: produce recomendaciones basadas en hallazgos
        self.generate_recommendations()
        
        # paso 5: exporta todos los resultados a archivo json
        self.save_analysis_report()
        
        # confirma finalizacion exitosa del analisis completo
        print("\n🎉 ¡Análisis completado exitosamente!")

# bloque de ejecucion principal cuando se ejecuta directamente el script
if __name__ == "__main__":
    # configuracion de rutas del sistema
    # ruta al directorio principal que contiene YESDOG y NODOG
    dataset_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS"
    
    # crea instancia del analizador con la ruta especificada
    analyzer = DatasetAnalyzer(dataset_path)
    
    # ejecuta el analisis completo automatizado
    analyzer.run_complete_analysis()