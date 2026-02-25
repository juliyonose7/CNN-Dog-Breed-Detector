"""
analizador of datasets for classification binaria dog vs no-dog
script especializado for analizar estructura, calidad and distribution
of images en dataset of training, optimized for hardware amd

funcionalidades principales:
- analisis of estructura of directories and distribution of classes
- estadisticas of images: resolucion, format, calidad
- detection of desbalance entre classes
- validation of integridad of files
- generacion of reportes visuales and metricas
- identificacion of problemas potenciales en data
- recomendaciones for improvement of the dataset
"""

# imports of the system operating and handling of files
import os                    # operations of the system operating
import json                  # handling of data json
from pathlib import Path     # handling moderno of paths
from collections import defaultdict, Counter  # estructuras of data especializadas
import warnings              # control of advertencias
warnings.filterwarnings('ignore')  # suprime advertencias no criticas

# imports of processing of images
import cv2                   # computer vision opencv
import numpy as np           # operations numericas
from tqdm import tqdm        # barras of progress

# imports of analisis of data and visualizacion
import pandas as pd          # manipulacion of dataframes
import matplotlib.pyplot as plt  # graficas and plots
import seaborn as sns        # visualizaciones estadisticas avanzadas

# class main for analisis complete of datasets of images
# proporciona herramientas for evaluar calidad and estructura of data
class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        """
initializes analizador with ruta to the dataset main
        
parameters:
- dataset_path: ruta to the directory raiz that contiene YESDOG and NODOG
        """
        self.dataset_path = Path(dataset_path)     # ruta main of the dataset
        self.yesdog_path = self.dataset_path / "YESDOG"  # subdirectory of dogs
        self.nodog_path = self.dataset_path / "NODOG"    # subdirectory of no-dogs
        self.stats = {}                            # diccionario for almacenar estadisticas
        
        # extensiones of image soportadas for validation
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
    def analyze_dataset_structure(self):
        """analiza the estructura complete of the dataset and distribution of classes"""
        print("🔍 Analizando estructura del dataset...")
        
        # analisis of the categoria YESDOG with all the breeds of dogs
        dog_breeds = []           # list for almacenar info of cada breed
        dog_image_count = 0       # contador total of images of dogs
        
        # itera about cada subdirectory of breed en YESDOG
        for breed_folder in self.yesdog_path.iterdir():
            if breed_folder.is_dir():  # verifies that sea directory valido
                breed_name = breed_folder.name  # name of the breed
                images = self._count_images_in_folder(breed_folder)  # cuenta images
                
                # almacena informacion of the breed
                dog_breeds.append({
                    'breed': breed_name,
                    'folder': str(breed_folder),
                    'image_count': images
                })
                dog_image_count += images  # suma to the total
        
        # analisis of the categoria NODOG with objetos that no son dogs
        nodog_categories = []     # list for almacenar info of cada categoria
        nodog_image_count = 0     # contador total of images no-dogs
        
        # itera about cada subdirectory of categoria en NODOG
        for category_folder in self.nodog_path.iterdir():
            if category_folder.is_dir():  # verifies that sea directory valido
                category_name = category_folder.name  # name of the categoria
                images = self._count_images_in_folder(category_folder)  # cuenta images
                
                # almacena informacion of the categoria
                nodog_categories.append({
                    'category': category_name,
                    'folder': str(category_folder),
                    'image_count': images
                })
                nodog_image_count += images  # suma to the total
        
        # almacena estadisticas completas en the objeto main
        self.stats['dog_breeds'] = dog_breeds               # list of breeds with conteos
        self.stats['nodog_categories'] = nodog_categories   # list of categorias with conteos
        self.stats['total_dog_images'] = dog_image_count    # total images of dogs
        self.stats['total_nodog_images'] = nodog_image_count # total images no-dogs
        self.stats['total_images'] = dog_image_count + nodog_image_count  # total general
        
        # calcula metricas of balance entre classes principales
        self.stats['class_balance'] = {
            'dogs': dog_image_count,      # cantidad absoluta dogs
            'no_dogs': nodog_image_count, # cantidad absoluta no-dogs
            'ratio': dog_image_count / max(nodog_image_count, 1)  # ratio evitando division for cero
        }
        
        # imprime resumen ejecutivo of the analisis
        print(f"✅ Análisis completado:")
        print(f"   - Razas de perros: {len(dog_breeds)}")
        print(f"   - Categorías no-perro: {len(nodog_categories)}")
        print(f"   - Total imágenes perros: {dog_image_count:,}")
        print(f"   - Total imágenes no-perros: {nodog_image_count:,}")
        print(f"   - Ratio perros/no-perros: {self.stats['class_balance']['ratio']:.2f}")
        
    def _count_images_in_folder(self, folder_path: Path) -> int:
        """
cuenta images validas en a directory especifico
filtra only files with extensiones of image reconocidas
        
parameters:
- folder_path: ruta to the directory a analizar
        
retorna:
- number entero of images validas encontradas
        """
        count = 0  # initializes contador
        
        # examina cada file en the directory
        for file in folder_path.iterdir():
            # verifies that sea file and tenga extension of image valid
            if file.is_file() and file.suffix.lower() in self.image_extensions:
                count += 1  # incrementa contador if es image valid
                    
        return count  # retorna total of images encontradas
    
    def analyze_image_properties(self, sample_size: int = 1000):
        """
analiza propiedades tecnicas of images mediante muestreo estadistico
examina dimensiones, calidad and caracteristicas visuales of the images
        
parameters:
- sample_size: number total of images a muestrear of the dataset
        
funcionalidades:
- distribution of dimensiones: ancho, alto, canales of color
Technical documentation in English.
- detection of files corruptos or no legibles
- analisis of proporciones and resolucion of images
        """
        print(f"📊 Analizando propiedades de imágenes muestra de {sample_size}...")
        
        # diccionario for recopilar all the propiedades medidas
        image_properties = {
            'widths': [],         # list of anchos en pixeles
            'heights': [],        # list of alturas en pixeles
            'channels': [],       # list of number of canales of color
            'file_sizes': [],     # Implementation note.
            'corrupted': [],      # Implementation note.
            'aspect_ratios': []   # list of proporciones ancho/alto
        }
        
        # gets samples balanceadas of ambas classes principales
        dog_samples = self._sample_images_from_class('dog', sample_size // 2)      # mitad dogs
        nodog_samples = self._sample_images_from_class('nodog', sample_size // 2)  # mitad no-dogs
        
        # combina all the samples for analisis unificado
        all_samples = dog_samples + nodog_samples
        
        # procesa cada image individual with bar of progress
        for img_path, label in tqdm(all_samples, desc="Analizando imágenes"):
            try:
                # load image usando opencv for analisis
                img = cv2.imread(str(img_path))
                
                # verifies if the image se cargo correctamente
                if img is None:
                    image_properties['corrupted'].append(str(img_path))  # marca como corrupta
                    continue  # salta to the siguiente file
                
                # extrae dimensiones basicas of the image
                h, w, c = img.shape  # alto, ancho, canales
                image_properties['heights'].append(h)      # almacena altura
                image_properties['widths'].append(w)       # almacena ancho
                image_properties['channels'].append(c)     # almacena canales
                image_properties['aspect_ratios'].append(w/h)  # calcula and almacena ratio
                
                # Implementation note.
                file_size = Path(img_path).stat().st_size / 1024  # convierte bytes a KB
                image_properties['file_sizes'].append(file_size)  # Implementation note.
                
            except Exception as e:
                # Implementation note.
                image_properties['corrupted'].append(str(img_path))  # marca como corrupta
        
        # calcula estadisticas descriptivas completas of the propiedades
        self.stats['image_properties'] = {
            'width_stats': {               # estadisticas of ancho
                'mean': np.mean(image_properties['widths']),    # average
                'std': np.std(image_properties['widths']),      # deviation estandar
                'min': np.min(image_properties['widths']),      # minimo
                'max': np.max(image_properties['widths']),      # maximo
                'median': np.median(image_properties['widths']) # mediana
            },
            'height_stats': {              # estadisticas of altura
                'mean': np.mean(image_properties['heights']),   # average
                'std': np.std(image_properties['heights']),     # deviation estandar
                'min': np.min(image_properties['heights']),     # minimo
                'max': np.max(image_properties['heights']),     # maximo
                'median': np.median(image_properties['heights'])# mediana
            },
            'aspect_ratio_stats': {        # estadisticas of proporcion
                'mean': np.mean(image_properties['aspect_ratios']),   # average
                'std': np.std(image_properties['aspect_ratios']),     # deviation
                'min': np.min(image_properties['aspect_ratios']),     # more cuadrada
                'max': np.max(image_properties['aspect_ratios'])      # more rectangular
            },
            'file_size_stats': {           # Implementation note.
                'mean_kb': np.mean(image_properties['file_sizes']),   # average en KB
                'median_kb': np.median(image_properties['file_sizes']),# mediana en KB
                'min_kb': np.min(image_properties['file_sizes']),     # minimo en KB
                'max_kb': np.max(image_properties['file_sizes'])      # maximo en KB
            },
            'corrupted_count': len(image_properties['corrupted']),   # total files corruptos
            'total_analyzed': len(all_samples)                       # total images analizadas
        }
        
        # imprime resumen ejecutivo of the analisis of propiedades
        print(f"✅ Propiedades analizadas:")
        print(f"   - Imágenes corruptas: {len(image_properties['corrupted'])}")
        print(f"   - Dimensión promedio: {self.stats['image_properties']['width_stats']['mean']:.0f}x{self.stats['image_properties']['height_stats']['mean']:.0f}")
        print(f"   - Tamaño promedio: {self.stats['image_properties']['file_size_stats']['mean_kb']:.1f} KB")
        
    def _sample_images_from_class(self, class_type: str, sample_size: int):
        """
muestrea images of a class especifica of manera proporcional
distribuye the muestreo equitativamente entre subcategorias
        
parameters:
- class_type: tipo of class a muestrear 'dog' or 'nodog'
- sample_size: number total of images a get
        
retorna:
- list of tuplas ruta_imagen, etiqueta_clase
        
strategy of muestreo:
- muestreo proporcional for subcategoria for evitar sesgo
- distribution equitativa entre breeds or categorias
- selection aleatoria dentro of cada subcategoria
        """
        images = []  # list for almacenar images muestreadas
        
        # determina directories segun the tipo of class solicitado
        if class_type == 'dog':
            # gets all the folders of breeds of dogs
            folders = [breed['folder'] for breed in self.stats['dog_breeds']]
        else:
            # gets all the folders of categorias no-dogs
            folders = [cat['folder'] for cat in self.stats['nodog_categories']]
        
        # procesa cada subcategoria individualmente
        for folder in folders:
            folder_path = Path(folder)  # convierte a objeto Path
            folder_images = []          # images encontradas en this folder
            
            # recopila all the images validas of the folder actual
            for file in folder_path.iterdir():
                # verifies that sea file and tenga extension of image
                if file.is_file() and file.suffix.lower() in self.image_extensions:
                    folder_images.append((file, class_type))  # adds tupla image-label
            
            # realiza muestreo proporcional if hay images disponibles
            if folder_images:
                # calcula number of samples for this folder
                n_samples = min(len(folder_images), max(1, sample_size // len(folders)))
                
                # selecciona indices random without reemplazo
                sampled = np.random.choice(len(folder_images), 
                                         size=min(n_samples, len(folder_images)), 
                                         replace=False)  # avoids duplicados
                
                # adds images selected a the list final
                for idx in sampled:
                    images.append(folder_images[idx])
        
        return images  # retorna list complete of images muestreadas
    
    def create_visualization_report(self):
        """
creates visualizaciones comprehensivas of the analisis of the dataset
genera graficas for entender distribution and caracteristicas of data
        
funcionalidades:
- graficas of distribution of classes principales
- analisis visual of breeds and categorias more representadas
- histogramas of propiedades tecnicas of images
- mapas of calor and correlaciones entre variables
- reportes visuales exportables en format PNG
        """
        print("📈 Creando reporte visual...")
        
        # configura figura main with multiples subplots organizados
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # grid 2x3 for 6 graficas
        fig.suptitle('Análisis del Dataset PERRO vs NO-PERRO', fontsize=16, fontweight='bold')
        
        # plot 1: distribution of classes principales with plot of torta
        ax1 = axes[0, 0]
        classes = ['Perros', 'No-Perros']                    # labels of classes
        counts = [self.stats['total_dog_images'], self.stats['total_nodog_images']]  # conteos
        colors = ['# FF6B6B', '#4ECDC4'] # colores distintivos
        
        # creates plot of torta with porcentajes automaticos
        wedges, texts, autotexts = ax1.pie(counts, labels=classes, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('Distribución de Clases')  # titulo descriptivo
        
        # plot 2: top 10 breeds of dogs with more images disponibles
        ax2 = axes[0, 1]
        dog_df = pd.DataFrame(self.stats['dog_breeds'])      # convierte a dataframe
        top_breeds = dog_df.nlargest(10, 'image_count')     # gets top 10
        
        # extrae names of breeds without prefijos tecnicos
        breed_names = [breed.split('-')[-1] for breed in top_breeds['breed']]
        
        # creates plot of barras horizontales for best legibilidad
        ax2.barh(breed_names, top_breeds['image_count'], color='# FF6B6B', alpha=0.7)
        ax2.set_title('Top 10 Razas más imágenes')     # titulo descriptivo
        ax2.set_xlabel('Número de imágenes')            # label eje x
        
        # plot 3: categorias no-dog with mayor representacion
        ax3 = axes[0, 2]
        nodog_df = pd.DataFrame(self.stats['nodog_categories'])  # convierte a dataframe
        
        # limpia names of categorias removiendo sufijos tecnicos
        cat_names = [cat.replace('_final', '') for cat in nodog_df['category']]
        
        # creates plot of barras verticales for categorias no-dog
        ax3.bar(range(len(cat_names)), nodog_df['image_count'], color='# 4ECDC4', alpha=0.7)
        ax3.set_title('Categorías No-Perro')         # titulo descriptivo
        ax3.set_xlabel('Categoría')                   # label eje x
        ax3.set_ylabel('Número de imágenes')        # label eje and
        ax3.set_xticks(range(len(cat_names)))       # posiciones of labels
        ax3.set_xticklabels(cat_names, rotation=45, ha='right')  # labels rotadas
        
        # graficas 4-6: analisis of propiedades tecnicas if estan disponibles
        if 'image_properties' in self.stats:
            # Implementation note.
            ax4 = axes[1, 0]
            width_stats = self.stats['image_properties']['width_stats']   # estadisticas ancho
            height_stats = self.stats['image_properties']['height_stats'] # estadisticas alto
            
            # prepara data for plot of barras with deviation estandar
            dimensions = ['Ancho', 'Alto']                              # labels
            means = [width_stats['mean'], height_stats['mean']]         # promedios
            stds = [width_stats['std'], height_stats['std']]            # desviaciones
            
            # Implementation note.
            ax4.bar(dimensions, means, yerr=stds, capsize=5, 
                   color=['# 95E1D3', '#F38BA8'], alpha=0.7)
            ax4.set_title('Dimensiones Promedio de Imágenes')    # titulo
            ax4.set_ylabel('Píxeles')                          # unidades
            
            # plot 5: estadisticas of aspect ratio en format texto
            ax5 = axes[1, 1]
            ar_stats = self.stats['image_properties']['aspect_ratio_stats']  # estadisticas ratio
            
            # shows metricas clave como texto formateado
            ax5.text(0.1, 0.8, f"Aspect Ratio Promedio: {ar_stats['mean']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.6, f"Desviación Estándar: {ar_stats['std']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.text(0.1, 0.4, f"Rango: {ar_stats['min']:.2f} - {ar_stats['max']:.2f}", 
                    fontsize=12, transform=ax5.transAxes)
            ax5.set_title('Estadísticas de Aspect Ratio')  # titulo
            ax5.axis('off')  # hidden ejes for presentacion limpia
            
            # plot 6: calidad of the dataset with proporcion validas vs corrupted
            ax6 = axes[1, 2]
            total_analyzed = self.stats['image_properties']['total_analyzed']  # total analizado
            corrupted = self.stats['image_properties']['corrupted_count']     # files corruptos
            valid = total_analyzed - corrupted                               # files valid
            
            # data for plot of torta of calidad
            quality_data = ['Válidas', 'Corruptas']         # labels
            quality_counts = [valid, corrupted]            # conteos
            quality_colors = ['# 90EE90', '#FFB6C1'] # colores semanticos
            
            # creates plot of torta for show proporcion of calidad
            ax6.pie(quality_counts, labels=quality_data, autopct='%1.1f%%', 
                   colors=quality_colors, startangle=90)
            ax6.set_title('Calidad de Imágenes Muestra')  # titulo
        
        # ajusta espaciado and guarda reporte como image of alta calidad
        plt.tight_layout()  # optimiza espaciado entre subplots
        plt.savefig(self.dataset_path / 'dataset_analysis_report.png', 
                   dpi=300, bbox_inches='tight')  # exporta en alta resolucion
        plt.show()  # shows en pantalla
        
        # confirma ubicacion of the file generado
        print(f"✅ Reporte guardado en: {self.dataset_path / 'dataset_analysis_report.png'}")
    
    def generate_recommendations(self):
        """
genera recomendaciones inteligentes for optimizar the training
analiza metricas of the dataset for sugerir best practicas
        
funcionalidades:
- evaluacion of the balance entre classes principales
- recomendaciones of tecnicas of balanceo
- sugerencias of preprocesamiento optimized
- recomendaciones of arquitectura of model
- configuration of hiperparametros sugerida
        """
        print("\n💡 RECOMENDACIONES PARA EL MODELO:")
        print("="*50)
        
        # analiza balance of classes and sugiere correcciones if es necesario
        ratio = self.stats['class_balance']['ratio']  # ratio dogs/no-dogs
        
        # verifies if exists desbalance significativo entre classes
        if ratio > 2 or ratio < 0.5:  # threshold of desbalance critico
            print(f"⚠️  DESBALANCE DE CLASES detectado ratio: {ratio:.2f}")
            print("   → Usar técnicas de balanceo oversampling, undersampling, o class weights")
        else:
            print(f"✅ Balance de clases aceptable ratio: {ratio:.2f}")
        
        # Implementation note.
        total = self.stats['total_images']  # total of images disponibles
        
        # determina if the dataset es suficientemente grande for training robusto
        if total < 10000:  # threshold minimo recomendado
            print(f"⚠️  Dataset pequeño {total:,} imágenes")
            print("   → Usar augmentación agresiva de datos")
            print("   → Considerar transfer learning con modelos preentrenados")
        else:
            print(f"✅ Tamaño de dataset adecuado {total:,} imágenes")
        
        # analiza propiedades tecnicas if estan disponibles
        if 'image_properties' in self.stats:
            # evalua tasa of corrupcion of files
            corruption_rate = (self.stats['image_properties']['corrupted_count'] / 
                             self.stats['image_properties']['total_analyzed'])
            
            # Implementation note.
            if corruption_rate > 0.01:  # more of the 1% corrupto es preocupante
                print(f"⚠️  Alto porcentaje de imágenes corruptas {corruption_rate*100:.1f}%")
                print("   → Implementar validación robusta de imágenes")
            
            # extrae dimensiones average for recomendaciones of preprocesamiento
            avg_width = self.stats['image_properties']['width_stats']['mean']
            avg_height = self.stats['image_properties']['height_stats']['mean']
            
            # seccion of recomendaciones of preprocesamiento optimized
            print(f"\n📋 PREPROCESAMIENTO RECOMENDADO:")
            print(f"   • Redimensionar a: 224x224 estándar para transfer learning")
            print(f"   • Normalización: ImageNet stats [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]")
            print(f"   • Augmentación: rotación ±15°, flip horizontal, crop aleatorio")
            
        # seccion of recomendaciones of model and arquitectura
        print(f"\n🎯 MODELO RECOMENDADO:")
        print(f"   • EfficientNet-B3 o ResNet-50 preentrenado en ImageNet")
        print(f"   • Transfer learning: congelar capas iniciales, fine-tune últimas capas")
        print(f"   • Optimizador: AdamW con learning rate scheduling")
        print(f"   • Loss: BCEWithLogitsLoss con class weights si hay desbalance")
        
    def save_analysis_report(self):
        """
guarda the reporte complete of the analisis en format json
exporta all the estadisticas and metricas recopiladas
        
funcionalidades:
- serializa all the estadisticas of the objeto stats
- formatea json with indentacion readable
- preserva caracteres unicode for names of breeds
- convierte objetos no serializables a strings
- genera file persistente for referencia futura
        """
        # define ruta of the file of reporte en the directory of the dataset
        report_path = self.dataset_path / 'dataset_analysis_report.json'
        
        # escribe file json with configuration optimizada for legibilidad
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats,           # diccionario complete of estadisticas
                     f,                     # file of destino
                     indent=2,              # indentacion for legibilidad
                     ensure_ascii=False,    # preserva caracteres especiales
                     default=str)           # convierte objetos no serializables
        
        # confirma ubicacion of the file saved
        print(f"💾 Reporte completo guardado en: {report_path}")
        
    def run_complete_analysis(self):
        """
ejecuta the flujo complete of analisis of the dataset
orquesta all the methods of analisis en secuencia logica
        
flujo of ejecucion:
1. analiza estructura of directories and distribution of classes
2. examina propiedades tecnicas of images mediante muestreo
3. genera visualizaciones comprehensivas of the analisis
4. produce recomendaciones inteligentes for training
5. guarda reporte complete en format json persistente
        
ideal for:
- evaluacion initial of datasets nuevos
- auditorias of calidad of data
- planificacion of estrategias of training
- documentacion of caracteristicas of the dataset
        """
        print("🚀 Iniciando análisis completo del dataset...")
        print("="*60)
        
        # paso 1: analiza estructura of directories and cuenta images for categoria
        self.analyze_dataset_structure()
        
        # paso 2: examina propiedades tecnicas with shows representativa
        self.analyze_image_properties(sample_size=2000)  # shows of 2000 images
        
        # paso 3: genera graficas and visualizaciones of the analisis
        self.create_visualization_report()
        
        # paso 4: produce recomendaciones basadas en hallazgos
        self.generate_recommendations()
        
        # paso 5: exporta all the resultados a file json
        self.save_analysis_report()
        
        # confirma finalizacion exitosa of the analisis complete
        print("\n🎉 ¡Análisis completado exitosamente!")

# bloque of ejecucion main when se ejecuta directly the script
if __name__ == "__main__":
    # configuration of paths of the system
    # ruta to the directory main that contiene YESDOG and NODOG
    dataset_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG\DATASETS"
    
    # creates instancia of the analizador with the ruta especificada
    analyzer = DatasetAnalyzer(dataset_path)
    
    # ejecuta the analisis complete automatizado
    analyzer.run_complete_analysis()