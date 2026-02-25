#!/usr/bin/env python3
"""
🔍 ANÁLISIS DE SESGOS EN EL MODELO DE CLASIFICACIÓN DE PERROS
============================================================

Este script analiza diferentes tipos de sesgos en el sistema de clasificación:
1. Sesgo de representación (dataset)
2. Sesgo demográfico/geográfico 
3. Sesgo de popularidad cultural
4. Sesgo técnico/arquitectural
5. Sesgo de evaluación

Autor: Sistema IA
Fecha: 2024
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class BiasDetectionAnalyzer:
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.breed_data_path = self.workspace_path / "breed_processed_data" / "train"
        self.yesdog_path = self.workspace_path / "DATASETS" / "YESDOG"
        
        # Información geográfica y cultural de razas
        self.breed_geography = {
            # Razas Europeas
            'German_shepherd': {'region': 'Europe', 'country': 'Germany', 'popularity': 'very_high'},
            'Rottweiler': {'region': 'Europe', 'country': 'Germany', 'popularity': 'high'},
            'Doberman': {'region': 'Europe', 'country': 'Germany', 'popularity': 'high'},
            'Great_Dane': {'region': 'Europe', 'country': 'Germany', 'popularity': 'medium'},
            'Weimaraner': {'region': 'Europe', 'country': 'Germany', 'popularity': 'medium'},
            'Scottish_deerhound': {'region': 'Europe', 'country': 'Scotland', 'popularity': 'low'},
            'Border_collie': {'region': 'Europe', 'country': 'UK', 'popularity': 'very_high'},
            'English_setter': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Irish_setter': {'region': 'Europe', 'country': 'Ireland', 'popularity': 'medium'},
            'Airedale': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Norfolk_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Norwich_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Yorkshire_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'very_high'},
            'Bedlington_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Border_terrier': {'region': 'Europe', 'country': 'UK', 'popularity': 'medium'},
            'Kerry_blue_terrier': {'region': 'Europe', 'country': 'Ireland', 'popularity': 'low'},
            'Lakeland_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'Sealyham_terrier': {'region': 'Europe', 'country': 'Wales', 'popularity': 'low'},
            'Saint_Bernard': {'region': 'Europe', 'country': 'Switzerland', 'popularity': 'high'},
            'Bernese_mountain_dog': {'region': 'Europe', 'country': 'Switzerland', 'popularity': 'high'},
            'EntleBucher': {'region': 'Europe', 'country': 'Switzerland', 'popularity': 'low'},
            'Great_Pyrenees': {'region': 'Europe', 'country': 'France', 'popularity': 'medium'},
            'Newfoundland': {'region': 'North America', 'country': 'Canada', 'popularity': 'medium'},
            'Italian_greyhound': {'region': 'Europe', 'country': 'Italy', 'popularity': 'medium'},
            'Norwegian_elkhound': {'region': 'Europe', 'country': 'Norway', 'popularity': 'medium'},
            
            # Razas Asiáticas
            'Shih-Tzu': {'region': 'Asia', 'country': 'China', 'popularity': 'very_high'},
            'Lhasa': {'region': 'Asia', 'country': 'Tibet', 'popularity': 'medium'},
            'Pug': {'region': 'Asia', 'country': 'China', 'popularity': 'very_high'},
            'Japanese_spaniel': {'region': 'Asia', 'country': 'Japan', 'popularity': 'medium'},
            'Tibetan_terrier': {'region': 'Asia', 'country': 'Tibet', 'popularity': 'low'},
            'chow': {'region': 'Asia', 'country': 'China', 'popularity': 'medium'},
            'Samoyed': {'region': 'Asia', 'country': 'Russia', 'popularity': 'high'},
            'Siberian_husky': {'region': 'Asia', 'country': 'Russia', 'popularity': 'very_high'},
            
            # Razas Americanas
            'Boston_bull': {'region': 'North America', 'country': 'USA', 'popularity': 'high'},
            'Labrador_retriever': {'region': 'North America', 'country': 'Canada', 'popularity': 'very_high'},
            'Chesapeake_Bay_retriever': {'region': 'North America', 'country': 'USA', 'popularity': 'medium'},
            
            # Razas del Medio Oriente/África
            'Afghan_hound': {'region': 'Middle East', 'country': 'Afghanistan', 'popularity': 'medium'},
            'Saluki': {'region': 'Middle East', 'country': 'Middle East', 'popularity': 'low'},
            'basenji': {'region': 'Africa', 'country': 'Central Africa', 'popularity': 'low'},
            
            # Razas Australianas
            'Australian_terrier': {'region': 'Oceania', 'country': 'Australia', 'popularity': 'medium'},
            
            # Otras razas (clasificación por características)
            'beagle': {'region': 'Europe', 'country': 'England', 'popularity': 'very_high'},
            'basset': {'region': 'Europe', 'country': 'France', 'popularity': 'high'},
            'bloodhound': {'region': 'Europe', 'country': 'Belgium', 'popularity': 'medium'},
            'bluetick': {'region': 'North America', 'country': 'USA', 'popularity': 'low'},
            'whippet': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Ibizan_hound': {'region': 'Europe', 'country': 'Spain', 'popularity': 'low'},
            'Irish_wolfhound': {'region': 'Europe', 'country': 'Ireland', 'popularity': 'medium'},
            'Rhodesian_ridgeback': {'region': 'Africa', 'country': 'Zimbabwe', 'popularity': 'medium'},
            'Maltese_dog': {'region': 'Europe', 'country': 'Malta', 'popularity': 'high'},
            'papillon': {'region': 'Europe', 'country': 'France', 'popularity': 'medium'},
            'Pomeranian': {'region': 'Europe', 'country': 'Germany', 'popularity': 'high'},
            'silky_terrier': {'region': 'Oceania', 'country': 'Australia', 'popularity': 'low'},
            'toy_terrier': {'region': 'Europe', 'country': 'England', 'popularity': 'medium'},
            'Blenheim_spaniel': {'region': 'Europe', 'country': 'England', 'popularity': 'low'},
            'cairn': {'region': 'Europe', 'country': 'Scotland', 'popularity': 'medium'},
            'Dandie_Dinmont': {'region': 'Europe', 'country': 'Scotland', 'popularity': 'low'},
            'malamute': {'region': 'North America', 'country': 'USA', 'popularity': 'medium'},
            'miniature_pinscher': {'region': 'Europe', 'country': 'Germany', 'popularity': 'medium'},
            'Pembroke': {'region': 'Europe', 'country': 'Wales', 'popularity': 'high'},
            'Leonberg': {'region': 'Europe', 'country': 'Germany', 'popularity': 'low'}
        }
        
        # Características físicas que pueden introducir sesgo
        self.breed_characteristics = {
            'size': {
                'toy': ['Japanese_spaniel', 'Maltese_dog', 'papillon', 'Pomeranian', 'silky_terrier', 'toy_terrier', 'Yorkshire_terrier'],
                'small': ['beagle', 'basset', 'cairn', 'Dandie_Dinmont', 'miniature_pinscher', 'Norfolk_terrier', 'Norwich_terrier', 'Pug', 'Shih-Tzu'],
                'medium': ['Australian_terrier', 'Bedlington_terrier', 'Border_terrier', 'Boston_bull', 'chow', 'Pembroke', 'whippet'],
                'large': ['Afghan_hound', 'Airedale', 'bloodhound', 'German_shepherd', 'Labrador_retriever', 'Rottweiler', 'Samoyed'],
                'giant': ['Great_Dane', 'Great_Pyrenees', 'Irish_wolfhound', 'Newfoundland', 'Saint_Bernard']
            },
            'coat_type': {
                'short': ['beagle', 'basset', 'bloodhound', 'Boston_bull', 'German_shepherd', 'Labrador_retriever', 'Pug', 'Rottweiler', 'whippet'],
                'medium': ['Australian_terrier', 'cairn', 'chow', 'Norwegian_elkhound', 'Pembroke', 'Siberian_husky'],
                'long': ['Afghan_hound', 'Bernese_mountain_dog', 'Irish_setter', 'Maltese_dog', 'Newfoundland', 'Shih-Tzu', 'Yorkshire_terrier'],
                'curly': ['Bedlington_terrier', 'Dandie_Dinmont', 'Kerry_blue_terrier', 'Pomeranian'],
                'wire': ['Airedale', 'Border_terrier', 'Lakeland_terrier', 'Norfolk_terrier', 'Norwich_terrier', 'Sealyham_terrier']
            },
            'color_patterns': {
                'solid': ['German_shepherd', 'Rottweiler', 'chow', 'Pug', 'Samoyed'],
                'bicolor': ['Bernese_mountain_dog', 'Boston_bull', 'Border_collie', 'Saint_Bernard'],
                'tricolor': ['beagle', 'basset', 'Airedale', 'Yorkshire_terrier'],
                'spotted': ['bluetick', 'English_setter', 'German_short-haired_pointer'],
                'brindle': ['Boston_bull', 'cairn', 'whippet']
            }
        }
        
    def analyze_dataset_representation_bias(self):
        """Analiza sesgo de representación en el dataset"""
        print("🔍 ANÁLISIS DE SESGO DE REPRESENTACIÓN")
        print("="*60)
        
        if not self.breed_data_path.exists():
            print("❌ No se encontró el dataset de razas balanceadas")
            return None
            
        breed_stats = {}
        total_images = 0
        
        # Contar imágenes por raza
        for breed_dir in self.breed_data_path.iterdir():
            if breed_dir.is_dir():
                images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.png"))
                count = len(images)
                breed_stats[breed_dir.name] = count
                total_images += count
        
        if not breed_stats:
            print("❌ No se encontraron datos de razas")
            return None
            
        # Estadísticas básicas
        counts = list(breed_stats.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        cv = std_count / mean_count  # Coeficiente de variación
        
        print(f"📊 Estadísticas del Dataset:")
        print(f"   Total de razas: {len(breed_stats)}")
        print(f"   Total de imágenes: {total_images:,}")
        print(f"   Promedio por raza: {mean_count:.1f}")
        print(f"   Desviación estándar: {std_count:.1f}")
        print(f"   Coeficiente de variación: {cv:.3f}")
        
        # Interpretación del balance
        if cv < 0.05:
            balance_status = "✅ PERFECTAMENTE BALANCEADO"
        elif cv < 0.1:
            balance_status = "✅ MUY BIEN BALANCEADO"
        elif cv < 0.2:
            balance_status = "⚠️ MODERADAMENTE BALANCEADO"
        else:
            balance_status = "❌ DESBALANCEADO"
            
        print(f"   Estado del balance: {balance_status}")
        
        return {
            'breed_stats': breed_stats,
            'total_images': total_images,
            'mean_count': mean_count,
            'std_count': std_count,
            'cv': cv,
            'balance_status': balance_status
        }
    
    def analyze_geographical_bias(self):
        """Analiza sesgo geográfico/cultural"""
        print("\n🌍 ANÁLISIS DE SESGO GEOGRÁFICO Y CULTURAL")
        print("="*60)
        
        # Obtener razas del dataset
        dataset_breeds = []
        if self.breed_data_path.exists():
            dataset_breeds = [d.name for d in self.breed_data_path.iterdir() if d.is_dir()]
        
        if not dataset_breeds:
            print("❌ No se encontraron razas en el dataset")
            return None
            
        # Analizar distribución geográfica
        regional_distribution = defaultdict(list)
        popularity_distribution = defaultdict(list)
        missing_breeds = []
        
        for breed in dataset_breeds:
            if breed in self.breed_geography:
                info = self.breed_geography[breed]
                regional_distribution[info['region']].append(breed)
                popularity_distribution[info['popularity']].append(breed)
            else:
                missing_breeds.append(breed)
        
        print(f"📍 Distribución por Región:")
        total_classified = sum(len(breeds) for breeds in regional_distribution.values())
        
        for region, breeds in regional_distribution.items():
            percentage = len(breeds) / total_classified * 100
            print(f"   {region:15}: {len(breeds):2d} razas ({percentage:5.1f}%)")
            
        print(f"   Sin clasificar:      {len(missing_breeds):2d} razas")
        
        print(f"\n⭐ Distribución por Popularidad:")
        for popularity, breeds in popularity_distribution.items():
            percentage = len(breeds) / total_classified * 100
            print(f"   {popularity:12}: {len(breeds):2d} razas ({percentage:5.1f}%)")
        
        # Detectar sesgos
        biases_detected = []
        
        # Sesgo regional
        europe_pct = len(regional_distribution.get('Europe', [])) / total_classified * 100
        if europe_pct > 60:
            biases_detected.append(f"⚠️ SESGO EUROPEO: {europe_pct:.1f}% de razas son europeas")
            
        asia_pct = len(regional_distribution.get('Asia', [])) / total_classified * 100
        if asia_pct < 15:
            biases_detected.append(f"⚠️ SUBREPRESENTACIÓN ASIÁTICA: Solo {asia_pct:.1f}% de razas asiáticas")
            
        africa_pct = len(regional_distribution.get('Africa', [])) / total_classified * 100
        if africa_pct < 5:
            biases_detected.append(f"⚠️ SUBREPRESENTACIÓN AFRICANA: Solo {africa_pct:.1f}% de razas africanas")
        
        # Sesgo de popularidad
        very_high_pct = len(popularity_distribution.get('very_high', [])) / total_classified * 100
        low_pct = len(popularity_distribution.get('low', [])) / total_classified * 100
        
        if very_high_pct > 25:
            biases_detected.append(f"⚠️ SESGO HACIA RAZAS POPULARES: {very_high_pct:.1f}% son muy populares")
            
        if low_pct < 20:
            biases_detected.append(f"⚠️ SUBREPRESENTACIÓN DE RAZAS RARAS: Solo {low_pct:.1f}% son poco populares")
        
        print(f"\n🚨 SESGOS DETECTADOS:")
        if biases_detected:
            for bias in biases_detected:
                print(f"   {bias}")
        else:
            print("   ✅ No se detectaron sesgos geográficos/culturales significativos")
        
        return {
            'regional_distribution': dict(regional_distribution),
            'popularity_distribution': dict(popularity_distribution),
            'missing_breeds': missing_breeds,
            'biases_detected': biases_detected,
            'total_classified': total_classified
        }
    
    def analyze_physical_characteristics_bias(self):
        """Analiza sesgo en características físicas"""
        print("\n🐕 ANÁLISIS DE SESGO EN CARACTERÍSTICAS FÍSICAS")
        print("="*60)
        
        # Obtener razas del dataset
        dataset_breeds = []
        if self.breed_data_path.exists():
            dataset_breeds = [d.name for d in self.breed_data_path.iterdir() if d.is_dir()]
        
        if not dataset_breeds:
            return None
            
        # Analizar distribución de tamaños
        size_distribution = defaultdict(list)
        coat_distribution = defaultdict(list)
        color_distribution = defaultdict(list)
        
        for breed in dataset_breeds:
            # Clasificar por tamaño
            for size, breeds in self.breed_characteristics['size'].items():
                if breed in breeds:
                    size_distribution[size].append(breed)
                    break
            
            # Clasificar por tipo de pelaje
            for coat_type, breeds in self.breed_characteristics['coat_type'].items():
                if breed in breeds:
                    coat_distribution[coat_type].append(breed)
                    break
                    
            # Clasificar por patrones de color
            for color_pattern, breeds in self.breed_characteristics['color_patterns'].items():
                if breed in breeds:
                    color_distribution[color_pattern].append(breed)
                    break
        
        print(f"📏 Distribución por Tamaño:")
        total_size_classified = sum(len(breeds) for breeds in size_distribution.values())
        for size, breeds in size_distribution.items():
            percentage = len(breeds) / len(dataset_breeds) * 100
            print(f"   {size:8}: {len(breeds):2d} razas ({percentage:5.1f}%)")
        
        print(f"\n🧥 Distribución por Tipo de Pelaje:")
        for coat_type, breeds in coat_distribution.items():
            percentage = len(breeds) / len(dataset_breeds) * 100
            print(f"   {coat_type:8}: {len(breeds):2d} razas ({percentage:5.1f}%)")
        
        print(f"\n🎨 Distribución por Patrones de Color:")
        for color_pattern, breeds in color_distribution.items():
            percentage = len(breeds) / len(dataset_breeds) * 100
            print(f"   {color_pattern:8}: {len(breeds):2d} razas ({percentage:5.1f}%)")
        
        # Detectar sesgos físicos
        physical_biases = []
        
        # Sesgo de tamaño
        small_breeds = len(size_distribution.get('small', [])) + len(size_distribution.get('toy', []))
        large_breeds = len(size_distribution.get('large', [])) + len(size_distribution.get('giant', []))
        
        if small_breeds > large_breeds * 1.5:
            physical_biases.append(f"⚠️ SESGO HACIA PERROS PEQUEÑOS: {small_breeds} pequeños vs {large_breeds} grandes")
        elif large_breeds > small_breeds * 1.5:
            physical_biases.append(f"⚠️ SESGO HACIA PERROS GRANDES: {large_breeds} grandes vs {small_breeds} pequeños")
        
        # Sesgo de pelaje
        long_coat = len(coat_distribution.get('long', []))
        short_coat = len(coat_distribution.get('short', []))
        
        if long_coat > short_coat * 1.5:
            physical_biases.append(f"⚠️ SESGO HACIA PELO LARGO: {long_coat} pelo largo vs {short_coat} pelo corto")
        elif short_coat > long_coat * 1.5:
            physical_biases.append(f"⚠️ SESGO HACIA PELO CORTO: {short_coat} pelo corto vs {long_coat} pelo largo")
        
        print(f"\n🚨 SESGOS FÍSICOS DETECTADOS:")
        if physical_biases:
            for bias in physical_biases:
                print(f"   {bias}")
        else:
            print("   ✅ No se detectaron sesgos físicos significativos")
        
        return {
            'size_distribution': dict(size_distribution),
            'coat_distribution': dict(coat_distribution),
            'color_distribution': dict(color_distribution),
            'physical_biases': physical_biases
        }
    
    def analyze_model_architecture_bias(self):
        """Analiza posibles sesgos introducidos por la arquitectura del modelo"""
        print("\n🏗️ ANÁLISIS DE SESGO EN ARQUITECTURA DEL MODELO")
        print("="*60)
        
        # Analizar el sistema híbrido
        print("🤖 Sistema Híbrido Actual:")
        print("   1. Modelo Binario: ResNet18 (perro/no perro)")
        print("   2. Modelo Principal: ResNet50 (50 razas)")
        print("   3. Modelo Selectivo: ResNet34 (6 razas problemáticas)")
        
        # Razas en modelo selectivo
        selective_breeds = ['basset', 'beagle', 'Labrador_retriever', 'Norwegian_elkhound', 'pug', 'Samoyed']
        
        print(f"\n🎯 Razas con Modelo Especializado:")
        for breed in selective_breeds:
            print(f"   • {breed}")
        
        # Analizar posibles sesgos arquitecturales
        architectural_biases = []
        
        # 1. Sesgo de arquitectura diferente
        architectural_biases.append("⚠️ SESGO ARQUITECTURAL: Diferentes arquitecturas (ResNet18/34/50) pueden tener diferentes capacidades")
        
        # 2. Sesgo de modelo selectivo
        architectural_biases.append("⚠️ SESGO DE ESPECIALIZACIÓN: 6 razas tienen modelo dedicado, ventaja injusta")
        
        # 3. Sesgo de temperatura scaling
        architectural_biases.append("⚠️ SESGO DE CALIBRACIÓN: Temperature scaling puede favorecer ciertas predicciones")
        
        # 4. Analizar si las razas selectivas tienen características comunes
        print(f"\n🔍 Análisis de Razas Selectivas:")
        
        selective_characteristics = {
            'regions': [],
            'sizes': [],
            'popularities': []
        }
        
        for breed in selective_breeds:
            if breed in self.breed_geography:
                info = self.breed_geography[breed]
                selective_characteristics['regions'].append(info['region'])
                selective_characteristics['popularities'].append(info['popularity'])
        
        # Verificar si hay patrones en las razas selectivas
        region_counter = Counter(selective_characteristics['regions'])
        popularity_counter = Counter(selective_characteristics['popularities'])
        
        print(f"   Distribución regional: {dict(region_counter)}")
        print(f"   Distribución popularidad: {dict(popularity_counter)}")
        
        # Sesgo si las razas selectivas están concentradas geográficamente
        most_common_region = region_counter.most_common(1)[0] if region_counter else None
        if most_common_region and most_common_region[1] >= 4:
            architectural_biases.append(f"⚠️ SESGO GEOGRÁFICO EN SELECTIVAS: {most_common_region[1]}/6 razas son de {most_common_region[0]}")
        
        # Sesgo si las razas selectivas tienen popularidades similares
        most_common_popularity = popularity_counter.most_common(1)[0] if popularity_counter else None
        if most_common_popularity and most_common_popularity[1] >= 4:
            architectural_biases.append(f"⚠️ SESGO DE POPULARIDAD EN SELECTIVAS: {most_common_popularity[1]}/6 razas son {most_common_popularity[0]}")
        
        print(f"\n🚨 SESGOS ARQUITECTURALES DETECTADOS:")
        for bias in architectural_biases:
            print(f"   {bias}")
        
        return {
            'selective_breeds': selective_breeds,
            'architectural_biases': architectural_biases,
            'selective_characteristics': selective_characteristics
        }
    
    def analyze_evaluation_bias(self):
        """Analiza posibles sesgos en la evaluación del modelo"""
        print("\n📊 ANÁLISIS DE SESGO EN EVALUACIÓN")
        print("="*60)
        
        evaluation_biases = []
        
        print("🎯 Métricas de Evaluación Actuales:")
        print("   • Accuracy general: 88.14% (modelo principal)")
        print("   • Accuracy selectivo: 95.15% (6 razas)")
        print("   • Temperature scaling: 10.0 (calibración)")
        print("   • Umbral de confianza: 0.35")
        
        # Posibles sesgos en evaluación
        evaluation_biases.extend([
            "⚠️ SESGO DE MÉTRICA ÚNICA: Solo se usa accuracy, ignora precision/recall por clase",
            "⚠️ SESGO DE DATASET DE PRUEBA: ¿Es representativo de casos reales?",
            "⚠️ SESGO DE CALIBRACIÓN: Temperature scaling puede enmascarar problemas reales",
            "⚠️ SESGO DE UMBRAL: Umbral único (0.35) puede no ser óptimo para todas las razas",
            "⚠️ SESGO DE COMPARACIÓN DESIGUAL: Modelo selectivo vs principal no es comparación justa"
        ])
        
        print(f"\n🚨 SESGOS DE EVALUACIÓN DETECTADOS:")
        for bias in evaluation_biases:
            print(f"   {bias}")
        
        return {
            'evaluation_biases': evaluation_biases,
            'current_metrics': {
                'main_accuracy': 88.14,
                'selective_accuracy': 95.15,
                'temperature': 10.0,
                'confidence_threshold': 0.35
            }
        }
    
    def suggest_bias_mitigation_strategies(self, all_analyses):
        """Sugiere estrategias para mitigar los sesgos detectados"""
        print("\n💡 ESTRATEGIAS DE MITIGACIÓN DE SESGOS")
        print("="*60)
        
        strategies = []
        
        # Estrategias para sesgo de representación
        if all_analyses.get('representation'):
            cv = all_analyses['representation']['cv']
            if cv > 0.1:
                strategies.append({
                    'type': 'Representación',
                    'strategy': 'Rebalancear dataset',
                    'description': f'CV={cv:.3f} indica desbalance. Usar data augmentation o resampling.'
                })
        
        # Estrategias para sesgo geográfico
        if all_analyses.get('geographical'):
            biases = all_analyses['geographical']['biases_detected']
            if any('EUROPEO' in bias for bias in biases):
                strategies.append({
                    'type': 'Geográfico',
                    'strategy': 'Diversificación regional',
                    'description': 'Incluir más razas de Asia, África y América para balance global.'
                })
        
        # Estrategias para sesgo arquitectural
        if all_analyses.get('architectural'):
            strategies.extend([
                {
                    'type': 'Arquitectural',
                    'strategy': 'Unificar arquitecturas',
                    'description': 'Usar la misma arquitectura (ej. ResNet50) para todos los modelos.'
                },
                {
                    'type': 'Arquitectural',
                    'strategy': 'Modelo único multi-cabeza',
                    'description': 'Reemplazar sistema híbrido con un modelo único con múltiples salidas.'
                },
                {
                    'type': 'Arquitectural',
                    'strategy': 'Eliminación del modelo selectivo',
                    'description': 'Remover ventaja injusta de las 6 razas con modelo especializado.'
                }
            ])
        
        # Estrategias para sesgo de evaluación
        strategies.extend([
            {
                'type': 'Evaluación',
                'strategy': 'Métricas por clase',
                'description': 'Reportar precision, recall y F1-score para cada raza individual.'
            },
            {
                'type': 'Evaluación',
                'strategy': 'Dataset de prueba estratificado',
                'description': 'Asegurar representación equilibrada en conjunto de prueba.'
            },
            {
                'type': 'Evaluación',
                'strategy': 'Umbrales adaptativos',
                'description': 'Usar umbrales de confianza específicos por raza basados en rendimiento.'
            },
            {
                'type': 'Evaluación',
                'strategy': 'Validación cruzada estratificada',
                'description': 'Usar k-fold estratificado para evaluación más robusta.'
            }
        ])
        
        print("🛠️ ESTRATEGIAS RECOMENDADAS:")
        for i, strategy in enumerate(strategies, 1):
            print(f"\n{i:2d}. [{strategy['type']}] {strategy['strategy']}")
            print(f"    📝 {strategy['description']}")
        
        return strategies
    
    def create_bias_visualization(self, all_analyses):
        """Crea visualizaciones de los análisis de sesgo"""
        print("\n📊 CREANDO VISUALIZACIONES DE SESGO...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        # 1. Distribución regional
        if all_analyses.get('geographical'):
            regional_data = all_analyses['geographical']['regional_distribution']
            if regional_data:
                regions = list(regional_data.keys())
                counts = [len(breeds) for breeds in regional_data.values()]
                
                ax = axes[0]
                bars = ax.bar(regions, counts, color='lightblue', edgecolor='navy')
                ax.set_title('Distribución Geográfica de Razas', fontweight='bold')
                ax.set_ylabel('Número de Razas')
                ax.tick_params(axis='x', rotation=45)
                
                # Agregar valores en las barras
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
        
        # 2. Distribución de popularidad
        if all_analyses.get('geographical'):
            popularity_data = all_analyses['geographical']['popularity_distribution']
            if popularity_data:
                popularities = list(popularity_data.keys())
                counts = [len(breeds) for breeds in popularity_data.values()]
                
                ax = axes[1]
                colors = ['red', 'orange', 'yellow', 'lightgreen'][:len(popularities)]
                bars = ax.bar(popularities, counts, color=colors, edgecolor='black')
                ax.set_title('Distribución de Popularidad de Razas', fontweight='bold')
                ax.set_ylabel('Número de Razas')
                
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
        
        # 3. Distribución de tamaños
        if all_analyses.get('physical'):
            size_data = all_analyses['physical']['size_distribution']
            if size_data:
                sizes = list(size_data.keys())
                counts = [len(breeds) for breeds in size_data.values()]
                
                ax = axes[2]
                bars = ax.bar(sizes, counts, color='lightcoral', edgecolor='darkred')
                ax.set_title('Distribución de Tamaños de Razas', fontweight='bold')
                ax.set_ylabel('Número de Razas')
                
                for bar, count in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
        
        # 4. Balance del dataset
        if all_analyses.get('representation'):
            breed_stats = all_analyses['representation']['breed_stats']
            if breed_stats:
                counts = list(breed_stats.values())
                ax = axes[3]
                ax.hist(counts, bins=15, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
                ax.set_title('Distribución de Imágenes por Raza', fontweight='bold')
                ax.set_xlabel('Número de Imágenes')
                ax.set_ylabel('Número de Razas')
                ax.axvline(np.mean(counts), color='red', linestyle='--', 
                          label=f'Media: {np.mean(counts):.0f}')
                ax.legend()
        
        # 5. Razas selectivas vs principales
        ax = axes[4]
        categories = ['Modelo Principal\n(44 razas)', 'Modelo Selectivo\n(6 razas)']
        accuracies = [88.14, 95.15]
        colors = ['lightblue', 'orange']
        
        bars = ax.bar(categories, accuracies, color=colors, edgecolor='black')
        ax.set_title('Comparación de Accuracies', fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(80, 100)
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Resumen de sesgos
        ax = axes[5]
        ax.axis('off')
        
        # Contar sesgos por categoría
        bias_summary = {
            'Geográfico': len(all_analyses.get('geographical', {}).get('biases_detected', [])),
            'Físico': len(all_analyses.get('physical', {}).get('physical_biases', [])),
            'Arquitectural': len(all_analyses.get('architectural', {}).get('architectural_biases', [])),
            'Evaluación': len(all_analyses.get('evaluation', {}).get('evaluation_biases', []))
        }
        
        summary_text = "🚨 RESUMEN DE SESGOS DETECTADOS\n\n"
        total_biases = 0
        for category, count in bias_summary.items():
            summary_text += f"{category}: {count} sesgos\n"
            total_biases += count
        
        summary_text += f"\nTotal: {total_biases} sesgos detectados"
        
        if total_biases == 0:
            summary_text += "\n\n✅ MODELO LIBRE DE SESGOS"
            color = 'green'
        elif total_biases < 5:
            summary_text += "\n\n⚠️ SESGOS MENORES"
            color = 'orange'
        else:
            summary_text += "\n\n🚨 SESGOS SIGNIFICATIVOS"
            color = 'red'
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.tight_layout()
        plt.savefig('bias_analysis_report.png', dpi=300, bbox_inches='tight')
        print("   ✅ Visualización guardada: bias_analysis_report.png")
        
        return fig
    
    def run_complete_bias_analysis(self):
        """Ejecuta el análisis completo de sesgos"""
        print("🔍 ANÁLISIS COMPLETO DE SESGOS EN EL MODELO")
        print("="*80)
        
        all_analyses = {}
        
        # 1. Análisis de representación
        all_analyses['representation'] = self.analyze_dataset_representation_bias()
        
        # 2. Análisis geográfico/cultural
        all_analyses['geographical'] = self.analyze_geographical_bias()
        
        # 3. Análisis de características físicas
        all_analyses['physical'] = self.analyze_physical_characteristics_bias()
        
        # 4. Análisis arquitectural
        all_analyses['architectural'] = self.analyze_model_architecture_bias()
        
        # 5. Análisis de evaluación
        all_analyses['evaluation'] = self.analyze_evaluation_bias()
        
        # 6. Estrategias de mitigación
        strategies = self.suggest_bias_mitigation_strategies(all_analyses)
        
        # 7. Crear visualizaciones
        fig = self.create_bias_visualization(all_analyses)
        
        # Guardar reporte completo
        self.save_bias_report(all_analyses, strategies)
        
        return {
            'analyses': all_analyses,
            'strategies': strategies,
            'visualization': fig
        }
    
    def save_bias_report(self, analyses, strategies):
        """Guarda un reporte completo del análisis de sesgos"""
        print("\n💾 GUARDANDO REPORTE DE SESGOS...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_info': {
                'architecture': 'Hierarchical (ResNet18 + ResNet50 + ResNet34)',
                'total_breeds': 50,
                'dataset_balance': analyses.get('representation', {}).get('balance_status', 'Unknown')
            },
            'bias_analyses': analyses,
            'mitigation_strategies': strategies,
            'summary': {
                'total_biases_detected': sum([
                    len(analyses.get('geographical', {}).get('biases_detected', [])),
                    len(analyses.get('physical', {}).get('physical_biases', [])),
                    len(analyses.get('architectural', {}).get('architectural_biases', [])),
                    len(analyses.get('evaluation', {}).get('evaluation_biases', []))
                ]),
                'critical_issues': [],
                'recommendations': []
            }
        }
        
        # Identificar issues críticos
        if analyses.get('representation', {}).get('cv', 0) > 0.2:
            report['summary']['critical_issues'].append('Dataset significativamente desbalanceado')
            
        if any('EUROPEO' in bias for bias in analyses.get('geographical', {}).get('biases_detected', [])):
            report['summary']['critical_issues'].append('Sesgo geográfico hacia razas europeas')
            
        # Recomendaciones principales
        report['summary']['recommendations'] = [
            'Unificar arquitectura del modelo para eliminar ventajas injustas',
            'Diversificar dataset con más razas no-europeas',
            'Implementar métricas de evaluación por clase individual',
            'Usar validación cruzada estratificada para evaluación robusta'
        ]
        
        # Guardar como JSON
        with open('bias_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print("   ✅ Reporte guardado: bias_analysis_report.json")
        
        return report

def main():
    """Función principal"""
    workspace_path = r"c:\Users\juliy\OneDrive\Escritorio\NOTDOG YESDOG"
    
    analyzer = BiasDetectionAnalyzer(workspace_path)
    results = analyzer.run_complete_bias_analysis()
    
    return results

if __name__ == "__main__":
    results = main()