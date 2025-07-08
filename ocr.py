#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complet OCR Script with Tesseract
Compatible with Snapdragon X Plus - Windows ARM64

Auteur: Julien Balderiotti
Date: 07/07/2025
"""

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import sys
from pathlib import Path
import time

class TesseractOCR:
    def __init__(self, tesseract_path=None):
        """
        Initialiser Tesseract OCR
        
        Args:
            tesseract_path (str): Chemin vers tesseract.exe (optionnel si dans PATH)
        """
        # Configuration automatique du chemin Tesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Essayer les chemins par défaut
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                'tesseract'  # Si dans PATH
            ]
            
            for path in possible_paths:
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    # Test rapide
                    pytesseract.get_tesseract_version()
                    print(f"✅ Tesseract trouvé : {path}")
                    break
                except:
                    continue
            else:
                raise Exception("❌ Tesseract non trouvé. Vérifiez l'installation.")
        
        # Vérifier les langues disponibles
        try:
            self.available_languages = pytesseract.get_languages()
            print(f"📝 Langues disponibles : {self.available_languages}")
        except:
            self.available_languages = ['eng']
            print("⚠️  Impossible de récupérer la liste des langues, utilisation de 'eng' par défaut")
    
    def preprocess_image(self, image, enhancement_level='medium'):
        """
        Préprocessing intelligent de l'image
        
        Args:
            image: Image PIL ou chemin vers fichier
            enhancement_level: 'light', 'medium', 'strong'
        
        Returns:
            Image PIL préprocessée
        """
        # Charger l'image si c'est un chemin
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convertir en PIL Image si c'est un array numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        print(f"📸 Image originale : {image.size} pixels, mode : {image.mode}")
        
        # Convertir en niveaux de gris
        if image.mode != 'L':
            image = image.convert('L')
        
        # Redimensionner si trop petite (OCR fonctionne mieux sur images plus grandes)
        min_size = 300
        if min(image.size) < min_size:
            ratio = min_size / min(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"🔍 Image redimensionnée : {new_size}")
        
        # Améliorations selon le niveau
        if enhancement_level in ['medium', 'strong']:
            # Améliorer le contraste
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5 if enhancement_level == 'medium' else 2.0)
            
            # Améliorer la netteté
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2 if enhancement_level == 'medium' else 1.5)
        
        if enhancement_level == 'strong':
            # Réduction du bruit avec filtre médian
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Seuillage adaptatif pour améliorer le contraste
            img_array = np.array(image)
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            image = Image.fromarray(img_array)
        
        return image
    
    def extract_text_simple(self, image, lang='eng', config=None):
        """
        Extraction simple de texte
        
        Args:
            image: Image ou chemin vers fichier
            lang: Code langue ('eng', 'fra', 'eng+fra', etc.)
            config: Configuration Tesseract personnalisée
        
        Returns:
            str: Texte extrait
        """
        if config is None:
            config = '--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
        
        # Préprocessing
        processed_image = self.preprocess_image(image)
        
        # Extraction avec gestion d'erreur
        try:
            start_time = time.time()
            text = pytesseract.image_to_string(
                processed_image, 
                lang=lang, 
                config=config
            )
            end_time = time.time()
            
            print(f"⏱️  Temps de traitement : {end_time - start_time:.2f}s")
            return text.strip()
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction : {e}")
            return ""
    
    def extract_text_with_boxes(self, image, lang='eng', confidence_threshold=30):
        """
        Extraction avec coordonnées des boîtes de texte
        
        Args:
            image: Image ou chemin vers fichier
            lang: Code langue
            confidence_threshold: Seuil de confiance minimum (0-100)
        
        Returns:
            list: Liste des détections avec texte, coordonnées et confiance
        """
        processed_image = self.preprocess_image(image)
        
        try:
            # Obtenir les données détaillées
            data = pytesseract.image_to_data(
                processed_image,
                lang=lang,
                config='--oem 3 --psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            for i in range(len(data['text'])):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                # Filtrer par confiance et texte non vide
                if confidence >= confidence_threshold and text:
                    result = {
                        'text': text,
                        'confidence': confidence,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'level': data['level'][i]  # Niveau hiérarchique (mot, ligne, paragraphe)
                    }
                    results.append(result)
            
            print(f"📊 {len(results)} éléments détectés avec confiance >= {confidence_threshold}%")
            return results
            
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction avec boîtes : {e}")
            return []
    
    def draw_results(self, image_path, results, output_path="output_tesseract.png", 
                    show_confidence=True, min_confidence=50):
        """
        Dessiner les résultats OCR sur l'image
        
        Args:
            image_path: Chemin vers l'image originale
            results: Résultats de extract_text_with_boxes()
            output_path: Chemin de sauvegarde
            show_confidence: Afficher le pourcentage de confiance
            min_confidence: Confiance minimale pour l'affichage
        """
        # Charger l'image originale
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Impossible de charger l'image : {image_path}")
            return
        
        # Compteurs
        total_boxes = len(results)
        drawn_boxes = 0
        
        # Dessiner chaque détection
        for result in results:
            if result['confidence'] < min_confidence:
                continue
                
            x, y, w, h = result['x'], result['y'], result['width'], result['height']
            text = result['text']
            confidence = result['confidence']
            
            # Couleur selon la confiance
            if confidence >= 80:
                color = (0, 255, 0)  # Vert
            elif confidence >= 60:
                color = (0, 255, 255)  # Jaune
            else:
                color = (0, 165, 255)  # Orange
            
            # Dessiner le rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Préparer le label
            if show_confidence:
                label = f"{text} ({confidence}%)"
            else:
                label = text
            
            # Limiter la longueur du label
            if len(label) > 30:
                label = label[:27] + "..."
            
            # Dessiner le texte de fond
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (x, y - 25), (x + label_size[0], y), color, -1)
            
            # Dessiner le texte
            cv2.putText(image, label, (x, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            drawn_boxes += 1
        
        # Sauvegarder
        cv2.imwrite(output_path, image)
        print(f"💾 Résultat sauvegardé : {output_path}")
        print(f"📊 {drawn_boxes}/{total_boxes} boîtes affichées (confiance >= {min_confidence}%)")
    
    def analyze_image(self, image_path, languages=['eng'], save_results=True):
        """
        Analyse complète d'une image avec rapport détaillé
        
        Args:
            image_path: Chemin vers l'image
            languages: Liste des langues à tester
            save_results: Sauvegarder les images de résultats
        
        Returns:
            dict: Rapport complet d'analyse
        """
        print(f"\n🔍 ANALYSE COMPLÈTE DE : {image_path}")
        print("=" * 60)
        
        if not os.path.exists(image_path):
            print(f"❌ Fichier non trouvé : {image_path}")
            return {}
        
        # Informations sur l'image
        image = Image.open(image_path)
        file_size = os.path.getsize(image_path) / 1024  # KB
        
        print(f"📁 Taille fichier : {file_size:.1f} KB")
        print(f"📐 Dimensions : {image.size[0]} x {image.size[1]} pixels")
        print(f"🎨 Mode couleur : {image.mode}")
        
        # Résultats pour chaque langue
        report = {
            'image_info': {
                'path': image_path,
                'size': image.size,
                'mode': image.mode,
                'file_size_kb': file_size
            },
            'results': {}
        }
        
        for lang in languages:
            if lang not in self.available_languages:
                print(f"⚠️  Langue '{lang}' non disponible, ignorée")
                continue
                
            print(f"\n📝 Test avec langue : {lang}")
            print("-" * 30)
            
            # Extraction simple
            text = self.extract_text_simple(image_path, lang=lang)
            
            # Extraction avec boîtes
            boxes = self.extract_text_with_boxes(image_path, lang=lang)
            
            # Statistiques
            total_chars = len(text)
            total_words = len(text.split()) if text else 0
            avg_confidence = np.mean([b['confidence'] for b in boxes]) if boxes else 0
            
            # Stocker les résultats
            report['results'][lang] = {
                'text': text,
                'boxes': boxes,
                'stats': {
                    'total_characters': total_chars,
                    'total_words': total_words,
                    'detected_elements': len(boxes),
                    'average_confidence': avg_confidence
                }
            }
            
            # Affichage
            print(f"📝 Texte extrait ({total_chars} caractères, {total_words} mots) :")
            if text:
                preview = text[:200] + ("..." if len(text) > 200 else "")
                print(f"   '{preview}'")
            else:
                print("   (Aucun texte détecté)")
            
            print(f"📊 {len(boxes)} éléments détectés, confiance moyenne : {avg_confidence:.1f}%")
            
            # Sauvegarder les résultats visuels
            if save_results and boxes:
                output_path = f"output_{lang}_{Path(image_path).stem}.png"
                self.draw_results(image_path, boxes, output_path)
        
        return report
    
    def test_configurations(self, image_path):
        """
        Tester différentes configurations PSM pour trouver la meilleure
        
        Args:
            image_path: Chemin vers l'image à tester
        """
        print(f"\n🧪 TEST DE CONFIGURATIONS POUR : {image_path}")
        print("=" * 60)
        
        # Différents modes PSM (Page Segmentation Mode)
        psm_modes = {
            6: "Bloc de texte uniforme (défaut)",
            7: "Ligne de texte unique",
            8: "Mot unique",
            11: "Caractère unique sparse",
            12: "Texte sparse avec OSD",
            13: "Ligne brute - pas de hack"
        }
        
        results = {}
        
        for psm, description in psm_modes.items():
            print(f"\n🔧 PSM {psm}: {description}")
            
            config = f'--oem 3 --psm {psm}'
            
            try:
                start_time = time.time()
                text = self.extract_text_simple(image_path, config=config)
                end_time = time.time()
                
                word_count = len(text.split()) if text else 0
                char_count = len(text) if text else 0
                processing_time = end_time - start_time
                
                results[psm] = {
                    'text': text,
                    'word_count': word_count,
                    'char_count': char_count,
                    'processing_time': processing_time,
                    'success': bool(text.strip())
                }
                
                status = "✅" if text.strip() else "❌"
                print(f"   {status} {word_count} mots, {char_count} chars, {processing_time:.2f}s")
                
                if text.strip():
                    preview = text.strip()[:100] + ("..." if len(text.strip()) > 100 else "")
                    print(f"   Preview: '{preview}'")
                
            except Exception as e:
                print(f"   ❌ Erreur: {e}")
                results[psm] = {'error': str(e), 'success': False}
        
        # Recommandation
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if successful_results:
            # Choisir le mode avec le plus de mots détectés
            best_psm = max(successful_results.keys(), 
                          key=lambda x: successful_results[x]['word_count'])
            
            print(f"\n🏆 RECOMMANDATION: PSM {best_psm} - {psm_modes[best_psm]}")
            print(f"   Meilleur résultat: {successful_results[best_psm]['word_count']} mots")
        else:
            print(f"\n❌ Aucune configuration n'a donné de résultats satisfaisants")
        
        return results


def main():
    """Fonction principale avec exemples d'utilisation"""
    print("🚀 TESSERACT OCR - SCRIPT DE DÉMONSTRATION")
    print("=" * 50)
    
    try:
        # Initialiser Tesseract OCR
        ocr = TesseractOCR()
        
        # Vérifier qu'une image de test existe
        test_images = ["test_image.png", "test_image.jpg", "sample.png", "sample.jpg"]
        image_path = None
        
        for img in test_images:
            if os.path.exists(img):
                image_path = img
                break
        
        if not image_path:
            print("❌ Aucune image de test trouvée.")
            print("   Placez une image nommée 'test_image.png' dans le répertoire courant.")
            
            # Créer une image de test simple
            print("📝 Création d'une image de test...")
            create_test_image()
            image_path = "test_image_generated.png"
        
        print(f"📸 Image de test : {image_path}")
        
        # === DÉMONSTRATION 1: Extraction simple ===
        print(f"\n1️⃣  EXTRACTION SIMPLE")
        print("-" * 30)
        
        text = ocr.extract_text_simple(image_path, lang='eng')
        print(f"Texte extrait : '{text}'")
        
        # === DÉMONSTRATION 2: Extraction avec boîtes ===
        print(f"\n2️⃣  EXTRACTION AVEC COORDONNÉES")
        print("-" * 30)
        
        boxes = ocr.extract_text_with_boxes(image_path, lang='eng')
        
        for i, box in enumerate(boxes[:5]):  # Afficher les 5 premiers
            print(f"  {i+1}. '{box['text']}' - {box['confidence']}% - "
                  f"({box['x']},{box['y']}) {box['width']}x{box['height']}")
        
        if len(boxes) > 5:
            print(f"  ... et {len(boxes) - 5} autres éléments")
        
        # === DÉMONSTRATION 3: Analyse complète ===
        print(f"\n3️⃣  ANALYSE COMPLÈTE")
        
        # Tester avec anglais et français si disponible
        languages = ['eng']
        if 'fra' in ocr.available_languages:
            languages.append('fra')
        
        report = ocr.analyze_image(image_path, languages=languages)
        
        # === DÉMONSTRATION 4: Test de configurations ===
        print(f"\n4️⃣  TEST DE CONFIGURATIONS")
        
        ocr.test_configurations(image_path)
        
        print(f"\n✅ DÉMONSTRATION TERMINÉE")
        print(f"📁 Vérifiez les fichiers de sortie générés dans le répertoire courant")
        
    except Exception as e:
        print(f"❌ Erreur dans la démonstration : {e}")
        import traceback
        traceback.print_exc()


def create_test_image():
    """Créer une image de test simple si aucune n'est disponible"""
    from PIL import ImageDraw, ImageFont
    
    # Créer une image blanche
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Essayer d'utiliser une police par défaut
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    # Ajouter du texte
    texts = [
        "Tesseract OCR Test",
        "Ceci est un test d'OCR",
        "Numéros: 123456789",
        "Email: test@example.com"
    ]
    
    y_position = 50
    for text in texts:
        draw.text((50, y_position), text, fill='black', font=font)
        y_position += 60
    
    # Sauvegarder
    img.save("test_image_generated.png")
    print("✅ Image de test créée : test_image_generated.png")


def check_installation():
    """Vérifier l'installation de Tesseract"""
    print("🔍 VÉRIFICATION DE L'INSTALLATION")
    print("=" * 40)
    
    try:
        # Test Tesseract
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            'tesseract'
        ]
        tesseract_found = False
        for path in possible_paths:
            try:
                pytesseract.pytesseract.tesseract_cmd = path
                version = pytesseract.get_tesseract_version()
                print(f"   ✅ Tesseract trouvé: {path}")
                print(f"   📝 Version: {version}")
                tesseract_found = True
                break
            except:
                continue
        if not tesseract_found:
            raise Exception("Tesseract non trouvé. Vérifiez l'installation.")
        
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version : {version}")
        
        # Test langues
        languages = pytesseract.get_languages()
        print(f"✅ Langues disponibles : {', '.join(languages)}")
        
        # Test des bibliothèques
        import cv2
        print(f"✅ OpenCV version : {cv2.__version__}")
        
        from PIL import Image
        print(f"✅ Pillow installé")
        
        import numpy
        print(f"✅ NumPy version : {numpy.__version__}")
        
        print(f"\n🎉 Installation complète et fonctionnelle !")
        return True
        
    except Exception as e:
        print(f"❌ Problème détecté : {e}")
        print(f"\n🔧 Solutions possibles :")
        print(f"   1. Vérifiez que Tesseract est installé")
        print(f"   2. Ajoutez Tesseract au PATH")
        print(f"   3. Installez les packages Python manquants")
        return False


if __name__ == "__main__":
    # Vérifier d'abord l'installation
    if check_installation():
        print("\n" + "="*60)
        main()
    else:
        print(f"\n❌ Corrigez les problèmes d'installation avant de continuer.")
        sys.exit(1)