#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick  test of Tesseract Installation
For Snapdragon X Plus - Windows ARM64
"""
import sys
import os


def test_tesseract_installation():
    """Test complet de l'installation Tesseract"""
    
    print("🔍 TEST D'INSTALLATION TESSERACT")
    print("=" * 50)
    
    # Test 1: Import des bibliothèques
    print("\n1️⃣ Test des imports Python...")
    
    try:
        import pytesseract
        print("   ✅ pytesseract importé")
    except ImportError:
        print("   ❌ pytesseract manquant")
        print("   Solution: pip install pytesseract")
        return False
    
    try:
        from PIL import Image
        print("   ✅ PIL/Pillow importé")
    except ImportError:
        print("   ❌ Pillow manquant")
        print("   Solution: pip install pillow")
        return False
    
    try:
        import cv2
        print("   ✅ OpenCV importé")
    except ImportError:
        print("   ❌ OpenCV manquant")
        print("   Solution: pip install opencv-python")
        return False
    
    try:
        import numpy
        print("   ✅ NumPy importé")
    except ImportError:
        print("   ❌ NumPy manquant")
        print("   Solution: pip install numpy")
        return False
    
    # Test 2: Configuration Tesseract
    print("\n2️⃣ Test de la configuration Tesseract...")
    
    # Chemins possibles pour Tesseract
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        'tesseract'  # Si dans PATH
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
        print("   ❌ Tesseract non trouvé")
        print("   Solutions:")
        print("     1. Installez Tesseract depuis: https://github.com/UB-Mannheim/tesseract/wiki")
        print("     2. Ajoutez Tesseract au PATH système")
        print("     3. Spécifiez le chemin manuellement dans le code")
        return False
    
    # Test 3: Langues disponibles
    print("\n3️⃣ Test des langues disponibles...")
    
    try:
        languages = pytesseract.get_languages()
        print(f"   ✅ Langues trouvées: {', '.join(languages)}")
        
        # Vérifier les langues importantes
        important_langs = ['eng', 'fra']
        for lang in important_langs:
            if lang in languages:
                print(f"   ✅ {lang} disponible")
            else:
                print(f"   ⚠️  {lang} non disponible")
                
    except Exception as e:
        print(f"   ❌ Erreur langues: {e}")
        return False
    
    # Test 4: Test OCR sur image simple
    print("\n4️⃣ Test OCR basique...")
    
    try:
        # Créer une image de test simple
        test_image = Image.new('RGB', (400, 100), color='white')
        from PIL import ImageDraw, ImageFont
        
        draw = ImageDraw.Draw(test_image)
        
        # Essayer d'utiliser une police
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), "TEST OCR 123", fill='black', font=font)
        
        # Test OCR
        text = pytesseract.image_to_string(test_image, lang='eng').strip()
        
        if text:
            print(f"   ✅ OCR fonctionnel")
            print(f"   📝 Texte détecté: '{text}'")
        else:
            print(f"   ⚠️  OCR ne détecte rien")
            
    except Exception as e:
        print(f"   ❌ Erreur OCR: {e}")
        return False
    
    # Test 5: Performance
    print("\n5️⃣ Test de performance...")
    
    try:
        import time
        
        # Créer une image plus complexe
        test_image = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(test_image)
        
        texts = ["Performance Test", "Ligne 2: Numbers 456789", "Email: test@example.com"]
        y_pos = 20
        
        for text in texts:
            draw.text((20, y_pos), text, fill='black')
            y_pos += 50
        
        # Mesurer le temps
        start_time = time.time()
        result = pytesseract.image_to_string(test_image, lang='eng')
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"   ✅ Temps de traitement: {processing_time:.2f} secondes")
        
        if processing_time < 2.0:
            print(f"   🚀 Performance excellente")
        elif processing_time < 5.0:
            print(f"   ✅ Performance correcte")
        else:
            print(f"   ⚠️  Performance lente")
            
    except Exception as e:
        print(f"   ❌ Erreur performance: {e}")
    
    print(f"\n🎉 INSTALLATION TESSERACT RÉUSSIE !")
    print(f"✅ Vous pouvez maintenant utiliser Tesseract OCR")
    return True


def quick_ocr_test(image_path):
    """Test OCR rapide sur une image utilisateur"""
    
    if not os.path.exists(image_path):
        print(f"❌ Fichier non trouvé: {image_path}")
        return
    
    print(f"\n🖼️  TEST OCR SUR: {image_path}")
    print("-" * 40)
    
    try:
        import pytesseract
        from PIL import Image
        import time
        
        # Configurer Tesseract (chemins automatiques)
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            'tesseract'
        ]
        
        for path in possible_paths:
            try:
                pytesseract.pytesseract.tesseract_cmd = path
                pytesseract.get_tesseract_version()
                break
            except:
                continue
        
        # Charger et analyser l'image
        image = Image.open(image_path)
        print(f"📐 Taille: {image.size}")
        print(f"🎨 Mode: {image.mode}")
        
        # OCR simple
        print(f"\n📝 Extraction de texte...")
        start_time = time.time()
        
        text = pytesseract.image_to_string(image, lang='eng')
        
        end_time = time.time()
        
        print(f"⏱️  Temps: {end_time - start_time:.2f}s")
        print(f"📄 Texte extrait:")
        print("-" * 20)
        
        if text.strip():
            print(text)
        else:
            print("(Aucun texte détecté)")
        
        print("-" * 20)
        
        # Statistiques
        lines = text.split('\n')
        words = text.split()
        chars = len(text)
        
        print(f"📊 Statistiques:")
        print(f"   Lignes: {len(lines)}")
        print(f"   Mots: {len(words)}")
        print(f"   Caractères: {chars}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")


def main():
    """Fonction principale"""
    
    # Test de l'installation
    if test_tesseract_installation():
        
        # Proposer un test sur image utilisateur
        print(f"\n" + "="*50)
        print(f"🔧 VOULEZ-VOUS TESTER SUR VOTRE IMAGE ?")
        print(f"="*50)
        
        # Chercher des images dans le répertoire courant
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        found_images = []
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                found_images.append(file)
        
        if found_images:
            print(f"\n📁 Images trouvées dans le répertoire courant:")
            for i, img in enumerate(found_images[:5], 1):
                print(f"   {i}. {img}")
            
            print(f"\n💡 Pour tester une image, tapez:")
            print(f"   python {os.path.basename(__file__)} nom_de_votre_image.png")
        else:
            print(f"\n📁 Aucune image trouvée dans le répertoire courant")
            print(f"💡 Placez une image (PNG, JPG) et relancez le script")
    
    else:
        print(f"\n❌ Corrigez les problèmes d'installation avant de continuer")
        print(f"\n🔧 AIDE À L'INSTALLATION:")
        print(f"1. Téléchargez Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        print(f"2. Installez les packages Python: pip install pytesseract pillow opencv-python")
        print(f"3. Ajoutez Tesseract au PATH système")


if __name__ == "__main__":
    # Si un argument est fourni, l'utiliser comme image de test
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        quick_ocr_test(image_path)
    else:
        main()