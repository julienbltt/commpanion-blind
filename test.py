#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme de Test Caméra OCR
Utilise la caméra pour prendre des photos et extraire le texte en temps réel

Compatible Snapdragon X Plus - Windows ARM64
Utilise la classe TesseractOCR

Contrôles:
- ESPACE : Prendre une photo et faire l'OCR
- 'p' : Activer/désactiver le préprocessing
- 's' : Sauvegarder la dernière capture
- 'c' : Effacer le texte affiché
- 'q' ou ESC : Quitter

Auteur: Assistant IA
Date: 2025
"""

import cv2
import numpy as np
from PIL import Image
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import threading
import queue
import argparse

# Import de la classe TesseractOCR
# Assurez-vous que ocr.py est dans le même répertoire
try:
    from ocr import TesseractOCR
except ImportError:
    print("❌ Impossible d'importer TesseractOCR")
    print("   Assurez-vous que ocr.py est dans le même répertoire")
    exit(1)

def detect_available_cameras(max_cameras=10):
    """
    Détecter les caméras disponibles sur le système
    
    Args:
        max_cameras (int): Nombre maximum de caméras à tester
        
    Returns:
        list: Liste des caméras disponibles avec leurs informations
    """
    print("🔍 Détection des caméras disponibles...")
    available_cameras = []
    
    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            # Tester si on peut vraiment lire une frame
            ret, frame = cap.read()
            if ret and frame is not None:
                # Obtenir les informations de la caméra
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Obtenir le nom du backend
                backend = cap.getBackendName()
                
                camera_info = {
                    'id': camera_id,
                    'name': f"Caméra {camera_id}",
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'backend': backend,
                    'working': True
                }
                
                available_cameras.append(camera_info)
                print(f"✅ Caméra {camera_id} détectée: {width}x{height} @ {fps:.1f}fps ({backend})")
            
            cap.release()
        
        # Petite pause pour éviter les conflits
        time.sleep(0.1)
    
    return available_cameras

def select_camera_interactive(available_cameras):
    """
    Interface interactive pour sélectionner une caméra
    
    Args:
        available_cameras (list): Liste des caméras disponibles
        
    Returns:
        int: ID de la caméra sélectionnée
    """
    if not available_cameras:
        print("❌ Aucune caméra détectée!")
        return None
    
    print("\n📷 CAMÉRAS DISPONIBLES:")
    print("=" * 60)
    
    for i, camera in enumerate(available_cameras):
        print(f"  [{camera['id']}] {camera['name']}")
        print(f"      Résolution: {camera['resolution']}")
        print(f"      FPS: {camera['fps']:.1f}")
        print(f"      Backend: {camera['backend']}")
        print()
    
    while True:
        try:
            choice = input(f"🎯 Choisissez une caméra (0-{max([c['id'] for c in available_cameras])}): ").strip()
            
            if choice == '':
                # Utiliser la première caméra par défaut
                camera_id = available_cameras[0]['id']
                print(f"📷 Caméra par défaut sélectionnée: {camera_id}")
                return camera_id
            
            camera_id = int(choice)
            
            # Vérifier que la caméra choisie est dans la liste
            if any(c['id'] == camera_id for c in available_cameras):
                print(f"📷 Caméra {camera_id} sélectionnée!")
                return camera_id
            else:
                print(f"❌ Caméra {camera_id} non disponible. Choisissez parmi: {[c['id'] for c in available_cameras]}")
                
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
        except KeyboardInterrupt:
            print("\n🛑 Sélection annulée")
            return None

def test_camera_preview(camera_id):
    """
    Afficher un aperçu de la caméra pour validation
    
    Args:
        camera_id (int): ID de la caméra à tester
    """
    print(f"🔍 Test de la caméra {camera_id}...")
    print("   Appuyez sur ESPACE pour confirmer, ESC pour annuler")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la caméra {camera_id}")
        return False
    
    # Configuration de base
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    window_name = f"Test Caméra {camera_id} - ESPACE=OK, ESC=Annuler"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Impossible de lire la frame")
            break
        
        # Ajouter du texte informatif
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Camera {camera_id} - Test Preview", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "ESPACE = Confirmer | ESC = Annuler", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Espace - confirmer
            cap.release()
            cv2.destroyWindow(window_name)
            print(f"✅ Caméra {camera_id} confirmée!")
            return True
        elif key == 27:  # ESC - annuler
            cap.release()
            cv2.destroyWindow(window_name)
            print(f"❌ Test de la caméra {camera_id} annulé")
            return False
    
    cap.release()
    cv2.destroyWindow(window_name)
    return False

class CameraOCR:
    """Application de test caméra avec OCR en temps réel"""
    
    def __init__(self, camera_id=0):
        """
        Initialiser l'application caméra OCR
        
        Args:
            camera_id (int): ID de la caméra (0 pour caméra par défaut)
        """
        print("🚀 Initialisation Camera OCR...")
        
        # Initialiser Tesseract OCR
        try:
            self.ocr = TesseractOCR()
            print("✅ TesseractOCR initialisé")
        except Exception as e:
            print(f"❌ Erreur initialisation OCR: {e}")
            raise
        
        # Configuration caméra
        self.camera_id = camera_id
        self.cap = None
        
        # États de l'application
        self.current_text = ""
        self.last_capture = None
        self.processing = False
        self.use_preprocessing = True
        self.capture_counter = 0
        
        # Configuration OCR
        self.ocr_language = 'eng'
        self.ocr_config = '--oem 3 --psm 6'
        self.confidence_threshold = 50
        
        # Interface
        self.window_name = f"Camera OCR Test - Caméra {camera_id} - Appuyez sur ESPACE pour capturer"
        self.text_window_name = "Texte Détecté"
        
        # Dossier de sauvegarde
        self.save_folder = "camera_ocr_captures"
        Path(self.save_folder).mkdir(exist_ok=True)
        
        # Queue pour traitement asynchrone
        self.ocr_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        print(f"📁 Dossier de sauvegarde: {self.save_folder}")
    
    def initialize_camera(self):
        """Initialiser la caméra"""
        print(f"📷 Initialisation caméra {self.camera_id}...")
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise Exception(f"Impossible d'ouvrir la caméra {self.camera_id}")
        
        # Configuration caméra
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Vérifier les paramètres réels
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        backend = self.cap.getBackendName()
        
        print(f"✅ Caméra {self.camera_id} initialisée: {width}x{height} @ {fps:.1f}fps ({backend})")
        
        return True
    
    def process_ocr_async(self, image_array, timestamp):
        """Traiter l'OCR de manière asynchrone"""
        try:
            print(f"🔍 Début OCR asynchrone ({timestamp})")
            
            # Convertir en image PIL
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Préprocessing si activé
            if self.use_preprocessing:
                processed_image = self.ocr.preprocess_image(
                    pil_image, 
                    enhancement_level='medium'
                )
            else:
                processed_image = pil_image
            
            # OCR simple pour rapidité
            start_time = time.time()
            text = self.ocr.extract_text_simple(
                processed_image,
                lang=self.ocr_language,
                config=self.ocr_config
            )
            ocr_time = time.time() - start_time
            
            # OCR avec coordonnées pour analyse détaillée
            boxes = self.ocr.extract_text_with_boxes(
                processed_image,
                lang=self.ocr_language,
                confidence_threshold=self.confidence_threshold
            )
            
            # Calculer statistiques
            total_words = len(text.split()) if text else 0
            avg_confidence = np.mean([b['confidence'] for b in boxes]) if boxes else 0
            
            result = {
                'text': text.strip(),
                'boxes': boxes,
                'stats': {
                    'processing_time': ocr_time,
                    'word_count': total_words,
                    'detection_count': len(boxes),
                    'avg_confidence': avg_confidence
                },
                'timestamp': timestamp,
                'image': image_array.copy()
            }
            
            # Mettre le résultat dans la queue
            self.result_queue.put(result)
            
            print(f"✅ OCR terminé: {total_words} mots, {len(boxes)} détections, {ocr_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Erreur OCR asynchrone: {e}")
            error_result = {
                'text': f"Erreur OCR: {e}",
                'boxes': [],
                'stats': {'processing_time': 0, 'word_count': 0, 'detection_count': 0, 'avg_confidence': 0},
                'timestamp': timestamp,
                'image': image_array.copy(),
                'error': True
            }
            self.result_queue.put(error_result)
    
    def capture_and_ocr(self, original_frame):
        """Capturer une frame et lancer l'OCR sur l'image brute (sans UI)"""
        if self.processing:
            print("⚠️  OCR en cours, capture ignorée")
            return
        
        self.processing = True
        self.capture_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"📸 Capture #{self.capture_counter} - {timestamp} (image brute sans UI)")
        
        # Sauvegarder la capture ORIGINALE (sans interface)
        self.last_capture = {
            'image': original_frame.copy(),
            'timestamp': timestamp,
            'filename': f"capture_cam{self.camera_id}_{timestamp}.png"
        }
        
        # Lancer OCR en arrière-plan sur l'image ORIGINALE
        ocr_thread = threading.Thread(
            target=self.process_ocr_async,
            args=(original_frame, timestamp),
            daemon=True
        )
        ocr_thread.start()
    
    def save_last_capture(self):
        """Sauvegarder la dernière capture avec annotations"""
        if not self.last_capture:
            print("❌ Aucune capture à sauvegarder")
            return
        
        try:
            # Chemin de sauvegarde
            image_path = Path(self.save_folder) / self.last_capture['filename']
            text_path = Path(self.save_folder) / f"text_cam{self.camera_id}_{self.last_capture['timestamp']}.txt"
            
            # Sauvegarder l'image
            cv2.imwrite(str(image_path), self.last_capture['image'])
            
            # Sauvegarder le texte
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Capture: {self.last_capture['timestamp']}\n")
                f.write(f"Caméra: {self.camera_id}\n")
                f.write(f"Langue: {self.ocr_language}\n")
                f.write(f"Préprocessing: {'Activé' if self.use_preprocessing else 'Désactivé'}\n")
                f.write(f"Configuration: {self.ocr_config}\n\n")
                f.write("Texte détecté:\n")
                f.write("-" * 40 + "\n")
                f.write(self.current_text)
            
            print(f"💾 Capture sauvegardée:")
            print(f"   📷 Image: {image_path}")
            print(f"   📄 Texte: {text_path}")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")
    
    def draw_interface(self, frame):
        """Dessiner l'interface utilisateur sur la frame"""
        height, width = frame.shape[:2]
        
        # Zone d'informations (semi-transparente)
        overlay = frame.copy()
        
        # Rectangle d'informations en haut
        cv2.rectangle(overlay, (10, 10), (width - 10, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Informations système
        info_lines = [
            f"Camera OCR Test - Snapdragon X Plus | Caméra #{self.camera_id}",
            f"Langue: {self.ocr_language.upper()} | Preprocessing: {'ON' if self.use_preprocessing else 'OFF'}",
            f"Captures: {self.capture_counter} | OCR: {'PROCESSING...' if self.processing else 'READY'}",
            f"Controles: ESPACE=Capturer, P=Preprocessing, S=Sauver, C=Clear, L=Langue, Q=Quitter"
        ]
        
        for i, line in enumerate(info_lines):
            y_pos = 25 + i * 22
            cv2.putText(frame, line, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Zone de texte détecté (en bas)
        if self.current_text:
            text_height = 150
            text_y_start = height - text_height - 10
            
            # Rectangle pour le texte
            cv2.rectangle(overlay, (10, text_y_start), (width - 10, height - 10), (0, 50, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Titre
            cv2.putText(frame, "TEXTE DETECTE:", (15, text_y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Texte (limité en longueur)
            text_lines = self.current_text.split('\n')
            max_chars = 80
            
            y_offset = 45
            for line in text_lines[:5]:  # Max 5 lignes
                if len(line) > max_chars:
                    line = line[:max_chars] + "..."
                
                cv2.putText(frame, line, (15, text_y_start + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            if len(text_lines) > 5:
                cv2.putText(frame, "... (texte tronque)", (15, text_y_start + y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Indicateur de traitement
        if self.processing:
            # Animation de traitement
            spinner_chars = "|/-\\"
            spinner_char = spinner_chars[int(time.time() * 4) % 4]
            
            cv2.putText(frame, f"OCR {spinner_char}", (width - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def show_detailed_results(self, result):
        """Afficher les résultats détaillés dans une fenêtre séparée"""
        if result.get('error', False):
            return
        
        # Créer une image pour afficher les détails (utiliser l'image ORIGINALE)
        detail_image = result['image'].copy()
        
        # Dessiner les boîtes de détection sur l'image brute
        for box in result['boxes']:
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            confidence = box['confidence']
            
            # Couleur selon confiance
            if confidence >= 80:
                color = (0, 255, 0)  # Vert
            elif confidence >= 60:
                color = (0, 255, 255)  # Jaune
            else:
                color = (0, 165, 255)  # Orange
            
            # Rectangle
            cv2.rectangle(detail_image, (x, y), (x + w, y + h), color, 2)
            
            # Label avec le texte détecté (si disponible)
            label = f"{confidence}%"
            if 'text' in box and box['text'].strip():
                # Limiter la longueur du texte affiché
                text_preview = box['text'].strip()[:15]
                if len(box['text'].strip()) > 15:
                    text_preview += "..."
                label = f"{text_preview} ({confidence}%)"
            
            # Calculer la position du label pour qu'il soit visible
            label_y = y - 5 if y > 20 else y + h + 20
            cv2.putText(detail_image, label, (x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Informations en overlay (sur fond semi-transparent)
        stats = result['stats']
        info_text = [
            f"Camera: {self.camera_id} | Image brute (sans UI)",
            f"Mots: {stats['word_count']}",
            f"Detections: {stats['detection_count']}",
            f"Confiance moy.: {stats['avg_confidence']:.1f}%",
            f"Temps: {stats['processing_time']:.2f}s"
        ]
        
        # Rectangle d'informations avec transparence
        overlay = detail_image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, detail_image, 0.3, 0, detail_image)
        
        for i, info in enumerate(info_text):
            cv2.putText(detail_image, info, (15, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Afficher
        cv2.imshow(f"Résultats OCR Détaillés - Caméra {self.camera_id}", detail_image)
    
    def check_ocr_results(self):
        """Vérifier et traiter les résultats OCR"""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                
                # Mettre à jour le texte actuel
                self.current_text = result['text']
                self.processing = False
                
                # Afficher les résultats détaillés
                self.show_detailed_results(result)
                
                # Log des statistiques
                stats = result['stats']
                print(f"📊 Résultats OCR (Caméra {self.camera_id}):")
                print(f"   📝 Texte: '{self.current_text[:100]}{'...' if len(self.current_text) > 100 else ''}'")
                print(f"   📊 {stats['word_count']} mots, {stats['detection_count']} détections")
                print(f"   🎯 Confiance moyenne: {stats['avg_confidence']:.1f}%")
                print(f"   ⏱️  Temps de traitement: {stats['processing_time']:.2f}s")
                
                break
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"❌ Erreur traitement résultats: {e}")
            self.processing = False
    
    def handle_keyboard_input(self, key):
        """Gérer les entrées clavier"""
        if key == ord(' '):  # Barre d'espace
            return 'capture'
        elif key == ord('p') or key == ord('P'):  # Toggle preprocessing
            self.use_preprocessing = not self.use_preprocessing
            print(f"🎛️  Préprocessing: {'Activé' if self.use_preprocessing else 'Désactivé'}")
            return 'toggle_preprocessing'
        elif key == ord('s') or key == ord('S'):  # Sauvegarder
            return 'save'
        elif key == ord('c') or key == ord('C'):  # Clear
            self.current_text = ""
            print("🗑️  Texte effacé")
            return 'clear'
        elif key == ord('l') or key == ord('L'):  # Toggle language
            self.ocr_language = 'fra' if self.ocr_language == 'eng' else 'eng'
            print(f"🌍 Langue changée: {self.ocr_language.upper()}")
            return 'toggle_language'
        elif key == ord('q') or key == ord('Q') or key == 27:  # Q ou ESC
            return 'quit'
        
        return None
    
    def run(self):
        """Lancer l'application principale"""
        try:
            # Initialiser la caméra
            if not self.initialize_camera():
                return
            
            print(f"\n🎮 CONTRÔLES (Caméra {self.camera_id}):")
            print("   ESPACE    - Prendre une photo et faire l'OCR")
            print("   P         - Activer/désactiver le préprocessing")
            print("   S         - Sauvegarder la dernière capture")
            print("   C         - Effacer le texte affiché")
            print("   L         - Changer de langue (ENG/FRA)")
            print("   Q ou ESC  - Quitter")
            print(f"\n📷 Caméra {self.camera_id} démarrée. Appuyez sur ESPACE pour capturer!")
            
            # Boucle principale
            while True:
                # Lire une frame
                ret, original_frame = self.cap.read()
                if not ret:
                    print("❌ Impossible de lire la frame de la caméra")
                    break
                
                # Vérifier les résultats OCR
                self.check_ocr_results()
                
                # Créer une copie pour l'interface (l'original reste intact)
                display_frame = original_frame.copy()
                
                # Dessiner l'interface sur la copie
                display_frame = self.draw_interface(display_frame)
                
                # Afficher la frame avec interface
                cv2.imshow(self.window_name, display_frame)
                
                # Gérer les entrées clavier
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Une touche a été pressée
                    action = self.handle_keyboard_input(key)
                    
                    if action == 'capture':
                        # Passer l'image ORIGINALE (sans interface) à l'OCR
                        self.capture_and_ocr(original_frame)
                    elif action == 'save':
                        self.save_last_capture()
                    elif action == 'quit':
                        break
            
        except KeyboardInterrupt:
            print("\n🛑 Interruption utilisateur")
        except Exception as e:
            print(f"❌ Erreur application: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Nettoyer les ressources"""
        print("🧹 Nettoyage des ressources...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        print("✅ Application fermée proprement")


def parse_arguments():
    """Parser les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Camera OCR Test avec sélection de caméra')
    parser.add_argument('--camera', '-c', type=int, 
                       help='ID de la caméra à utiliser (ex: 0, 1, 2...)')
    parser.add_argument('--list-cameras', '-l', action='store_true',
                       help='Lister les caméras disponibles et quitter')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Utiliser automatiquement la première caméra disponible')
    parser.add_argument('--test', '-t', type=int,
                       help='Tester une caméra spécifique avec aperçu')
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    print("🚀 DÉMARRAGE CAMERA OCR TEST")
    print("=" * 50)
    
    try:
        # Parser les arguments
        args = parse_arguments()
        
        # Vérifier la disponibilité de Tesseract
        from ocr import TesseractOCR
        test_ocr = TesseractOCR()
        print(f"✅ Tesseract disponible, langues: {test_ocr.available_languages}")
        
        # Détecter les caméras disponibles
        available_cameras = detect_available_cameras()
        
        if not available_cameras:
            print("❌ Aucune caméra détectée!")
            print("   Vérifiez que votre caméra est connectée et accessible")
            return
        
        # Traiter les arguments de ligne de commande
        if args.list_cameras:
            print("\n📷 CAMÉRAS DISPONIBLES:")
            print("=" * 60)
            for camera in available_cameras:
                print(f"  [{camera['id']}] {camera['name']}")
                print(f"      Résolution: {camera['resolution']}")
                print(f"      FPS: {camera['fps']:.1f}")
                print(f"      Backend: {camera['backend']}")
                print()
            return
        
        if args.test is not None:
            camera_id = args.test
            if any(c['id'] == camera_id for c in available_cameras):
                if test_camera_preview(camera_id):
                    print(f"✅ Test de la caméra {camera_id} réussi!")
                else:
                    print(f"❌ Test de la caméra {camera_id} échoué!")
            else:
                print(f"❌ Caméra {camera_id} non disponible")
            return
        
        # Sélectionner la caméra
        if args.camera is not None:
            camera_id = args.camera
            if any(c['id'] == camera_id for c in available_cameras):
                print(f"📷 Caméra {camera_id} spécifiée via argument")
            else:
                print(f"❌ Caméra {camera_id} non disponible")
                print(f"   Caméras disponibles: {[c['id'] for c in available_cameras]}")
                return
        elif args.auto:
            camera_id = available_cameras[0]['id']
            print(f"🚀 Mode automatique: utilisation de la caméra {camera_id}")
        else:
            # Sélection interactive
            camera_id = select_camera_interactive(available_cameras)
            if camera_id is None:
                print("❌ Aucune caméra sélectionnée")
                return
        
        # Test optionnel de la caméra
        if not args.auto:
            print(f"\n🔍 Voulez-vous tester la caméra {camera_id} avant de continuer? (y/N): ", end="")
            test_choice = input().strip().lower()
            if test_choice in ['y', 'yes', 'o', 'oui']:
                if not test_camera_preview(camera_id):
                    print("❌ Test échoué, arrêt du programme")
                    return
        
        # Créer et lancer l'application
        print(f"\n🚀 Lancement de l'application avec la caméra {camera_id}...")
        app = CameraOCR(camera_id=camera_id)
        app.run()
        
    except ImportError:
        print("❌ Module ocr non trouvé")
        print("   Assurez-vous que ocr.py est dans le même répertoire")
    except Exception as e:
        print(f"❌ Erreur de démarrage: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()