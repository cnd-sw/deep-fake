"""
tinyeye pro - local reverse image search & forensics tool
replicates core identification technology using computer vision (orb features) and forensics (ela).
"""

import os
import io
import json
import sqlite3
import pickle
import numpy as np
import cv2
from datetime import datetime
from PIL import Image, ImageChops, ImageEnhance
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# initialize flask
app = Flask(__name__)
CORS(app)

# config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
DB_PATH = os.path.join(os.path.dirname(__file__), 'tinyeye_pro.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'}

for d in [UPLOAD_FOLDER]:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------------------
# core computer vision engine
# -------------------------------------------------------------------------

class FeatureExtractor:
    """
    state-of-the-art feature extraction using orb (oriented fast and rotated brief).
    this allows matching images even if they are cropped, rotated, or scaled.
    """
    def __init__(self):
        # orb is a robust local feature detector
        self.orb = cv2.ORB_create(nfeatures=1000)

    def extract_features(self, image_path_or_array):
        """extract keypoints and descriptors from an image."""
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array, cv2.IMREAD_GRAYSCALE)
        else:
            # assume it's a pil image or bytes, convert to numpy
            if isinstance(image_path_or_array, Image.Image):
                img = np.array(image_path_or_array.convert('L'))
            else:
                nparr = np.frombuffer(image_path_or_array, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return None, None

        # detect and compute
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2):
        """
        match two sets of descriptors using hamming distance (efficient for orb).
        returns the number of 'good' matches based on lowe's ratio test.
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return 0

        # bfmatcher with hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        try:
            # knn match
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            return len(good_matches)
        except Exception:
            return 0

# -------------------------------------------------------------------------
# forensics engine (deep fake / manipulation detection)
# -------------------------------------------------------------------------

class ForensicsEngine:
    """
    digital forensics utilizing error level analysis (ela) and noise profiling.
    """
    
    @staticmethod
    def error_level_analysis(image_path, quality=90):
        """
        perform error level analysis (ela).
        re-saves the image at a known quality and subtracts it from the original.
        high difference areas indicate potential manipulation (foreign artifacts).
        """
        original = Image.open(image_path).convert('RGB')
        
        # save compressed version to memory
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        
        # calculate difference
        # ela image = scale * |original - compressed|
        diff = ImageChops.difference(original, compressed)
        
        # amplify difference for visualization
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff > 0 else 1
        
        ela_image = ImageEnhance.Brightness(diff).enhance(scale)
        
        # calculate statistic score (mean brightness of ela)
        # higher score = more potential artifacts/noise inconsistencies
        stat = np.array(ela_image).mean()
        
        return ela_image, stat

    @staticmethod
    def analyze(image_path):
        """run comprehensive analysis."""
        results = {}
        
        # 1. ela
        _, ela_score = ForensicsEngine.error_level_analysis(image_path)
        
        # ela score interpretation
        # this acts as a heuristic. > 15-20 often implies significant high-freq noise or editing.
        results['ela_score'] = round(ela_score, 2)
        
        # 2. noise/blur analysis (opencv)
        img = cv2.imread(image_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # laplacian variance = blur metric
            blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            results['blur_metric'] = round(blur_var, 2)
            results['is_blurry'] = blur_var < 100
        
        # verdict logic
        risk_score = 0
        if results['ela_score'] > 20: risk_score += 40
        if results.get('blur_metric', 200) < 50: risk_score += 20  # too blurry is suspicious for ai sometimes
        
        results['risk_score'] = min(100, risk_score)
        results['verdict'] = "Suspicious" if risk_score > 50 else "Likely Authentic"
        
        return results

# -------------------------------------------------------------------------
# database & application
# -------------------------------------------------------------------------

extractor = FeatureExtractor()

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE,
                descriptors BLOB,
                keypoint_count INTEGER,
                added_at TEXT
            )
        ''')

init_db()

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/image/<path:filename>')
def serve_image(filename):
    # security: ensure file serves from allowed dirs (simplified for local demo)
    if filename.startswith('/'):
        return send_from_directory(os.path.dirname(filename), os.path.basename(filename))
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/index', methods=['POST'])
def api_index():
    data = request.json
    directory = data.get('directory')
    if not directory or not os.path.isdir(directory):
        return jsonify(error="Invalid directory"), 400

    count = 0
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
                    path = os.path.join(root, file)
                    try:
                        # extract features
                        kps, desc = extractor.extract_features(path)
                        if desc is not None:
                            # serialize descriptors
                            desc_blob = pickle.dumps(desc)
                            
                            cursor.execute('''
                                INSERT OR REPLACE INTO images (filepath, descriptors, keypoint_count, added_at)
                                VALUES (?, ?, ?, ?)
                            ''', (path, desc_blob, len(kps), datetime.now().isoformat()))
                            count += 1
                            if count % 10 == 0: print(f"indexed {count} images...")
                    except Exception as e:
                        print(f"skipping {file}: {e}")
        conn.commit()
    
    return jsonify(success=True, indexed=count)

@app.route('/api/search', methods=['POST'])
def api_search():
    if 'image' not in request.files:
        return jsonify(error="No image"), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # 1. extract query features
    img_np = np.frombuffer(img_bytes, np.uint8)
    query_img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
         return jsonify(error="Invalid image"), 400

    kp, query_desc = extractor.orb.detectAndCompute(query_img, None)
    if query_desc is None:
        return jsonify(results=[], message="No features found in query image")

    matches = []
    
    # 2. linear scan & match (professional approach uses index/flann, using sqlite + loop for simplicity)
    # note: for production, we would use faiss. here we load descriptors from db.
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT filepath, descriptors FROM images")
        rows = cursor.fetchall()
        
        for filepath, desc_blob in rows:
            if not desc_blob: continue
            
            db_desc = pickle.loads(desc_blob)
            
            # match
            score = extractor.match_features(query_desc, db_desc)
            
            # matches > 10 is usually a strong indicator for orb
            if score > 5:
                # calculate simple percentage based on query keypoints
                pct = min(100, int((score / len(kp)) * 100 * 2)) # *2 is a heuristic factor
                if pct > 10:
                    matches.append({
                        'filepath': filepath,
                        'similarity_score': pct,
                        'matches_count': score,
                        'match_type': 'Match' if pct > 80 else 'Partial Match'
                    })

    # sort by matches score
    matches.sort(key=lambda x: x['matches_count'], reverse=True)
    return jsonify(results=matches[:50])

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'image' not in request.files:
        return jsonify(error="No image"), 400
    
    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    
    try:
        report = ForensicsEngine.analyze(path)
        return jsonify(report)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    print(f"Server running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
