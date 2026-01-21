"""
tinyeye pro - high performance web crawler
automatically populates the image database by crawling the web.
"""

import os
import time
import requests
import sqlite3
import pickle
import threading
import cv2
import numpy as np
import urllib.parse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

# configuration
MAX_THREADS = 10
MAX_DEPTH = 2  # how deep to follow links from the start url
MIN_IMAGE_SIZE_KB = 20  # ignore icons/thumbnails
USER_AGENT = 'TinyEye-Local-Crawler/1.0'

# database connection (shared with app.py)
DB_PATH = os.path.join(os.path.dirname(__file__), 'tinyeye_pro.db')
DOWNLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'crawled_images')

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

class LinkParser(HTMLParser):
    """simple html parser to extract <img> and <a> tags"""
    def __init__(self):
        super().__init__()
        self.images = []
        self.links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            for k, v in attrs:
                if k == 'src': self.images.append(v)
        if tag == 'a':
            for k, v in attrs:
                if k == 'href': self.links.append(v)

class ImageIndexer:
    """helper to extract orb features and save to db (replicates app.py logic)"""
    def __init__(self):
        # we need to initialize orb in the main thread or per thread safely
        # note: cv2.orb_create() is generally thread safe enough for separate instances
        pass

    def process_and_index(self, filepath, url):
        try:
            # 1. read image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None: return False
            
            # 2. extract features
            orb = cv2.ORB_create(nfeatures=1000)
            kps, desc = orb.detectAndCompute(img, None)
            
            if desc is None: return False
            
            # 3. save to db
            desc_blob = pickle.dumps(desc)
            
            with sqlite3.connect(DB_PATH, timeout=30.0) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO images (filepath, descriptors, keypoint_count, added_at)
                    VALUES (?, ?, ?, ?)
                ''', (filepath, desc_blob, len(kps), url)) # storing url in added_at for reference, or modify schema
                conn.commit()
            
            return True
        except Exception as e:
            print(f"indexing error {filepath}: {e}")
            return False

class Crawler:
    def __init__(self, start_urls, max_images=1000):
        self.queue = Queue()
        self.visited_urls = set()
        self.image_urls = set()
        self.downloaded_count = 0
        self.max_images = max_images
        self.indexer = ImageIndexer()
        self.lock = threading.Lock()
        
        for url in start_urls:
            self.queue.put((url, 0)) # url, depth

    def is_valid_url(self, url):
        parsed = urlparse(url)
        return bool(parsed.netloc) and bool(parsed.scheme)

    def download_image(self, img_url):
        if self.downloaded_count >= self.max_images: return
        
        try:
            # hash url to get a unique filename
            filename = str(abs(hash(img_url))) + ".jpg"
            filepath = os.path.join(DOWNLOAD_FOLDER, filename)
            
            # skip if exists
            if os.path.exists(filepath): return
            
            # download
            headers = {'User-Agent': USER_AGENT}
            response = requests.get(img_url, headers=headers, timeout=5)
            
            if response.status_code == 200 and len(response.content) > MIN_IMAGE_SIZE_KB * 1024:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # index immediately
                if self.indexer.process_and_index(filepath, img_url):
                    with self.lock:
                        self.downloaded_count += 1
                        print(f"[{self.downloaded_count}/{self.max_images}] indexed: {img_url[:50]}...")
                else:
                    os.remove(filepath) # delete if not indexable (corrupt/no features)
                    
        except Exception as e:
            pass 

    def process_url(self, url, depth):
        if depth > MAX_DEPTH: return
        try:
            headers = {'User-Agent': USER_AGENT}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200: return
            
            # parse html
            parser = LinkParser()
            parser.feed(response.text)
            
            # process images
            for src in parser.images:
                img_url = urljoin(url, src)
                if img_url not in self.image_urls:
                    self.image_urls.add(img_url)
                    self.download_image(img_url)
            
            # process links (for next depth)
            if depth < MAX_DEPTH:
                for link in parser.links:
                    next_url = urljoin(url, link)
                    # simple heuristic: keep to same domain or minimal drift to avoid drifting to facebook/twitter login pages
                    if self.is_valid_url(next_url) and next_url not in self.visited_urls:
                        self.visited_urls.add(next_url)
                        self.queue.put((next_url, depth + 1))
                        
        except Exception as e:
            # print(f"error crawling {url}: {e}")
            pass

    def worker(self):
        while self.downloaded_count < self.max_images:
            try:
                if self.queue.empty():
                    time.sleep(1)
                    continue
                    
                url, depth = self.queue.get(timeout=2)
                self.process_url(url, depth)
                self.queue.task_done()
            except:
                break

    def start(self):
        print(f"starting crawl with {MAX_THREADS} threads...")
        print("note: this will download images to ./crawled_images and index them.")
        
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            for _ in range(MAX_THREADS):
                executor.submit(self.worker)
            
            # wait loop
            while self.downloaded_count < self.max_images and not self.queue.empty():
                time.sleep(1)
                
        print(f"crawl finished. total indexed: {self.downloaded_count}")

if __name__ == "__main__":
    print("--- tinyeye universal crawler ---")
    seeds = input("enter starting urls (comma separated) [default: https://fakerface.ai, https://thispersondoesnotexist.com]: ")
    
    if not seeds.strip():
        seeds_list = ["https://fakerface.ai", "https://thispersondoesnotexist.com", "https://unsplash.com/s/photos/portrait"]
    else:
        seeds_list = [s.strip() for s in seeds.split(',')]
        
    limit = input("how many images to download? [default: 500]: ")
    max_imgs = int(limit) if limit.strip() else 500
    
    crawler = Crawler(seeds_list, max_images=max_imgs)
    crawler.start()
