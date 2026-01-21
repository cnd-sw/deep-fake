# tinyeye pro

a professional-grade local reverse image search and digital forensics tool.

this application replicates the core functionality of advanced reverse image search engines using computer vision descriptors (orb) instead of simple perceptual hashing, allowing for the detection of cropped, rotated, and modified images. additionally, it includes an error level analysis (ela) engine for detecting digital manipulation and deep fakes.

## key features

- **robust reverse image search**: utilizes oriented fast and rotated brief (orb) for feature extraction and matching, capable of identifying images despite geometric transformations.
- **digital forensics**: implements error level analysis (ela) to visualize compression artifacts and identify potential manipulation.
- **high-performance crawler**: multi-threaded web crawler to automatically populate the local dataset from specified web sources.
- **privacy-focused**: operates entirely locally with no external api dependencies.

## technical architecture

- **backend**: python 3.10, flask
- **computer vision**: opencv (orb descriptors, brute-force matcher)
- **image processing**: pillow, numpy, scikit-image
- **database**: sqlite (blob storage for serialized feature descriptors)

## installation

ensure you have a python 3.10 environment active.

1.  install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *note: opencv-python-headless is required for the vision engine.*

## usage

### 1. starting the server

run the application server:

```bash
python app.py
```

access the interface at `http://localhost:5001`.

### 2. populating the database

you can index images in two ways:

- **local indexing**: use the web interface to point to a local directory of images.
- **web crawling**: use the included crawler to build a dataset from the web.

to run the crawler:

```bash
python crawler.py
```
follow the prompts to specify seed urls and download limits.

### 3. api endpoints

- `POST /api/search`: accepts an image file, returns matching records with similarity scores.
- `POST /api/analyze`: accepts an image file, returns forensic analysis report (ela score, blur metric).
- `POST /api/index`: accepts a directory path, recursively indexes images.

## license

proprietary software. built for local deployment.
