import csv
import os

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png'}


def _is_image(filename):
    return os.path.splitext(filename.lower())[1] in SUPPORTED_EXTS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'folder1' not in request.files or 'folder2' not in request.files:
        return 'Missing folder1 or folder2 in request', 400

    folder1_files = [f for f in request.files.getlist('folder1') if f.filename and _is_image(f.filename)]
    folder2_files = [f for f in request.files.getlist('folder2') if f.filename and _is_image(f.filename)]

    if not folder1_files or not folder2_files:
        return 'No valid images found in one or both folders', 400

    folder1_path = os.path.join(app.root_path, 'folder1')
    folder2_path = os.path.join(app.root_path, 'folder2')
    os.makedirs(folder1_path, exist_ok=True)
    os.makedirs(folder2_path, exist_ok=True)

    for f in folder1_files:
        f.save(os.path.join(folder1_path, secure_filename(f.filename)))
    for f in folder2_files:
        f.save(os.path.join(folder2_path, secure_filename(f.filename)))

    return render_template('save_as.html', folder1='folder1', folder2='folder2')


@app.route('/download', methods=['POST'])
def download():
    # secure_filename prevents path traversal; CSV is always written inside data/
    raw_name = request.form.get('filename', 'labels').strip() or 'labels'
    filename = secure_filename(raw_name) or 'labels'

    data_dir = os.path.join(app.root_path, 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, f'{filename}.csv')

    folder1_path = os.path.join(app.root_path, 'folder1')
    folder2_path = os.path.join(app.root_path, 'folder2')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Folder', 'Full Path'])
        for folder_path, label in [(folder1_path, 'thumbs_up'), (folder2_path, 'thumbs_down')]:
            for root, _, files in os.walk(folder_path):
                for fname in sorted(files):
                    if _is_image(fname):
                        writer.writerow([label, os.path.join(root, fname)])

    return send_file(csv_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
