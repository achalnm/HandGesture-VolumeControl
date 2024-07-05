from flask import Flask, render_template, request, send_file
import os
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)

def process_folders(folder_paths, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder', 'Full Path'])  # Write header
        
        for idx, folder_path in enumerate(folder_paths):
            folder_name = 'thumbs_up' if idx == 0 else 'thumbs_down'
            for root, _, files in os.walk(folder_path):
                for file_name in files:
                    if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
                        full_path = os.path.join(root, file_name)
                        csv_writer.writerow([folder_name, full_path])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'folder1' not in request.files or 'folder2' not in request.files:
        return 'Missing folder part'

    folder1 = request.files.getlist('folder1')
    folder2 = request.files.getlist('folder2')

    if not folder1 or not folder2:
        return 'No selected folder'

    folder1_files = []
    folder2_files = []

    for file in folder1:
        if file.filename:
            if file.filename.endswith('.jpg') or file.filename.endswith('.png') or file.filename.endswith('.jpeg'):
                folder1_files.append(file)
            elif os.path.isdir(file):
                for root, _, files in os.walk(file):
                    for f in files:
                        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                            folder1_files.append(os.path.join(root, f))

    for file in folder2:
        if file.filename:
            if file.filename.endswith('.jpg') or file.filename.endswith('.png') or file.filename.endswith('.jpeg'):
                folder2_files.append(file)
            elif os.path.isdir(file):
                for root, _, files in os.walk(file):
                    for f in files:
                        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'):
                            folder2_files.append(os.path.join(root, f))

    folder1_path = os.path.join(app.root_path, 'folder1')
    folder2_path = os.path.join(app.root_path, 'folder2')

    os.makedirs(folder1_path, exist_ok=True)
    os.makedirs(folder2_path, exist_ok=True)

    for file in folder1_files:
        if isinstance(file, str):
            file_object = open(file, 'rb')
            filename = os.path.basename(file)
        else:
            file_object = file
            filename = secure_filename(file.filename)
        file_object.save(os.path.join(folder1_path, filename))

    for file in folder2_files:
        if isinstance(file, str):
            file_object = open(file, 'rb')
            filename = os.path.basename(file)
        else:
            file_object = file
            filename = secure_filename(file.filename)
        file_object.save(os.path.join(folder2_path, filename))

    return render_template('save_as.html', folder1='folder1', folder2='folder2')

@app.route('/download', methods=['POST'])
def download():
    filename = request.form['filename']
    save_location = request.form['location']
    csv_file_path = os.path.join(save_location, f'{filename}.csv')

    folder1_path = os.path.join(app.root_path, 'folder1')
    folder2_path = os.path.join(app.root_path, 'folder2')

    process_folders([folder1_path, folder2_path], csv_file_path)

    return send_file(csv_file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

