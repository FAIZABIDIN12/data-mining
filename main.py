from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from io import BytesIO, StringIO
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    predicted_sentiment = db.Column(db.String(20), nullable=True)

# Buat tabel di database (jalankan sekali pada awal pembuatan database)
with app.app_context():
    db.create_all()

# Fungsi untuk training model Naive Bayes
def train_naive_bayes(X, y):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X, y)
    return model

# Fungsi untuk melakukan prediksi sentimen
def predict_sentiment(model, data):
    return model.predict(data)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Tidak ada bagian berkas')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='Tidak ada berkas yang dipilih')

        if file:
            try:
                # Simpan data file ke database
                content = file.read().decode('utf-8')
                uploaded_file = UploadedFile(filename=file.filename, content=content)
                db.session.add(uploaded_file)
                db.session.commit()

                # Baca file CSV setelah menyimpannya ke database
                df = pd.read_csv(StringIO(content))

                # Pernyataan cetak untuk mengecek konten DataFrame
                print(df.head())

                # Periksa apakah DataFrame memiliki data
                if df.empty:
                    return render_template('index.html', error='DataFrame kosong setelah membaca berkas')

            except pd.errors.EmptyDataError:
                return render_template('index.html', error='DataFrame kosong setelah membaca berkas')
            except Exception as e:
                return render_template('index.html', error=f'Error: {str(e)}')

    # Ambil semua file yang telah diunggah dari database
    uploaded_files = UploadedFile.query.all()

    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/analyze/<int:file_id>')
def analyze(file_id):
    uploaded_file = UploadedFile.query.get(file_id)

    if not uploaded_file:
        return render_template('index.html', error='File not found.')

    try:
        # Periksa apakah konten merupakan byte atau string
        if isinstance(uploaded_file.content, bytes):
            # Gunakan BytesIO jika konten merupakan bytes
            df = pd.read_csv(BytesIO(uploaded_file.content))
        else:
            # Jika konten merupakan string, tidak perlu menggunakan BytesIO
            df = pd.read_csv(StringIO(uploaded_file.content))

        # Pernyataan cetak untuk mengecek konten DataFrame
        print(df.head())

        # Periksa apakah DataFrame memiliki data
        if df.empty:
            return render_template('index.html', error='DataFrame kosong setelah membaca berkas')

    except pd.errors.EmptyDataError:
        return render_template('index.html', error='DataFrame kosong setelah membaca berkas')
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}')

    # Melanjutkan dengan analisis jika tidak ada kesalahan
    texts = df['ulasan'].values
    labels = df['label'].values

    # Training model Naive Bayes
    model = train_naive_bayes(texts, labels)

    # Prediksi sentimen
    predictions = predict_sentiment(model, texts)

    # Menambahkan prediksi ke DataFrame
    df['predicted_sentiment'] = predictions

    # Render template dengan DataFrame
    return render_template('index.html', file_content=df.to_dict(orient='records'), show_button=True)

# fungsi delete
@app.route('/delete_file/<int:file_id>', methods=['GET'])
def delete_file(file_id):
    uploaded_file = UploadedFile.query.get(file_id)

    if not uploaded_file:
        return render_template('index.html', error='File not found.')

    try:
        # Hapus file dari database
        db.session.delete(uploaded_file)
        db.session.commit()

        # Redirect kembali ke halaman utama setelah menghapus
        return redirect(url_for('index'))
    except Exception as e:
        # Tangani kesalahan jika terjadi
        return render_template('index.html', error=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
