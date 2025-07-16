# import library
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from sqlalchemy import create_engine, inspect, text
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import load_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from scipy.stats import boxcox
from scipy.special import inv_boxcox 
import numpy as np
import joblib
import warnings
import itertools 
warnings.filterwarnings("ignore")

# membuat koneksi ke database PostgreSQL
# engine = create_engine('postgresql://alvina:alvina@127.0.0.1:6543/harga_komoditas')
engine = create_engine('postgresql://alvina:alvina@192.168.60.108:5432/harga_komoditas')
# Inisiasi Object Flask
app = Flask(__name__)

# Inisiasi Object Flask Restful
api = Api(app)

# Inisiasi Object Flask CORS
CORS(app)

# model_cabeRawit = None
model_bawang_merah = None
model_tomat_merah = None
model_cabe_besar = None
model_bawang_putih = load_model('models/model_bwgPth.keras')
scaler_bawang_putih = joblib.load('models/scaler_bwgPth.joblib')
model_cabe_rawit = load_model('models/model_cabe_rawit.keras')
scaler_cabe_rawit = joblib.load('models/scaler_cabe_rawit.joblib')
data = None
last_optimized_bawang_merah = None
last_optimized_tomat_merah = None
optimized_params_bawang_merah = None
optimized_params_tomat_merah = None

@app.route('/api/forecast/nama_komoditas', methods=['GET'])
def get_komoditas_names():
    query = '''
        SELECT DISTINCT nama_komoditas
        FROM hasil_prediksi
        ORDER BY nama_komoditas;
    '''
    with engine.connect() as conn:
        result = conn.execute(text(query))
        komoditas_names = [row[0] for row in result]
    return komoditas_names

def load_data():
    global data
    query = '''
        SELECT 
            krr.tanggal,  
            krr.komoditas_nama,
            krr.harga
        FROM "komoditas_rata-rata" as krr
        WHERE komoditas_nama IN (
            'Cabe Rawit Merah',
            'Bawang Merah',
            'Cabe Merah Besar',
            'Bawang Putih Sinco/Honan',
            'Tomat Merah'
        )
        ORDER BY krr.tanggal ASC;
    '''
    data = pd.read_sql(query, engine)
    data['tanggal'] = pd.to_datetime(data['tanggal'])
    data.set_index('tanggal', inplace=True)
    return data

def optimize_ets(train, test, seasonal_periods=30, max_alpha=0.9, max_beta=0.9, max_gamma=0.9):
        
        alpha = np.round(np.arange(0.1, max_alpha, 0.1), 2)
        beta = np.round(np.arange(0.1, max_beta, 0.1), 2)
        gamma = np.round(np.arange(0.1, max_gamma, 0.1), 2)
        
        abg_combinations = list(itertools.product(alpha, beta, gamma))
        
        best_params = {
            'alpha': None,
            'beta': None,
            'gamma': None,
            'mae': float('inf')
        }
        
        for alpha, beta, gamma in abg_combinations:
            try:
                # Model untuk data harian dengan seasonal period 7 (mingguan)
                model = ExponentialSmoothing(
                    train,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods,  # 7 untuk pola mingguan
                    initialization_method='estimated'
                ).fit(
                    smoothing_level=alpha,
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma
                )
                
                # Forecast
                forecast = model.forecast(len(test))
                
                # Evaluasi
                current_mae = mean_absolute_error(test, forecast)
                
                # Update best params
                if current_mae < best_params['mae']:
                    best_params.update({
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'mae': current_mae
                    })
                
                print(f"Alpha: {alpha:.2f} | Beta: {beta:.2f} | Gamma: {gamma:.2f} | MAE: {current_mae:.2f}")
                
            except Exception as e:
                print(f"Error with alpha={alpha}, beta={beta}, gamma={gamma}: {str(e)}")
                continue
        print(f"Best Params: {best_params}")
        print(f"Best MAE: {best_params['mae']:.2f}")
        return best_params

@app.route('/hasil_prediksi/kolom', methods=['GET'])
def get_hasil_prediksi_columns():
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns('hasil_prediksi')
        column_names = [col['name'] for col in columns]
        return jsonify({
            'columns': column_names
        })
    except Exception as e:
        return jsonify({'error': f"Gagal mengambil nama kolom: {str(e)}"}), 500

@app.route('/data', methods=['GET'])
def get_data():
    # query = '''
    #     SELECT *
    #     FROM hasil_prediksi where tanggal <= current_date - interval '30 days';
    # '''
    query = '''
        SELECT * FROM hasil_prediksi;
    '''
    data = pd.read_sql(query, engine)
    data['tanggal'] = pd.to_datetime(data['tanggal'])
    data.set_index('tanggal', inplace=True)

    # Convert to JSON format
    data_json = data.to_dict(orient='records')
    return jsonify(data_json)

# 
def forecast_bawang_merah(data=None, steps=90):
    global model_bawang_merah, last_optimized_bawang_merah, optimized_params_bawang_merah

    # Filter data hanya untuk Bawang Merah
    bawang_merah_data = data[data['komoditas_nama'] == 'Bawang Merah'].copy()

    # Gunakan hanya kolom harga
    harga_series = bawang_merah_data['harga']

    # Cek apakah sudah lewat 7 hari dari terakhir optimasi
    if last_optimized_bawang_merah is None or (datetime.now().date() - last_optimized_bawang_merah).days >= 7:
        print("🔁 Re-optimizing parameters...")
        optimized_params_bawang_merah = optimize_ets(
            harga_series.iloc[:-90],
            harga_series.iloc[-90:]
        )
        last_optimized_bawang_merah = datetime.now().date()
    else:
        print("✅ Using existing optimized parameters")

    # Fit model harian (selalu dilakukan)
    model_bawang_merah = ExponentialSmoothing(
        harga_series,
        trend="add",
        seasonal="add",
        seasonal_periods=30,
        initialization_method='estimated'
    ).fit(
        smoothing_level=optimized_params_bawang_merah['alpha'],
        smoothing_trend=optimized_params_bawang_merah['beta'],
        smoothing_seasonal=optimized_params_bawang_merah['gamma']
    )

    # Forecast
    forecast_result = model_bawang_merah.forecast(steps=steps)
    nama_komoditas = 'Bawang Merah'
    return forecast_result, nama_komoditas

def forecast_tomat_merah(data=None, steps=90):
    global model_tomat_merah, last_optimized_tomat_merah, optimized_params_tomat_merah

    # Filter data hanya untuk Bawang Merah
    tomat_merah_data = data[data['komoditas_nama'] == 'Tomat Merah'].copy()

    # Gunakan hanya kolom harga
    harga_series = tomat_merah_data['harga']

    # Cek apakah sudah lewat 7 hari dari terakhir optimasi
    if last_optimized_tomat_merah is None or (datetime.now().date() - last_optimized_tomat_merah).days >= 7:
        print("🔁 Re-optimizing parameters...")
        optimized_params_tomat_merah = optimize_ets(
            harga_series.iloc[:-90],
            harga_series.iloc[-90:]
        )
        last_optimized_tomat_merah = datetime.now().date()
    else:
        print("✅ Using existing optimized parameters")

    # Fit model harian (selalu dilakukan)
    model_tomat_merah = ExponentialSmoothing(
        harga_series,
        trend="add",
        seasonal="add",
        seasonal_periods=30,
        initialization_method='estimated'
    ).fit(
        smoothing_level=optimized_params_tomat_merah['alpha'],
        smoothing_trend=optimized_params_tomat_merah['beta'],
        smoothing_seasonal=optimized_params_tomat_merah['gamma']
    )

    # Forecast
    forecast_result = model_tomat_merah.forecast(steps=steps)
    nama_komoditas = 'Tomat Merah'
    return forecast_result, nama_komoditas

def forecast_cabe_besar(data, order=(2,0,2), seasonal_order=(1,0,1,30), steps=90):
    global model_cabe_besar

    cabe_besar_data = data[data['komoditas_nama'] == 'Cabe Merah Besar'].copy()
    # Gunakan hanya kolom harga
    harga_series = cabe_besar_data['harga']
    
    model_cabe_besar = SARIMAX(harga_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model_cabe_besar.fit()
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    nama_komoditas = 'Cabe Merah Besar'
    return forecast_mean, nama_komoditas

def forecast_cabe_rawit(data, order=(2,0,2), seasonal_order=(1,0,1,30), steps=90):
    global model_cabe_besar

    cabe_besar_data = data[data['komoditas_nama'] == 'Cabe Merah Besar'].copy()
    # Gunakan hanya kolom harga
    harga_series = cabe_besar_data['harga']
    
    model_cabe_besar = SARIMAX(harga_series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model_cabe_besar.fit()
    forecast = results.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    return forecast_mean

# Fungsi untuk membuat input sequence
def create_input_sequence(data, look_back=60):
    sequence = data[-look_back:].reshape(1, look_back, 1)
    return sequence

def forecast_bawang_putih(data):
    print("🔮 Melakukan prediksi bawang putih...")
    try:
        bawang_putih_data = data[data['komoditas_nama'] == 'Bawang Putih Sinco/Honan'].copy()
        prices = bawang_putih_data['harga'].values

        print(f"Jumlah data harga bawang putih: {len(prices)}")

        if np.any(prices <= 0):
            raise ValueError("Data harga mengandung nilai <= 0, tidak bisa gunakan Box-Cox")

        # === Box-Cox Transform ===
        boxcox_transformed, lambda_bc = boxcox(prices)

        # Scaling
        harga_scaled = scaler_bawang_putih.transform(boxcox_transformed.reshape(-1, 1))

        # Buat input untuk prediksi
        look_back = 90
        preds_scaled = []
        input_seq = create_input_sequence(harga_scaled, look_back)
        steps = 90  # Jumlah langkah prediksi

        # for _ in range(steps):
        #     pred = model.predict(input_seq, verbose=0)
        #     preds_scaled.append(pred[0][0])
        #     input_seq = np.append(input_seq[:, 1:, :], [[[pred[0][0]]]], axis=1)
        # Lakukan prediksi sekali saja
        preds_scaled = model_bawang_putih.predict(input_seq)

        # Invers transformasi
        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds_boxcox = scaler_bawang_putih.inverse_transform(preds_scaled).flatten()
        preds = inv_boxcox(preds_boxcox, lambda_bc)

        print(f"Prediksi bawang putih: {preds}")
        komoditas_nama = 'Bawang Putih Sinco/Honan'

        # Buat tanggal hasil prediksi
        last_date = bawang_putih_data.index.max()
        forecast_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(steps)]

        return preds, komoditas_nama

    except Exception as e:
        return [], f"❌ Error: {str(e)}"

def forecast_cabe_rawit(data):
    print("🔮 Melakukan prediksi cabe rawit...")
    try:
        cabe_rawit_data = data[data['komoditas_nama'] == 'Cabe Rawit Merah'].copy()
        prices = cabe_rawit_data['harga'].values

        # print(f"Jumlah data harga cabe rawit: {prices.head()}")

        if np.any(prices <= 0):
            raise ValueError("Data harga mengandung nilai <= 0, tidak bisa gunakan Box-Cox")

        # === Box-Cox Transform ===
        boxcox_transformed, lambda_bc = boxcox(prices)

        # Scaling
        harga_scaled = scaler_cabe_rawit.transform(boxcox_transformed.reshape(-1, 1))

        # Buat input untuk prediksi
        look_back = 180
        preds_scaled = []
        input_seq = create_input_sequence(harga_scaled, look_back)
        steps = 90  # Jumlah langkah prediksi

        # Lakukan prediksi sekali saja
        preds_scaled = model_cabe_rawit.predict(input_seq)

        # Invers transformasi
        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        preds_boxcox = scaler_cabe_rawit.inverse_transform(preds_scaled).flatten()
        preds = inv_boxcox(preds_boxcox, lambda_bc)
        nama_komoditas = 'Cabe Rawit Merah'

        print(f"Prediksi bawang putih: {preds}")

        # Buat tanggal hasil prediksi
        # last_date = bawang_putih_data.index.max()
        # forecast_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(steps)]

        return preds, nama_komoditas

    except Exception as e:
        return [], f"❌ Error: {str(e)}"

def insert_data(values, nama_komoditas):
    if isinstance(values, str):  # Artinya error string dikembalikan
        return jsonify({'error': values}), 500

    try:
        if len(values) < 90:
            raise ValueError("Jumlah data prediksi kurang dari 90")

        tanggal_prediksi = datetime.today().strftime('%Y-%m-%d')
        kolom_nilai = {f"val_{i + 1}": float(values[i]) for i in range(90)}

        columns = ', '.join(['tanggal', 'nama_komoditas'] + list(kolom_nilai.keys()))
        placeholders = ', '.join([':tanggal', ':nama_komoditas'] + [f':{k}' for k in kolom_nilai])

        params = {
            'tanggal': tanggal_prediksi,
            'nama_komoditas': nama_komoditas,
            **kolom_nilai
        }

        query = text(f"""
            INSERT INTO hasil_prediksi ({columns})
            VALUES ({placeholders})
        """)

        with engine.begin() as conn:
            conn.execute(query, params)

        print(f"✅ Prediksi untuk {nama_komoditas} berhasil disimpan.")
    except Exception as e:
        print(f"❌ Gagal simpan ke database: {e}")
        return jsonify({'error': f"Gagal simpan ke database: {str(e)}"}), 500

        # Gabung ke dalam 1
@app.route('/forecast', methods=['GET'])
def forecast():
    global data
    data = load_data()

    forecast_functions = [
        forecast_bawang_merah,
        forecast_tomat_merah,
        forecast_cabe_besar,
        forecast_cabe_rawit,
        forecast_bawang_putih
    ]

    for func in forecast_functions:
        try:
            values, nama_komoditas = func(data)
            if isinstance(values, np.ndarray):
                values = values.tolist()
            insert_data(values, nama_komoditas)
        except Exception as e:
            print(f"❌ Gagal menyimpan hasil dari {func.__name__}: {e}")

    return jsonify({'status': '✅ Semua prediksi selesai diproses'})
    # return jsonify({
    #     # 'forecast_dates': dates,
    #     'forecast_values': values,
    # })
    # return jsonify({
    #     'forecast_dates': forecast_result.index.strftime('%Y-%m-%d').tolist(),
    #     'forecast_values': forecast_result.tolist(),
    #     'last_optimized': last_optimized.strftime('%Y-%m-%d') if last_optimized else None
    # })

# Menjalankan aplikasi
if __name__ == '__main__':
    # initialize()
    app.run(debug=True, host='0.0.0.0', port=5000)