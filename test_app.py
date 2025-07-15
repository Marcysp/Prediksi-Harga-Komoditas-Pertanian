from sqlalchemy import create_engine, inspect, text

engine = create_engine('postgresql://alvina:alvina@127.0.0.1:6543/harga_komoditas')
conn = engine.connect()

inspector = inspect(engine)

columns = inspector.get_columns('hasil_prediksi')
column_names = [col['name'] for col in columns]

# Tampilkan kolom
print("📋 Kolom pada tabel 'hasil_prediksi':")
for col in column_names:
    print(f"- {col}")



# # Ambil semua nama kolom di tabel hasil_prediksi
# existing_columns = [col['name'] for col in inspector.get_columns('hasil_prediksi')]

# # Kolom yang ingin ditambahkan
# new_columns = ['val_16', 'val_46']

# with engine.connect() as conn:
#     trans = conn.begin()  # Mulai transaksi
#     try:
#         for col in new_columns:
#             if col not in existing_columns:
#                 print(f"Menambahkan kolom: {col}")
#                 conn.execute(text(f'ALTER TABLE hasil_prediksi ADD COLUMN "{col}" FLOAT'))
#             else:
#                 print(f"Kolom {col} sudah ada.")
#         trans.commit()  # Simpan perubahan
#     except Exception as e:
#         trans.rollback()  # Batalkan jika error
        # print("Gagal:", e)





# with engine.begin() as conn:  # Begin = otomatis commit
#     inspector = inspect(conn)
#     existing_columns = [col['name'] for col in inspector.get_columns('hasil_prediksi')]

#     # Tambah kolom 1..90 jika belum ada
#     for i in range(1, 91):
#         col_raw = str(i)
#         col_final = f"val_{i}"

#         if col_raw not in existing_columns and col_final not in existing_columns:
#             try:
#                 conn.execute(text(f'ALTER TABLE hasil_prediksi ADD COLUMN "{col_raw}" INTEGER;'))
#                 print(f"✅ Kolom '{col_raw}' dibuat sebagai INTEGER")
#             except Exception as e:
#                 print(f"⚠️ Gagal membuat kolom '{col_raw}': {e}")

#     # Refresh kolom setelah penambahan
#     inspector = inspect(conn)
#     existing_columns = [col['name'] for col in inspector.get_columns('hasil_prediksi')]

#     # Rename dan ubah tipe
#     for i in range(1, 91):
#         col_raw = str(i)
#         col_final = f"val_{i}"

#         # Rename kolom
#         if col_raw in existing_columns:
#             try:
#                 conn.execute(text(f'ALTER TABLE hasil_prediksi RENAME COLUMN "{col_raw}" TO "{col_final}";'))
#                 print(f"✅ Kolom '{col_raw}' diubah menjadi '{col_final}'")
#             except Exception as e:
#                 print(f"⚠️ Gagal rename kolom '{col_raw}': {e}")

#         # Ubah tipe data jadi DOUBLE PRECISION
#         if col_final in existing_columns or col_raw in existing_columns:
#             try:
#                 conn.execute(text(
#                     f'ALTER TABLE hasil_prediksi ALTER COLUMN "{col_final}" TYPE DOUBLE PRECISION USING "{col_final}"::DOUBLE PRECISION;'
#                 ))
#                 print(f"✅ Tipe data kolom '{col_final}' diubah ke DOUBLE PRECISION")
#             except Exception as e:
#                 print(f"⚠️ Gagal ubah tipe data '{col_final}': {e}")





# with engine.begin() as conn:
#     conn.execute(text("TRUNCATE TABLE hasil_prediksi RESTART IDENTITY CASCADE"))
#     print("✅ Semua data dari tabel 'hasil_prediksi' berhasil dihapus.")