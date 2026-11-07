import os
import glob
import SimpleITK as sitk

def convert_all(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "slice_*.vtk")))
    
    if not files:
        print(f"В папке {input_dir} нет VTK файлов.")
        return

    print(f"Конвертирую {len(files)} файлов из {input_dir}...")
    
    for f in files:
        img = sitk.ReadImage(f)
        # Сохраняем под тем же именем, но с расширением .nii.gz
        filename = os.path.basename(f).replace(".npz.vtk", ".nii.gz")
        out_path = os.path.join(output_dir, filename)
        sitk.WriteImage(img, out_path)
        print(f"  Сохранено: {filename}")

# --- ЗАПУСК ---
# Указываем, куда сохранять готовые NIfTI файлы
os.makedirs("D:/vesselFM/my_data20_ready", exist_ok=True)

# 1. Конвертируем неинвертированные
convert_all("D:/vesselFM/new_data_no_inv", "D:/vesselFM/my_data20_ready/no_inv")

# 2. Конвертируем инвертированные
convert_all("D:/vesselFM/new_data_inv", "D:/vesselFM/my_data20_ready/inv")