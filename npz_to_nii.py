import numpy as np
import nibabel as nib
import os

input_dir = "D:/vesselFM/npz_data"
output_dir = "D:/vesselFM/my_data"

# Создаем папку, если ее нет
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Создана папка: {output_dir}")

# Смотрим, что вообще лежит в input_dir
print("Файлы в папке с данными:", os.listdir(input_dir))

for filename in os.listdir(input_dir):
    if filename.endswith(".npz"):
        print(f"Обрабатываю: {filename}...")
        filepath = os.path.join(input_dir, filename)
        
        try:
            # Загружаем
            data = np.load(filepath)
            array = data['my_array']
            
            # Сохраняем
            out_name = filename.replace(".npz", ".nii.gz")
            out_path = os.path.join(output_dir, out_name)
            
            nifti_img = nib.Nifti1Image(array, affine=np.eye(4))
            nib.save(nifti_img, out_path)
            
            print(f"  УСПЕХ! Файл сохранен: {out_path}")
            print(f"  Проверка: файл существует? {os.path.exists(out_path)}")
            
        except Exception as e:
            print(f"  ОШИБКА: {e}")

print("Готово.")