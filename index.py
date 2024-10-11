import os

# Путь к папке с аннотациями
annotations_path = 'D:\\LableImage\\data\\labels\\val'

for filename in os.listdir(annotations_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(annotations_path, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        with open(file_path, 'w') as file:
            for line in lines:
                # Заменяем 15 на 0
                file.write(line.replace('15', '0'))