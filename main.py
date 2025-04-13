import cv2
import os
import easyocr
import numpy as np
from ultralytics import YOLO

model = YOLO('carplate_model/weights/best.pt')

l = os.listdir('plates/')

score_all = [] #Список всех значений уверенности
score_b5 = [] #Список всех уверенностей >0.5

#перебираем изображения из папки
for i in l:
    img = cv2.imread(f'plates/{i}')

    #Подрубаем модель и получаем координаты боксов
    results = model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)

    #Вырезаем номера по координатам
    for x, y, w, h in boxes:
        carplate_img = img[y:h, x:w]
        carplate_img_gray = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2GRAY)

        #Считываем номера
        reader = easyocr.Reader(['ru'], gpu = False)
        data = reader.readtext(carplate_img_gray, allowlist='АВEКМНОРСТУХ0123456789',
                                       contrast_ths=8, adjust_contrast=0.85, add_margin=0.015, width_ths=20,
                                       decoder='beamsearch', text_threshold=0.1, batch_size=8, beamWidth=32)
        text_full = ''
        for l in data:
            bbox, text, score = l
            score_all.append(score)
            if score > 0.5:
                score_b5.append(score)
                text_full += text
                text_full = text_full.upper().replace(' ', '')
            print(score)
        print(text_full)

        #Отрисовываем всю инфу на изображении
        # final_img = cv2.putText(img, text_full, (x, y - 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
        # final_img = cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        # final_img = cv2.resize(final_img, (final_img.shape[1] // 2, final_img.shape[0] // 2))

        # cv2.imshow('result', carplate_img)
        # cv2.waitKey(0)


#Средние значения уверенности
average_score_all = sum(score_all) / len(score_all)
average_score_b5 = sum(score_b5) / len(score_b5)
print('Средний score:', average_score_all)
print('Средний score>0.5:',average_score_b5)