import json
import os

import numpy as np
from keras import backend as K
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image
from yolo import YOLO

#----------------------------------------------------------------------------#
#   map_mode viene utilizzato per specificare cosa viene calcolato quando viene eseguito il file
# map_mode è 0 per rappresentare l'intero processo di calcolo della mappa, compreso l'ottenimento del risultato della previsione e il calcolo della mappa.
# map_mode è 1 significa che si ottiene solo il risultato della previsione.
# map_mode di 2 significa che si ottiene solo la mappa di calcolo.
#--------------------------------------------------------------------------#
map_mode            = 0
#-------------------------------------------------------#
#   Punta all'etichetta del set di convalida e al percorso dell'immagine
#-------------------------------------------------------#
cocoGt_path         = 'coco_dataset/annotations/instances_val2017.json'
dataset_img_path    = 'coco_dataset/val2017'
#-------------------------------------------------------#
#   La cartella per l'output dei risultati, l'impostazione predefinita è mappa
#-------------------------------------------------------#
temp_save_path      = 'map_out/coco_eval'

class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   Rileva immagini
    #---------------------------------------------------#
    def detect_image(self, image_id, image, results, clsid2catid):
        #---------------------------------------------------------#
        #   Converti qui l'immagine in un'immagine RGB per evitare che l'immagine in scala di grigi commetta errori durante la previsione.
        # Il codice supporta solo la previsione delle immagini RGB, tutti gli altri tipi di immagini verranno convertiti in RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Aggiungi barre grigie all'immagine per ottenere un ridimensionamento senza distorsioni
        # Puoi anche ridimensionare direttamente per l'identificazione
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Aggiungi la dimensione della dimensione del batch e normalizzala
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        for i, c in enumerate(out_classes):
            result                      = {}
            top, left, bottom, right    = out_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(out_scores[i])
            results.append(result)

        return results

if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = mAP_YOLO(confidence = 0.001, nms_iou = 0.65)

        with open(os.path.join(temp_save_path, 'eval_results.json'),"w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path  = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
                image       = Image.open(image_path)
                results     = yolo.detect_image(image_id, image, results)
            json.dump(results, f)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print("Get map done.")
