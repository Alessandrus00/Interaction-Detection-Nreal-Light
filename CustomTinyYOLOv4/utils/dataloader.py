import math
from random import shuffle, sample

import cv2
import keras
import numpy as np
from PIL import Image
from utils.utils import cvtColor, preprocess_input


class YoloDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, epoch_now, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.anchors_mask       = anchors_mask
        self.epoch_now          = epoch_now - 1
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data  = []
        box_data    = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            #---------------------------------------------------#
            #   Aumento casuale dei dati durante l'allenamento
            # Non eseguire aumenti casuali dei dati durante la convalida
            #---------------------------------------------------#
            if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
                lines = sample(self.annotation_lines, 3)
                lines.append(self.annotation_lines[i])
                shuffle(lines)
                image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                    
                if self.mixup and self.rand() < self.mixup_prob:
                    lines           = sample(self.annotation_lines, 1)
                    image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                    image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
            else:
                image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            image_data.append(preprocess_input(np.array(image, np.float32)))
            box_data.append(box)

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)
        y_true      = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def on_epoch_end(self):
        self.epoch_now += 1
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, max_boxes=500, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   leggi l'immagine e converti in immagine rgb
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   Ottieni l'altezza e la larghezza dell'immagine e l'altezza e la larghezza di destinazione
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   ottenere la casella di previsione
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   Aggiungi barre grigie alle parti extra dell'immagine
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   Regola la scatola reale
            #---------------------------------#
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0]  = 0
                box[:, 2][box[:, 2]>w]      = w
                box[:, 3][box[:, 3]>h]      = h
                box_w   = box[:, 2] - box[:, 0]
                box_h   = box[:, 3] - box[:, 1]
                box     = box[np.logical_and(box_w>1, box_h>1)]
                if len(box)>max_boxes: box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data
                
        #------------------------------------------#
        #   Ridimensiona l'immagine e distorce la lunghezza e la larghezza
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   Aggiungi barre grigie alle parti extra dell'immagine
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   capovolgere l'immagine
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   Trasformazione della gamma di colori di un'immagine
        # Calcola i parametri della trasformazione della gamma cromatica
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   trasferire l'immagine in hsv
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   applicare la trasformazione
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   Regola la scatola reale
        #---------------------------------#
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box
        
        return image_data, box_data

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=500, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   dividere ogni riga
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   immagine aperta
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   dimensione dell'immagine
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   salva la posizione della casella
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            #---------------------------------#
            #   se capovolgere l'immagine
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            #------------------------------------------#
            #   Ridimensiona l'immagine e distorce la lunghezza e la larghezza
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   Posizionare le immagini corrispondenti alle posizioni delle quattro immagini divise
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #  Rielabora la scatola
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   Dividi le immagini e mettile insieme
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   Trasformazione della gamma di colori di un'immagine
        # Calcola i parametri della trasformazione della gamma cromatica
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   trasferire l'immagine in hsv
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   applicare la trasformazione
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   Eseguire ulteriori elaborazioni sulla scatola
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        #---------------------------------#
        #   regolare la scatola
        #---------------------------------#
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes)>0:
            if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2, max_boxes=500):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        
        box_1_wh    = box_1[:, 2:4] - box_1[:, 0:2]
        box_1_valid = box_1_wh[:, 0] > 0
        
        box_2_wh    = box_2[:, 2:4] - box_2[:, 0:2]
        box_2_valid = box_2_wh[:, 0] > 0
        
        new_boxes = np.concatenate([box_1[box_1_valid, :], box_2[box_2_valid, :]], axis=0)
        #---------------------------------#
        #   regolare la scatola
        #---------------------------------#
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes)>0:
            if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
        #-----------------------------------------------------------#
        #   Ottieni le coordinate della scatola e le dimensioni dell'immagine
        #-----------------------------------------------------------#
        true_boxes  = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        
        #-----------------------------------------------------------#
        #   Sono disponibili tre livelli di funzionalità
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        #-----------------------------------------------------------#
        #  M è il numero di immagini, le forme della griglia è la forma della griglia
        #-----------------------------------------------------------#
        m           = true_boxes.shape[0]
        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        #-----------------------------------------------------------#
        #   Il formato di Y vero è (m,13,13,3,85)(m,26,26,3,85)
        #-----------------------------------------------------------#
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                    dtype='float32') for l in range(num_layers)]

        #-----------------------------------------------------------#
        #   Ottieni il centro, la larghezza e l'altezza della scatola reale mediante calcolo
        # punto centrale(m,n,2) larghezza e altezza(m,n,2)
        #-----------------------------------------------------------#
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]
        #-----------------------------------------------------------#
        #   Normalizza la casella della verità fondamentale in forma decimale
        #-----------------------------------------------------------#
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        #-----------------------------------------------------------#
        #   [9,2] -> [1,9,2]
        #-----------------------------------------------------------#
        anchors         = np.expand_dims(anchors, 0)
        anchor_maxes    = anchors / 2.
        anchor_mins     = -anchor_maxes

        #-----------------------------------------------------------#
        #   La lunghezza e la larghezza devono essere maggiori di 0 per essere valide
        #-----------------------------------------------------------#
        valid_mask = boxes_wh[..., 0]>0

        for b in range(m):
            #-----------------------------------------------------------#
            #   elaborare ogni immagine
            #-----------------------------------------------------------#
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            #-----------------------------------------------------------#
            #   [n,2] -> [n,1,2]
            #-----------------------------------------------------------#
            wh          = np.expand_dims(wh, -2)
            box_maxes   = wh / 2.
            box_mins    = - box_maxes

            #-----------------------------------------------------------#
            #   Calcola il rapporto di intersezione di tutte le caselle della verità e delle caselle precedenti
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            #-----------------------------------------------------------#
            intersect_mins  = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area    = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)
            #-----------------------------------------------------------#
            #  La dimensione è [n,] grazie per il promemoria
            #-----------------------------------------------------------#
            best_anchor = np.argmax(iou, axis=-1)
            sort_anchor = np.argsort(iou, axis=-1)

            def check_in_anchors_mask(index, anchors_mask):
                for sub_anchors_mask in anchors_mask:
                    if index in sub_anchors_mask:
                        return True
                return False

            for t, n in enumerate(best_anchor):
                #----------------------------------------#
                #   Impedire che la casella a priori abbinata non sia nella maschera degli ancoraggi
                #----------------------------------------#
                if not check_in_anchors_mask(n, self.anchors_mask):
                    for index in sort_anchor[t][::-1]:
                        if check_in_anchors_mask(index, self.anchors_mask):
                            n = index
                            break
                #-----------------------------------------------------------#
                #   Trova il livello di funzionalità a cui appartiene ciascuna scatola della verità di base
                #-----------------------------------------------------------#
                for l in range(num_layers):
                    if n in self.anchors_mask[l]:
                        #-----------------------------------------------------------#
                        #   Il pavimento viene utilizzato per arrotondare per difetto per trovare le coordinate degli assi x e y corrispondenti al feature layer a cui appartiene il frame reale
                        #-----------------------------------------------------------#
                        i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                        #-----------------------------------------------------------#
                        #   Il k-esimo fotogramma a priori del punto di caratteristica corrente indicato da K
                        #-----------------------------------------------------------#
                        k = self.anchors_mask[l].index(n)
                        #-----------------------------------------------------------#
                        #   C si riferisce al tipo della scatola reale corrente
                        #-----------------------------------------------------------#
                        c = true_boxes[b, t, 4].astype('int32')
                        #-----------------------------------------------------------#
                        #   La forma di y_true è (m, 13, 13, 3, 85) (m, 26, 26, 3, 85)
                        # Gli ultimi 85 possono essere suddivisi in 4+1+80, 4 rappresenta il centro, la larghezza e l'altezza del riquadro,
                        # 1 rappresenta la fiducia, 80 rappresenta la categoria
                        #-----------------------------------------------------------#
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1

        return y_true
