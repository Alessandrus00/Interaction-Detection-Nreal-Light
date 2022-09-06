import math
from functools import partial

import tensorflow as tf
from keras import backend as K
from utils.utils_bbox import get_anchors_and_decode

def box_ciou(b1, b2):
    """
    输入为：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回为：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    #-----------------------------------------------------------#
    #   Trova l'angolo in alto a sinistra e l'angolo in basso a destra della casella di previsione
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    #-----------------------------------------------------------#
    #   Trova l'angolo in alto a sinistra e l'angolo in basso a destra della scatola reale
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    #-----------------------------------------------------------#
    #   Trova tutti gli iou della scatola reale e della scatola prevista
    #   iou         (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    #-----------------------------------------------------------#
    #   Divario del centro di calcolo
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    #-----------------------------------------------------------#
    #   Calcola la distanza diagonale
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())
    
    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    return ciou

#---------------------------------------------------#
#   etichetta liscia
#---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    
#---------------------------------------------------#
#   L'iou utilizzato per calcolare ogni casella prevista rispetto alla casella della verità fondamentale
#---------------------------------------------------#
def box_iou(b1, b2):
    #---------------------------------------------------#
    #   num_anchor,1,4
    #   Calcola le coordinate dell'angolo in alto a sinistra e le coordinate dell'angolo in basso a destra
    #---------------------------------------------------#
    b1          = K.expand_dims(b1, -2)
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    #---------------------------------------------------#
    #   1,n,4
    #   Calcola le coordinate degli angoli in alto a sinistra e in basso a destra
    #---------------------------------------------------#
    b2          = K.expand_dims(b2, 0)
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #---------------------------------------------------#
    #   Calcola l'area di sovrapposizione
    #---------------------------------------------------#
    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)
    return iou

#---------------------------------------------------#
#   Calcolo del valore di perdita
#---------------------------------------------------#
def yolo_loss(
    args, 
    input_shape, 
    anchors, 
    anchors_mask, 
    num_classes, 
    ignore_thresh   = 0.5,
    balance         = [0.4, 1.0, 4], 
    box_ratio       = 0.05, 
    obj_ratio       = 1, 
    cls_ratio       = 0.5 / 4, 
    label_smoothing = 0.1,
    print_loss      = False, 
):
    num_layers = len(anchors_mask)
    #---------------------------------------------------------------------------------------------------#
    # Separare il risultato della previsione dalla verità di base effettiva, args è [*model_body.output, *y_true]
    # y_true è un elenco contenente tre feature layer con forme:
    # (m,13,13,3,85)
    # (m,26,26,3,85)
    # yolo_outputs è un elenco contenente tre livelli di funzionalità con forme:
    # (m,13,13,3,85)
    # (m,26,26,3,85)
    #---------------------------------------------------------------------------------------------------#
    y_true          = args[num_layers:]
    yolo_outputs    = args[:num_layers]

    #-----------------------------------------------------------#
    #   ottieni input shpae come 416.416
    #-----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))

    #-----------------------------------------------------------#
    #   scatta ogni foto
    # Il valore di m è batch_size
    #-----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]

    loss    = 0
    #---------------------------------------------------------------------------------------------------#
    #   y_true è un elenco contenente tre feature layer con forme (m, 13, 13, 3, 85), (m, 26, 26, 3, 85).
    # yolo_outputs è un elenco contenente tre livelli di funzionalità con forme (m, 13, 13, 3, 85), (m, 26, 26, 3, 85).
    #---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        #-----------------------------------------------------------#
        #   Prendi il primo feature layer (m, 13, 13, 3, 85) come esempio
        # Elimina la posizione del punto in cui si trova il target nel feature layer. (m,13,13,3,1)
        #-----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        #-----------------------------------------------------------#
        #   Estrarre il tipo corrispondente (m, 13, 13, 3, 80)
        #-----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        #-----------------------------------------------------------#
        #   Elabora l'output del feature layer di yolo_outputs e ottieni quattro valori di ritorno
        #   in:
        # coordinate della griglia (13,13,1,2).
        # raw_pred (m,13,13,3,85) risultati di previsione non elaborati
        # pred_xy (m,13,13,3,2) coordinate centrali decodificate
        # pred_wh (m,13,13,3,2) decodificate le coordinate di larghezza e altezza
        #-----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
             anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)
        
        #-----------------------------------------------------------#
        #   pred_box è la posizione della casella prevista decodificata
        # (m,13,13,3,4)
        #-----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        #-----------------------------------------------------------#
        #   Per trovare gruppi di campioni negativi, il primo passaggio consiste nel creare un array, []
        #-----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        
        #-----------------------------------------------------------#
        #   Calcola ignora maschera per ogni immagine
        #-----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            #-----------------------------------------------------------#
            #  Estrarre n scatole reali: n, 4
            #-----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            #-----------------------------------------------------------#
            #   Calcola l'iou della casella prevista e della casella della verità fondamentale
            # pred_box 13,13,3,4 coordinate della casella di previsione
            # true_box n,4 coordinate del true_box
            # iou 13,13,3,n Iou di scatola predetta e scatola della verità fondamentale
            #-----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            #-----------------------------------------------------------#
            #  best_iou 13,13,3 Il massimo grado di coincidenza tra ciascun punto caratteristico e la scatola reale
            #-----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            #-----------------------------------------------------------#
            #   A giudicare che il massimo iou della casella prevista e della casella reale è inferiore a ignore_thresh
            # pensare che la casella prevista non abbia una casella reale corrispondente
            # Lo scopo di questa operazione è di:
            # Ignora i punti caratteristici che indicano che il risultato della previsione corrisponde molto bene al riquadro reale, perché questi riquadri sono già relativamente precisi
            # non è adatto come campione negativo, quindi ignoralo.
            #-----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask

        #-----------------------------------------------------------#
        #   Fai un loop in questo posto, il loop viene eseguito per ogni immagine
        #-----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])

        #-----------------------------------------------------------#
        #   ignore_mask viene utilizzato per estrarre punti caratteristica come campioni negativi
        # (m,13,13,3)
        #-----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        #-----------------------------------------------------------#
        #   Più grande è la scatola reale, minore è la proporzione e maggiore è la proporzione della scatola piccola.
        # Quando si utilizza la perdita di iou, la perdita di regressione di obiettivi grandi, medi e piccoli non presenta il problema dello squilibrio proporzionale, quindi viene scartata
        #-----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        #-----------------------------------------------------------#
        #   Calcola la perdita di Ciou
        #-----------------------------------------------------------#
        raw_true_box    = y_true[l][...,0:4]
        ciou            = box_ciou(pred_box, raw_true_box)
        ciou_loss       = object_mask * (1 - ciou)
        location_loss   = K.sum(ciou_loss)
        
        #------------------------------------------------------------------------------#
        #   Se c'è una casella nella posizione, calcola l'entropia incrociata di 1 e la confidenza
        # Se non c'è una casella nella posizione, calcolare l'entropia incrociata di 0 e la confidenza
        # Alcuni campioni verranno ignorati in questo, e questi campioni ignorati soddisfano la condizione best_iou<ignore_thresh
        # Lo scopo di questa operazione è di:
        # Ignora i punti caratteristici che indicano che il risultato della previsione corrisponde molto bene al riquadro reale, perché questi riquadri sono già relativamente precisi
        # non è adatto come campione negativo, quindi ignoralo.
        #------------------------------------------------------------------------------#
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
                    (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        
        class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        #-----------------------------------------------------------#
        #   Calcola il numero di campioni positivi
        #-----------------------------------------------------------#
        num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)
        
        #-----------------------------------------------------------#
        #   Somma tutte le perdite
        #-----------------------------------------------------------#
        location_loss   = location_loss * box_ratio / num_pos
        confidence_loss = K.sum(confidence_loss) * balance[l] * obj_ratio / (num_pos + num_neg)
        class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        loss            += location_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, tf.shape(ignore_mask)], summarize=100, message='loss: ')
    return loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
