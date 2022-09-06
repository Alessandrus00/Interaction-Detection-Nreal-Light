import datetime
import os
import sys

import keras.backend as K
import tensorflow as tf
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard)
from keras.layers import Conv2D, Dense, DepthwiseConv2D
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.utils.multi_gpu_utils import multi_gpu_model

from nets.yolo import get_train_model, yolo_body
from nets.yolo_training import get_lr_scheduler
from utils.callbacks import (EvalCallback, ExponentDecayScheduler, LossHistory,
                             ParallelModelCheckpoint,
                             WarmUpCosineDecayScheduler)
from utils.dataloader import YoloDatasets
from utils.utils import get_anchors, get_classes, show_config

tf.logging.set_verbosity(tf.logging.ERROR)

'''
When training your own target detection model, you must pay attention to the following points:
1. Before training, carefully check whether your format meets the requirements. The library requires the data set format to be VOC format, and the content to be prepared includes input pictures and labels
   The input image is a .jpg image, no fixed size is required, and it will be automatically resized before being passed into training.
   Grayscale images will be automatically converted to RGB images for training, no need to modify them yourself.
   If the suffix of the input image is not jpg, you need to convert it into jpg in batches before starting training.

   The tag is in .xml format, and the file contains target information to be detected. The tag file corresponds to the input image file.

2. The size of the loss value is used to judge whether or not to converge. The more important thing is that there is a trend of convergence, that is, the loss of the validation set continues to decrease. If the loss of the validation set basically does not change, the model basically converges.
   The specific size of the loss value does not make much sense. The big and small only depend on the calculation method of the loss, and it is not good to be close to 0. If you want to make the loss look better, you can directly divide 10000 into the corresponding loss function.
   The loss value during training will be saved in the loss_%Y_%m_%d_%H_%M_%S folder under the logs folder
3. The trained weight file is saved in the logs folder. Each training generation (Epoch) contains several training steps (Step), and each training step (Step) performs a gradient descent.
   If you only train a few Steps, it will not be saved. The concepts of Epoch and Step should be clarified.
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------#
    # train_gpu GPU used for training
    # Default is the first card, [0, 1] for two cards, [0, 1, 2] for three cards
    # When using multiple GPUs, the batch on each card is the total batch divided by the number of cards.
    #---------------------------------------------------------------------#
    train_gpu       = [0,]
    #---------------------------------------------------------------------#
    # classes_path points to the txt under model_data, which is related to the data set trained by yourself
    # Be sure to modify classes_path before training so that it corresponds to your own dataset
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/enigma_classes.txt'
    #---------------------------------------------------------------------#
    # Anchors_path represents the txt file corresponding to the a priori box, which is generally not modified.
    # anchors_mask is used to help the code find the corresponding a priori box and is generally not modified.
    # In YoloV4-Tiny, since the tiny model has a general recognition effect on small targets,
    # The official use is [[3, 4, 5], [1, 2, 3]],
    # The a priori box with the serial number of 0 is not used, so there is no need to worry too much.
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[3, 4, 5], [1, 2, 3]]
    #----------------------------------------------------------------------------------------------------------------------------#
    # Please refer to the README for the download of the weight file, which can be downloaded from the network disk. The pretrained weights of the model are common to different datasets because the features are common.
    # The more important part of the pre-training weight of the model is the weight part of the backbone feature extraction network, which is used for feature extraction.
    # Pre-training weights must be used in 99% of cases. If they are not used, the weights of the main part are too random, the feature extraction effect is not obvious, and the results of network training will not be good.
    #
    # If there is an operation that interrupts training during the training process, you can set model_path to the weights file in the logs folder, and load the weights that have been trained for part of the training again.
    # At the same time, modify the parameters of the freeze phase or thaw phase below to ensure the continuity of the model epoch.
    #
    # When model_path = '', do not load the weights of the entire model.
    #
    # The weights of the entire model are used here, so they are loaded in train.py.
    # If you want the model to start training from the pre-trained weights of the backbone, set model_path as the weight of the backbone network, and only load the backbone at this time.
    # If you want the model to start training from 0, set model_path = '', Freeze_Train = Fasle, then start training from 0, and there is no process of freezing the backbone.
    #
    # Generally speaking, the training effect of the network starting from 0 will be very poor, because the weights are too random, and the feature extraction effect is not obvious, so it is very, very, very not recommended for everyone to start training from 0!
    # There are two options for training from 0:
    # 1. Thanks to the powerful data enhancement capability of the Mosaic data enhancement method, when UnFreeze_Epoch is set to a larger value (300 and above), a larger batch (16 and above), and a large amount of data (above 10,000),
    # You can set mosaic=True and start training with random initialization parameters, but the effect is still not as good as pre-training. (big datasets like COCO can do this)
    # 2. To understand the imagenet data set, first train the classification model to obtain the weights of the backbone part of the network. The backbone part of the classification model is common to the model, and training is based on this.
    #----------------------------------------------------------------------------------------------------------------------------#
    if len(sys.argv)>1:
        model_path      = sys.argv[1]
    else:
        model_path      = 'model_data/yolov4_tiny_weights_coco.h5'
    #------------------------------------------------------#
    # input_shape     The size of the input shape must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [416, 416]
    #-------------------------------#
    # The type of attention mechanism used
    # phi = 0 to not use the attention mechanism
    # phi = 1 is SE
    # phi = 2 for CBAM
    # phi = 3 for ECA
    # phi = 4 is CA
    #-------------------------------#
    phi             = 0
    #------------------------------------------------------------------#
    # mosaic Mosaic data augmentation.
    # mosaic_prob How much probability each step uses mosaic data augmentation, the default is 50%.
    #
    # mixup Whether to use mixup data augmentation, only valid when mosaic=True.
    # Only the mosaic-enhanced images will be mixed up.
    # mixup_prob How many probability to use mixup data augmentation after mosaic, default 50%.
    # The total mixup probability is mosaic_prob *mixup_prob.
    #
    # special_aug_ratio Referring to YoloX, the training images generated by Mosaic are far away from the real distribution of natural images.
    # When mosaic=True, this code will enable mosaic in the range of special_aug_ratio.
    # The default is the first 70% of epochs, and 100 generations will start 70 generations.
    #
    # The parameters of the cosine annealing algorithm are set in the following lr_decay_type
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    # label_smoothing     Label smoothing. Generally below 0.01. Such as 0.01, 0.005.
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #----------------------------------------------------------------------------------------------------------------------------#
    # The training is divided into two phases, the freezing phase and the thawing phase. The freezing stage is set to meet the training needs of students with insufficient machine performance.
    # Freeze training requires a small amount of video memory, and when the graphics card is very poor, you can set Freeze_Epoch equal to UnFreeze_Epoch, and only freeze training at this time.
    #
    # Here are some suggestions for parameter settings, and trainers can flexibly adjust according to their own needs:
    # (1) Start training from the pre-trained weights of the entire model:
    # Adam:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (not frozen)
    # SGD:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (not frozen)
    # Where: UnFreeze_Epoch can be adjusted between 100-300.
    # (2) Start training from the pre-trained weights of the backbone network:
    # Adam:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (not frozen)
    # SGD:
    # Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (freeze)
    # Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (not frozen)
    # Among them: Since the training starts from the pre-training weights of the backbone network, the weights of the backbone are not necessarily suitable for target detection, and more training is required to jump out of the local optimal solution.
    # UnFreeze_Epoch can be adjusted between 150-300, 300 is recommended for both YOLOV5 and YOLOX.
    # Adam converges faster than SGD. Therefore, UnFreeze_Epoch can theoretically be smaller, but more Epochs are still recommended.
    # (3) Training from 0:
    # Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (do not freeze training)
    # Among them: UnFreeze_Epoch should not be less than 300 as much as possible. optimizer_type='sgd', Init_lr=1e-2, mosaic=True.
    # (4) Setting of batch_size:
    # Within the acceptable range of the graphics card, it is better to be great. Insufficient video memory has nothing to do with the size of the data set. If it indicates insufficient video memory (OOM or CUDA out of memory), please reduce the batch_size.
    # Affected by the BatchNorm layer, the minimum batch_size is 2 and cannot be 1.
    # Normally Freeze_batch_size is recommended to be 1-2 times of Unfreeze_batch_size. It is not recommended to set the gap too large, because it is related to the automatic adjustment of the learning rate.
    #----------------------------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------#
    # Freeze phase training parameters
    # At this time, the backbone of the model is frozen, and the feature extraction network does not change
    # Occupy less memory, only fine-tune the network
    # Init_Epoch The current training generation of the model, its value can be greater than Freeze_Epoch, such as setting:
    # Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100
    # will skip the freezing phase, start directly from generation 60, and adjust the corresponding learning rate.
    # (Used when retraining from a breakpoint)
    # Freeze_Epoch model Freeze_Epoch for freezing training
    # (failed when Freeze_Train=False)
    # Freeze_batch_size model freeze training batch_size
    # (failed when Freeze_Train=False)
    #------------------------------------------------------------------#
    if len(sys.argv)>2:
        Init_Epoch          = int(sys.argv[2])
    else:
        Init_Epoch          = 0

    Freeze_Epoch        = 50
    Freeze_batch_size   = 32
    #------------------------------------------------------------------#
    # Thawing phase training parameters
    # At this time, the backbone of the model is not frozen, and the feature extraction network will change
    # The occupied video memory is large, and all the parameters of the network will be changed
    # UnFreeze_Epoch The total epoch of model training
    # SGD takes longer to converge, so set a larger UnFreeze_Epoch
    # Adam can use a relatively small UnFreeze_Epoch
    # Unfreeze_batch_size The batch_size of the model after thawing
    #------------------------------------------------------------------#
    if len(sys.argv)>3:
        UnFreeze_Epoch      = int(sys.argv[3])
    else:
        UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    # Freeze_Train whether to perform freeze training
    # By default, the backbone training is frozen first and then the training is thawed.
    #------------------------------------------------------------------#
    Freeze_Train        = False
    
    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate drop related
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    # Init_lr maximum learning rate for the model
    # It is recommended to set Init_lr=1e-3 when using Adam optimizer
    # It is recommended to set Init_lr=1e-2 when using the SGD optimizer
    # Min_lr The minimum learning rate of the model, the default is 0.01 of the maximum learning rate
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    # optimizer_type The type of optimizer used, optional adam, sgd
    # It is recommended to set Init_lr=1e-3 when using Adam optimizer
    # It is recommended to set Init_lr=1e-2 when using the SGD optimizer
    # momentum parameter used internally by the momentum optimizer
    # weight_decay weight decay to prevent overfitting
    # adam will cause weight_decay error, it is recommended to set it to 0 when using adam.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   The learning rate decline method used by lr_decay_type, optional 'step', 'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period How many epochs to save the weights once
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        Folder where weights and log files are saved
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    # eval_flag Whether to evaluate during training, the evaluation object is the validation set
    # After installing the pycocotools library, the evaluation experience is better.
    # eval_period represents how many epochs to evaluate once, frequent evaluation is not recommended
    # Evaluation takes a lot of time, frequent evaluation will lead to very slow training
    # The mAP obtained here will be different from that obtained by get_map.py for two reasons:
    # (1) The mAP obtained here is the mAP of the validation set.
    # (2) The evaluation parameters are set conservatively here, in order to speed up the evaluation.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    # num_workers is used to set whether to use multi-threading to read data, 1 means to turn off multi-threading
    # When enabled, it will speed up data reading, but it will take up more memory
    # When multi-threading is enabled in keras, sometimes the speed is much slower
    # Turn on multithreading when IO is the bottleneck, that is, the GPU operation speed is much faster than the speed of reading pictures.
    #------------------------------------------------------------------#
    num_workers         = 1

    #------------------------------------------------------#
    # train_annotation_path training image path and label
    # val_annotation_path Validate image path and label
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #------------------------------------------------------#
    #   Set the graphics card used
    #------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    #----------------------------------------------------#
    #  Get classes and anchors
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    K.clear_session()
    #------------------------------------------------------#
    #   Create a yolo model
    #------------------------------------------------------#
    model_body  = yolo_body((input_shape[0], input_shape[1], 3), anchors_mask, num_classes, phi = phi)
    if model_path != '':
        #------------------------------------------------------#
        #   Load pretrained weights
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
        
    if ngpus_per_node > 1:
        model = multi_gpu_model(model_body, gpus=ngpus_per_node)
        model = get_train_model(model, input_shape, num_classes, anchors, anchors_mask, label_smoothing)
    else:
        model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing)
        
    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    show_config(
        classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )
    #---------------------------------------------------------#
    # The total training generation refers to the total number of times to traverse all the data
    # The total training step size refers to the total number of gradient descents
    # Each training epoch contains several training steps, and each training step performs a gradient descent.
    # Only the minimum training generation is recommended here, the upper limit is not capped, and only the thawed part is considered in the calculation
    #----------------------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] use%sWhen optimizing the optimizer, it is recommended to set the total training step size to%dthat's all.\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] The total amount of training data for this run is%d，Unfreeze_batch_sizefor%d，total training%d个Epoch，The total training step size is calculated as%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] Since the total training step size is%d，less than the recommended total step size%d，It is recommended to set the total generation to %d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    for layer in model_body.layers:
        if isinstance(layer, DepthwiseConv2D):
            layer.add_loss(l2(weight_decay)(layer.depthwise_kernel))
        elif isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.add_loss(l2(weight_decay)(layer.kernel))
    
    #------------------------------------------------------#
    # The backbone feature extraction network features are common, and freezing training can speed up training
    # Also prevents weights from being corrupted at the beginning of training.
    # Init_Epoch is the starting generation
    # Freeze_Epoch is the epoch to freeze training
    # UnFreeze_Epoch total training generation
    # Prompt OOM or insufficient video memory, please reduce the Batch_size
    #------------------------------------------------------#
    if True:
        if Freeze_Train:
            freeze_layers = 60
            for i in range(freeze_layers): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
            
        #-------------------------------------------------------------------#
        #   If you do not freeze training, directly set the batch size to unfreeze batch size
        #-------------------------------------------------------------------#
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
        
        #-------------------------------------------------------------------#
        #   Determine the current batch size and adjust the learning rate adaptively
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : Adam(lr = Init_lr_fit, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]
        model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    
        #---------------------------------------#
        #   The formula to get the learning rate drop
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step          = num_train // batch_size
        epoch_step_val      = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')

        train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

        #-------------------------------------------------------------------------------#
        # Setting of training parameters
        # logging is used to set the save address of tensorboard
        # checkpoint is used to set the details of weight saving, period is used to modify how many epochs are saved once
        # lr_scheduler is used to set the way the learning rate drops
        # early_stopping is used to set early stopping, val_loss will automatically end the training without falling for many times, indicating that the model has basically converged
        #-------------------------------------------------------------------------------#
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        logging         = TensorBoard(log_dir)
        loss_history    = LossHistory(log_dir)
        if ngpus_per_node > 1:
            checkpoint      = ParallelModelCheckpoint(model_body, os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            checkpoint_last = ParallelModelCheckpoint(model_body, os.path.join(save_dir, "last_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ParallelModelCheckpoint(model_body, os.path.join(save_dir, "best_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        else:
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        eval_callback   = EvalCallback(model_body, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, \
                                        eval_flag=eval_flag, period=eval_period)
        callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]

        if start_epoch < end_epoch:
            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
        #---------------------------------------#
        # If the model has a frozen learning part
        # Then unfreeze and set parameters
        #---------------------------------------#
        if Freeze_Train:
            batch_size  = Unfreeze_batch_size
            start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
            end_epoch   = UnFreeze_Epoch
                
            #-------------------------------------------------------------------#
            #   Determine the current batch size and adjust the learning rate adaptively
            #-------------------------------------------------------------------#
            nbs             = 64
            lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 5e-2
            lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            #---------------------------------------#
            #   The formula to get the learning rate drop
            #---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]
            
            for i in range(len(model_body.layers)): 
                model_body.layers[i].trainable = True
            model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

            train_dataloader.batch_size    = Unfreeze_batch_size
            val_dataloader.batch_size      = Unfreeze_batch_size

            print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
            model.fit_generator(
                generator           = train_dataloader,
                steps_per_epoch     = epoch_step,
                validation_data     = val_dataloader,
                validation_steps    = epoch_step_val,
                epochs              = end_epoch,
                initial_epoch       = start_epoch,
                use_multiprocessing = True if num_workers > 1 else False,
                workers             = num_workers,
                callbacks           = callbacks
            )
