import os
import pandas as pd
import tensorflow as tf
import keras
import json

from argparse import ArgumentParser

from m3cv.ConfigHandler import Config
from m3cv.pipeline.io import DataLoader, Handler
from m3cv.models.Resnet import Resnet3DBuilder
from m3cv.models.ViT import ViTBuilder
from m3cv.pipeline.evaluation.reports import classification_report

"""Script that houses primary entry point for model training runs.
"""

def run(args):
    # === Load configuration file ===
    parser = ArgumentParser()
    parser.add_argument('configpath')
    args = parser.parse_args(args)
    with open(args.configpath, 'r') as f:
        config = Config(f)
    # === Instantiate data loader off config ===
    loader = DataLoader(config)
    loader.scout_files(verbose=True)
    # === Configure data loader ===
    problem_type = config.data.preprocessing.dynamic.endpoint.type
    if problem_type == 'classification':
        loader.make_binary_labels()
    elif problem_type == 'segmentation':
        raise Exception("Segmentation support coming soon")
    else:
        raise Exception("Unrecognized problem type.")
    loader.build_encoders()
    # === Wrap data loader in handler ===
    batch_size = config.model.train_args.batch_size
    handler = Handler(loader, batch_size=batch_size)
    # === Set up splits ===
    splits_config = config.data.runtime
    handler.kfolds(
        seed=splits_config.seed,
        nfolds=splits_config.kfolds,
        testsplit=splits_config.test_fold
    )
    if splits_config.preload:
        handler.preload()
    # === Set up tf.Dataset for training data ===
    train_dataset = tf.data.Dataset.from_generator(
    handler,
    output_signature=handler.loader.output_sig
    )
    train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
    # === Build model ===
    build_args = config.model.build_args
    if build_args.fusion is True:
        fusion_plan = {
            build_args.fusion_point:handler.supp_vector_len
            }
    else:
        fusion_plan = {}

    inputshape = handler.loader.volshape
    if args.model.lower() == 'resnet-34':
        builder = Resnet3DBuilder.build_resnet_34
    elif args.model.lower() == 'resnet-18':
        builder = Resnet3DBuilder.build_resnet_18
    elif args.model.lower() == 'vit':
        builder = ViTBuilder
    else:
        raise Exception(
            "Unrecognized model argument, defaulting to ResNet-18"
            )
    model = builder(
        inputshape,
        num_outputs=1,
        fusions=fusion_plan,
        basefilters=build_args.basefilters
        )
    if build_args.optimizer == 'adam':
        optim = keras.optimizers.Adam(
            learning_rate=build_args.learning_rate
            )
    elif build_args.optimizer == 'sgd':
        optim = keras.optimizers.SGD(
            learning_rate=build_args.learning_rate
            )
    metrics = []
    for m in build_args.metrics:
        if m == 'AUC':
            metrics.append(keras.metrics.AUC(name='auc'))
        elif m == 'binary_accuracy':
            metrics.append(keras.metrics.BinaryAccuracy(name='acc'))
        elif m == 'precision':
            metrics.append(keras.metrics.Precision(name='prec'))
        elif m == 'recall':
            metrics.append(keras.metrics.Recall(name='rec'))
    model.compile(
        optimizer=optim,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics
        )
    
    train_args = config.model.train_args
    # === Set up callbacks ===
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            config.model.artifact_output,
            "model.{epoch:02d}-loss_{val_loss:.2f}-auc_{val_auc:.2f}.h5"
            ),
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True
        )
    earlystopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=train_args.early_stopping,
        restore_best_weights=True,
        verbose=1
        )
    # === Configure run settings ===
    fitargs = {
        'x':train_dataset,
        'validation_data':((handler.vXv, handler.vXnv), handler.vY),
        'steps_per_epoch':(len(handler.train) // batch_size),
        'epochs': train_args.epochs,
        'callbacks':[checkpoint, earlystopping],
        'verbose' : train_args.verbose,
        'class_weight' : {
            0: train_args.class_weight[0],
            1: train_args.class_weight[1]
            }
        }
    # === Run model ===
    history = model.fit(**fitargs)
    pd.DataFrame(data=history.history).to_csv(
        os.path.join(config.model.artifact_output,"history.csv")
        )   
    model.save(
        os.path.join(config.model.artifact_output,"full_final_model.h5")
        )


    testXvol, testXnonvol, testY = handler.bulk_load(handler.test)
    preds = model.predict((testXvol, testXnonvol))
    results = {
    'patients' : gen.test,
    'true' : np.squeeze(testY),
    'preds' : np.squeeze(preds)
    }
    pd.DataFrame(data=results).to_csv(
        os.path.join(config.model.artifact_output,"results.csv")
        )

    report = classification_report(
        config.model.evaluation, testY, preds
    )
    print(report)
    with open(
        os.path.join(config.model.artifact_output,'report.json'), 'w+'
        ) as f:
        json.dump(report, f)


if __name__ == '__main__':
    run([r'F:\repos\M3CV\src\m3cv\templates\config_test.yaml'])