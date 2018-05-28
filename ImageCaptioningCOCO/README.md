## Twin Networks for Image Captioning
PyTorch implementation for ICLR 2018 accepted paper *Twin Networks: Matching the Future for Sequence Generation* [[pdf]](https://openreview.net/pdf?id=BydLzGb0Z) for verifying the results for image captioning on MSCOCO dataset.

### Setup
This repository is compatible with python 2. </br>
- Follow instructions outlined on [PyTorch Homepage](https://pytorch.org/) for installing PyTorch (Python2).

### Data
Download the MSCOCO Dataset from [http://cocodataset.org/#download](http://cocodataset.org/#download). The 2014 version dataset and Karpathy's train-val-test split is used for this repo.

Download preprocessed captions from [[link]](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) on Karpathy's homepage. Extract ```dataset_coco.json``` from the zip file and copy it in to ```data/```.

### Preprocess Data
Please refer to [nke001/neuraltalk2.pytorch](https://github.com/nke001/neuraltalk2.pytorch) for detailed instructions for downloading and prerprocessing data. 

Run the following commands which will process the COCO images and ```data/dataset_coco.json``` and create a dataset (two feature folders, a hdf5 label file and a json file).
```
python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```
Check ```prepro_labels.py``` and ```prepro_feats.py``` for more information on the various parameters to be used while running the scripts.

The features extracted are using the ResNet architecture for which pretrained models are available at [[link]](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM).

### Train the model
To train the model, run the following command:
```
python train.py --id st --caption_model show_tell --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_st --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 25
```
The parameter ```--caption_model``` is used to specify the different type of models which can be utilised - *show and tell*, *show, attend and tell*, *show and tell twin* and *show, attend and tell twin*.

Please refer to ```opts.py``` for more information on different parameters which are to be used while training.

For evaluating metric scores during training in addition to validation cross entropy loss, use ```--language_eval 1```.

For more details, please refer to [nke001/neuraltalk2.pytorch](https://github.com/nke001/neuraltalk2.pytorch).

### Evaluation
To evaluate the model on the Karpathy's test split, run the command:
```
python eval.py --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1
```

```--language_eval 1``` can be used to compute the BLEU/CIDEr/METEOR/ROUGE_L metric scores.

The code also generates the captions for the testset images which can be stored in the path specified by ```--dump_path```.

Please refer to ```eval.py``` for more information on different parameters which are to be used while issuing the command.

### Results
The results on various metrics are specified in [Reproducibility_Report.pdf](https://github.com/ap229997/Twin-Networks-for-Sequence-Generation/blob/master/Reproducibility_Report.pdf).

Additional metric scores for variations in different parameters and model architectures are also provided in ```reproducibility_results.txt```.

The generated captions for different model architectures are stored in json files in ```eval_results```.
