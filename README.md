# This is the repository for the master thesis of Mathis Jürgen Arend, at the RSiM group of Technische Universität Berlin, titled Multi-Query, Cross-Modal and Scalable Content-Based Image Retrieval in Remote Sensing. It contains all the necessary code to reproduce the experiments described therein.


Supervisor: Prof. Begüm Demir

Advisors: Gencer Sümbül, Leonard Hackel



## Structure

### Text.py

Embeds captions using pretrained BERT or SBERT OR SimCLR strategy (toggle by bool). The datasets are assumed to be in the folders '../Datasets/UCM' and '../Datasets/RSICD' 
respectively.

CLI Arguments:


**--rsicd** - use the RSICD dataset (default: True)

**--ucm** - use the UCM captions dataset (default: False)

**--embedding** - embedding technique to use (defuault: CLS, other options: last (last layer), last_n (last 5 layers), sbert (SBERT))


The data modules for simclr and non simclr can be found in /data/text (simclr_data_module, data_module)

The models for simclr and non simclr can be found in /model (simclr_text_model, text_embedding)

### Image.py

Embeds images using pretraiend ResNet50 OR SimCLR strategy (toggle by bool) The datasets are assumed to be in the folders '../Datasets/UCM' and '../Datasets/RSICD' respectively.

The data modules for simclr and non simclr can be found in /data/image (simclr_data_module, data_module)

The models for simclr and non simclr can be found in /model (simclr_model, image_embedding)

### /utility/helpers.py

Helper functions. 

*closest_indices* - returns top 5 pairs of closest indices according to pairwise cosine similarity

### /utility/argument_paser.py

Helper for parsing the CLI arguments


### main.py

Complete pipeline. 

Current Goal: Use pretrained BERT/SBERT for text embedding + pretrained Resnet for image embedding. In a batch use NT-Xent loss for learning. A batch contains pairs of image with caption like this:

Image-Caption
Image-Caption
...

The agreement between image and fitting caption will be maximized and to other images/ captions minimized. When there are multiple captions per image we either chose only one or multiply the image like this:

Image1-(caption1, caption2, caption3) --> Image1-caption1, Image1-caption2, Image1-caption3


The full pipeline is not fully functional yet!

The model can be found in /model (full_pipeline). This uses the respective data modules and models for text and image, that are either trained (simclr) or only used for embedding (non-simclr).
