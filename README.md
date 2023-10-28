# thesis



## Structure

### Text.py

Embeds captions using pretrained BERT or SBERT. The datasets are assumed to be in the folders '../Datasets/UCM' and '../Datasets/RSICD' 
respectively.

CLI Arguments:

*--rsicd* - use the RSICD dataset (default: True)
*--ucm* - use the UCM captions dataset (default: False)
*--embedding* - embedding technique to use (defuault: CLS, other options: last (last layer), last_n (last 5 layers), sbert (SBERT))

### Image.py

Embeds images using pretraiend ResNet50 The datasets are assumed to be in the folders '../Datasets/UCM' and '../Datasets/RSICD' respectively.

### util.py

Utility functions. 

*closest_indices* - returns top 5 pairs of closest indices according to pairwise cosine similarity

### argument_paser.py

Helper for parsing the CLI arguments
