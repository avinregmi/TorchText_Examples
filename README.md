# TorchText Examples
### Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install TorchText:

``` bash
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install both the English and German models or any language with:

``` bash
python -m spacy download en
python -m spacy download de
```

### Field
One of the main concepts of TorchText is the `Field`. These define how your data should be processed. In our sentiment classification task the data consists of both the raw string of the review and the sentiment, either "pos" or "neg".

The parameters of a `Field` specify how the data should be processed. We use the `TEXT` field to define how the review should be processed, and the `LABEL` field to process the sentiment. Our `TEXT` field has `tokenize='spacy'` as an argument. This defines that the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the [spaCy](https://spacy.io) tokenizer. If no `tokenize` argument is passed, the default is simply splitting the string on spaces.

`LABEL` is defined by a `LabelField`, a special subset of the `Field` class specifically used for handling labels.
```
    Field Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        tensor_type: The torch.Tensor class that represents a batch of examples
            of this kind of data. Default: torch.LongTensor.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list,
            the field's Vocab, and train (a bool).
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy English tokenizer is
            used. Default: str.split.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
```


| Name        | Description           | Use Case  |
| ------------- |:-------------:| -----:|
| Field      | A regular field that defines preprocessing and postprocessing |  Non-text fields and text fields where you don't need to map integers back to words |
| ReversibleField	 | An extension of the field that allows reverse mapping of word ids to words |Text fields if you want to map the integers back to natural language (such as in the case of language modeling) |
| NestedField | A field that takes processes non-tokenized text into a set of smaller fields |  Char-based models |
| LabelField (New!) | A regular field with sequential=False and no <unk> token. |  Label fields in text classification. |


### Dataset
The fields know what to do when given raw data. Now, we need to tell the fields what data it should work on. This is where we use Datasets. There are various built-in Datasets in torchtext that handle common data formats. For csv/tsv files, the TabularDataset class is convenient. 

| Name        | Description           | Use Case  |
| ------------- |:-------------:| -----:|
| TabularDataset      |Takes paths to csv/tsv files and json files or Python dictionaries as inputs. | Any problem that involves a label (or labels) for each piece of text |
|  LanguageModelingDataset	 |Takes the path to a text file as input. |Language modeling |
|  TranslationDataset | Takes a path and extensions to a file for each language.e.g. If the files are English: "hoge.en", French: "hoge.fr", path="hoge", exts=("en","fr") |  Translation |
|  SequenceTaggingDataset |Takes a path to a file with the input sequence and output sequence separated by tabs.|  Sequence tagging |

### DataIterator
The final step of preparing the data is creating the iterators. We iterate over these in the training/evaluation loop, and they return a batch of examples (indexed and converted into tensors) at each iteration.In torchvision and PyTorch, the processing and batching of data is handled by DataLoaders. For some reason, torchtext has renamed the objects that do the exact same thing to Iterators. The basic functionality is the same, but Iterators, as we will see, have some convenient functionality that is unique to NLP.

| Name        | Description           | Use Case  |
| ------------- |:-------------:| -----:|
| Iterator      | Iterates over the data in the order of the dataset. |  Test data, or any other data where the order is important. |
| BucketIterator | Buckets sequences of similar lengths together.	      |   Text classification, sequence tagging, etc. (use cases where the input is of variable length) |
| BPTTIterator | An iterator built especially for language modeling that also generates the input sequence delayed by one timestep. It also varies the BPTT (backpropagation through time) length. This iterator deserves its own post, so I'll omit the details here. |    Language modeling |

#### BucketIterator Attributes
```
data.BucketIterator(
    dataset,
    batch_size,
    sort_key=None,
    device=None,
    batch_size_fn=None,
    train=True,
    repeat=False,
    shuffle=None,
    sort=None,
    sort_within_batch=None,
)
```
In order to sort all tensors within a batch by their lengths, we set `sort_within_batch = True`.



### build_vocab():

**Arguments:**

            counter: collections.Counter object holding the frequencies of
                each value found in the data.
                
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
                
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
                
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ['<unk'>, '<pad>']
                
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
                
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
                
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
            
            specials_first: Whether to add special tokens into the vocabulary at first.
                If it is False, they are added into the vocabulary at last.
                Default: True.
