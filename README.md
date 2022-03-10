# cscw_21_wikipedia

The datasets folder contains the datasets for the tasks.
For each task of "citations", "point-of-view" and "clarifications", there are
two types of files:

* \*\_sentences.tsv.gz contains only positive examples for each task - sentence,
  label, reivision id, aritcle title, and section name the example was extracted
from. Note that **positive example** means that the sentence needs the specific
improvement/edit (citation, neutral point-of-view or clarification)
* \*\_data.tsv.gz contains both positive and negative examples mixed up.

 sentences for the three tasks extracted from
Wikipedia. These tasks correspond to "citations", "point-of-view", and
"clarifications". Each row of the dataset files contain a Wikipedia sentence, a
weakly extracted label for the specific task ('1' if the sentence needs the improvement, '0' otherwise), 
It may additionally contain the article title, section name, and the Wikipedia revision from which the sentence is extracted. 

Please cite the work if using the dataset

```
@article{10.1145/3479503,
author = {Asthana, Sumit and Tobar Thommel, Sabrina and Halfaker, Aaron Lee and Banovic, Nikola},
title = {Automatically Labeling Low Quality Content on Wikipedia By Leveraging Patterns in Editing Behaviors},
year = {2021},
issue_date = {October 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {5},
number = {CSCW2},
url = {https://doi.org/10.1145/3479503},
doi = {10.1145/3479503},
journal = {Proc. ACM Hum.-Comput. Interact.},
month = {oct},
articleno = {359},
numpages = {23},
keywords = {wikipedia, machine learning, content labeling}
}
```
