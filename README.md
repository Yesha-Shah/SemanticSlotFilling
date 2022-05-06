# Semantic Slot Filling
Conversational dialogue systems like Amazon Alexa, Google Assistant, Apple Siri, and Microsoft Cortana have exploded in popularity thanks to the proliferation of mobile internet and smart gadgets. These systems rely heavily on natural language understanding (NLU). Slot filling is often considered as a sequence labeling issue, in which semantic labels are applied to contiguous sequences of words (slots). Deep learning has outperformed most standard techniques in the slot filling problem in NLU. Deep learning, on the other hand, is infamous for doing badly when given minimal labelled data.

# Statement of purpose
Currently, all the models that exist require high computation and are very complex. They require more resources and some rely on fine-tuning pretrained models such as BERT. Our goal is to achieve similar outputs even with a simple DL based model. We plan on using the same dataset as used in other pre-existing models. The dataset is SNIPS by Snips.ai

# Approach
The plan to do so is through a simple BiGRU along with segment tagging and Named Entity Recognition, followed by 2 linear layers. Made use of the dataset SNIPS - dataset by Snips.ai for Intent Detection and Slot Filling benchmarking. 
Packages needed:
* pytorch
* pandas
* numpy
* matplotlib

# Running the code
Simply use the .ipynb notebook to run the entire code - or use the .pt model for testing (the .pt is a pretrained model)

# Evaluation 
The following metrics are calculated on the test data:
•	Accuracy – NumberOfExcatMatches / TotalNumberOfPredictions
•	Missing Slot – predicted slot doesn’t exactly match actual slot
•	Spurious Slot – label of predicted slot doesn’t match actual labels
•	Wrong Boundary – predicted label is a substring of actual (or vice versa)
•	Wrong Label – the label predicted is wrong but the slot matches
| Metric  | Value in Evaluation |
| ------  | ------------------- |
| Accuracy | 96.75% |
| Missing slots | 108 |
| Spurious slots | 271 | 
| Wrong boundary | 49 |
| Wrong label | 1 |
| Total predictions | 3197 |

# Comparision with existing works
| Model name / type |	Accuracy |
| ----------------  | -------  |
| SlotRefine + BERT	| 99.05% |
| SlotRefine	| 97.44% |
| Stack-Propagation + BERT | 99% |
| Stack-Propagation	| 98% |
| SF-ID (BLSTM) network	| 97.43% |
| Capsule-NLU	| 97.70% |
| Slot-Gated BLSTM with Attention	| 97.00% |
| BiGRU (Our approach)	| 96.75% |
