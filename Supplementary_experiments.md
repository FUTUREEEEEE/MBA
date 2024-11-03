### The results of using T5-Large as our MBA query encoder are as follows:
**Table 1: Performance on FLAN-T5-XL (3B) as the LLM (generation model)**
Method               | Exact Match (EM) | F1 Score | Accuracy (%) | Avg Steps
---------------------|------------------|----------|--------------|-----------
No Retrieval         | 14.87           | 21.12    | 15.97        | 0.00
Adaptive Retrieval   | 23.87           | 32.24    | 26.73        | 0.50
Self-RAG             | 9.90            | 20.79    | 31.57        | 0.72
Adaptive-RAG         | 37.17           | 46.94    | 42.10        | 2.17
MBA-RAG (T5-Large)  | 38.40 | 48.38 | 43.30 | 1.95
MBA-RAG (Distill-BERT)       | **38.80**       | **48.61**| **43.57**    | 1.80

**Table 2: Results on individual Single-step Datasets with FLAN-T5-XL (3B) as the LLM**
| Data         | Methods               | SQuAD EM | SQuAD F1 | SQuAD Acc | SQuAD Step | NQ EM | NQ F1 | NQ Acc | NQ Step | TriviaQA EM | TriviaQA F1 | TriviaQA Acc | TriviaQA Step |
|--------------|-----------------------|----------|----------|-----------|------------|-------|-------|--------|---------|-------------|-------------|--------------|---------------|
| Single-step  | No Retrieval*         | 3.60     | 10.50    | 5.00      | 0.00       | 14.20 | 19.00 | 15.60  | 0.00    | 25.00       | 31.80       | 27.00        | 0.00          |
|              | Adaptive Retrieval*   | 13.40    | 23.10    | 17.60     | 0.50       | 28.20 | 36.00 | 33.00  | 0.50    | 38.40       | 46.90       | 42.60        | 0.50          |
|              | Self-RAG*             | 2.20     | 11.20    | 18.40     | 0.63       | 31.40 | 39.00 | 33.60  | 0.63    | 12.80       | 29.30       | 57.00        | 0.68          |
|              | DRAGIN†               | 18.70    | 28.70    | --        | --         | 23.20 | 33.20 | --     | --      | 54.00       | 62.30       | --           | --            |
|              | SEAKR†                | 27.10    | 36.50    | --        | --         | 25.60 | 35.50 | --     | --      | **54.40**   | **63.10**   | --           | --            |
|              | Adaptive-RAG*         | 26.80    | 38.30    | 33.00     | 1.37       | 37.80 | 47.30 | 44.60  | 1.00    | 52.20       | 60.70       | 58.20        | 1.23          |
|              | MBA-RAG (Distill-BERT)        | **27.60** | **39.10** | **33.80** | 1.11     | **37.80** | **47.50** | **44.60** | 1.23 | 53.60 | 62.40 | **60.20** | 1.06 |
|              | MBA-RAG (T5-Large) | 27.20     | 19.00     | 33.40      | 1.32       | **37.80**  | 47.30  | 44.55  | 1.10    | 53.20        | 62.3        | 60.00         | 1.00          |

**Table 3: Results on indivdual Multi-step Datasets with FLAN-T5-XL (3B) as the LLM**
| Data          | Methods               | MuSiQue EM | MuSiQue F1 | MuSiQue Acc | MuSiQue Step | HotpotQA EM | HotpotQA F1 | HotpotQA Acc | HotpotQA Step | 2WikiMultiHopQA EM | 2WikiMultiHopQA F1 | 2WikiMultiHopQA Acc | 2WikiMultiHopQA Step |
|---------------|-----------------------|------------|------------|-------------|--------------|-------------|-------------|--------------|---------------|---------------------|---------------------|----------------------|-----------------------|
| Multi-step    | No Retrieval*         | 2.40       | 10.70      | 3.20        | 0.00         | 16.60       | 22.71       | 17.20        | 0.00          | 27.40               | 32.04               | 27.80               | 0.00                 |
|               | Adaptive Retrieval*   | 6.40       | 15.80      | 8.00        | 0.50         | 23.60       | 32.22       | 25.00        | 0.50          | 33.20               | 39.44               | 34.20               | 0.50                 |
|               | Self-RAG*             | 1.60       | 8.10       | 12.00       | 0.73         | 6.80        | 17.53       | 29.60        | 0.73          | 4.60                | 19.59               | 38.80               | 0.93                 |
|               | DRAGIN†               | --         | --         | --          | --           | 23.70       | 34.20       | --           | --            | 22.40               | 30.0                | --                  | --                   |
|               | SEAKR†                | --         | --         | --          | --           | 27.90       | 39.70       | --           | --            | 30.20               | 36.0                | --                  | --                   |
|               | Adaptive-RAG*         | 23.60      | 31.80      | **26.00**   | 3.22         | **42.00**   | **53.82**   | **44.40**    | 3.55          | 40.60               | 49.75               | 46.40               | 2.63                 |
|               | MBA-RAG (Distill-BERT)        | **23.80**  | **31.90**  | 25.40       | 2.56         | 40.60       | 52.44       | 42.60        | 2.25          | **49.40**           | **58.33**           | **54.60**           | 2.57                 |
|               | MAB-RAG (T5-Large) |21.2      | 30.9       | 23.8        | 2.31         | 42.00        | 53.30       | 44.30         | 2.70          | 49.20                | 58.30               | 54.60                | 2.93                 |


These results suggest that T5-large (783M parameters) may not be the best choice for query routing due to limited data size. In this scenario, a lighter DistilBERT (67M parameters) proves more efficient. Although MAB-RAG with T5-Large achieves higher performance on complex queries, such as those from the HotpotQA dataset, MBA-RAG with DistilBERT offers a more efficient routing method.


### The results of using DistillBERT for the adaptive-RAG method are as follows:
**Table 4: Performance on FLAN-T5-XL (3B)**
Method               | Exact Match (EM) | F1 Score | Accuracy (%) | Avg Steps
---------------------|------------------|----------|--------------|-----------
No Retrieval         | 14.87           | 21.12    | 15.97        | 0.00
Adaptive Retrieval   | 23.87           | 32.24    | 26.73        | 0.50
Self-RAG             | 9.90            | 20.79    | 31.57        | 0.72
Adaptive-RAG         | 37.17           | 46.94    | 42.10        | 2.17
MBA-RAG (Ours)       | **38.80**       | **48.61**| **43.57**    | 1.80
DistilBERT-Classifier  | 34.37           | 43.80    | 38.50        | 1.69

**Table 5: Results on individual Single-step with FLAN-T5-XL (3B) as the LLM**
| Data         | Methods               | SQuAD EM | SQuAD F1 | SQuAD Acc | SQuAD Step | NQ EM | NQ F1 | NQ Acc | NQ Step | TriviaQA EM | TriviaQA F1 | TriviaQA Acc | TriviaQA Step |
|--------------|-----------------------|----------|----------|-----------|------------|-------|-------|--------|---------|-------------|-------------|--------------|---------------|
| Single-step  | No Retrieval*         | 3.60     | 10.50    | 5.00      | 0.00       | 14.20 | 19.00 | 15.60  | 0.00    | 25.00       | 31.80       | 27.00        | 0.00          |
|              | Adaptive Retrieval*   | 13.40    | 23.10    | 17.60     | 0.50       | 28.20 | 36.00 | 33.00  | 0.50    | 38.40       | 46.90       | 42.60        | 0.50          |
|              | Self-RAG*             | 2.20     | 11.20    | 18.40     | 0.63       | 31.40 | 39.00 | 33.60  | 0.63    | 12.80       | 29.30       | 57.00        | 0.68          |
|              | DRAGIN†               | 18.70    | 28.70    | --        | --         | 23.20 | 33.20 | --     | --      | 54.00       | 62.30       | --           | --            |
|              | SEAKR†                | 27.10    | 36.50    | --        | --         | 25.60 | 35.50 | --     | --      | **54.40**   | **63.10**   | --           | --            |
|              | Adaptive-RAG*         | 26.80    | 38.30    | 33.00     | 1.37       | 37.80 | 47.30 | 44.60  | 1.00    | 52.20       | 60.70       | 58.20        | 1.23          |
|              | MBA-RAG (Ours)        | **27.60** | **39.10** | **33.80** | 1.11     | **37.80** | **47.50** | **44.60** | 1.23 | 53.60 | 62.40 | **60.20** | 1.06 |
|              | DistilBert-Classifier | 22.60     | 33.90     | 28.20      | 1.57       | 34.60  | 43.00  | 39.8 0  | 1.58    | 49.60        | 57.50        | 54.80         | 1.32          |


**Table 6: Results on indivdual Multi-step with FLAN-T5-XL (3B) as the LLM**
| Data          | Methods               | MuSiQue EM | MuSiQue F1 | MuSiQue Acc | MuSiQue Step | HotpotQA EM | HotpotQA F1 | HotpotQA Acc | HotpotQA Step | 2WikiMultiHopQA EM | 2WikiMultiHopQA F1 | 2WikiMultiHopQA Acc | 2WikiMultiHopQA Step |
|---------------|-----------------------|------------|------------|-------------|--------------|-------------|-------------|--------------|---------------|---------------------|---------------------|----------------------|-----------------------|
| Multi-step    | No Retrieval*         | 2.40       | 10.70      | 3.20        | 0.00         | 16.60       | 22.71       | 17.20        | 0.00          | 27.40               | 32.04               | 27.80               | 0.00                 |
|               | Adaptive Retrieval*   | 6.40       | 15.80      | 8.00        | 0.50         | 23.60       | 32.22       | 25.00        | 0.50          | 33.20               | 39.44               | 34.20               | 0.50                 |
|               | Self-RAG*             | 1.60       | 8.10       | 12.00       | 0.73         | 6.80        | 17.53       | 29.60        | 0.73          | 4.60                | 19.59               | 38.80               | 0.93                 |
|               | DRAGIN†               | --         | --         | --          | --           | 23.70       | 34.20       | --           | --            | 22.40               | 30.0                | --                  | --                   |
|               | SEAKR†                | --         | --         | --          | --           | 27.90       | 39.70       | --           | --            | 30.20               | 36.0                | --                  | --                   |
|               | Adaptive-RAG*         | 23.60      | 31.80      | **26.00**   | 3.22         | **42.00**   | **53.82**   | **44.40**    | 3.55          | 40.60               | 49.75               | 46.40               | 2.63                 |
|               | MBA-RAG (Ours)        | **23.80**  | **31.90**  | 25.40       | 2.56         | 40.60       | 52.44       | 42.60        | 2.25          | **49.40**           | **58.33**           | **54.60**           | 2.57                 |
|               | DistilBert-Classifier | 22.2       | 30.9       | 23.8        | 2.22         | 35.8        | 47.68       | 38.0         | 1.68          | 41.4                | 49.7                | 46.4                | 1.78                 |

The experimental results show that using DistilBert as a classifier leads to a noticeable decrease in performance compared to t5-large, which is understandable given the significant difference in their parameter sizes. 
However, our MBA method achieves good results even with DistilBert, demonstrating that the reward-based approach can effectively select a more optimal retrieval path.
