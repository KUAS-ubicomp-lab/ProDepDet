# <B> ProDepDet </B>
ProDepDet is a framework which is specifically designed to use the knowledge of a pre-trained language model (PLM) in structure and semantic modelling in multi-party conversations to perform depression detection which is an unseen out-of-domain task. To our knowledge, this study is the first attempt to adapt the acquired knowledge of a PLM for out-of-domain task modelling using prompt tuning (PT)-based cross-task transferability.

# The Approach

![Prompt Transferability across Different Tasks drawio](https://github.com/KUAS-ubicomp-lab/ProDepDet/assets/4902204/7a032da5-5167-4d39-b00b-44e4059531f3)

The main contributions are: 
- A novel method is proposed for enhancing out-of-domain task transferability of PT. 
- A soft verbalizer is introduced along with a soft PT template for PT transferring for the first time. 
- Multiple downstream tasks including <I> depressed utterance classification (DUC) </I> and <I> depressed speaker identification (DSI) </I> are used to evaluate the generalization and interpretability of the novel methods. 
 
 # System Design 

![IJCNN 2024 Full Paper Design drawio](https://github.com/KUAS-ubicomp-lab/ProDepDet/assets/4902204/7a66711c-ca94-432e-981a-1d6d5671d837)

# Settings
Python 3.8 and PyTorch 2.0 were used as the main programming language and machine learning framework, respectively . We separated MPC data into three categories based on the session length such as Len-5, Len-10, and Len-15 and used three different prompt lengths <I> (l) </I> such as 25, 50, and 75. Hyper-parameters were used such as GELU activations, Adam optimizer, with learning rate 0.0005, warmup proportion 0.1, and frozen model hyper-parameters, θ1 and θ2 both True.

# Baseline Models
We adopted several pre-trained models and large language models as source frozen baselines. To evaluate <I> DUC </I>, we used WSW, BERT, RoBERTa, SA-BERT, MPC-BERT, and DisorBERT as pre-trained models. For the evaluations of <I> DSI </I>, WSW, BERT, RoBERTa, ELECTRA, SA-BERT, MDFN, and MPC-BERT were used as pre-trained models. GPT-3, ChatGPT, and GPT-4 were adopted as large language models to evaluate both <I> DUC </I> and <I> DSI </I>.

# Datasets
- Download and extract the [Reddit SDD Corpus](https://ir.cs.georgetown.edu/resources/rsdd.html)
- Download and use the [Reddit eRisk 18 T2 2018](https://link.springer.com/chapter/10.1007/978-3-319-98932-7_30)
- Download and use the [Reddit eRisk 22 T2 2022](https://books.google.co.jp/books?hl=en&lr=&id=LzaFEAAAQBAJ&oi=fnd&pg=PA231&dq=Overview+of+eRisk+2022:+Early+Risk+Prediction+on+the+Internet&ots=LnO4GFgjt7&sig=lgSXnAWqqgjiPUp-jYV3HKIv4z8&redir_esc=y#v=onepage&q=Overview%20of%20eRisk%202022%3A%20Early%20Risk%20Prediction%20on%20the%20Internet&f=false)
- Download and extract the [Twitter Depression 2022](https://www.nature.com/articles/s41599-022-01313-2)

 # Experimental Results
Evaluation results of <I> DUC </I> in terms of R10@1 which denotes the first correctly classified depressed utterances from 10 candidates. Ablation results are shown in the last two rows.

![DUC](https://github.com/KUAS-ubicomp-lab/ProDepDet/assets/4902204/29880de6-1f20-466f-9a01-0830a7fe72a6)

Evaluation results of <I> DSI </I> in terms of F1 score. Ablation results are shown in the last two rows.

![DSI](https://github.com/KUAS-ubicomp-lab/ProDepDet/assets/4902204/18e56079-4082-4a40-9a79-d96b3c32bcd7)
