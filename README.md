# ChatGraph
PyTorch implementation for paper "ChatGraph: Interpretable Text Classification by Converting ChatGPT Knowledge to Graphs"

## Download Data

[Google Drive](https://drive.google.com/drive/folders/1I07H-dcfDFIAWA9d9QuXz4gFj-NXzTwK?usp=sharing) 

## Training & Evaluation
### ChatGraph
```
sh run.sh
```
## Prompt for ChatGPT Zero/Few Shot Text Classification
```
You are a text classifier and your task is to classifiy a given text into the following categories: ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade']. You should directly output the predicted label only. You answer should be either one of ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade']. Do not output a sentence.

Good example: 
###Input###:
champion products approves stock split champion products inc said board directors approved two one stock split common shares shareholders record april company also said board voted recommend shareholders annual meeting april increase authorized capital stock five mln mln shares reuter. 
###Output###:
earn

Bad example:
###Input###:
champion products approves stock split champion products inc said board directors approved two one stock split common shares shareholders record april company also said board voted recommend shareholders annual meeting april increase authorized capital stock five mln mln shares reuter. 
###Output###:
loss
```
## GPT API
Please refer to this link: [GPT_API](https://github.com/JacksonWuxs/chatgpt_callers/blob/main/openaigpt.py)

