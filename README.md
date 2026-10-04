# AI Projects

A collection of small machine learning and NLP notebooks: tabular regression, text sentiment classification, and a LoRA fine-tune of Llama-2. They are learning and portfolio projects, and each one is summarised below with what it does and its limits.

| Notebook | What it does | Result shown in the notebook |
|---|---|---|
| `Data Cleaning and Preprocessing.ipynb` | Cleans and explores the World Happiness Report 2019 dataset (`2019.csv`) | Exploratory only |
| `Happiness Prediction.ipynb` | Linear regression predicting the happiness `Score` from the other columns, with standardised features and an 80/20 split | MAE 0.108, MSE 0.019, R² 0.977 on the test split |
| `Sentiment_Analysis.ipynb` | Binary sentiment classifier on the IMDB movie-review dataset: tokenizer (10,000 words, 200 tokens), embedding, two bidirectional LSTM layers, trained 5 epochs | Test accuracy about 0.89 |
| `Testing_and_Validation.ipynb` | Starts a sentiment workflow on a public tweet dataset (31,962 tweets): cleaning and an 80/20 split | Unfinished, see notes |
| `Llama Model.ipynb` | Fine-tunes `NousResearch/Llama-2-7b-chat-hf` on `mlabonne/guanaco-llama2-1k` using LoRA (r=8, alpha 16, learning rate 3e-4) with 4-bit loading, then compares generated answers before and after | Qualitative samples only, no metrics |

Saved artefacts from the IMDB model: `sentiment_analysis_model.h5` and `tokenizer.pkl`.

## Notes and limitations

- **Happiness model:** the 2019 file includes an "Overall rank" column, which is derived from the score. The notebook drops only the country column, so rank is likely still a feature and the R² of 0.977 is probably inflated. Removing it is the first fix to make.
- **Testing and validation notebook:** it maps tweet labels with `{'negative': 1, 'positive': 0}`, but the labels are already numeric (0/1), so the mapping produces missing values. It ends after the train/test split without training a model.
- **Llama fine-tune:** it needs a GPU (it was run on a hosted notebook runtime) and downloads the base model from Hugging Face. No quantitative evaluation is included.
- Notebook paths (for example the local path to `2019.csv`) are from the author's machine and need adjusting to run elsewhere.

## Running

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
# Llama notebook only:
pip install transformers peft trl bitsandbytes datasets accelerate
jupyter notebook
```

The IMDB sentiment notebook needs `IMDB Dataset.csv` (the Kaggle "IMDB Dataset of 50K Movie Reviews" file) and the happiness notebooks need `2019.csv` (World Happiness Report 2019); neither is included here.
