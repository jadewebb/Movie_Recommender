# Movie Recommender

Performing DPO to align QLoRA-configured Llama 3.1 for accurate movie recommendations based on user preferences 

## Dataset

**DistillRecDial**: diverse multi-turn conversational movie recommendation dataset formatted according to the Llama 3.1 chat template
   
   - Source: https://huggingface.co/datasets/swap-uniba/DistillRecDial
   
   - Original Paper: A. F. M. Martina, A. Petruzzelli, C. Musto, M. de Gemmis, P. Lops, and G. Semeraro, "DistillRecDial: A knowledge-distilled dataset capturing user diversity in conversational recommendation," in Proc. 19th ACM Conf. Recommender Syst. (RecSys 2025), Sep. 2025, pp. 726-735, doi: 10.1145/3705328.3748161
   
   - 21039 samples of multi-turn conversations between a user and an assistant that involve thoughtful dialogue, clarifying questions, and movie recommendations based on user preferences

## Preprocessing

### Augmentation

Subset of 500 dataset samples were randomly selected for tuning efficiency

Samples were preprocessed and augmented to fit the DPO pairwise preference structure. **Preprocessing** was performed by splitting conversations at the recommendation turn. **Augmentation** was performed by prompting **GPT-4o-mini** to generate rejected recommendations for each conversation

   - Prompt: Entire conversation context up until the active recommendation turn
   - Chosen: Assistant's recommendation turn, accurate to user preferences
   - Rejected: Augmented recommendation, inaccurate to user preferences

### Cleaning

Dataset was inspected and cleaned of duplicates, headers/prefixes (e.g., "rejected response:") and special reserved tokens. End-of-turn suffix tokens were enforced for all chosen and rejected responses to maintain Llama 3.1 structure

Final dataset was saved as a JSON file in the format:

```
    [
        {
            "prompt": "...",
            "chosen": "...",
            "rejected": "..."
        },
        ...
    ]
```

## Llama 3.1 DPO using QLoRA

Dataset was shuffled and split into training and validation sets (90:10)

**Llama 3.1 8B Instruct** tokenizer and pretrained model were loaded with 4-bit normal float quantization using transformers. Two Llama 3.1 instances were loaded to perform DPO, one as the **base** (policy) model and another as the static **reference** model to prevent drift

QLoRA PEFT configuration and DPO trainer hyperparameters were tuned for optimal fine-tuning efficiency, memory usage, and metric accuracy

Fine-tuning was performed for 1 epoch to prevent overfitting, with the final metrics being:

Step | Training Loss | Validation Loss | Entropy | Logits/chosen | Logits/rejected | Mean Token Accuracy | Rewards/chosen | Rewards/rejected | Rewards/accuracies | Rewards/margins | LogPs/chosen | LogPs/rejected
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
63 | 0.126721 | 0.12113 | 1.393301 | -0.944293 | -1.233962 | 0.661907 | 0.109644 | -2.141406 | 1 | 2.251049 | -128.649515 | -240.750747

Visualizations were created to gain the following insights:

   - Cross-entropy loss minimization
   - Gradient stability
   - Learning rate decay
   - Alignment preference reward margin growth
   - Predictive vocabilary entropy compression
   - Policy divergence
   - Implicit reward shifts
   - Training policy log-probability suppression

<p align="center">
<img src="https://raw.githubusercontent.com/jadewebb/Movie_Recommender/main/DPO_metrics.png" width="1000"></p>
