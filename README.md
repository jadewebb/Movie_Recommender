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

Samples were preprocessed and augmented to fit the **DPO pairwise preference structure**

   - Prompt: Entire conversation context up until the active recommendation turn
   - Chosen: Assistant's recommendation turn, accurate to user preferences
   - Rejected: Augmented recommendation, inaccurate to user preferences

**Preprocessing** was performed by splitting conversations at the recommendation turn

**Augmentation** was performed by prompting **GPT-4o-mini** to generate rejected recommendations for each conversation

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

**QLoRA PEFT configuration** and **DPO trainer** hyperparameters were tuned for optimal fine-tuning efficiency, memory usage, and metric accuracy

Fine-tuning was performed for 1 epoch to prevent overfitting, with the final metrics being:

Step | Training Loss | Validation Loss | Entropy | Logits/chosen | Logits/rejected | Mean Token Accuracy | Rewards/chosen | Rewards/rejected | Rewards/accuracies | Rewards/margins | LogPs/chosen | LogPs/rejected
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
63 | 0.126721 | 0.12113 | 1.393301 | -0.944293 | -1.233962 | 0.661907 | 0.109644 | -2.141406 | 1 | 2.251049 | -128.649515 | -240.750747

Visualizations were created to gain the following insights:

   - **Cross-Entropy Loss Minimization**: The training and validation losses decrease together, demonstrating ideal optimization trajectory and proper alignment generalization without overfitting
   - **Gradient Stability**: The smooth L2 gradient norm decay verifies strong training stability
   - **Cosine Learning Rate Decay Schedule**: The learning rate follows the expected cosine decay trajectory, ensuring stable convergence
   - **Alignment Preference Reward Margin Growth**: The parallel growth of the training and validation reward margins demonstrates that the DPO objective successfully widened the behavioral gap between chosen and rejected responses
   - **Predictive Vocabulary Entropy Compression**: The Shannon entropy remains bounded, demonstrating that the model avoided mode collapse and preserved conversational diversity
   - **Policy Divergence**: The simultaneous growth of the training and validation delta log-probabilities demonstrates that the policy shifted its probability mass to learn broad structural rules that differentiate chosen and rejected responses
   - **Implicit Reward Shifts**: For both training and validation, the chosen sequences achieved sustained positive rewards while the rejected sequences were increasingly penalized
   - **Training Policy Log-Probability Suppression**: The model maintained a steady sequence log-probability for chosen responses while suppressing the likelihood of rejected responses

<p align="center">
<img src="https://raw.githubusercontent.com/jadewebb/Movie_Recommender/main/DPO_metrics.png" width="1200"></p>

## Simple Evaluation

The final model was evaluated by feeding it a novel test conversation and analyzing the recommender output. Performance demonstrates strong signals of the **conversational tone and length** established by the dataset, as well as the **strict adherence to multiple user preferences** expressed across two conversational turns. Further experimentation is needed to develop an interactive movie recommender system, rather than a static model that focuses only on the recommendation step.
