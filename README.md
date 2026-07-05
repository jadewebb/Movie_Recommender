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

Dataset was inspected and cleaned of duplicates, headers/prefixes (e.g., "rejected response:") and special reserved tokens. End-of-turn suffix tokens were enforced for all chosen and rejected responses for compatability with Llama 3.1

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





