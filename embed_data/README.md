Scripts for embedding the image-text datasets (COCO, CC3M, Visual Genome) with LLMs (Qwen 3, LLAMA-3.2, Gemma-3 families), Text Embedding Models (Codefuse, Nomic, BGE), 
Multimodal Text-Image Models (SigLIP, OpenAI-CLIP, LAION-CLIP), Image Foundation Models (DINOv2, BEIT, VIT-MAE).

Once you embed all the data, you can normalize it by running the "center_embeddings_bash.sh"

Given that you fix the input/output folders in the slurms scripts and cluster meta-data, you should be good to go and embed the different datasets!
