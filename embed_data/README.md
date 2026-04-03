Scripts for embedding the image-text datasets (COCO, CC3M, Visual Genome):
- With LLMs (Qwen 3, LLAMA-3.2, Gemma-3 families) ,
- Text Embedding Models (Codefuse, Nomic, BGE), 
- Multimodal Text-Image Models (SigLIP, OpenAI-CLIP, LAION-CLIP),
- Image Foundation Models (DINOv2, BEIT, VIT-MAE).
Given that you fix the input/output folders in the slurms scripts and cluster meta-data, you should be good to go and embed the different datasets!

Once you embed all the data, you can center and re-normalize it by running the "center_embeddings_bash.sh".
- We do recommend centering the data and doing all experimnts withe centered data. This leads to consistently higher alignment.


